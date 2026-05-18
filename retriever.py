import json
import re
import duckdb
import numpy as np
from clients.voyage_client import embed_texts

DB_PATH = "./data/recipes.duckdb"
EMB_PATH = "./data/embeddings.npy"
IDS_PATH = "./data/embedding_ids.json"

# Load once at import (fine for <1k)
EMB_MAT = np.load(EMB_PATH)  # normalized vectors
with open(IDS_PATH, "r", encoding="utf-8") as f:
    IDX_TO_ID = json.load(f)
ID_TO_IDX = {rid: i for i, rid in enumerate(IDX_TO_ID)}

def parse_filters(question: str) -> dict:
    q = question.lower()
    filters = {
        "max_total_time_minutes": None,
        "dietary_tags": [],
        "cuisine": None,
        "meal_type": None,
        "include_ingredients": [],
        "exclude_ingredients": []
    }

    # time: "under 30 minutes", "less than 45 min"
    m = re.search(r"(under|less than)\s+(\d+)\s*(minutes|min)\b", q)
    if m:
        filters["max_total_time_minutes"] = int(m.group(2))

    # dietary (extend as you like)
    for tag in ["vegetarian", "vegan", "gluten free", "gluten-free", "keto"]:
        if tag in q:
            filters["dietary_tags"].append(tag.replace("gluten-free", "gluten free"))

    # simple cuisine detection (add more)
    for c in ["italian", "mexican", "indian", "thai", "french", "turkish"]:
        if c in q:
            filters["cuisine"] = c.capitalize()

    # meal type (add more)
    for mt in ["breakfast", "lunch", "dinner", "snack"]:
        if mt in q:
            filters["meal_type"] = mt

    # exclusions: "no mushrooms", "without garlic"
    m2 = re.findall(r"(no|without)\s+([a-zA-Z ]+)", q)
    for _, item in m2:
        filters["exclude_ingredients"].append(item.strip())

    return filters

def sql_candidates(filters: dict) -> list[str]:
    con = duckdb.connect(DB_PATH)

    where = []
    params = []

    if filters["max_total_time_minutes"] is not None:
        where.append("total_time_minutes <= ?")
        params.append(filters["max_total_time_minutes"])

    for tag in filters["dietary_tags"]:
        where.append("list_contains(dietary_tags, ?)")
        params.append(tag)

    if filters["cuisine"]:
        where.append("lower(cuisine) = lower(?)")
        params.append(filters["cuisine"])

    if filters["meal_type"]:
        where.append("lower(meal_type) = lower(?)")
        params.append(filters["meal_type"])

    # Exclude ingredients (simple: check ingredient items table)
    # For <1k, do it as NOT EXISTS subqueries
    for ex in filters["exclude_ingredients"]:
        where.append("""
            NOT EXISTS (
              SELECT 1 FROM ingredients i
              WHERE i.recipe_id = recipes.id
                AND lower(i.item) LIKE lower(?)
            )
        """)
        params.append(f"%{ex}%")

    sql = "SELECT id FROM recipes"
    if where:
        sql += " WHERE " + " AND ".join(where)

    ids = [r[0] for r in con.execute(sql, params).fetchall()]
    con.close()
    return ids

def embed_query(question: str) -> np.ndarray:
    qv = np.array(embed_texts([question])[0], dtype=np.float32)
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    return qv

def top_k(question: str, k: int = 5) -> list[dict]:
    filters = parse_filters(question)
    candidates = sql_candidates(filters)

    # If filters are too strict and return nothing, fallback to all recipes
    if not candidates:
        candidates = IDX_TO_ID

    qv = embed_query(question)
    sims = EMB_MAT @ qv  # cosine similarity because both normalized

    # rank only candidates
    cand_idxs = [ID_TO_IDX[rid] for rid in candidates if rid in ID_TO_IDX]
    scored = [(IDX_TO_ID[i], float(sims[i])) for i in cand_idxs]
    scored.sort(key=lambda x: x[1], reverse=True)
    best = scored[:k]

    # Hydrate recipe info
    con = duckdb.connect(DB_PATH)
    results = []
    for rid, score in best:
        row = con.execute("""
            SELECT id, title, cuisine, meal_type, total_time_minutes, dietary_tags, tags
            FROM recipes WHERE id = ?
        """, [rid]).fetchone()
        results.append({
            "recipe_id": row[0],
            "title": row[1],
            "cuisine": row[2],
            "meal_type": row[3],
            "total_time_minutes": row[4],
            "dietary_tags": row[5],
            "tags": row[6],
            "score": score,
            "filters": filters
        })
    con.close()
    return results