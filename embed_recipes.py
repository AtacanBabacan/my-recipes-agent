import os, json
import duckdb
import numpy as np
from dotenv import load_dotenv
from clients.voyage_client import embed_texts

load_dotenv()

DB_PATH = "./data/recipes.duckdb"
OUT_EMB = "./data/embeddings.npy"
OUT_IDS = "./data/embedding_ids.json"

BATCH_SIZE = 64  # safe default; adjust if you want

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def main():
    con = duckdb.connect(DB_PATH)

    rows = con.execute("""
        SELECT id, text_for_embedding
        FROM recipes
        ORDER BY id
    """).fetchall()

    ids = [r[0] for r in rows]
    texts = [r[1] or "" for r in rows]

    all_embs: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        embs = embed_texts(batch)
        all_embs.extend(embs)
        print(f"Embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}")

    mat = np.array(all_embs, dtype=np.float32)
    mat = l2_normalize(mat)

    np.save(OUT_EMB, mat)
    with open(OUT_IDS, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print("Saved:", OUT_EMB, "shape=", mat.shape)
    print("Saved:", OUT_IDS, "count=", len(ids))

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    main()