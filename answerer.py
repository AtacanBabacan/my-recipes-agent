import duckdb
from clients.openrouter_client import chat
from retriever import top_k, DB_PATH

def answer(question: str, k: int = 5) -> dict:
    hits = top_k(question, k=k)

    # Pull full recipe details for grounding
    con = duckdb.connect(DB_PATH)
    contexts = []
    citations = []
    for h in hits:
        rid = h["recipe_id"]
        recipe = con.execute("""
            SELECT id, title, cuisine, meal_type, total_time_minutes, servings, dietary_tags, tags, text_for_embedding
            FROM recipes WHERE id = ?
        """, [rid]).fetchone()
        contexts.append(f"[{recipe[0]}] {recipe[1]} | {recipe[2]} | {recipe[3]} | total_time={recipe[4]} mins | servings={recipe[5]} | dietary={recipe[6]} | tags={recipe[7]}\n{recipe[8]}")
        citations.append(rid)
    con.close()

    system = (
        "You are a cooking assistant. Answer ONLY using the provided recipes context. "
        "If you are missing info, say so. Always cite recipes by their [recipe_id]."
    )

    user = f"Question: {question}\n\nRecipes context:\n\n" + "\n\n---\n\n".join(contexts)

    response = chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=450,
        temperature=0.2
    )

    return {"answer": response, "citations": citations, "retrieved": hits}