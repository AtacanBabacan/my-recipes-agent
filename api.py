# api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import duckdb

from retriever import top_k
from answerer import answer

DB_PATH = "./data/recipes.duckdb"

app = FastAPI(title="My Recipes Agent", version="0.1.0")


class SearchRequest(BaseModel):
    question: str
    k: int = 5


class AskRequest(BaseModel):
    question: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/search")
def api_search(req: SearchRequest):
    return {"results": top_k(req.question, k=req.k)}


@app.get("/api/recipe/{recipe_id}")
def api_recipe(recipe_id: str):
    con = duckdb.connect(DB_PATH)
    row = con.execute(
        """
        SELECT id, title, cuisine, meal_type, prep_time_minutes, cook_time_minutes,
               total_time_minutes, servings, dietary_tags, tags, personal_notes, source
        FROM recipes WHERE id = ?
        """,
        [recipe_id],
    ).fetchone()

    if not row:
        con.close()
        raise HTTPException(status_code=404, detail="Recipe not found")

    ingredients = con.execute(
        """
        SELECT line_no, quantity, unit, item
        FROM ingredients
        WHERE recipe_id = ?
        ORDER BY line_no
        """,
        [recipe_id],
    ).fetchall()

    instructions = con.execute(
        """
        SELECT step_no, instruction
        FROM instructions
        WHERE recipe_id = ?
        ORDER BY step_no
        """,
        [recipe_id],
    ).fetchall()

    con.close()

    return {
        "id": row[0],
        "title": row[1],
        "cuisine": row[2],
        "meal_type": row[3],
        "prep_time_minutes": row[4],
        "cook_time_minutes": row[5],
        "total_time_minutes": row[6],
        "servings": row[7],
        "dietary_tags": row[8],
        "tags": row[9],
        "personal_notes": row[10],
        "source": row[11],
        "ingredients": [
            {"line_no": i[0], "quantity": i[1], "unit": i[2], "item": i[3]} for i in ingredients
        ],
        "instructions": [{"step_no": s[0], "instruction": s[1]} for s in instructions],
    }


@app.post("/api/ask")
def api_ask(req: AskRequest):
    return answer(req.question, k=req.k)


@app.get("/", response_class=HTMLResponse)
def home():
    # Minimal UI so you can use it from phone/laptop.
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>My Recipes Agent</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 12px; }
    input, button { font-size: 16px; padding: 10px; }
    input { width: 72%; }
    button { width: 24%; margin-left: 2%; }
    pre { background: #f6f6f6; padding: 12px; border-radius: 8px; overflow:auto; }
    .row { display:flex; gap: 8px; }
  </style>
</head>
<body>
  <h2>My Recipes Agent</h2>
  <p>Ask a question and get grounded answers + citations.</p>

  <div class="row">
    <input id="q" placeholder="e.g., Vegetarian recipes under 30 minutes, no mushrooms" />
    <button onclick="ask()">Ask</button>
  </div>

  <h3>Answer</h3>
  <pre id="answer">—</pre>

  <h3>Retrieved</h3>
  <pre id="retrieved">—</pre>

<script>
async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q) return;

  document.getElementById('answer').textContent = "Thinking...";
  document.getElementById('retrieved').textContent = "Loading...";

  const res = await fetch('/api/ask', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({question:q, k:5})
  });

  const data = await res.json();
  document.getElementById('answer').textContent = data.answer || JSON.stringify(data, null, 2);
  document.getElementById('retrieved').textContent = JSON.stringify(data.retrieved || data, null, 2);
}
</script>
</body>
</html>
"""