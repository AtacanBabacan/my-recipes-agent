import os, json
import duckdb
import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_KEY = os.environ["S3_KEY"]

DB_PATH = "./data/recipes.duckdb"
LOCAL_JSONL = "./data/recipes.jsonl"

os.makedirs("./data", exist_ok=True)

def download_jsonl():
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_KEY, LOCAL_JSONL)

def create_schema(con: duckdb.DuckDBPyConnection):
    con.execute("""
    CREATE TABLE IF NOT EXISTS recipes (
        id TEXT PRIMARY KEY,
        title TEXT,
        cuisine TEXT,
        meal_type TEXT,
        prep_time_minutes INTEGER,
        cook_time_minutes INTEGER,
        total_time_minutes INTEGER,
        servings INTEGER,
        dietary_tags TEXT[],
        tags TEXT[],
        source TEXT,
        personal_notes TEXT,
        text_for_embedding TEXT
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS ingredients (
        recipe_id TEXT,
        line_no INTEGER,
        quantity DOUBLE,
        unit TEXT,
        item TEXT,
        PRIMARY KEY (recipe_id, line_no)
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS instructions (
        recipe_id TEXT,
        step_no INTEGER,
        instruction TEXT,
        PRIMARY KEY (recipe_id, step_no)
    );
    """)

    # Helpful indexes (small data, but good practice)
    con.execute("CREATE INDEX IF NOT EXISTS idx_recipes_time ON recipes(total_time_minutes);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_recipes_cuisine ON recipes(cuisine);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_recipes_mealtype ON recipes(meal_type);")

def upsert_data(con: duckdb.DuckDBPyConnection):
    # Start fresh each run (simplest for <1k recipes)
    con.execute("DELETE FROM ingredients;")
    con.execute("DELETE FROM instructions;")
    con.execute("DELETE FROM recipes;")

    with open(LOCAL_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)

            con.execute("""
                INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                r.get("id"),
                r.get("title"),
                r.get("cuisine"),
                r.get("meal_type"),
                int(r.get("prep_time_minutes") or 0),
                int(r.get("cook_time_minutes") or 0),
                int(r.get("total_time_minutes") or 0),
                int(r.get("servings") or 0),
                r.get("dietary_tags") or [],
                r.get("tags") or [],
                r.get("source") or "",
                r.get("personal_notes") or "",
                r.get("text_for_embedding") or ""
            ])

            for i, ing in enumerate(r.get("ingredients") or []):
                con.execute("""
                    INSERT INTO ingredients VALUES (?, ?, ?, ?, ?)
                """, [
                    r["id"],
                    i,
                    float(ing.get("quantity") or 0),
                    ing.get("unit") or "",
                    ing.get("item") or ""
                ])

            for s, step in enumerate(r.get("instructions") or []):
                con.execute("""
                    INSERT INTO instructions VALUES (?, ?, ?)
                """, [r["id"], s, step])

def quick_test(con: duckdb.DuckDBPyConnection):
    # Example: vegetarian under 30 minutes
    rows = con.execute("""
        SELECT id, title, total_time_minutes
        FROM recipes
        WHERE total_time_minutes < 30
          AND list_contains(dietary_tags, 'vegetarian')
        ORDER BY total_time_minutes ASC
        LIMIT 5;
    """).fetchall()
    print("Sample query results:", rows)

def main():
    download_jsonl()
    con = duckdb.connect(DB_PATH)
    create_schema(con)
    upsert_data(con)
    quick_test(con)
    con.close()
    print("Ingestion complete:", DB_PATH)

if __name__ == "__main__":
    main()