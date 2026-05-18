# mcp_server.py
import os, sys, logging
import requests
from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)

BASE_URL = os.environ.get("AGENT_API_BASE", "http://localhost:8000")
session = requests.Session()

mcp = FastMCP("my-recipes-agent")


def _post(path: str, payload: dict):
    url = f"{BASE_URL}{path}"
    r = session.post(url, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()


def _get(path: str):
    url = f"{BASE_URL}{path}"
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


@mcp.tool()
def search_recipes(question: str, k: int = 5) -> dict:
    """Search recipes using hybrid retrieval (SQL filters + cosine similarity)."""
    return _post("/api/search", {"question": question, "k": k})


@mcp.tool()
def get_recipe(recipe_id: str) -> dict:
    """Fetch full recipe details by id."""
    return _get(f"/api/recipe/{recipe_id}")


@mcp.tool()
def ask_recipes(question: str, k: int = 5) -> dict:
    """Ask a question; returns grounded answer + citations + retrieved items."""
    return _post("/api/ask", {"question": question, "k": k})


if __name__ == "__main__":
    mcp.run()