import os
import requests

VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]
VOYAGE_MODEL = os.environ.get("VOYAGE_EMBED_MODEL")

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Returns embeddings for a batch of texts.
    """
    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": VOYAGE_MODEL,
        "input": texts,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    # Voyage returns embeddings in data[i].embedding
    return [item["embedding"] for item in data["data"]]