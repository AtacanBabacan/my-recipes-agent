import os
import time
import requests

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY")
VOYAGE_MODEL = os.environ.get("VOYAGE_EMBED_MODEL")

# Free tier: 3 requests/min -> ~20s per request
RPM = int(os.environ.get("VOYAGE_RPM"))
MIN_INTERVAL_SEC = 60.0 / RPM + 1.0  # add 1s buffer

_last_call_ts = 0.0
_session = requests.Session()

def _throttle():
    global _last_call_ts
    now = time.time()
    wait = (_last_call_ts + MIN_INTERVAL_SEC) - now
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()

def embed_texts(texts: list[str], max_retries: int = 6) -> list[list[float]]:
    """
    Batch-embed texts using Voyage, respecting RPM and retrying on 429.
    """
    url = "https://api.voyageai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": VOYAGE_MODEL, "input": texts}

    backoff = 2.0
    for attempt in range(max_retries):
        _throttle()
        r = _session.post(url, headers=headers, json=payload, timeout=90)

        if r.status_code == 429:
            # Prefer server-provided Retry-After if present
            retry_after = r.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else max(MIN_INTERVAL_SEC, backoff)
            time.sleep(sleep_s)
            backoff = min(backoff * 2, 60.0)
            continue

        if r.status_code >= 500:
            # transient server error
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue

        r.raise_for_status()
        data = r.json()
        return [item["embedding"] for item in data["data"]]

    raise RuntimeError("Voyage embeddings failed after retries (rate limited or server errors).")