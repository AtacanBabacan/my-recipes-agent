import os
import requests

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_MODEL = os.environ["OPENROUTER_MODEL"]

def chat(messages: list[dict], max_tokens: int = 500, temperature: float = 0.2) -> str:
    """
    messages: [{"role":"system|user|assistant","content":"..."}]
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]