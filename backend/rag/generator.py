# backend/rag/generator.py
from __future__ import annotations
import os, time
from typing import List, Dict
from openai import OpenAI, RateLimitError

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        base_url = os.getenv("OPENAI_BASE_URL")  # e.g., http://localhost:8000/v1
        api_key = os.getenv("OPENAI_API_KEY", "sk-dummy")
        # Pass base_url when using OpenAI-compatible local servers
        _client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    return _client

SYSTEM_PROMPT = (
    "You are a helpful assistant. If provided with context snippets, use them to answer. "
    "Always be accurate and concise. If the answer is not in the provided context, say you "
    "don't have enough information and provide a helpful general answer if possible."
)

def generate_answer(query: str, contexts: List[str], conversation_history: List[Dict] = None) -> str:
    try:
        client = _get_client()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            recent_history = conversation_history[-6:]
            for msg in recent_history[:-1]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts[:8])) if contexts else ""
        current_content = (("Context:\n" + context_block + "\n\n") if context_block else "") + f"Question: {query}"
        messages.append({"role": "user", "content": current_content})

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))

        # Simple exponential backoff for transient 429s
        attempts, delay = 4, 2
        for attempt in range(attempts):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1000
                )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                if attempt == attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    except Exception as e:
        return f"Error generating answer: {str(e)}"
