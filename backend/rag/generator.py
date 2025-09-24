from __future__ import annotations
import os
from typing import List,Dict

from openai import OpenAI


_client: OpenAI | None = None


def _get_client() -> OpenAI:
	global _client
	if _client is None:
		_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
	return _client


SYSTEM_PROMPT = (
	"You are a helpful assistant. If provided with context snippets, use them to answer. "
	"Always be accurate and concise. If the answer is not in the provided context, say you "
	"don't have enough information and provide a helpful general answer if possible."
)


def generate_answer(query: str, contexts: List[str], conversation_history: List[Dict] = None) -> str:
    try:
        client = _get_client()
        
        # Build messages with conversation history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history (last 4-6 exchanges)
        if conversation_history:
            # Take last 6 messages to maintain context but avoid token limits
            recent_history = conversation_history[-6:]
            for msg in recent_history[:-1]:  # All except the current query
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current query with context
        context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts[:8])) if contexts else ""
        
        current_content = (
            ("Context:\n" + context_block + "\n\n" if context_block else "") +
            f"Question: {query}"
        )
        
        messages.append({"role": "user", "content": current_content})
        
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=1000  # Add token limit
        )
        
        return resp.choices[0].message.content or ""
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"
