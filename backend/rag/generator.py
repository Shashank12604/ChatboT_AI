from __future__ import annotations
import os
from typing import List

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


def generate_answer(query: str, contexts: List[str]) -> str:
	try:
		client = _get_client()
		context_block = "\n\n".join(f"[Context {i+1}]\n{c}" for i, c in enumerate(contexts[:8])) if contexts else ""
		user_prompt = (
			("Context:\n" + context_block + "\n\n" if context_block else "") +
			f"Question: {query}\n\n" 
			"Answer:"
		)
		model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
		resp = client.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": SYSTEM_PROMPT},
				{"role": "user", "content": user_prompt},
			],
			temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
		)
		return resp.choices[0].message.content or ""
	except Exception as e:
		return f"Error generating answer: {str(e)}"


