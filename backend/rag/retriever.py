from __future__ import annotations
import os
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


class RAGRetriever:

	def __init__(self) -> None:
		persist_dir = os.getenv("CHROMA_DB_DIR", os.path.join("data", "chroma"))
		# Use sentence-transformers for embeddings
		self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
		self._client = chromadb.PersistentClient(path=persist_dir)

	def _collection(self, namespace: str):
		name = f"kb_{namespace}"
		try:
			return self._client.get_collection(name=name)
		except Exception:
			# Collection doesn't exist, return empty results
			return None

	def search(self, query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
		col = self._collection(namespace)
		if col is None or col.count() == 0:
			return []
		
		# Generate query embedding
		query_embedding = self._embed_model.encode([query]).tolist()[0]
		
		res = col.query(query_embeddings=[query_embedding], n_results=max(1, top_k), include=["metadatas", "documents", "distances"])
		results: List[Dict[str, Any]] = []
		ids = res.get("ids", [[]])[0]
		docs = res.get("documents", [[]])[0]
		metas = res.get("metadatas", [[]])[0]
		dists = res.get("distances", [[]])[0]
		for i in range(len(ids)):
			meta = metas[i] if i < len(metas) else {}
			results.append({
				"id": ids[i],
				"text": docs[i] if i < len(docs) else "",
				"score": float(1.0 / (1.0 + (dists[i] if i < len(dists) else 0.0))),
				"source": meta.get("source") or meta.get("file_name") or namespace,
			})
		return results


