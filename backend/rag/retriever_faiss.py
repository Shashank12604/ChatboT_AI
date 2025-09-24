from __future__ import annotations
import os
import pickle
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self) -> None:
        self.persist_dir = os.getenv("CHROMA_DB_DIR", os.path.join("data", "chroma"))
        self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._indices = {}
        self._metadata = {}
        self._load_data()

    def _load_data(self):
        """Load FAISS indices and metadata for each namespace"""
        for namespace in ["nec", "wattmonk"]:
            index_path = os.path.join(self.persist_dir, f"{namespace}_index.faiss")
            meta_path = os.path.join(self.persist_dir, f"{namespace}_metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(meta_path):
                self._indices[namespace] = faiss.read_index(index_path)
                with open(meta_path, 'rb') as f:
                    self._metadata[namespace] = pickle.load(f)
            else:
                self._indices[namespace] = None
                self._metadata[namespace] = []

    def search(self, query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if namespace not in self._indices or self._indices[namespace] is None:
            return []
        
        # Generate query embedding
        query_embedding = self._embed_model.encode([query])
        
        # Search FAISS index
        scores, indices = self._indices[namespace].search(query_embedding, min(top_k, len(self._metadata[namespace])))
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self._metadata[namespace]):
                meta = self._metadata[namespace][idx]
                results.append({
                    "id": str(idx),
                    "text": meta.get("text", ""),
                    "score": float(1.0 / (1.0 + score)),
                    "source": meta.get("source", namespace),
                })
        
        return results
