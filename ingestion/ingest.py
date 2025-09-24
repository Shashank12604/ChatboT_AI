from __future__ import annotations
import os
import uuid
from typing import List, Tuple

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()


def read_pdf_text(path: str) -> str:
	reader = PdfReader(path)
	texts: List[str] = []
	for page in reader.pages:
		try:
			texts.append(page.extract_text() or "")
		except Exception:
			texts.append("")
	return "\n".join(texts)


def chunk_text(text: str, chunk_chars: int = 1200, overlap: int = 150) -> List[str]:
	text = text.replace("\r", "\n")
	spans: List[str] = []
	start = 0
	while start < len(text):
		end = min(len(text), start + chunk_chars)
		spans.append(text[start:end].strip())
		start = end - overlap
		if start < 0:
			start = 0
	return [s for s in spans if s]


def ensure_collection(client: chromadb.PersistentClient, name: str, embed_model: str) -> chromadb.Collection:
	# Use sentence-transformers for embeddings
	sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
	return client.get_or_create_collection(name=name)


def ingest_dir(root: str, namespace: str, embed_model: str = "text-embedding-3-small") -> Tuple[int, int]:
	persist_dir = os.getenv("CHROMA_DB_DIR", os.path.join("data", "chroma"))
	client = chromadb.PersistentClient(path=persist_dir)
	collection = ensure_collection(client, f"kb_{namespace}", embed_model)

	added_chunks = 0
	files = []
	for dirpath, _, filenames in os.walk(root):
		for fn in filenames:
			if fn.lower().endswith((".pdf",)):
				files.append(os.path.join(dirpath, fn))

	for fpath in files:
		text = read_pdf_text(fpath)
		chunks = chunk_text(text)
		if not chunks:
			continue
		ids = [str(uuid.uuid4()) for _ in chunks]
		# Build distinct metadata dicts per chunk (avoid shared references)
		metas = [{
			"source": os.path.relpath(fpath, start=root),
			"namespace": str(namespace),
			"file_name": os.path.basename(fpath),
		} for _ in chunks]
		
		# Generate embeddings using sentence-transformers
		embeddings = sentence_model.encode(chunks).tolist()
		
		collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
		added_chunks += len(chunks)

	return len(files), added_chunks


if __name__ == "__main__":
	nec_dir = os.getenv("NEC_DIR", os.path.join("data", "nec"))
	wm_dir = os.getenv("WATTMONK_DIR", os.path.join("data", "wattmonk"))
	embed_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
	print("Ingesting NEC ...")
	nec_files, nec_chunks = ingest_dir(nec_dir, "nec", embed_model)
	print(f"NEC: files={nec_files} chunks={nec_chunks}")
	print("Ingesting Wattmonk ...")
	wm_files, wm_chunks = ingest_dir(wm_dir, "wattmonk", embed_model)
	print(f"Wattmonk: files={wm_files} chunks={wm_chunks}")
	print("Done.")


