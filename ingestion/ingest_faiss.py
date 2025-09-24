from __future__ import annotations
import os
import pickle
from typing import List, Tuple

import faiss
import numpy as np
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


def ingest_dir(root: str, namespace: str) -> Tuple[int, int]:
    persist_dir = os.getenv("CHROMA_DB_DIR", os.path.join("data", "chroma"))
    os.makedirs(persist_dir, exist_ok=True)
    
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    added_chunks = 0
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".pdf",)):
                files.append(os.path.join(dirpath, fn))

    all_texts = []
    all_metadata = []
    
    for fpath in files:
        text = read_pdf_text(fpath)
        chunks = chunk_text(text)
        if not chunks:
            continue
        
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadata.append({
                "source": os.path.relpath(fpath, start=root),
                "namespace": str(namespace),
                "file_name": os.path.basename(fpath),
                "text": chunk,
            })
        added_chunks += len(chunks)

    if all_texts:
        # Generate embeddings
        embeddings = sentence_model.encode(all_texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        index_path = os.path.join(persist_dir, f"{namespace}_index.faiss")
        meta_path = os.path.join(persist_dir, f"{namespace}_metadata.pkl")
        
        faiss.write_index(index, index_path)
        with open(meta_path, 'wb') as f:
            pickle.dump(all_metadata, f)
    
    return len(files), added_chunks


if __name__ == "__main__":
    nec_dir = os.getenv("NEC_DIR", os.path.join("data", "nec"))
    wm_dir = os.getenv("WATTMONK_DIR", os.path.join("data", "wattmonk"))
    
    print("Ingesting NEC ...")
    nec_files, nec_chunks = ingest_dir(nec_dir, "nec")
    print(f"NEC: files={nec_files} chunks={nec_chunks}")
    
    print("Ingesting Wattmonk ...")
    wm_files, wm_chunks = ingest_dir(wm_dir, "wattmonk")
    print(f"Wattmonk: files={wm_files} chunks={wm_chunks}")
    
    print("Done.")
