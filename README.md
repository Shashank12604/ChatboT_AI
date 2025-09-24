RAG Chatbot (NEC + Wattmonk)
============================

Quick start
-----------

1. Create and fill `.env` from `.env.example` (add `OPENAI_API_KEY`).
2. Put PDFs:
   - `data/nec/` for NEC code docs
   - `data/wattmonk/` for company docs
3. Install deps: `pip install -r requirements.txt`
4. Ingest docs: `python -m ingestion.ingest`
5. Run API: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`
6. Run UI: `streamlit run streamlit_app.py`

Features
--------

- Intent routing: general vs `nec` vs `wattmonk`
- Chroma vector store, OpenAI embeddings
- RAG answers with source attribution and confidence
- Streamlit chat with memory, search panel, and settings

Configuration
-------------

See `.env.example`. Important variables:

- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`
- `CHROMA_DB_DIR`, `NEC_DIR`, `WATTMONK_DIR`
- `API_URL`, `CORS_ORIGINS`

Deployment
----------

- Streamlit Cloud: set secrets as env vars, deploy repo, run `streamlit_app.py`
- Hugging Face Spaces: Space (Streamlit), set env vars, point to `streamlit_app.py`
- Backend on Railway/Render/Heroku: Procfile `web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

API
---

- `POST /chat` with `{ messages: [{role, content}], top_k?, include_sources? }`
- `POST /search` with `{ query, namespace: "nec"|"wattmonk", top_k? }`





