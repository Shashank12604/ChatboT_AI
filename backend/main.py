from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal
import os
from dotenv import load_dotenv

load_dotenv()

# Check required environment variables
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY not set in environment")

app = FastAPI(title="RAG Chatbot API", version="0.1.0")

origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: int = 5
    include_sources: bool = True


class SourceDoc(BaseModel):
    source: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    snippet: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    source: Optional[str] = None
    sources: Optional[List[SourceDoc]] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None


_retriever = None
_intent = None
_generator = None


def _ensure_components():
    global _retriever, _intent, _generator
    if _retriever is None:
        from backend.rag.retriever_faiss import RAGRetriever
        _retriever = RAGRetriever()
    if _intent is None:
        from backend.rag.intent import classify_intent
        _intent = classify_intent
    if _generator is None:
        from backend.rag.generator import generate_answer
        _generator = generate_answer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        _ensure_components()

        if not req.messages:
            raise HTTPException(status_code=400, detail="messages required")

        user_msg = req.messages[-1].content

        intent = _intent(user_msg)

        if intent == "general":
            answer = _generator(user_msg, [])
            return ChatResponse(answer=answer, intent=intent, confidence=0.6)

        results = _retriever.search(user_msg, namespace=intent, top_k=req.top_k)

        contexts = [r["text"] for r in results]
        sources = [
            SourceDoc(
                source=r.get("source", intent),
                chunk_id=r.get("id"),
                score=r.get("score"),
                snippet=r.get("text", "")[:400],
            )
            for r in results
        ] if req.include_sources else None

        answer = _generator(user_msg, contexts)

        scores = [r.get("score") for r in results if isinstance(r.get("score"), (int, float))]
        confidence = float(sum(scores) / len(scores)) if scores else 0.5

        return ChatResponse(
            answer=answer,
            intent=intent,
            sources=sources,
            source=intent,
            confidence=confidence,
        )
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class SearchRequest(BaseModel):
    query: str
    namespace: Literal["nec", "wattmonk"]
    top_k: int = 5


@app.post("/search")
def search(req: SearchRequest) -> List[SourceDoc]:
    _ensure_components()
    if not req.query:
        raise HTTPException(status_code=400, detail="query required")
    results = _retriever.search(req.query, namespace=req.namespace, top_k=req.top_k)
    return [
        SourceDoc(
            source=r.get("source", req.namespace),
            chunk_id=r.get("id"),
            score=r.get("score"),
            snippet=r.get("text", "")[:400],
        )
        for r in results
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8001, reload=True)