import os
import urllib.request
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from rag_logic import ask_rag_pipeline

# Inicjalizacja aplikacji FastAPI
app = FastAPI(title="RAG PDF API", version="1.0")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_HTTP_PORT", "6333")


# Schemat danych wejściowych od klienta
class QueryRequest(BaseModel):
    question: str
    top_k: int = 2


# 1. Liveness / Startup: czy sam proces FastAPI żyje i odpowiada
@app.get("/health/live", tags=["Health"], status_code=status.HTTP_200_OK)
def liveness_check():
    return {"status": "alive"}


# 2. Readiness: czy aplikacja ma aktywne połączenie z Qdrantem
@app.get("/health/ready", tags=["Health"], status_code=status.HTTP_200_OK)
def readiness_check():
    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/readyz"
    try:
        with urllib.request.urlopen(qdrant_url, timeout=2) as response:
            if response.status == 200:
                return {"status": "ready", "qdrant": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unavailable: {str(e)}",
        )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Qdrant returned non-200 status",
    )


# Główny endpoint RAG
@app.post("/ask", tags=["RAG"])
def ask_endpoint(payload: QueryRequest):
    try:
        result = ask_rag_pipeline(payload.question, payload.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))