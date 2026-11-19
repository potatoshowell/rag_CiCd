from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_logic import ask_rag_pipeline

# Inicjalizacja aplikacji FastAPI
app = FastAPI(title="RAG PDF API", version="1.0")

# Schemat danych wejściowych od klienta
class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

# Endpoint API
@app.post("/ask")
def ask_endpoint(payload: QueryRequest):
    try:
        # Wywołujemy Twoją funkcję z pliku logiki\
        result = ask_rag_pipeline(payload.question, payload.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))