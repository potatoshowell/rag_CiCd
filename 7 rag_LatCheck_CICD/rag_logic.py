import time # za pomocą tego będziemy obliczacz opóznienia
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from qdrant_client import QdrantClient

print("Ładowanie modeli")

# 1. Model do embeddingów
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
embed_model = AutoModel.from_pretrained(embed_model_name)
embed_model.eval()

# 2. Model generujący
gen_model_name = "google/flan-t5-small"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
gen_model.eval()

print("Modele załadowane")


# Mean Pooling, sklejenie embeddingów jednego zdania
def get_embedding(text):
    inputs = embed_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = embed_model(**inputs)

    token_embeddings = outputs[0]
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings*input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return (sum_embeddings / sum_mask).squeeze().tolist()


# --- POŁĄCZENIE Z QDRANTEM W DOCKERZE ---
print("Łączenie z bazą Qdrant...")
client = QdrantClient(host="localhost", port=6333)
collection_name = "pdf_rag_collection"


# --- KROK 1: RETRIEVAL (Pobieranie wielu chunków z metadanymi) ---
def retrieve_chunks(query, top_k=2):
    query_vector = get_embedding(query)
    
    # Odpytujemy Qdranta o top_k najbardziej podobnych chunków
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k
    ).points
    
    retrieved_data = []
    for point in search_result:
        text = point.payload.get("text", "")
        source = point.payload.get("source", "nieznane")
        chunk_idx = point.payload.get("chunk_index", 0)
        score = point.score
        
        retrieved_data.append({
            "text": text,
            "source": source,
            "chunk_index": chunk_idx,
            "score": score
        })
        
    return retrieved_data


# --- KROK 2: GENERATION (Sklejanie kontekstu i generowanie odpowiedzi) ---
def generate_answer(query, contexts):
    # Sklejamy teksty znalezionych chunków w jeden spójny kontekst
    combined_context = " ".join([ctx["text"] for ctx in contexts])
    # Prompt dla modelu instrukcyjnego 
    prompt = f"answer the question based on the context. Context: {combined_context} Question: {query}"
    inputs = gen_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = gen_model.generate(**inputs, max_length=128, num_beams=2, early_stopping=True)
        
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)


# --- URUCHAMIAMY PĘTLĘ RAG ---
def ask_rag_pipeline(user_query: str, top_k: int = 2):
    start_time = time.time() # Włączamy count czasu

    retrieved_results = retrieve_chunks(user_query, top_k=top_k)
    
    if not retrieved_results:
        elapsed_time = (time.time() - start_time) * 1000 # Czas który upłynął w ms
        return {
            "question": user_query,
            "answer": "Brak wyników w bazie dla tego zapytania.",
            "sources": [],
            "latency_ms": round(elapsed_time, 2) # Zwracamy czas
        }

    print(f"[Qdrant Retrieval] Znaleziono {len(retrieved_results)} chunki:")
    sources = []
    for i, res in enumerate(retrieved_results):
        print(f"  -> [{i+1}] Plik: {res['source']} (Chunk #{res['chunk_index']}, score: {res['score']:.4f})")
        print(f"      Fragment: \"{res['text'][:120]}...\"\n")
        sources.append({"source": res['source'], "score": float(res['score'])})\

    final_answer = generate_answer(user_query, retrieved_results)

    end_time = time.time() # Czas który upłynął
    elapsed_ms = (end_time - start_time) * 1000


    print(f"[Generator] Odpowiedź modelu:")
    print(f"-> {final_answer}")

    return {
        "question": user_query,
        "answer": final_answer,
        "sources": sources,
        "latency_ms": round(elapsed_ms, 2)
    }
