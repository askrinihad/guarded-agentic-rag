from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

COLLECTION_NAME = "guarded_rag_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_K = 6

app = FastAPI(title="Guarded Agentic RAG - Phase 1")

embedder = SentenceTransformer(EMBEDDING_MODEL)
qdrant = QdrantClient(host="localhost", port=6333)
generator = pipeline(
    "text-generation",
    model=GENERATION_MODEL,
    device_map="auto",
)


class Question(BaseModel):
    question: str


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Excerpt {i+1}]\n{c}" for i, c in enumerate(context_chunks))
    return (
        "Answer the question using only the excerpts below. "
        "If the answer isn't in the excerpts, say you don't know.\n\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


@app.post("/ask")
def ask(payload: Question):
    query_vector = embedder.encode(payload.question).tolist()

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
    ).points

    context_chunks = [point.payload["text"] for point in results]
    sources = list({point.payload["source"] for point in results})

    prompt = build_prompt(payload.question, context_chunks)
    output = generator(prompt, max_new_tokens=100, do_sample=False)
    answer = output[0]["generated_text"][len(prompt):].strip()

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": context_chunks,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
