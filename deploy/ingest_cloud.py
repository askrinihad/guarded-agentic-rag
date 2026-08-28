"""
Ingests the paper into Qdrant CLOUD (not local Docker) for the hosted demo.
Reuses the same sentence-aware chunking logic as src/ingestion/ingest.py.

Run locally, once, before deploying:
    python deploy/ingest_cloud.py

Requires environment variables (set these in your shell before running,
or create a .env file in this folder - see .env.example):
    QDRANT_URL
    QDRANT_API_KEY
"""

import os
import re
import uuid
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

PAPERS_DIR = Path(__file__).parent.parent / "data" / "papers"
COLLECTION_NAME = "guarded_rag_papers"
CHUNK_SIZE = 500
OVERLAP_SENTENCES = 1
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap_sentences: int = OVERLAP_SENTENCES) -> list[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)
    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


def main():
    qdrant_url = (os.environ.get("QDRANT_URL") or "").strip()
    qdrant_api_key = (os.environ.get("QDRANT_API_KEY") or "").strip()

    if not qdrant_url or not qdrant_api_key:
        print("ERROR: set QDRANT_URL and QDRANT_API_KEY environment variables first.")
        print("  export QDRANT_URL='https://your-cluster-url.cloud.qdrant.io:6333'")
        print("  export QDRANT_API_KEY='your-database-api-key'")
        return

    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR}/")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = embedder.get_sentence_embedding_dimension()

    print(f"Connecting to Qdrant Cloud at {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    all_points = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path.name}...")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        print(f"  -> {len(chunks)} chunks")

        embeddings = embedder.encode(chunks, show_progress_bar=False)
        for chunk, vector in zip(chunks, embeddings):
            all_points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector.tolist(),
                    payload={"text": chunk, "source": pdf_path.name},
                )
            )

    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"Upserted {len(all_points)} chunks into Qdrant Cloud collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
