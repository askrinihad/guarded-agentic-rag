
import uuid
from pathlib import Path

from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

PAPERS_DIR = Path("data/papers")
COLLECTION_NAME = "guarded_rag_papers"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAPERS_DIR}/ - add your paper there first.")
        return

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = embedder.get_sentence_embedding_dimension()

    client = QdrantClient(host="localhost", port=6333)

    if not client.collection_exists(COLLECTION_NAME):
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
    print(f"Upserted {len(all_points)} chunks into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
