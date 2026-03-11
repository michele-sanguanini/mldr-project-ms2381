import os
import sys
import uuid
from pathlib import Path

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

DOCS_DIR = Path("data/docs")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = "docs"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama")
VECTOR_SIZE = 768  # nomic-embed-text output dimension
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(c: QdrantClient):
    names = [col.name for col in c.get_collections().collections]
    if COLLECTION not in names:
        c.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def embed(text: str) -> list[float]:
    return ollama.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]


def load_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(errors="replace")


def chunk(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE].strip()
        if piece:
            chunks.append(piece)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files = [p for p in DOCS_DIR.iterdir() if p.suffix.lower() in (".pdf", ".txt", ".md")]
    if not files:
        print(f"No .pdf / .txt / .md files found in {DOCS_DIR}")
        return

    c = client()
    ensure_collection(c)

    for path in files:
        pieces = chunk(load_file(path))
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(piece),
                payload={"source": path.name, "text": piece},
            )
            for piece in pieces
        ]
        c.upsert(collection_name=COLLECTION, points=points)
        print(f"  {path.name}: {len(pieces)} chunks")

    print("Ingestion complete.")


def query(question: str, top_k: int = 3):
    c = client()
    results = c.query_points(
        collection_name=COLLECTION,
        query=embed(question),
        limit=top_k,
        with_payload=True,
    )
    context = "\n\n".join(h.payload["text"] for h in results.points)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    print(response["message"]["content"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  uv run python rag.py ingest")
        print("  uv run python rag.py query '<question>'")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "ingest":
        ingest()
    elif cmd == "query" and len(sys.argv) > 2:
        query(sys.argv[2])
    else:
        print("Usage:")
        print("  uv run python rag.py ingest")
        print("  uv run python rag.py query '<question>'")
        sys.exit(1)
