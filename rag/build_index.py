from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "index"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def chunk_text(text: str, size: int = 500, overlap: int = 100):
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_documents():
    docs = []
    for path in DOCS_DIR.glob("*.txt"):
        print(f"Laster: {path}")
        content = path.read_text(encoding="utf-8")

        chunks = chunk_text(content)
        print(f"  → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            docs.append({
                "source": path.name,
                "chunk_id": i,
                "text": chunk
            })

    return docs


def main():
    if not DOCS_DIR.exists():
        raise ValueError(f"Fant ikke mappe: {DOCS_DIR}")

    docs = load_documents()
    print(f"Totalt docs: {len(docs)}")

    if not docs:
        raise ValueError("Ingen dokumenter funnet i rag/documents")

    texts = [d["text"] for d in docs]
    print(f"Antall tekster: {len(texts)}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Embeddings shape:", embeddings.shape)

    if embeddings.ndim != 2:
        raise ValueError(f"Forventet 2D embeddings, fikk {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    (INDEX_DIR / "metadata.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"✅ Ferdig! Lagret {len(docs)} chunks i rag/index")


if __name__ == "__main__":
    main()