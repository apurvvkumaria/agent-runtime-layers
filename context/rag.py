"""RAG over the docs/ directory.

Indexes docs/*.md (paragraph chunks) into a separate ChromaDB "docs" collection
using the same sentence-transformers embedder as conversation memory, then
retrieves the most relevant chunks for a question. Used to inject internal
documentation (e.g. SLA thresholds) into the prompt for storage/latency questions.
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings

from memory.vector_store import SentenceTransformerEmbedder

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
STORE_DIR = "./chroma_db"  # shared client, separate collection from conversation memory
COLLECTION = "docs"

# Questions mentioning any of these trigger RAG injection from docs/.
RAG_KEYWORDS = ("latency", "p99", "sla", "metrics", "cluster")

_collection = None
_embedder = None


def needs_rag(question: str) -> bool:
    """True if the question is about storage/latency topics covered by docs/."""
    q = (question or "").lower()
    return any(kw in q for kw in RAG_KEYWORDS)


def _chunks(text: str) -> list[str]:
    """Split a markdown doc into paragraph-ish chunks."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _ensure_indexed(collection, embedder) -> None:
    """Index docs/*.md once (only if the collection is empty)."""
    if collection.count() > 0:
        return
    ids, docs, embs, metas = [], [], [], []
    for md in sorted(DOCS_DIR.glob("*.md")):
        for i, chunk in enumerate(_chunks(md.read_text(encoding="utf-8"))):
            ids.append(f"{md.name}:{i}")
            docs.append(chunk)
            embs.append(embedder.embed(chunk))
            metas.append({"source": md.name, "chunk": i})
    if ids:
        collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)


def _get() -> tuple[object, object]:
    global _collection, _embedder
    if _collection is None:
        client = chromadb.PersistentClient(
            path=STORE_DIR, settings=Settings(anonymized_telemetry=False)
        )
        _collection = client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        _embedder = SentenceTransformerEmbedder()
        _ensure_indexed(_collection, _embedder)
    return _collection, _embedder


def relevant_docs(question: str, k: int = 3) -> list[dict]:
    """Top-k doc chunks for the question, as [{source, text}, ...]."""
    collection, embedder = _get()
    count = collection.count()
    if not count:
        return []
    res = collection.query(
        query_embeddings=[embedder.embed(question)], n_results=min(k, count)
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    return [{"source": m.get("source", ""), "text": d} for d, m in zip(docs, metas)]


def retrieve(question: str, k: int = 3) -> str:
    """Relevant doc chunks for the question, formatted as a single string."""
    hits = relevant_docs(question, k)
    return "\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
