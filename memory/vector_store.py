"""Vector-store conversation memory implementing LangChain's BaseMemory.

Instead of replaying the *entire* conversation into the prompt (ConversationBufferMemory),
this stores each turn as an embedding and, for a new question, retrieves only the top-k
*semantically nearest* past turns. That keeps the {chat_history} slot bounded no matter how
long the conversation gets.

Backend (native arm64): ChromaDB `PersistentClient` (stored in ./chroma_db/) for storage and
nearest-neighbor search, with embeddings from sentence-transformers `all-MiniLM-L6-v2`
(free, local, no API key). The embedder is the swap point — on a platform without torch it
can be replaced with any object exposing `.embed(text) -> list[float]`.
"""

import time
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_classic.base_memory import BaseMemory
from pydantic import ConfigDict, PrivateAttr
from sentence_transformers import SentenceTransformer

DEFAULT_STORE_DIR = "./chroma_db"
_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_DIM = 384  # all-MiniLM-L6-v2 embedding size (known; avoids loading just for stats)


class SentenceTransformerEmbedder:
    """Local, free embedder using sentence-transformers all-MiniLM-L6-v2.

    The model (~90MB) is loaded lazily on first use and cached by huggingface.
    """

    name = _MODEL_NAME
    dim = _MODEL_DIM

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


class VectorStoreMemory(BaseMemory):
    """BaseMemory that retrieves the top-k semantically similar past turns via ChromaDB."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_key: str = "chat_history"
    k: int = 3
    store_dir: str = DEFAULT_STORE_DIR
    collection_name: str = "conversation_turns"

    # Heavy/runtime objects kept off the pydantic schema.
    _embedder: object = PrivateAttr(default=None)
    _client: object = PrivateAttr(default=None)
    _collection: object = PrivateAttr(default=None)

    def __init__(self, embedder: object | None = None, **data) -> None:
        super().__init__(**data)
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self._client = chromadb.PersistentClient(
            path=self.store_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()

    def _open_collection(self):
        # cosine space pairs with our L2-normalized embeddings.
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # --- BaseMemory interface ------------------------------------------------
    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        """Return the top-k past turns most similar to the current question."""
        query = inputs.get("input") or next(iter(inputs.values()), "")
        count = self._collection.count()
        if not count:
            return {self.memory_key: ""}

        result = self._collection.query(
            query_embeddings=[self._embedder.embed(query)],
            n_results=min(self.k, count),
        )
        metas = result["metadatas"][0]
        # Present retrieved turns in chronological order so they read naturally.
        metas.sort(key=lambda m: m.get("ts", 0))
        history = "\n".join(f"Human: {m['input']}\nAI: {m['output']}" for m in metas)
        return {self.memory_key: history}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Embed and store one conversation turn."""
        question = inputs.get("input") or next(iter(inputs.values()), "")
        answer = outputs.get("output") or next(iter(outputs.values()), "")
        self._collection.add(
            ids=[uuid.uuid4().hex],
            embeddings=[self._embedder.embed(f"{question}\n{answer}")],
            documents=[f"{question}\n{answer}"],
            metadatas=[{"input": question, "output": answer, "ts": time.time()}],
        )

    def clear(self) -> None:
        """Wipe the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()

    # --- stats ---------------------------------------------------------------
    def stats(self) -> dict:
        """Counts and an estimate of tokens saved vs. replaying the full buffer."""
        turns = self._collection.count()
        docs = self._collection.get().get("documents", []) if turns else []
        chars = sum(len(d) for d in docs)
        per_turn_tokens = (chars // 4 // turns) if turns else 0  # ~4 chars/token
        buffer_tokens = turns * per_turn_tokens               # buffer sends everything
        vector_tokens = min(self.k, turns) * per_turn_tokens   # vector sends top-k
        saved = buffer_tokens - vector_tokens
        pct = round(100 * saved / buffer_tokens, 1) if buffer_tokens else 0.0
        return {
            "turns": turns,
            "embedding_dim": getattr(self._embedder, "dim", None),
            "embedder": getattr(self._embedder, "name", type(self._embedder).__name__),
            "on_disk_bytes": sum(
                f.stat().st_size for f in Path(self.store_dir).rglob("*") if f.is_file()
            ) if Path(self.store_dir).exists() else 0,
            "buffer_tokens": buffer_tokens,
            "vector_tokens": vector_tokens,
            "estimated_savings": saved,
            "savings_pct": pct,
        }
