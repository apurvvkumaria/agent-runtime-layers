"""Vector-store conversation memory implementing LangChain's BaseMemory.

Instead of replaying the *entire* conversation into the prompt (ConversationBufferMemory),
this stores each turn as an embedding and, for a new question, retrieves only the top-k
*semantically nearest* past turns. That keeps the {chat_history} slot bounded no matter how
long the conversation gets.

Embedding backend: a dependency-light, torch-free hashing embedder (numpy only). It produces
fixed-dimension normalized vectors so cosine similarity is a dot product. The embedder is the
single swap-in point — on a native arm64 toolchain, replace `HashingEmbedder` with a
sentence-transformers (all-MiniLM-L6-v2) embedder and nothing else changes.

Persistence: turns are stored as JSON under `./vector_store/` (a stand-in for what would be a
ChromaDB PersistentClient on a torch/onnxruntime-capable platform), so memory survives across
processes — `agent memory-stats` reads the same store `agent chat` writes.
"""

import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
from langchain_classic.base_memory import BaseMemory
from pydantic import ConfigDict, PrivateAttr

DEFAULT_STORE_DIR = "./vector_store"
_EMBED_DIM = 512


def _tokenize(text: str) -> list[str]:
    """Lowercase word unigrams + bigrams, for a bit more lexical signal."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class HashingEmbedder:
    """Torch-free embedder via the hashing trick — deterministic, local, free.

    Swap point: replace with a sentence-transformers embedder (same `.embed` /
    `.name` interface) once on an arm64 venv where torch installs.
    """

    name = f"hashing-{_EMBED_DIM}"

    def __init__(self, dim: int = _EMBED_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 8) & 1 == 0 else -1.0
            vec[h % self.dim] += sign
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class VectorStoreMemory(BaseMemory):
    """BaseMemory that retrieves the top-k semantically similar past turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_key: str = "chat_history"
    k: int = 3
    store_dir: str = DEFAULT_STORE_DIR

    # Heavy/runtime objects kept off the pydantic schema.
    _embedder: object = PrivateAttr(default=None)
    _records: list = PrivateAttr(default_factory=list)

    def __init__(self, embedder: object | None = None, **data) -> None:
        super().__init__(**data)
        self._embedder = embedder if embedder is not None else HashingEmbedder()
        self._records = self._load()

    # --- persistence ---------------------------------------------------------
    @property
    def _store_file(self) -> Path:
        return Path(self.store_dir) / "turns.json"

    def _load(self) -> list:
        if not self._store_file.exists():
            return []
        raw = json.loads(self._store_file.read_text(encoding="utf-8"))
        for rec in raw:
            rec["embedding"] = np.asarray(rec["embedding"], dtype=np.float32)
        return raw

    def _persist(self) -> None:
        self._store_file.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {**rec, "embedding": rec["embedding"].tolist()} for rec in self._records
        ]
        self._store_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    # --- BaseMemory interface ------------------------------------------------
    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict) -> dict:
        """Return the top-k past turns most similar to the current question."""
        query = inputs.get("input") or next(iter(inputs.values()), "")
        if not self._records:
            return {self.memory_key: ""}

        qv = self._embedder.embed(query)
        scored = sorted(
            self._records,
            key=lambda r: float(np.dot(qv, r["embedding"])),
            reverse=True,
        )[: self.k]
        # Present the retrieved turns in chronological order so they read naturally.
        scored.sort(key=lambda r: r["ts"])
        history = "\n".join(f"Human: {r['input']}\nAI: {r['output']}" for r in scored)
        return {self.memory_key: history}

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Embed and store one conversation turn."""
        question = inputs.get("input") or next(iter(inputs.values()), "")
        answer = outputs.get("output") or next(iter(outputs.values()), "")
        self._records.append(
            {
                "id": uuid.uuid4().hex,
                "input": question,
                "output": answer,
                "embedding": self._embedder.embed(f"{question}\n{answer}"),
                "ts": time.time(),
            }
        )
        self._persist()

    def clear(self) -> None:
        """Wipe the collection (in memory and on disk)."""
        self._records = []
        if self._store_file.exists():
            self._store_file.unlink()

    # --- stats ---------------------------------------------------------------
    def stats(self) -> dict:
        """Counts and an estimate of tokens saved vs. replaying the full buffer."""
        turns = len(self._records)
        chars = sum(len(r["input"]) + len(r["output"]) for r in self._records)
        per_turn_tokens = (chars // 4 // turns) if turns else 0  # ~4 chars/token
        buffer_tokens = turns * per_turn_tokens             # buffer sends everything
        vector_tokens = min(self.k, turns) * per_turn_tokens  # vector sends top-k
        saved = buffer_tokens - vector_tokens
        pct = round(100 * saved / buffer_tokens, 1) if buffer_tokens else 0.0
        return {
            "turns": turns,
            "embedding_dim": getattr(self._embedder, "dim", None),
            "embedder": getattr(self._embedder, "name", type(self._embedder).__name__),
            "on_disk_bytes": self._store_file.stat().st_size if self._store_file.exists() else 0,
            "buffer_tokens": buffer_tokens,
            "vector_tokens": vector_tokens,
            "estimated_savings": saved,
            "savings_pct": pct,
        }
