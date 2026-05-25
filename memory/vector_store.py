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

import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_classic.base_memory import BaseMemory
from pydantic import ConfigDict, Field, PrivateAttr
from sentence_transformers import SentenceTransformer

DEFAULT_STORE_DIR = "./chroma_db"
_MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL_DIM = 384  # all-MiniLM-L6-v2 embedding size (known; avoids loading just for stats)

# Memory decay: a turn's tier downgrades as it ages, so old context is compressed
# (and eventually dropped) rather than kept verbatim forever.
TIERS = ("full", "summary", "marker", "archived")
_AGE_DAYS = {"summary": 3, "marker": 30, "archived": 90}  # lower bound (days) per tier


def _tier_for_age(age_days: float, thresholds: dict | None = None) -> str:
    t = thresholds or _AGE_DAYS
    if age_days < t["summary"]:
        return "full"
    if age_days < t["marker"]:
        return "summary"
    if age_days < t["archived"]:
        return "marker"
    return "archived"


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
    # Age thresholds (days) at which a turn enters each tier. Override fully or
    # partially, e.g. VectorStoreMemory(decay_days={"archived": 365}); missing
    # keys fall back to the defaults.
    decay_days: dict[str, int] = Field(default_factory=lambda: dict(_AGE_DAYS))

    # Heavy/runtime objects kept off the pydantic schema.
    _embedder: object = PrivateAttr(default=None)
    _client: object = PrivateAttr(default=None)
    _collection: object = PrivateAttr(default=None)

    def __init__(self, embedder: object | None = None, **data) -> None:
        super().__init__(**data)
        self.decay_days = {**_AGE_DAYS, **(self.decay_days or {})}  # fill missing keys
        self._embedder = embedder if embedder is not None else SentenceTransformerEmbedder()
        self._client = chromadb.PersistentClient(
            path=self.store_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._open_collection()
        # Age out old turns on startup. Idempotent and cheap when nothing has
        # crossed a tier boundary; guarded so a failure never breaks construction.
        try:
            self.decay_memory()
        except Exception:
            pass

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
        lines = [self._render(m) for m in metas]
        return {self.memory_key: "\n".join(line for line in lines if line)}

    @staticmethod
    def _render(meta: dict) -> str:
        """Format a retrieved turn according to its decay tier."""
        tier = meta.get("tier", "full")
        if tier == "summary":
            return f"(summary) {meta.get('summary') or meta.get('input', '')}"
        if tier == "marker":
            day = datetime.fromtimestamp(meta.get("ts", 0)).strftime("%Y-%m-%d")
            topic = meta.get("topic") or (meta.get("input", "")[:40])
            return f"[Topic: {topic} discussed on {day}]"
        if tier == "archived":
            return ""  # archived turns are deleted, but exclude defensively
        return f"Human: {meta.get('input', '')}\nAI: {meta.get('output', '')}"

    def save_context(self, inputs: dict, outputs: dict) -> None:
        """Embed and store one conversation turn."""
        question = inputs.get("input") or next(iter(inputs.values()), "")
        answer = outputs.get("output") or next(iter(outputs.values()), "")
        self._collection.add(
            ids=[uuid.uuid4().hex],
            embeddings=[self._embedder.embed(f"{question}\n{answer}")],
            documents=[f"{question}\n{answer}"],
            metadatas=[{"input": question, "output": answer, "ts": time.time(), "tier": "full"}],
        )

    def clear(self) -> None:
        """Wipe the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._open_collection()

    # --- decay ---------------------------------------------------------------
    def _summarize(self, question: str, answer: str) -> str:
        """One-sentence summary of a turn (LLM, with a heuristic fallback)."""
        try:
            from langchain_anthropic import ChatAnthropic

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("no API key")
            llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
            prompt = (
                "Summarize this exchange in ONE short sentence.\n"
                f"User: {question}\nAssistant: {answer}\nSummary:"
            )
            content = llm.invoke(prompt).content
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            return text.strip().splitlines()[0]
        except Exception:
            base = f"{question} -> {answer}"
            return base[:120] + ("…" if len(base) > 120 else "")

    @staticmethod
    def _topic(text: str) -> str:
        """A short topic tag from text (a few salient words)."""
        stop = {"the", "and", "what", "that", "with", "this", "your", "from", "have",
                "about", "into", "does", "would", "much", "there"}
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
                 if w.lower() not in stop]
        return " ".join(words[:5]) if words else text[:40]

    def decay_memory(self) -> dict:
        """Downgrade each stored turn's tier by age; return the resulting counts.

        full -> summary (LLM summary), summary -> marker (topic tag), marker ->
        archived (deleted). A turn jumps straight to its age-appropriate tier,
        performing whatever transforms that requires. Idempotent.
        """
        now = time.time()
        stored = self._collection.get()
        counts = {tier: 0 for tier in TIERS}
        to_delete: list[str] = []
        for cid, meta in zip(stored["ids"], stored["metadatas"]):
            target = _tier_for_age((now - meta.get("ts", now)) / 86400, self.decay_days)
            counts[target] += 1
            if target == "archived":
                to_delete.append(cid)
                continue
            if target == meta.get("tier", "full"):
                continue  # already at the right tier
            meta = dict(meta)
            meta["tier"] = target
            if target in ("summary", "marker") and not meta.get("summary"):
                meta["summary"] = self._summarize(meta.get("input", ""), meta.get("output", ""))
            if target == "marker" and not meta.get("topic"):
                meta["topic"] = self._topic(meta.get("summary") or meta.get("input", ""))
            self._collection.update(ids=[cid], metadatas=[meta])
        if to_delete:
            self._collection.delete(ids=to_delete)
        return counts

    # --- stats ---------------------------------------------------------------
    def stats(self) -> dict:
        """Counts and an estimate of tokens saved vs. replaying the full buffer."""
        turns = self._collection.count()
        stored = self._collection.get() if turns else {"documents": [], "metadatas": []}
        docs = stored.get("documents", [])
        tier_counts = {tier: 0 for tier in TIERS}
        for meta in stored.get("metadatas", []):
            tier_counts[meta.get("tier", "full")] = tier_counts.get(meta.get("tier", "full"), 0) + 1
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
            "tiers": tier_counts,
        }
