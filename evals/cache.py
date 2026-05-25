"""Eval response cache (Layer 22) — JSON-on-disk cache for LLM-as-judge calls.

Every eval run re-judges with Claude, even when the (prompt, input, model) is
unchanged — wasteful at scale. This caches judge responses so a re-run on
unchanged prompts costs zero tokens.

This module is the **disk tier**: a JSON file at `evals/.cache/eval_cache.json`
keyed by `SHA256(prompt, input, model)`. The **memory tier** (an in-process
`functools.lru_cache`) lives in `judge.py`, layered on top of this — see
`ClaudeJudge`. Together they're two-tier: lru within a run, JSON across runs.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_FILE = CACHE_DIR / "eval_cache.json"
COST_PER_1K_TOKENS = 0.003  # rough $/1K tokens, for the "cost saved" estimate


def make_key(prompt: str, input: str, model: str) -> str:
    """SHA256 of (prompt, input, model) — the cache key for one judge call."""
    h = hashlib.sha256()
    for part in (model, prompt, input or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class EvalCache:
    """JSON-file cache of judge responses, with session hit/miss accounting.

    `get`/`set` persist to disk; `hits`/`misses`/`tokens_saved` count activity
    for *this process* (so run_all can report what the cache saved this run,
    while the CLI reports the persistent on-disk totals).
    """

    def __init__(self, path: Path = CACHE_FILE) -> None:
        self.path = Path(path)
        self._store: dict[str, dict] = self._load()
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    def _load(self) -> dict[str, dict]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def get(self, key: str) -> str | None:
        """Return the cached value (str) or None, counting a hit/miss."""
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.tokens_saved += entry.get("token_count", 0)
        return entry["value"]

    def set(self, key: str, value: str, *, prompt_hash: str, model: str, token_count: int) -> None:
        self._store[key] = {
            "value": value,
            "timestamp": time.time(),
            "prompt_hash": prompt_hash,
            "model": model,
            "token_count": token_count,
        }
        self._persist()

    def clear(self) -> int:
        """Wipe the cache (memory + disk); return how many entries were removed."""
        n = len(self._store)
        self._store = {}
        if self.path.exists():
            self.path.unlink()
        return n

    def stats(self) -> dict:
        total = self.hits + self.misses
        cached_tokens = sum(e.get("token_count", 0) for e in self._store.values())
        return {
            # session (this process) — meaningful inside an eval run
            "hits": self.hits,
            "misses": self.misses,
            "total": total,
            "hit_rate": (self.hits / total) if total else 0.0,
            "tokens_saved": self.tokens_saved,
            "cost_saved": self.tokens_saved / 1000 * COST_PER_1K_TOKENS,
            # persistent (on disk) — meaningful for the standalone CLI
            "entries": len(self._store),
            "cached_tokens": cached_tokens,
            "cached_cost": cached_tokens / 1000 * COST_PER_1K_TOKENS,
            "disk_bytes": self.path.stat().st_size if self.path.exists() else 0,
        }


# Process-wide singleton so the judge, run_all, and the CLI share one cache.
_cache = EvalCache()


def get_cache() -> EvalCache:
    return _cache
