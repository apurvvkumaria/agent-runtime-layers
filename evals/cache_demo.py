"""Cache demo (Layer 22): judge the same cases twice, show miss -> hit.

Run 1 is all cache misses (real Claude judge calls, tokens spent). Between runs
we drop the in-process memory tier so run 2 is served by the JSON disk cache —
all hits, zero tokens. Prints a before/after comparison.

Run directly:  uv run python evals/cache_demo.py   (or `agent eval-cache-demo`)
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import AnswerRelevancyMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.cache import get_cache  # noqa: E402
from evals.judge import claude_judge, clear_memory_tier  # noqa: E402

# Fixed (question, answer) pairs — the judge sees identical prompts both runs.
CASES = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 12 times 12?", "12 times 12 is 144."),
]


def _judge_all() -> None:
    metric = AnswerRelevancyMetric(model=claude_judge(), threshold=0.5, async_mode=False)
    for question, answer in CASES:
        metric.measure(LLMTestCase(input=question, actual_output=answer))


def run() -> None:
    cache = get_cache()
    print(f"Starting from {cache.stats()['entries']} cached entr(ies). "
          "Clear with `agent eval-cache-stats` → `EvalCache().clear()` for a cold demo.\n")

    print("===== RUN 1 (cold: expect cache misses, real judge calls) =====")
    before = cache.stats()
    _judge_all()
    after1 = cache.stats()
    r1_hits = after1["hits"] - before["hits"]
    r1_misses = after1["misses"] - before["misses"]
    r1_tokens = sum(  # tokens actually spent this run = on misses
        e.get("token_count", 0) for e in cache._store.values()
    ) if before["entries"] == 0 else None

    print("\n===== dropping memory tier so run 2 hits the JSON disk cache =====")
    clear_memory_tier()

    print("\n===== RUN 2 (warm: expect all cache hits, zero tokens) =====")
    mid = cache.stats()
    _judge_all()
    after2 = cache.stats()
    r2_hits = after2["hits"] - mid["hits"]
    r2_misses = after2["misses"] - mid["misses"]
    r2_saved = after2["tokens_saved"] - mid["tokens_saved"]

    print("\n===== COMPARISON =====")
    print(f"Run 1 (cold): {r1_hits} hits, {r1_misses} misses  -> tokens spent on judging")
    print(f"Run 2 (warm): {r2_hits} hits, {r2_misses} misses  -> {r2_saved} tokens saved, $0 spent")
    s = cache.stats()
    print(f"\nCache now: {s['entries']} entries, {s['cached_tokens']} tokens represented, "
          f"{s['disk_bytes']} bytes on disk.")


if __name__ == "__main__":
    run()
