"""LangFuse evals: deepeval metrics scored onto LangFuse traces.

For each question we open a LangFuse span (so we control/know the trace id), run
the agent inside it (its callback nests under the span, sharing the trace), grade
the answer with deepeval's `AnswerRelevancyMetric` (judged by Claude), and attach
that score to the trace. Requires LANGFUSE_* keys; degrades to a skip otherwise.

Run directly:  uv run python evals/langfuse_evals.py
"""

import os
import pathlib
import sys

# Repo root on the path so `import core` works regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import AnswerRelevancyMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from core import build_single_shot_agent  # noqa: E402
from evals.judge import claude_judge  # noqa: E402
from hooks import _langfuse_configured, get_callbacks  # noqa: E402

QUESTIONS = [
    "What is the capital of France?",
    "What is 15 times 15?",
    "Explain what a hash table is in one sentence.",
]


def run() -> float | None:
    """Run each question, score relevancy onto its trace; return the average (0-1)."""
    if not _langfuse_configured():
        print("LangFuse not configured (set LANGFUSE_* keys) — skipping LangFuse evals.")
        return None

    from langfuse import get_client

    langfuse = get_client()
    executor = build_single_shot_agent()
    # AnswerRelevancyMetric is LLM-judged; use Claude as the judge model.
    metric = AnswerRelevancyMetric(model=claude_judge(), threshold=0.5, async_mode=False)
    scores: list[float] = []

    for question in QUESTIONS:
        with langfuse.start_as_current_observation(name="langfuse-eval", as_type="span"):
            trace_id = langfuse.get_current_trace_id()
            # Run inside the span so the agent's trace nests under it (same trace_id).
            result = executor.invoke({"input": question}, config={"callbacks": get_callbacks()})
            answer = result.get("output", "")

            metric.measure(LLMTestCase(input=question, actual_output=answer))
            score = metric.score or 0.0
            langfuse.score_current_trace(
                name="answer_relevancy",
                value=score,
                data_type="NUMERIC",
                comment=metric.reason or "deepeval AnswerRelevancyMetric (Claude judge)",
            )
        scores.append(score)
        print(f"[trace {trace_id}] answer_relevancy={score:.2f}  Q: {question}")

    langfuse.flush()
    avg = sum(scores) / len(scores)
    print(f"\nLangFuse evals: avg answer_relevancy {avg:.2f}/1.0")
    return avg


if __name__ == "__main__":
    run()
