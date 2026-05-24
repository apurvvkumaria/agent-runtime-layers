"""LangFuse evals: run the agent, judge each answer with an LLM, score the trace.

For each question we open a LangFuse span (so we control/know the trace id), run
the agent inside it (its callback nests under the span, sharing the trace), ask
Claude to rate the answer 1-5 for relevance, and attach that as a score on the
trace. Requires LANGFUSE_* keys; degrades to a skip when not configured.

Run directly:  uv run python evals/langfuse_evals.py
"""

import pathlib
import re
import sys

# Repo root on the path so `import core` works regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic  # noqa: E402

from core import build_single_shot_agent  # noqa: E402
from hooks import _langfuse_configured, get_callbacks  # noqa: E402

QUESTIONS = [
    "What is the capital of France?",
    "What is 15 times 15?",
    "Explain what a hash table is in one sentence.",
]


def _judge(question: str, answer: str) -> int:
    """LLM-as-judge: ask Claude to rate the answer 1-5 for relevance/quality."""
    judge = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    prompt = (
        "You are grading an AI assistant's answer for relevance and quality on a scale "
        "of 1 to 5 (5 = excellent, fully relevant and correct; 1 = irrelevant or wrong).\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Respond with ONLY a single integer from 1 to 5."
    )
    resp = judge.invoke(prompt)
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else 3


def run() -> float | None:
    """Run every question, score its trace, and return the average score (or None)."""
    if not _langfuse_configured():
        print("LangFuse not configured (set LANGFUSE_* keys) — skipping LangFuse evals.")
        return None

    from langfuse import get_client

    langfuse = get_client()
    executor = build_single_shot_agent()
    scores: list[int] = []

    for question in QUESTIONS:
        with langfuse.start_as_current_observation(name="langfuse-eval", as_type="span"):
            trace_id = langfuse.get_current_trace_id()
            # Run inside the span so the agent's trace nests under it (same trace_id).
            result = executor.invoke({"input": question}, config={"callbacks": get_callbacks()})
            answer = result.get("output", "")
            score = _judge(question, answer)
            langfuse.score_current_trace(
                name="relevance",
                value=score,
                data_type="NUMERIC",
                comment="LLM-as-judge relevance (1-5)",
            )
        scores.append(score)
        print(f"[trace {trace_id}] relevance={score}/5  Q: {question}")

    langfuse.flush()
    avg = sum(scores) / len(scores)
    print(f"\nLangFuse evals: avg score {avg:.1f}/5")
    return avg


if __name__ == "__main__":
    run()
