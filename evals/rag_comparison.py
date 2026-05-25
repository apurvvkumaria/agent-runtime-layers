"""Compare answering a docs question with vs. without RAG context.

Asks the same storage/SLA question two ways through the single-shot agent:
plain, and with the relevant docs/ chunks prepended. Reports the token cost of
the injected context and whether each answer reflects the project's specific SLA
doc (e.g. "< 5 ms = Excellent") — i.e. grounding vs. guessing.

Run directly:  uv run python evals/rag_comparison.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_anthropic import ChatAnthropic  # noqa: E402

from context.manager import ContextManager  # noqa: E402
from context.rag import retrieve  # noqa: E402

# "OpenShell" is defined in docs/openShell_overview.md as a sandbox runtime/broker for
# agents. The real-world OpenShell is a Windows Start-menu tool — so without our docs the
# model answers about the wrong thing entirely. This makes RAG's value visible.
QUESTION = "In our system, what is OpenShell and what does its broker do?"

# Doc-specific terms that don't appear in the question and aren't obvious guesses for a
# fictional system — a grounded (vs. hallucinated) answer should contain at least one.
_GROUNDING_TERMS = ("least-privilege", "declarative", "yaml", "mediat")


def _grounded(answer: str) -> bool:
    return any(t in answer.lower() for t in _GROUNDING_TERMS)


def _snippet(text: str, n: int = 150) -> str:
    one_line = " ".join(text.split())
    return one_line[:n] + ("…" if len(one_line) > n else "")


def _ask(llm, text: str) -> str:
    content = llm.invoke(text).content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def run() -> dict:
    """Run both variants; print token + quality comparison; return the numbers.

    Uses the raw LLM (no tools) so the only path to our docs is RAG injection — the
    agent's filesystem tool would otherwise let even the no-RAG run read docs/.
    """
    cm = ContextManager()
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    # Without RAG: plain question.
    plain_answer = _ask(llm, QUESTION)
    plain_tokens = cm.count_tokens(QUESTION)

    # With RAG: prepend retrieved doc chunks.
    rag_context = retrieve(QUESTION)
    augmented = f"Relevant documentation:\n{rag_context}\n\nQuestion: {QUESTION}"
    rag_answer = _ask(llm, augmented)
    rag_tokens = cm.count_tokens(augmented)

    print(f"Without RAG: {plain_tokens} input tokens | grounded in docs: {_grounded(plain_answer)}")
    print(f"   -> {_snippet(plain_answer)}")
    print(f"With RAG:    {rag_tokens} input tokens | grounded in docs: {_grounded(rag_answer)}")
    print(f"   -> {_snippet(rag_answer)}")
    print(f"RAG context cost: +{rag_tokens - plain_tokens} tokens")
    return {
        "plain_tokens": plain_tokens,
        "rag_tokens": rag_tokens,
        "plain_grounded": _grounded(plain_answer),
        "rag_grounded": _grounded(rag_answer),
    }


if __name__ == "__main__":
    run()
