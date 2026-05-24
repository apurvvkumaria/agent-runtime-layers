"""Deterministic evals: run the agent on fixed cases and check tool + answer.

Each case asserts two things: the expected tool was actually called, and the
final answer contains an expected substring. Unlike the unit tests, these run the
real agent (real Claude calls) — they measure behavior, not just plumbing.

Run directly:  uv run python evals/deterministic_evals.py
"""

import pathlib
import sys

# Repo root on the path so `import core` works regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from core import build_single_shot_agent  # noqa: E402

# question -> which tool we expect the agent to call, and a substring the final
# answer should contain.
CASES = [
    {"question": "What is 25 times 4?", "expected_tool": "calculator", "expected_contains": "100"},
    {"question": "What is 1000 divided by 8?", "expected_tool": "calculator", "expected_contains": "125"},
    {"question": "What is 256 minus 56?", "expected_tool": "calculator", "expected_contains": "200"},
    {
        "question": "Show the storage metrics for cluster prod-east-1.",
        "expected_tool": "storage_metrics",
        "expected_contains": "prod-east-1",
    },
    {
        "question": "Search the web and tell me what year the Eiffel Tower was completed.",
        "expected_tool": "duckduckgo_search",
        "expected_contains": "1889",
    },
]


class ToolRecorder(BaseCallbackHandler):
    """Records the names of tools the agent calls during a run."""

    def __init__(self) -> None:
        self.tools: list[str] = []

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.tools.append((serialized or {}).get("name", "unknown"))


def run() -> tuple[int, int]:
    """Run every case; return (passed, total) and print per-case results."""
    executor = build_single_shot_agent()  # stateless, safe to reuse across cases
    passed = 0

    for case in CASES:
        recorder = ToolRecorder()
        result = executor.invoke({"input": case["question"]}, config={"callbacks": [recorder]})
        answer = result.get("output", "")

        tool_ok = case["expected_tool"] in recorder.tools
        contains_ok = case["expected_contains"].lower() in answer.lower()
        ok = tool_ok and contains_ok
        passed += ok

        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']}")
        print(f"        tool: want {case['expected_tool']}, used {recorder.tools or '[]'}"
              f" -> {'ok' if tool_ok else 'MISS'}")
        print(f"        answer contains {case['expected_contains']!r}: "
              f"{'ok' if contains_ok else 'MISS'}")

    print(f"\nDeterministic evals: {passed}/{len(CASES)} passed")
    return passed, len(CASES)


if __name__ == "__main__":
    run()
