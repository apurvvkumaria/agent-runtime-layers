"""Deterministic evals via deepeval.

Each case becomes a deepeval `LLMTestCase` scored by two deterministic metrics
(no LLM judge needed, so these stay cheap and repeatable):

  - `ToolCorrectnessMetric` — was the expected tool actually called?
  - `SubstringMetric` (custom) — does the answer contain the expected string?

The agent itself still runs for real (real Claude calls); only the grading is
deterministic.

Run directly:  uv run python evals/deterministic_evals.py
"""

import os
import pathlib
import sys

# Repo root on the path so `import core` works regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import BaseMetric, ToolCorrectnessMetric  # noqa: E402
from deepeval.test_case import LLMTestCase, ToolCall  # noqa: E402
from langchain_core.callbacks import BaseCallbackHandler  # noqa: E402

from core import build_single_shot_agent  # noqa: E402

# question -> expected tool the agent should call, and a substring the answer
# should contain.
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


class SubstringMetric(BaseMetric):
    """Custom deepeval metric: pass iff the answer contains an expected substring."""

    def __init__(self, substring: str, threshold: float = 1.0) -> None:
        self.substring = substring
        self.threshold = threshold
        self.evaluation_cost = 0.0  # deterministic — no model calls

    def measure(self, test_case: LLMTestCase) -> float:
        contained = self.substring.lower() in (test_case.actual_output or "").lower()
        self.score = 1.0 if contained else 0.0
        self.success = contained
        self.reason = f"answer {'contains' if contained else 'is missing'} {self.substring!r}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):  # shown in deepeval output
        return "Substring"


class ToolRecorder(BaseCallbackHandler):
    """Records the names of tools the agent calls during a run."""

    def __init__(self) -> None:
        self.tools: list[str] = []

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.tools.append((serialized or {}).get("name", "unknown"))


def run() -> tuple[int, int]:
    """Run every case through the agent + deepeval metrics; return (passed, total)."""
    executor = build_single_shot_agent()  # stateless, safe to reuse across cases
    passed = 0

    for case in CASES:
        recorder = ToolRecorder()
        result = executor.invoke({"input": case["question"]}, config={"callbacks": [recorder]})
        answer = result.get("output", "")

        test_case = LLMTestCase(
            input=case["question"],
            actual_output=answer,
            tools_called=[ToolCall(name=name) for name in recorder.tools],
            expected_tools=[ToolCall(name=case["expected_tool"])],
        )

        tool_metric = ToolCorrectnessMetric(threshold=0.5, async_mode=False)
        substring_metric = SubstringMetric(case["expected_contains"])
        tool_metric.measure(test_case)
        substring_metric.measure(test_case)

        ok = tool_metric.is_successful() and substring_metric.is_successful()
        passed += ok

        print(f"[{'PASS' if ok else 'FAIL'}] {case['question']}")
        print(f"        ToolCorrectness: {tool_metric.score:.2f} "
              f"(used {recorder.tools or '[]'}, want {case['expected_tool']})")
        print(f"        Substring: {substring_metric.score:.2f} ({substring_metric.reason})")

    print(f"\nDeterministic evals: {passed}/{len(CASES)} passed")
    return passed, len(CASES)


if __name__ == "__main__":
    run()
