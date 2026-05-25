"""Skill-level evals (Layer 24): one behavioral contract per skill.

Layer 9 evaluated the *agent*; this evaluates each *skill* in isolation — run the
skill directly (via the registry, the direct/no-ctx path) and assert its output
contract with deterministic deepeval metrics. The skill still runs for real (web
search, metrics, the SLA doc, Claude); only the grading is deterministic, so these
stay cheap and repeatable.

Contracts asserted:
  - research_and_summarize -> a markdown report with the three required sections.
  - storage_health_check   -> names the cluster, cites p99, and emits a valid SLA
    rating + an overall verdict (regardless of the randomized metric values).

Run directly:  uv run python evals/skill_evals.py
"""

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.metrics import BaseMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from evals.deterministic_evals import SubstringMetric  # noqa: E402
from skills.registry import SkillRegistry  # noqa: E402


class SubstringAnyMetric(BaseMetric):
    """Pass iff the answer contains at least one of a set of substrings."""

    def __init__(self, options: list[str], threshold: float = 1.0) -> None:
        self.options = options
        self.threshold = threshold
        self.evaluation_cost = 0.0  # deterministic — no model calls

    def measure(self, test_case: LLMTestCase) -> float:
        out = (test_case.actual_output or "").lower()
        hit = next((o for o in self.options if o.lower() in out), None)
        self.score = 1.0 if hit else 0.0
        self.success = hit is not None
        self.reason = f"matched {hit!r}" if hit else f"none of {self.options} present"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))

    @property
    def __name__(self):
        return "SubstringAny"


# Each case: which skill, the input, required substrings (all), and any-of groups.
CASES = [
    {
        "skill": "research_and_summarize",
        "input": "OpenShell sandbox runtime",
        "all_of": ["## Research Findings", "## Storage Context", "## Summary"],
        "any_groups": [],
    },
    {
        "skill": "storage_health_check",
        "input": "prod-us-east-1",
        "all_of": ["prod-us-east-1", "p99"],
        "any_groups": [
            ["Excellent", "Normal", "Degraded", "Critical"],  # a valid SLA rating
            ["HEALTHY", "NEEDS ATTENTION"],                    # an overall verdict
        ],
    },
]


def run() -> tuple[int, int]:
    """Run every skill case + its metrics; return (passed, total)."""
    registry = SkillRegistry().auto_discover()
    passed = 0

    for case in CASES:
        tool = registry.get_skill(case["skill"])
        output = tool.invoke(case["input"])
        tc = LLMTestCase(input=case["input"], actual_output=output)

        checks: list[tuple[str, bool, str]] = []
        for sub in case["all_of"]:
            m = SubstringMetric(sub)
            m.measure(tc)
            checks.append(("Substring", m.is_successful(), m.reason))
        for group in case["any_groups"]:
            m = SubstringAnyMetric(group)
            m.measure(tc)
            checks.append(("SubstringAny", m.is_successful(), m.reason))

        ok = all(c[1] for c in checks)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {case['skill']}({case['input']!r})")
        for name, success, reason in checks:
            print(f"        {name}: {'✓' if success else '✗'} {reason}")

    print(f"\nSkill evals: {passed}/{len(CASES)} passed")
    return passed, len(CASES)


if __name__ == "__main__":
    run()
