"""Run the whole quality gate: unit/integration tests, then both eval suites.

Prints a final summary. Invoked by `agent test`, or directly:
    uv run python evals/run_all.py
"""

import pathlib
import sys

# Repo root on the path so `import evals.*` and the top-level modules resolve.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from evals.cache import get_cache  # noqa: E402
from evals.deterministic_evals import run as run_deterministic  # noqa: E402
from evals.langfuse_evals import run as run_langfuse  # noqa: E402
from evals.skill_evals import run as run_skill_evals  # noqa: E402


class _ResultCollector:
    """Counts passed/failed test calls so we can report X/Y."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def main() -> None:
    print("\n========== UNIT + INTEGRATION TESTS ==========")
    collector = _ResultCollector()
    pytest.main(["tests", "-q"], plugins=[collector])
    unit_passed = collector.passed
    unit_total = collector.passed + collector.failed

    print("\n========== DETERMINISTIC EVALS ==========")
    det_passed, det_total = run_deterministic()

    print("\n========== SKILL EVALS ==========")
    skill_passed, skill_total = run_skill_evals()

    print("\n========== LANGFUSE EVALS ==========")
    lf_avg = run_langfuse()

    print("\n========== SUMMARY ==========")
    print(f"Unit tests: {unit_passed}/{unit_total} passed")
    print(f"Deterministic evals: {det_passed}/{det_total} passed")
    print(f"Skill evals: {skill_passed}/{skill_total} passed")
    if lf_avg is None:
        print("LangFuse evals: skipped (not configured)")
    else:
        print(f"LangFuse evals: avg answer_relevancy {lf_avg:.2f}/1.0")

    # Layer 22: what the judge-response cache saved this run.
    c = get_cache().stats()
    print(f"Cache: {c['hits']} hits, {c['misses']} misses, {c['tokens_saved']} tokens saved "
          f"(~${c['cost_saved']:.4f})")


if __name__ == "__main__":
    main()
