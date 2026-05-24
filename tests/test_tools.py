"""Unit tests for the agent's tools (no LLM involved)."""

import ast

import pytest

from tools import _safe_eval, calculator, storage_metrics

_METRIC_FIELDS = [
    "requests_per_second",
    "p99_latency_ms",
    "disk_utilization_pct",
    "replication_lag_ms",
    "active_partitions",
]


def test_calculator_addition():
    assert calculator.invoke("2 + 2") == "4"


def test_calculator_multiplication():
    assert calculator.invoke("150 * 223.48") == "33522.0"


def test_calculator_subtraction():
    assert calculator.invoke("100 - 42") == "58"


def test_calculator_blocks_dangerous_input():
    # The safe evaluator must reject anything that isn't pure arithmetic.
    tree = ast.parse("__import__('os')", mode="eval")
    with pytest.raises(ValueError):
        _safe_eval(tree.body)
    # And the tool wrapper degrades to an error string rather than executing it.
    assert calculator.invoke("__import__('os')").startswith("Error")


def test_storage_metrics_has_all_fields():
    output = storage_metrics.invoke("test-cluster")
    for field in _METRIC_FIELDS:
        assert field in output


def _parse_metrics(text: str) -> dict[str, float]:
    """Pull the numeric fields out of the formatted metrics block."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip().replace(",", "")
        if not val:
            continue
        try:
            values[key] = float(val)
        except ValueError:
            pass
    return values


def test_storage_metrics_values_in_range():
    # Values are randomized, so sample several times to exercise the bounds.
    for _ in range(20):
        metrics = _parse_metrics(storage_metrics.invoke("c"))
        assert 1 <= metrics["p99_latency_ms"] <= 5
        assert 50_000 <= metrics["requests_per_second"] <= 100_000
