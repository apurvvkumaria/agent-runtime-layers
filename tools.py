"""Tool definitions for the agent: web search, calculator, storage metrics.

Each tool is a plain callable the LLM can choose to invoke. `get_tools()` returns
the full list wired into the agent; individual tools are also imported directly by
the CLI/API for no-LLM "direct tool" access.
"""

import ast
import operator
import random

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Allowlist of AST node types -> the operator function they map to. Anything
# outside this set is rejected, so the calculator can't execute arbitrary code
# (the safety goal behind the ast.literal_eval pattern).
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST, allowing only numbers and math ops.

    ast.literal_eval can't handle operators like "2 + 2", so we parse to an AST
    ourselves and walk an explicit allowlist of node types — same safety idea,
    extended to arithmetic.
    """
    if isinstance(node, ast.Constant):  # a literal number
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Use this for any math, e.g. "2 + 2", "100 * 0.15", or "150 * 1234.56".
    Supports + - * / // % ** and parentheses. Input must be a single expression.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: could not evaluate {expression!r} ({exc})"


@tool
def storage_metrics(cluster_name: str) -> str:
    """Return current distributed-storage metrics for a named cluster.

    Use this to look up live operational metrics (throughput, latency, disk,
    replication, partitions) for a storage cluster by name, e.g. "prod-us-east-1".
    """
    # Fake but realistic numbers — a stand-in for a real metrics backend.
    rps = random.randint(50_000, 100_000)
    p99_latency_ms = round(random.uniform(1, 5), 2)
    disk_utilization_pct = round(random.uniform(40, 85), 1)
    replication_lag_ms = random.randint(0, 50)
    active_partitions = random.randint(100, 500)

    return (
        f"Storage metrics for cluster '{cluster_name}':\n"
        f"  requests_per_second:  {rps:,}\n"
        f"  p99_latency_ms:       {p99_latency_ms}\n"
        f"  disk_utilization_pct: {disk_utilization_pct}\n"
        f"  replication_lag_ms:   {replication_lag_ms}\n"
        f"  active_partitions:    {active_partitions}"
    )


def get_tools() -> list:
    """The full tool list the agent can choose from.

    Web search + calculator + storage metrics + the MCP-backed filesystem reader
    (imported lazily so the mcp dependency only loads when an agent is built).
    """
    from mcp_integration.client import filesystem

    return [DuckDuckGoSearchRun(), calculator, storage_metrics, filesystem]
