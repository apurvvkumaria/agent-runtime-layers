"""storage_health_check — assess a cluster against the SLA thresholds.

OpenClaw-pattern skill (see research_and_summarize/skill.py for the ctx/no-ctx
contract). This one is rule-based — no LLM, no network — which matches its
filesystem-only policy: it pulls live metrics, reads the SLA bands from
docs/sla_thresholds.md via the filesystem tool, and compares them.
"""

from __future__ import annotations

import re

from skills.context import SkillContext

_CLUSTER = "prod-us-east-1"


def _num(metrics: str, field: str) -> float | None:
    m = re.search(rf"{field}:\s*([\d.]+)", metrics)
    return float(m.group(1)) if m else None


def _assess(cluster: str, metrics: str, thresholds: str) -> str:
    p99 = _num(metrics, "p99_latency_ms")
    disk = _num(metrics, "disk_utilization_pct")
    repl = _num(metrics, "replication_lag_ms")
    if p99 is None:
        return f"Could not assess '{cluster}' — no p99_latency_ms in metrics.\n\n{metrics}"

    # SLA bands from docs/sla_thresholds.md.
    if p99 < 5:
        rating = "Excellent"
    elif p99 < 20:
        rating = "Normal"
    elif p99 < 50:
        rating = "Degraded"
    else:
        rating = "Critical"

    flags = []
    if disk is not None and disk > 85:
        flags.append(f"disk utilization {disk}% > 85%")
    if repl is not None and repl > 50:
        flags.append(f"replication lag {repl}ms > 50ms")

    healthy = rating in ("Excellent", "Normal") and not flags
    verdict = "HEALTHY" if healthy else "NEEDS ATTENTION"
    thresholds_note = (
        "SLA bands per docs/sla_thresholds.md"
        if thresholds and not thresholds.startswith("Error")
        else "SLA bands (docs/sla_thresholds.md unavailable; using known bands)"
    )

    lines = [
        f"## Health assessment — {cluster}: {verdict}",
        f"- p99 latency: {p99} ms → **{rating}** (<5 Excellent, 5–20 Normal, 20–50 Degraded, >50 Critical)",
        f"- disk utilization: {disk}%" + ("  ⚠️" if disk is not None and disk > 85 else ""),
        f"- replication lag: {repl} ms" + ("  ⚠️" if repl is not None and repl > 50 else ""),
    ]
    if flags:
        lines.append("- flags: " + "; ".join(flags))
    lines.append(f"\n_{thresholds_note}._")
    return "\n".join(lines)


async def storage_health_check(cluster: str = _CLUSTER, ctx: SkillContext | None = None) -> str:
    """Assess a storage cluster's health against its SLA thresholds."""
    cluster = (cluster or _CLUSTER).strip() or _CLUSTER
    if ctx is not None:
        metrics = await ctx.call_tool("storage_metrics", cluster=cluster)
        thresholds = await ctx.call_tool("filesystem", path="sla_thresholds.md")
    else:
        from mcp_integration.client import filesystem

        from tools import storage_metrics

        metrics = storage_metrics.invoke(cluster)
        thresholds = filesystem.invoke("sla_thresholds.md")
    return _assess(cluster, metrics, thresholds)
