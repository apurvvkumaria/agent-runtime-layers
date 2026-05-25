"""capacity_planner — a marketplace skill that DEPENDS on other skills (Layer 28).

Requires `error_budget` (a marketplace skill) and `storage_health_check` (core).
The dependency resolver installs `error_budget` first, which is exactly why the
direct-path import below resolves at run time. Then it composes them (Layer 26)
into a capacity plan.
"""

from __future__ import annotations

from skills.context import SkillContext

_CLUSTER = "prod-us-east-1"
_TARGET = "99.9"


async def capacity_planner(cluster: str = _CLUSTER, ctx: SkillContext | None = None) -> str:
    """Capacity plan: the SLA-target error budget + the cluster's current health."""
    cluster = (cluster or _CLUSTER).strip() or _CLUSTER
    if ctx is not None:
        budget = await ctx.call_skill("error_budget", _TARGET)
        health = await ctx.call_skill("storage_health_check", cluster)
    else:
        from skills.error_budget.skill import error_budget  # installed as a dependency
        from skills.storage_health_check.skill import storage_health_check

        budget = await error_budget(_TARGET)
        health = await storage_health_check(cluster)

    return (
        f"# Capacity Plan — {cluster}\n\n"
        f"## SLA Target ({_TARGET}%)\n\n{budget}\n\n"
        f"## Current Health\n\n{health}"
    )
