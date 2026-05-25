"""cluster_briefing — a skill composed of other skills (Layer 26).

Unlike the other skills (which compose *tools*), this one composes *skills*:
`storage_health_check` for the live SLA assessment and `research_and_summarize`
for background. With a `ctx` it goes through `await ctx.call_skill(...)` (which
runs each sub-skill with a depth-incremented child context, so composition is
bounded); with `ctx=None` it calls the sub-skill functions directly.
"""

from __future__ import annotations

from skills.context import SkillContext

_CLUSTER = "prod-us-east-1"
# A fixed, relevant research topic — the briefing pairs this cluster's live health
# with general reliability/SLA background.
_TOPIC = "distributed storage cluster reliability and SLA best practices"


async def cluster_briefing(cluster: str = _CLUSTER, ctx: SkillContext | None = None) -> str:
    """Full briefing on a cluster: live health check + background research."""
    cluster = (cluster or _CLUSTER).strip() or _CLUSTER

    if ctx is not None:
        health = await ctx.call_skill("storage_health_check", cluster)
        research = await ctx.call_skill("research_and_summarize", _TOPIC)
    else:
        from skills.research_and_summarize.skill import research_and_summarize
        from skills.storage_health_check.skill import storage_health_check

        health = await storage_health_check(cluster)
        research = await research_and_summarize(_TOPIC)

    return (
        f"# Cluster Briefing — {cluster}\n\n"
        f"## Health\n\n{health}\n\n"
        f"## Background\n\n{research}"
    )
