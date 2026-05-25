"""storage_sla_report — DEPRECATED (v0.9.0), replaced by storage_health_check.

Kept only for backward compatibility through the deprecation window (remove after
2026-09-30). It delegates entirely to `storage_health_check`; the SkillRegistry
hides it from the agent's tool list and emits a `[deprecated]` warning on use.
"""

from __future__ import annotations

from skills.context import SkillContext

_CLUSTER = "prod-us-east-1"


async def storage_sla_report(cluster: str = _CLUSTER, ctx: SkillContext | None = None) -> str:
    """(Deprecated) Assess a cluster's health — delegates to storage_health_check."""
    from skills.storage_health_check.skill import storage_health_check

    return await storage_health_check(cluster, ctx)
