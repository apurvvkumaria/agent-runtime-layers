"""error_budget — a skill distributed via the marketplace (Layer 27).

Pure arithmetic: no tools, no network. Follows the OpenClaw skill contract
(`async def <name>(arg, ctx=None)`) so the SkillRegistry loads it like any other
skill once it's been verified and installed from the marketplace.
"""

from __future__ import annotations

from skills.context import SkillContext

_MINUTES_PER_MONTH = 30 * 24 * 60  # 30-day month


async def error_budget(availability: str = "99.9", ctx: SkillContext | None = None) -> str:
    """Monthly error budget (allowed downtime) for a target availability percent."""
    pct = float(str(availability).strip().rstrip("%") or "99.9")
    allowed = _MINUTES_PER_MONTH * (1 - pct / 100)
    return (
        f"At {pct}% availability, the monthly error budget is {allowed:.1f} minutes "
        f"(~{allowed / 60:.2f} hours) of allowed downtime per 30-day month."
    )
