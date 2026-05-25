"""Skill dependency resolution (Layer 28).

A skill declares `## Requires` in its SKILL.md — other skills it needs, with
optional semver constraints (`error_budget >= 1.0.0`). `SkillResolver` walks that
graph for a target marketplace skill and returns the **install order** (deps
first), partitioning each dependency into:

  - already satisfied — installed and meets the constraint (skipped),
  - to install — available in the marketplace and meets the constraint,

and raising `DependencyError` on a missing/unsatisfiable dependency or a cycle.

Requires are read from each skill's SKILL.md (hash-covered by the marketplace
index, Layer 27), so the dependency declaration can't be tampered with undetected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from skills.marketplace import SkillMarketplace
from skills.registry import SkillRegistry, _parse_skill_md


class DependencyError(RuntimeError):
    """A dependency is missing, version-unsatisfiable, or forms a cycle."""


def _ver(v: str) -> tuple:
    nums = [int(x) for x in re.findall(r"\d+", v)][:3]
    return tuple(nums + [0] * (3 - len(nums)))


def _satisfies(version: str, constraint: str) -> bool:
    """Does `version` satisfy a constraint like `>=1.0.0`? Empty constraint = any."""
    if not constraint:
        return True
    m = re.match(r"(>=|<=|==|!=|>|<)\s*([0-9][0-9.]*)", constraint)
    if not m:
        return True
    op, target = m.group(1), _ver(m.group(2))
    cur = _ver(version)
    return {
        ">=": cur >= target, "<=": cur <= target, "==": cur == target,
        "!=": cur != target, ">": cur > target, "<": cur < target,
    }[op]


@dataclass
class Resolution:
    target: str
    order: list[str]                       # marketplace skills to install, deps first
    already_satisfied: list[str] = field(default_factory=list)
    graph: dict[str, list] = field(default_factory=dict)  # name -> [{name, constraint}]


class SkillResolver:
    def __init__(self, marketplace: SkillMarketplace | None = None) -> None:
        self.mp = marketplace or SkillMarketplace()
        self._available = {e["name"]: e for e in self.mp.available()}
        self._installed = {s["name"]: s["version"] for s in SkillRegistry().auto_discover().list_skills()}

    def _requires(self, name: str) -> list[dict]:
        """Requires for an available marketplace skill (read from its SKILL.md)."""
        entry = self._available[name]
        return _parse_skill_md(self.mp.source / entry["path"] / "SKILL.md")["requires"]

    def resolve(self, target: str) -> Resolution:
        if target not in self._available:
            raise DependencyError(f"{target!r} is not in the marketplace")
        res = Resolution(target=target, order=[])
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str, stack: list[str]) -> None:
            if name in done:
                return
            if name in visiting:
                raise DependencyError("dependency cycle: " + " -> ".join(stack + [name]))
            visiting.add(name)
            reqs = self._requires(name)
            res.graph[name] = reqs
            for r in reqs:
                rn, rc = r["name"], r["constraint"]
                installed = self._installed.get(rn)
                if installed is not None and _satisfies(installed, rc):
                    res.already_satisfied.append(f"{rn} (installed {installed})")
                elif rn in self._available and _satisfies(self._available[rn]["version"], rc):
                    visit(rn, stack + [name])  # needs installing (deps appended first)
                elif installed is not None:
                    raise DependencyError(
                        f"{name} requires {rn}{rc}, but installed {installed} and the "
                        f"marketplace can't satisfy it"
                    )
                else:
                    raise DependencyError(
                        f"{name} requires {rn}{rc} — not installed and not in the marketplace"
                    )
            visiting.discard(name)
            done.add(name)
            res.order.append(name)  # after its deps

        visit(target, [])
        # de-dupe already_satisfied while keeping order
        res.already_satisfied = list(dict.fromkeys(res.already_satisfied))
        return res
