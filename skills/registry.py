"""SkillRegistry — discover OpenClaw-style skills and expose them as agent tools.

Each skill is a directory `skills/<name>/` with `SKILL.md` (capability declaration
the LLM reads), `skill.py` (an `async def <name>(arg, ctx=None)`), and `policy.yaml`
(skill-level capability policy). `auto_discover()` scans for them, parses each
`SKILL.md`, imports the function, and wraps it as a LangChain `Tool` so the ReAct
agent can call a skill by name exactly like any other tool.

Versioning & deprecation (Layer 25): `SKILL.md` declares `## Version` (semver) and
`## Status` (active|deprecated). A deprecated skill adds a `## Deprecation` block
(`Replaced by:` / `Remove after:`). Deprecated skills are **excluded from the
agent's tool list** (`as_tools()` defaults to active-only, so the agent won't route
to them), but stay resolvable via `get_skill()` for backward-compatible direct
callers — and emit a `[deprecated]` warning when invoked.
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

from langchain_core.tools import Tool

from skills.context import run_coro

_SKILLS_DIR = Path(__file__).resolve().parent


def _section(text: str, title: str) -> str:
    """Return the body of a `## <title>` section from a SKILL.md."""
    m = re.search(rf"^##\s*{re.escape(title)}\s*\n(.*?)(?=\n##\s|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def _first_line(text: str, title: str, default: str = "") -> str:
    body = _section(text, title)
    return body.splitlines()[0].strip() if body else default


def _parse_skill_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    tools_block = _section(text, "Tools used internally")
    deprecation = _section(text, "Deprecation")
    status = _first_line(text, "Status", "active").lower()
    replaced = re.search(r"[Rr]eplaced by:\s*`?([A-Za-z0-9_]+)`?", deprecation)
    remove_after = re.search(r"[Rr]emove after:\s*([0-9][0-9-]+)", deprecation)
    return {
        "what": _section(text, "What it does"),
        "when": _section(text, "When to use it"),
        "tools_used": re.findall(r"^[-*]\s*`?([A-Za-z0-9_]+)`?", tools_block, re.M),
        "version": _first_line(text, "Version", "0.0.0"),
        "status": status,
        "deprecated": status == "deprecated" or bool(deprecation),
        "replaced_by": replaced.group(1) if replaced else None,
        "remove_after": remove_after.group(1) if remove_after else None,
    }


class SkillRegistry:
    """Discovers `skills/<name>/` packages and serves them as tools."""

    def __init__(self) -> None:
        self._skills: dict[str, dict] = {}

    def auto_discover(self) -> "SkillRegistry":
        """Scan `skills/` for `<name>/SKILL.md` + `skill.py`, importing each."""
        for sub in sorted(p for p in _SKILLS_DIR.iterdir() if p.is_dir()):
            md, py = sub / "SKILL.md", sub / "skill.py"
            if not (md.exists() and py.exists()):
                continue
            meta = _parse_skill_md(md)
            module = importlib.import_module(f"skills.{sub.name}.skill")
            fn = getattr(module, sub.name, None)
            if fn is None:  # convention: skill.py defines a function named after its dir
                continue
            self._skills[sub.name] = {"fn": fn, "meta": meta}
        return self

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": n,
                "description": s["meta"]["what"],
                "tools_used": s["meta"]["tools_used"],
                "version": s["meta"]["version"],
                "status": s["meta"]["status"],
                "deprecated": s["meta"]["deprecated"],
                "replaced_by": s["meta"]["replaced_by"],
                "remove_after": s["meta"]["remove_after"],
            }
            for n, s in self._skills.items()
        ]

    def _deprecation_note(self, name: str, meta: dict) -> str:
        note = f"[deprecated] skill '{name}' (v{meta['version']}) is deprecated"
        if meta["replaced_by"]:
            note += f" — use '{meta['replaced_by']}'"
        if meta["remove_after"]:
            note += f"; removed after {meta['remove_after']}"
        return note

    def _as_tool(self, name: str, entry: dict) -> Tool:
        fn, meta = entry["fn"], entry["meta"]
        description = meta["what"]
        if meta["when"]:
            description += f"\n\nWhen to use: {meta['when']}"
        if meta["deprecated"]:
            replacement = f" — use {meta['replaced_by']}" if meta["replaced_by"] else ""
            description = f"[DEPRECATED{replacement}] " + description

        def _warn() -> None:
            if meta["deprecated"]:
                print(self._deprecation_note(name, meta), file=sys.stderr)

        def _sync(arg: str) -> str:  # ReAct passes a single Action Input string
            _warn()
            return run_coro(fn(arg))

        async def _async(arg: str) -> str:
            _warn()
            return await fn(arg)

        return Tool(name=name, description=description, func=_sync, coroutine=_async)

    def get_skill(self, name: str) -> Tool:
        """Return a single skill as a callable LangChain `Tool` (use `.invoke(arg)`).

        Works for deprecated skills too — they stay resolvable for compatibility and
        warn on use. (`as_tools()` is what hides them from the agent's tool list.)
        """
        return self._as_tool(name, self._skills[name])

    def as_tools(self, include_deprecated: bool = False) -> list[Tool]:
        """The skills to wire into an agent — active only by default.

        Deprecated skills are dropped so the agent never routes to them; pass
        `include_deprecated=True` to keep them callable from the agent for a
        compatibility window.
        """
        return [
            self._as_tool(n, e)
            for n, e in self._skills.items()
            if include_deprecated or not e["meta"]["deprecated"]
        ]

    def inject_into_agent(self, executor, include_deprecated: bool = False):
        """Append discovered skills to a built executor's tool list.

        Note: the ReAct prompt enumerates tools at *build* time, so the canonical
        wiring is including `as_tools()` in `get_tools()`. This is for dynamic
        addition (e.g. native tool-calling agents that re-read the tool list).
        """
        existing = list(getattr(executor, "tools", []))
        executor.tools = existing + self.as_tools(include_deprecated=include_deprecated)
        return executor
