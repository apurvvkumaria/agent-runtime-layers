"""SkillRegistry — discover OpenClaw-style skills and expose them as agent tools.

Each skill is a directory `skills/<name>/` with `SKILL.md` (capability declaration
the LLM reads), `skill.py` (an `async def <name>(arg, ctx=None)`), and `policy.yaml`
(skill-level capability policy). `auto_discover()` scans for them, parses each
`SKILL.md`, imports the function, and wraps it as a LangChain `Tool` so the ReAct
agent can call a skill by name exactly like any other tool.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from langchain_core.tools import Tool

from skills.context import run_coro

_SKILLS_DIR = Path(__file__).resolve().parent


def _section(text: str, title: str) -> str:
    """Return the body of a `## <title>` section from a SKILL.md."""
    m = re.search(rf"^##\s*{re.escape(title)}\s*\n(.*?)(?=\n##\s|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def _parse_skill_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    tools_block = _section(text, "Tools used internally")
    tools_used = re.findall(r"^[-*]\s*`?([A-Za-z0-9_]+)`?", tools_block, re.M)
    return {
        "what": _section(text, "What it does"),
        "when": _section(text, "When to use it"),
        "tools_used": tools_used,
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
            {"name": n, "description": s["meta"]["what"], "tools_used": s["meta"]["tools_used"]}
            for n, s in self._skills.items()
        ]

    def _as_tool(self, name: str, entry: dict) -> Tool:
        fn, meta = entry["fn"], entry["meta"]
        description = meta["what"]
        if meta["when"]:
            description += f"\n\nWhen to use: {meta['when']}"

        def _sync(arg: str) -> str:  # ReAct passes a single Action Input string
            return run_coro(fn(arg))

        async def _async(arg: str) -> str:
            return await fn(arg)

        return Tool(name=name, description=description, func=_sync, coroutine=_async)

    def get_skill(self, name: str) -> Tool:
        """Return a single skill as a callable LangChain `Tool` (use `.invoke(arg)`)."""
        return self._as_tool(name, self._skills[name])

    def as_tools(self) -> list[Tool]:
        return [self._as_tool(n, e) for n, e in self._skills.items()]

    def inject_into_agent(self, executor):
        """Append all discovered skills to a built executor's tool list.

        Note: the ReAct prompt enumerates tools at *build* time, so the canonical
        wiring is including `as_tools()` in `get_tools()`. This is for dynamic
        addition (e.g. native tool-calling agents that re-read the tool list).
        """
        executor.tools = list(getattr(executor, "tools", [])) + self.as_tools()
        return executor
