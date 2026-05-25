"""SkillContext — the OpenClaw-style execution context passed to a skill.

A skill is written once and runs two ways:

  - **with a `ctx`** — tool calls go through `await ctx.call_tool(name, **kwargs)`,
    which routes them through the broker/sandbox (this is exactly how an OpenClaw
    skill calls `ctx.call_claw(...)`; `sandbox_id` + `policy` are the run context).
  - **without a `ctx`** (`ctx=None`) — the skill falls back to calling the tools
    directly in-process.

Defined here (not duplicated in each `skill.py`) so both skills share one
`SkillContext`; the bridge to OpenClaw is the *pattern*, not the file location.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field


def run_coro(coro):
    """Run a coroutine to completion whether or not we're already in an event loop.

    Skills are async; the ReAct executor calls tools from both sync (`invoke`) and
    async (`astream_events`) contexts. Running in a fresh loop on a worker thread
    works in either case (same trick as the MCP filesystem tool).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


async def _dispatch(name: str, **kwargs) -> str:
    """The 'broker': map a logical tool name to its implementation.

    In OpenClaw this hop is where the sandbox/policy is enforced; here it just
    dispatches to the in-process tools, lazily imported so importing a skill stays
    cheap.
    """
    if name == "web_search":
        from langchain_community.tools import DuckDuckGoSearchRun
        return DuckDuckGoSearchRun().invoke(kwargs["query"])
    if name == "storage_metrics":
        from tools import storage_metrics
        return storage_metrics.invoke(kwargs["cluster"])
    if name == "filesystem":
        from mcp_integration.client import filesystem
        return filesystem.invoke(kwargs["path"])
    if name == "llm_summarize":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
        fmt = kwargs.get("format", "markdown")
        msg = llm.invoke(
            f"Summarize the material below into a clear {fmt} report.\n\n{kwargs['content']}"
        )
        content = msg.content
        if isinstance(content, str):
            return content
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    raise ValueError(f"unknown tool {name!r}")


@dataclass
class SkillContext:
    """Run context handed to a skill — mirrors OpenClaw's SkillContext."""

    sandbox_id: str = "local"
    policy: dict = field(default_factory=dict)
    depth: int = 0          # composition depth (Layer 26): skills calling skills
    max_depth: int = 5      # hard stop so composition can't recurse unbounded

    async def call_tool(self, name: str, **kwargs) -> str:
        """Invoke a tool by logical name through the broker (await-able)."""
        return await _dispatch(name, **kwargs)

    async def call_skill(self, name: str, arg: str) -> str:
        """Invoke another *skill* by name (Layer 26 — skill composition).

        Resolves the skill through the registry and runs it with a child context
        whose depth is incremented, so a chain of skills-calling-skills is bounded
        by `max_depth` (a self-directing runaway guard, like the heartbeat loop's).
        """
        if self.depth >= self.max_depth:
            raise RuntimeError(
                f"skill composition depth exceeded ({self.max_depth}) calling {name!r}"
            )
        from skills.registry import SkillRegistry  # lazy: avoids a context<->registry cycle

        fn = SkillRegistry().auto_discover().skill_function(name)
        child = SkillContext(
            sandbox_id=self.sandbox_id,
            policy=self.policy,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )
        return await fn(arg, ctx=child)
