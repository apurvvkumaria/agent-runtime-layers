"""research_and_summarize — compose web search + storage metrics + LLM summary.

OpenClaw-pattern skill: one async function that runs two ways. With a `ctx`, every
tool call goes through `await ctx.call_tool(...)` (the broker/sandbox path, exactly
like `ctx.call_claw(...)`); with `ctx=None` it falls back to calling the tools
directly in-process. The agent invokes it through the direct path (it's wrapped as
a LangChain tool by the SkillRegistry); the ctx path is the bridge to OpenClaw.
"""

from __future__ import annotations

from skills.context import SkillContext

_CLUSTER = "prod-us-east-1"


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


async def research_and_summarize(topic: str, ctx: SkillContext | None = None) -> str:
    """Research a topic and return a full structured markdown report.

    Composes web search + storage metrics + LLM summarization into one report with
    `## Research Findings`, `## Storage Context`, and `## Summary` sections.
    """
    if ctx is not None:
        # Broker path — identical in shape to an OpenClaw skill.
        search_results = await ctx.call_tool("web_search", query=topic)
        metrics = await ctx.call_tool("storage_metrics", cluster=_CLUSTER)
        content = (
            f"Topic: {topic}\n\nWeb search results:\n{search_results}\n\n"
            f"Storage metrics ({_CLUSTER}):\n{metrics}"
        )
        return await ctx.call_tool("llm_summarize", content=content, format="markdown")

    # Direct path (no broker) — call the tools in-process.
    from langchain_anthropic import ChatAnthropic
    from langchain_community.tools import DuckDuckGoSearchRun

    from tools import storage_metrics

    search_results = DuckDuckGoSearchRun().invoke(topic)
    metrics = storage_metrics.invoke(_CLUSTER)
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    prompt = (
        "Write a markdown report on the topic below using the provided material. "
        "Use EXACTLY these three sections, in this order:\n"
        "## Research Findings  — synthesize the web search results\n"
        f"## Storage Context    — interpret the {_CLUSTER} metrics\n"
        "## Summary            — a short overall takeaway tying them together\n\n"
        f"Topic: {topic}\n\n"
        f"Web search results:\n{search_results}\n\n"
        f"Storage metrics ({_CLUSTER}):\n{metrics}"
    )
    return _text(llm.invoke(prompt)).strip()
