"""A skill: one tool that composes several tools internally.

`research_and_summarize` looks like a single tool to the agent, but inside it
runs three steps and stitches the results into one structured report:

  1. web search (DuckDuckGo) for the topic
  2. storage metrics for the prod-us-east-1 cluster
  3. LLM summarization into a markdown report (Research Findings / Storage
     Context / Summary)

The agent calls one tool; the skill handles the orchestration — the same pattern
as OpenClaw skills.
"""

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from tools import storage_metrics

_CLUSTER = "prod-us-east-1"


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@tool
def research_and_summarize(topic: str) -> str:
    """Research a topic and return a full structured markdown report.

    Composes web search + storage metrics + LLM summarization into one report with
    `## Research Findings`, `## Storage Context`, and `## Summary` sections. Use
    this when the user asks for a full report, a research summary, or to "research
    and summarize" a topic — not for simple lookups.
    """
    # Step 1: search the web.
    search_results = DuckDuckGoSearchRun().invoke(topic)
    # Step 2: pull storage metrics for the reference cluster.
    metrics = storage_metrics.invoke(_CLUSTER)
    # Step 3: combine into a structured report.
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
