"""The research pipeline rebuilt with Strands Agents.

Same tools (search, calculator, storage_metrics) and the same orchestrator →
research/calculator structure as the LangGraph pipeline — but here there is *no*
explicit graph. The orchestrator is an agent that holds the two specialists as
tools (via `Agent.as_tool()`); the model itself decides which to call and when.
That's the contrast with LangGraph: emergent, model-driven routing vs. a fixed
topology.

Uses our Anthropic API key via Strands' AnthropicModel (not Bedrock).
"""

import os

from strands import Agent, tool
from strands.models.anthropic import AnthropicModel

from langchain_community.tools import DuckDuckGoSearchRun
from tools import calculator as _lc_calculator
from tools import storage_metrics as _lc_storage_metrics

_MODEL_ID = "claude-sonnet-4-6"


# --- tools (Strands @tool, delegating to the same underlying logic) --------- #
@tool
def web_search(query: str) -> str:
    """Search the web for current information about a topic."""
    return DuckDuckGoSearchRun().invoke(query)


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '5 * 223.48'."""
    return _lc_calculator.invoke(expression)


@tool
def storage_metrics(cluster_name: str) -> str:
    """Get current distributed-storage metrics for a named cluster."""
    return _lc_storage_metrics.invoke(cluster_name)


def _model() -> AnthropicModel:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    return AnthropicModel(
        client_args={"api_key": os.environ["ANTHROPIC_API_KEY"]},
        model_id=_MODEL_ID,
        max_tokens=2048,
    )


def build_orchestrator() -> tuple[Agent, list[Agent]]:
    """Build the orchestrator + its specialist sub-agents (returned for metrics)."""
    model = _model()
    # callback_handler=None silences Strands' default streaming-to-stdout so callers
    # control the output.
    research = Agent(
        model=model,
        name="research_specialist",
        system_prompt="You research topics with web_search and return concise factual findings.",
        tools=[web_search],
        callback_handler=None,
    )
    calc = Agent(
        model=model,
        name="calculator_specialist",
        system_prompt="You do precise arithmetic with the calculate tool and fetch storage metrics.",
        tools=[calculate, storage_metrics],
        callback_handler=None,
    )
    orchestrator = Agent(
        model=model,
        name="orchestrator",
        system_prompt=(
            "You answer research questions. Use research_specialist for factual or current "
            "information and calculator_specialist for any arithmetic. Call whichever you need "
            "(both if the question requires research and math), then combine their results into "
            "one clear final answer."
        ),
        tools=[
            research.as_tool(
                name="research_specialist",
                description="Research a topic via web search; returns factual findings.",
            ),
            calc.as_tool(
                name="calculator_specialist",
                description="Perform arithmetic or fetch storage metrics.",
            ),
        ],
        callback_handler=None,
    )
    return orchestrator, [research, calc]


def _answer_text(result) -> str:
    message = getattr(result, "message", None)
    if isinstance(message, dict):
        return "".join(
            b.get("text", "") for b in message.get("content", []) if isinstance(b, dict)
        ).strip()
    return str(result).strip()


def _usage_total(metrics) -> int:
    """Best-effort total-token extraction from a Strands EventLoopMetrics."""
    usage = getattr(metrics, "accumulated_usage", None)
    if usage is None and hasattr(metrics, "get_summary"):
        usage = (metrics.get_summary() or {}).get("accumulated_usage", {})
    usage = usage or {}
    return usage.get("totalTokens") or usage.get("total_tokens") or 0


def run_strands(question: str) -> dict:
    """Run the orchestrator on the question; return answer, tokens, and step count."""
    orchestrator, subs = build_orchestrator()
    result = orchestrator(question)

    answer = _answer_text(result)
    tokens = _usage_total(result.metrics)
    for sub in subs:  # add sub-agent usage (they run as tools, separate metrics)
        tokens += _usage_total(getattr(sub, "event_loop_metrics", None))
    steps = getattr(result.metrics, "cycle_count", None)
    return {"answer": answer, "tokens": tokens, "steps": steps}
