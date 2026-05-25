"""A three-stage, five-agent research pipeline built with LangGraph.

Flow (a StateGraph):

    START -> orchestrator -> {research | calculator}
    research -> {calculator (if "both") | writer}
    calculator -> writer -> reviewer
    reviewer -> research (retry, if quality < 0.7 and retry_count < 2) | END

Each node is a small "agent": orchestrator routes by inspecting the question;
research calls DuckDuckGo; calculator derives + evaluates an arithmetic expression;
writer drafts an answer; reviewer scores it and decides whether to loop back.
"""

import os
import re
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import END, START, StateGraph

from tools import calculator


class ResearchState(TypedDict):
    question: str
    search_results: str
    calculation: str
    draft_answer: str
    quality_score: float
    final_answer: str
    retry_count: int
    # The orchestrator's routing decision ("research" | "calculate" | "both"),
    # read by the conditional edges. (Added to the spec's fields because LangGraph
    # routes off state.)
    route: str


# --- helpers ---------------------------------------------------------------- #
def _llm() -> ChatAnthropic:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    return ChatAnthropic(model="claude-sonnet-4-6", temperature=0)


def _text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


# Math needed: an inline expression, or keywords implying arithmetic.
_MATH_HINTS = re.compile(
    r"\d+\s*[-+*/x×]\s*\d+|\b(times|multiplied|divided|plus|minus|percent|worth|"
    r"how much|calculate|sum of|product of|square root|total)\b",
    re.I,
)
# Current/external info needed.
_RESEARCH_HINTS = re.compile(
    r"\b(stock|price|shares?|latest|current|today|news|recent|weather|who is|ceo|"
    r"founded|release|version|capital of|population|look up|search)\b",
    re.I,
)


# --- nodes ------------------------------------------------------------------ #
def orchestrator_agent(state: ResearchState) -> dict:
    """Decide routing from the question: research, calculate, or both."""
    question = state["question"]
    needs_math = bool(_MATH_HINTS.search(question))
    needs_research = bool(_RESEARCH_HINTS.search(question))
    if needs_math and needs_research:
        route = "both"
    elif needs_math:
        route = "calculate"
    else:
        route = "research"
    return {"route": route}


def research_agent(state: ResearchState) -> dict:
    """Search the web for the question and store the raw results."""
    results = DuckDuckGoSearchRun().invoke(state["question"])
    return {"search_results": results}


def calculator_agent(state: ResearchState) -> dict:
    """Derive a single arithmetic expression (using any researched numbers) and evaluate it."""
    prompt = (
        "You produce ONE arithmetic expression that answers the question.\n"
        f"Question: {state['question']}\n"
        f"Search results (may contain a needed number such as a price):\n"
        f"{(state.get('search_results') or '(none)')[:1500]}\n\n"
        "Reply with a single expression using only digits and + - * / . ( ) "
        "(e.g. '200 * 178.5'). If no calculation is needed, reply exactly: NONE"
    )
    expr = _text(_llm().invoke(prompt)).strip().splitlines()[0].strip().strip("`")
    if expr.upper().startswith("NONE") or not re.search(r"\d", expr):
        return {"calculation": "No calculation needed."}
    return {"calculation": f"{expr} = {calculator.invoke(expr)}"}


def writer_agent(state: ResearchState) -> dict:
    """Combine research + calculation into a coherent draft answer."""
    prompt = (
        "Write a concise, accurate answer to the question using the gathered information. "
        "Lead with the answer and cite the key numbers.\n\n"
        f"Question: {state['question']}\n"
        f"Search results:\n{state.get('search_results') or '(none)'}\n\n"
        f"Calculation:\n{state.get('calculation') or '(none)'}"
    )
    return {"draft_answer": _text(_llm().invoke(prompt)).strip()}


def reviewer_agent(state: ResearchState) -> dict:
    """Score the draft 0.0-1.0 by an additive rubric; carry it as the final answer."""
    draft = (state.get("draft_answer") or "").strip()
    score = 0.0
    if draft:                                   # has content
        score += 0.4
    if re.search(r"\d", draft):                 # has specific facts/numbers
        score += 0.3
    q_words = {w for w in re.findall(r"[a-z]{4,}", state["question"].lower())}
    if any(w in draft.lower() for w in q_words):  # addresses the question
        score += 0.3
    return {
        "quality_score": round(score, 2),
        "final_answer": draft,
        "retry_count": state.get("retry_count", 0) + 1,
    }


# --- edges ------------------------------------------------------------------ #
def route_after_orchestrator(state: ResearchState) -> str:
    return "research" if state["route"] in ("research", "both") else "calculator"


def route_after_research(state: ResearchState) -> str:
    return "calculator" if state["route"] == "both" else "writer"


def route_after_reviewer(state: ResearchState) -> str:
    if state["quality_score"] < 0.7 and state["retry_count"] < 2:
        return "research"
    return "end"


def build_pipeline():
    """Compile the StateGraph."""
    g = StateGraph(ResearchState)
    g.add_node("orchestrator", orchestrator_agent)
    g.add_node("research", research_agent)
    g.add_node("calculator", calculator_agent)
    g.add_node("writer", writer_agent)
    g.add_node("reviewer", reviewer_agent)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges(
        "orchestrator", route_after_orchestrator,
        {"research": "research", "calculator": "calculator"},
    )
    g.add_conditional_edges(
        "research", route_after_research,
        {"calculator": "calculator", "writer": "writer"},
    )
    g.add_edge("calculator", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer", route_after_reviewer,
        {"research": "research", "end": END},
    )
    return g.compile()


def run_pipeline(question: str, on_node=None) -> dict:
    """Stream the pipeline, calling on_node(name) per executed node; return final state."""
    initial: ResearchState = {
        "question": question,
        "search_results": "",
        "calculation": "",
        "draft_answer": "",
        "quality_score": 0.0,
        "final_answer": "",
        "retry_count": 0,
        "route": "",
    }
    final = dict(initial)
    for chunk in build_pipeline().stream(initial, stream_mode="updates"):
        for node_name, update in chunk.items():
            if on_node:
                on_node(node_name)
            if isinstance(update, dict):
                final.update(update)
    return final
