"""MCP server: expose the agent and its tools as MCP endpoints.

Any MCP-compatible client (Claude Desktop, another agent, etc.) can connect and call:
  - ask_agent(question)        -> run the full ReAct agent
  - get_storage_metrics(cluster) -> the storage_metrics tool directly
  - calculate(expression)      -> the calculator tool directly

Run via the CLI: `uv run python agent.py mcp-serve --port 3000`.
"""

import pathlib
import sys

# Repo root on the path so `core`/`tools` import when run from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load .env before core reads ANTHROPIC_API_KEY / LANGFUSE_*.
load_dotenv()

from mcp.server.fastmcp import FastMCP  # noqa: E402

from core import build_single_shot_agent, new_session_id, stream_answer  # noqa: E402
from tools import calculator, storage_metrics  # noqa: E402

# The tools this server exposes (names match the functions below).
TOOL_NAMES = ["ask_agent", "get_storage_metrics", "calculate"]


async def ask_agent(question: str) -> str:
    """Ask the ReAct agent a question and return its final answer."""
    executor = build_single_shot_agent()
    return await stream_answer(executor, question, new_session_id(), echo=False)


def get_storage_metrics(cluster: str) -> str:
    """Return current distributed-storage metrics for a named cluster."""
    return storage_metrics.invoke(cluster)


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g. '150 * 223.48')."""
    return calculator.invoke(expression)


def create_server(host: str = "127.0.0.1", port: int = 3000) -> FastMCP:
    """Build a FastMCP server with the three tools registered."""
    server = FastMCP("agent-mcp", host=host, port=port)
    server.add_tool(ask_agent)
    server.add_tool(get_storage_metrics)
    server.add_tool(calculate)
    return server


if __name__ == "__main__":
    create_server().run(transport="sse")
