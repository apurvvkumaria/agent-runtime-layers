"""MCP client: expose the official MCP filesystem server as a LangChain tool.

The `filesystem` tool lets the agent read files from the project's `docs/` directory
via the Model Context Protocol. We launch the official filesystem server
(`@modelcontextprotocol/server-filesystem`) over stdio with `docs/` as its only
allowed directory, so reads are sandboxed there — the agent cannot reach anything
outside it (enforced both here and by the server).
"""

import asyncio
import concurrent.futures
from pathlib import Path

from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# The single allowed directory — the sandbox boundary for all reads.
DOCS_DIR = (Path(__file__).resolve().parent.parent / "docs").resolve()


def _run(coro):
    """Run a coroutine to completion whether or not we're already in an event loop.

    The agent calls tools from both sync (`invoke`) and async (`astream_events`)
    contexts; running in a fresh loop on a worker thread works in either case.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _resolve(path: str) -> Path:
    """Resolve a requested path to an absolute file inside DOCS_DIR, or raise."""
    name = (path or "").strip().lstrip("/")
    if name.startswith("docs/"):
        name = name[len("docs/"):]
    if not name:
        raise ValueError("provide a filename under docs/, e.g. 'sla_thresholds.md'")
    target = (DOCS_DIR / name).resolve()
    if target != DOCS_DIR and DOCS_DIR not in target.parents:
        raise ValueError("path escapes the docs/ sandbox")
    return target


async def _read_via_mcp(abs_path: str) -> str:
    """Open an MCP session to the filesystem server and read one file."""
    params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", str(DOCS_DIR)],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("read_text_file", {"path": abs_path})
            text = "".join(getattr(block, "text", "") for block in result.content)
            if result.isError:
                raise RuntimeError(text or "MCP filesystem server returned an error")
            return text


@tool
def filesystem(path: str) -> str:
    """Read a documentation file from the docs/ directory via the MCP filesystem server.

    Available files: openShell_overview.md, sla_thresholds.md, agent_patterns.md.
    Pass a filename relative to docs/ (e.g. "sla_thresholds.md"). Reads are sandboxed
    to docs/ — files outside it cannot be accessed. Use this to look up internal
    documentation before answering.
    """
    try:
        target = _resolve(path)
    except ValueError as exc:
        return f"Error: {exc}"
    try:
        return _run(_read_via_mcp(str(target)))
    except Exception as exc:  # npx missing, server error, bad path, etc.
        return f"Error reading {path!r} via MCP filesystem server: {exc}"
