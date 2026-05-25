"""Click CLI front door for the agent.

Thin command layer over `core` (agent runtime) and `api` (REST server). Commands
fall in two groups: LLM-backed (chat/ask/research, traced via LangFuse) and direct
tool/memory access (calc/metrics/history, no API call).

Run: `uv run python agent.py --help`
"""

import asyncio

import click
from dotenv import load_dotenv

from core import (
    build_chat_agent,
    build_single_shot_agent,
    load_recent_turns,
    new_session_id,
    stream_answer,
)
from hooks import flush_traces
from tools import calculator, storage_metrics


def _build_or_die(builder):
    """Build an agent, converting a missing API key into a clean CLI error."""
    try:
        return builder()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


def _show_memory(executor) -> None:
    """Print the full conversation buffer accumulated this session."""
    print("\n=== Full conversation memory ===")
    print(executor.memory.buffer or "(empty — no questions were answered)")


async def converse(executor, session_id: str | None = None) -> None:
    """Interactive REPL: read a question, stream the answer, repeat."""
    print("\nAgent ready. Type your question or 'exit' to quit.")

    try:
        while True:
            question = input("\n> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            # Same executor (and memory) every turn, so each answer is appended to
            # {chat_history} and visible to later turns.
            await stream_answer(executor, question, session_id)
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C, or end-of-stream when stdin is piped/redirected: end cleanly.
        print("\nSession ended.")

    _show_memory(executor)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Claude + LangChain ReAct agent with multiple front doors.

    \b
    Examples:
      agent chat                          start an interactive session
      agent ask "What is a Merkle tree?"  one question, one answer
      agent research "RAFT" -o raft.md    research a topic, save a report
      agent calc "150 * 223.48"           direct calculator (no LLM)
      agent metrics prod-us-east-1        direct storage metrics (no LLM)
      agent history                       show recent conversation turns
      agent serve --port 8000             start the REST API server
    """
    # Load .env up front so ANTHROPIC_API_KEY and LANGFUSE_* are available to every
    # command. (Langfuse is imported lazily, after this, in hooks.)
    load_dotenv()


@cli.command()
def chat() -> None:
    """Start an interactive multi-turn conversation (memory + streaming)."""
    executor = _build_or_die(build_chat_agent)
    session_id = new_session_id()
    try:
        asyncio.run(converse(executor, session_id))
    finally:
        flush_traces()


@cli.command()
@click.argument("question")
@click.option(
    "--timeout", type=float, default=None,
    help="Max seconds for the run; on timeout it's recorded to the DLQ as tool_timeout.",
)
def ask(question: str, timeout: float | None) -> None:
    """Answer a single QUESTION and exit (no interactive loop, no memory)."""
    executor = _build_or_die(build_single_shot_agent)
    try:
        asyncio.run(stream_answer(executor, question, new_session_id(), timeout=timeout))
    except Exception as exc:  # noqa: BLE001 — already recorded to the DLQ in core
        raise click.ClickException(
            f"Run failed ({type(exc).__name__}); recorded to the DLQ — see `agent dlq-stats`."
        ) from exc
    finally:
        flush_traces()


@cli.command()
@click.argument("topic")
@click.option(
    "--output", "-o", default="report.md", show_default=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Markdown file to write the report to.",
)
def research(topic: str, output: str) -> None:
    """Research TOPIC on the web and save a markdown report."""
    executor = _build_or_die(lambda: build_chat_agent(prompt_name="research_agent"))
    question = (
        f"Research this topic using web search and write a thorough, well-structured "
        f"summary with key points and sources: {topic}"
    )
    try:
        answer = asyncio.run(stream_answer(executor, question, new_session_id()))
    finally:
        flush_traces()

    if not answer:
        raise click.ClickException("The agent produced no answer; nothing was saved.")

    from pathlib import Path

    path = Path(output)
    path.write_text(f"# {topic}\n\n{answer}\n", encoding="utf-8")
    click.echo(f"\n✅ Report saved to {path.resolve()}")


@cli.command()
@click.argument("expression")
def calc(expression: str) -> None:
    """Evaluate a math EXPRESSION with the calculator tool (no LLM, no API call)."""
    click.echo(calculator.invoke(expression))


@cli.command()
@click.argument("cluster_name")
def metrics(cluster_name: str) -> None:
    """Show storage metrics for CLUSTER_NAME (no LLM, no API call)."""
    click.echo(storage_metrics.invoke(cluster_name))


@cli.command()
@click.argument("topic")
def skill(topic: str) -> None:
    """Run the research_and_summarize skill on TOPIC and print the report."""
    from skills.research_and_summarize import research_and_summarize

    try:
        report = research_and_summarize.invoke(topic)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    click.echo(report)


@cli.command(name="sandbox-ask")
@click.argument("question")
@click.option(
    "--keep/--no-keep", default=False, show_default=True,
    help="Keep the sandbox after the run instead of deleting it.",
)
@click.option(
    "--image", default=None,
    help="Sandbox image to create from (else $OPENSHELL_SANDBOX_IMAGE, default 'agent-sandbox').",
)
def sandbox_ask(question: str, keep: bool, image: str | None) -> None:
    """Answer QUESTION by running `ask` inside a policy-constrained OpenShell sandbox."""
    from sandbox_runner import SandboxError, run_in_sandbox

    try:
        answer = run_in_sandbox(question, keep=keep, image=image)
    except SandboxError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("\n--- Answer (from sandbox) ---")
    click.echo(answer)


@cli.command(name="sandbox-info")
def sandbox_info() -> None:
    """Show gateway status, running sandboxes, and the active sandbox policy."""
    from sandbox_runner import SandboxError, gateway_status, list_sandboxes, policy_text

    try:
        click.echo("=== Gateway status ===")
        click.echo(gateway_status())
    except SandboxError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo("\n=== Sandboxes ===")
    click.echo(list_sandboxes())
    click.echo("\n=== Active policy (openshell/policy.yaml) ===")
    click.echo(policy_text())


@cli.command()
def history() -> None:
    """Show the last 10 conversation turns from saved memory."""
    turns = load_recent_turns(10)
    if not turns:
        click.echo("No conversation history yet")
        return

    click.echo(f"Last {len(turns)} conversation turn(s):\n")
    for i, (question, answer) in enumerate(turns, start=1):
        one_line = " ".join(answer.split())
        snippet = one_line if len(one_line) <= 200 else one_line[:200] + "…"
        click.echo(f"{i}. Q: {question}")
        click.echo(f"   A: {snippet}\n")


@cli.command()
def test() -> None:
    """Run the unit/integration tests and evals, then print a summary."""
    from evals.run_all import main as run_all

    run_all()


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
def serve(host: str, port: int) -> None:
    """Start the FastAPI REST server via uvicorn."""
    import uvicorn

    from api import app  # lazy import so non-serve commands don't load FastAPI

    click.echo(f"Starting REST API server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command(name="mcp-serve")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
@click.option("--port", default=3000, show_default=True, type=int, help="Port to listen on.")
def mcp_serve(host: str, port: int) -> None:
    """Start the MCP server exposing the agent's tools to MCP clients."""
    from mcp_integration.server import TOOL_NAMES, create_server  # lazy import

    server = create_server(host=host, port=port)
    click.echo(f"MCP server running on port {port}")
    click.echo("Available tools:")
    for name in TOOL_NAMES:
        click.echo(f"  - {name}")
    server.run(transport="sse")


@cli.command(name="sync-prompt")
def sync_prompt() -> None:
    """Push the local single-shot prompt to LangFuse as 'react-agent-prompt'."""
    from core import LANGFUSE_PROMPT_NAME
    from hooks import _langfuse_configured
    from prompts.loader import load_prompt

    if not _langfuse_configured():
        raise click.ClickException(
            "LangFuse is not configured — set LANGFUSE_* keys in .env to sync prompts."
        )

    from langfuse import get_client

    text = load_prompt("single_shot_agent")
    try:
        prompt = get_client().create_prompt(
            name=LANGFUSE_PROMPT_NAME,
            prompt=text,
            type="text",
            labels=["production"],
        )
    except Exception as exc:
        raise click.ClickException(f"Failed to sync prompt to LangFuse: {exc}") from exc

    click.echo(f"Prompt synced to LangFuse: {LANGFUSE_PROMPT_NAME} v{prompt.version}")


@cli.command(name="memory-stats")
def memory_stats() -> None:
    """Show vector-store memory stats and estimated token savings vs. buffer memory."""
    from memory.vector_store import VectorStoreMemory

    mem = VectorStoreMemory()
    s = mem.stats()
    click.echo(f"Vector store: {mem.store_dir} ({s['on_disk_bytes']} bytes on disk)")
    click.echo(f"Total turns stored: {s['turns']}")
    click.echo(f"Embedding: {s['embedder']} (dim {s['embedding_dim']})")
    d = mem.decay_days
    click.echo(
        f"Decay thresholds (days): full <{d['summary']}, summary <{d['marker']}, "
        f"marker <{d['archived']}, archived >={d['archived']}"
    )
    if s["turns"]:
        t = s["tiers"]
        click.echo(
            f"Decay tiers: {t['full']} full, {t['summary']} summary, "
            f"{t['marker']} marker (archived turns are deleted)"
        )
        click.echo(
            f"Estimated tokens per turn — buffer (all {s['turns']} turns): "
            f"{s['buffer_tokens']}, vector (top-{mem.k}): {s['vector_tokens']}"
        )
        click.echo(
            f"Estimated savings: {s['estimated_savings']} tokens ({s['savings_pct']}%)"
        )
    else:
        click.echo("(no turns stored yet — run `agent chat` to populate the store)")


def _run_langgraph(question: str) -> None:
    import asyncio

    from langgraph_agents.pipeline import stream_pipeline

    click.echo("=== LangGraph (fixed graph, streaming) ===")
    state = {"answering": False}

    def on_node(name: str) -> None:
        if name == "writer":  # the writer's output is streamed below, not labelled
            return
        # separate a node label from any answer text streamed just before it
        click.echo(f"{chr(10) if state['answering'] else ''}[{name}]")

    def on_token(text: str) -> None:
        if not state["answering"]:
            click.echo("\nAnswer (streaming):\n", nl=False)
            state["answering"] = True
        click.echo(text, nl=False)

    result = asyncio.run(stream_pipeline(question, on_node=on_node, on_token=on_token))
    if not state["answering"]:  # nothing streamed — fall back to the final text
        click.echo(f"\nFinal answer:\n{result['final_answer']}")
    click.echo(f"\n\nQuality score: {result['quality_score']} | tokens: {result.get('token_count', 0)}")


def _run_strands(question: str) -> None:
    from strands_agent.agent import run_strands

    click.echo("=== Strands (model-driven) ===")
    result = run_strands(question)
    click.echo(f"steps: {result['steps']} | tokens: {result['tokens']}")
    click.echo(f"\nFinal answer:\n{result['answer']}")


@cli.command()
@click.argument("question")
@click.option(
    "--framework", type=click.Choice(["langgraph", "strands", "both"]),
    default="langgraph", show_default=True,
    help="Which agent framework to run the pipeline on.",
)
def pipeline(question: str, framework: str) -> None:
    """Run the multi-agent research pipeline on QUESTION."""
    try:
        if framework in ("langgraph", "both"):
            _run_langgraph(question)
        if framework == "both":
            click.echo()
        if framework in ("strands", "both"):
            _run_strands(question)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.argument("question")
@click.option("--cron", "cron_expr", required=True, help="Cron expression, e.g. '0 9 * * *'.")
@click.option("--output", default="report.md", show_default=True, help="File to append answers to.")
def schedule(question: str, cron_expr: str, output: str) -> None:
    """Run QUESTION on a cron schedule (runs once immediately, then blocks)."""
    from autonomy.scheduler import AgentScheduler

    try:
        AgentScheduler(on_log=click.echo).schedule(question, cron_expr, output)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option("--interval", default=60, show_default=True, type=int, help="Seconds between ticks.")
def heartbeat(interval: int) -> None:
    """Run a heartbeat loop that processes pending tasks from tasks.json (blocks)."""
    from autonomy.scheduler import HeartbeatLoop

    HeartbeatLoop(interval=interval, on_log=click.echo).run()


@cli.command(name="add-task")
@click.argument("description")
def add_task_cmd(description: str) -> None:
    """Add a pending task to tasks.json for the heartbeat loop to pick up."""
    from autonomy.scheduler import add_task

    task = add_task(description)
    click.echo(f"Added task {task['id'][:8]}: {description}")


@cli.command(name="dlq-stats")
def dlq_stats() -> None:
    """Show DLQ failure counts by reason and classification."""
    from dlq.manager import DLQManager

    s = DLQManager().stats()
    click.echo(f"Failures: {s['total']} total "
               f"({s['active_transient']} transient active, {s['permanent']} permanent)")
    if s["by_reason"]:
        click.echo("By reason:")
        for reason, n in sorted(s["by_reason"].items()):
            click.echo(f"  {reason}: {n}")
        click.echo(f"By classification: {s['by_classification']}")
    else:
        click.echo("(no failures recorded)")


@cli.command(name="dlq-retry")
def dlq_retry() -> None:
    """Retry all transient failures with exponential backoff (1s, 2s, 4s)."""
    from dlq.manager import DLQManager

    r = DLQManager().retry_transient()
    click.echo(
        f"Retried {r['retried']}: {r['succeeded']} succeeded, "
        f"{r['moved_to_permanent']} moved to permanent (max retries exceeded)."
    )


@cli.command(name="dlq-clear")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def dlq_clear(yes: bool) -> None:
    """Clear permanent failures after review."""
    from dlq.manager import DLQManager

    mgr = DLQManager()
    n = mgr.stats()["permanent"]
    if n == 0:
        click.echo("No permanent failures to clear.")
        return
    if not yes:
        click.confirm(f"Clear {n} permanent failure(s)?", abort=True)
    click.echo(f"Cleared {mgr.clear_permanent()} permanent failure(s).")


@cli.command()
@click.argument("question")
def compare(question: str) -> None:
    """Run QUESTION on both frameworks and compare steps, tokens, time, and quality."""
    import time

    from langgraph_agents.pipeline import run_pipeline, score_answer
    from strands_agent.agent import run_strands

    try:
        # LangGraph
        nodes: list[str] = []
        t0 = time.perf_counter()
        lg = run_pipeline(question, on_node=nodes.append)
        lg_time = time.perf_counter() - t0
        lg_answer = lg["final_answer"]
        lg_quality = lg.get("quality_score", score_answer(question, lg_answer))

        # Strands
        t1 = time.perf_counter()
        st = run_strands(question)
        st_time = time.perf_counter() - t1
        st_answer = st["answer"]
        st_quality = score_answer(question, st_answer)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("\nFramework    | Nodes/Steps     | Total Tokens | Time (s) | Quality")
    click.echo("-" * 70)
    click.echo(f"LangGraph    | {len(nodes)} nodes        | {lg.get('token_count', 0):<12} | "
               f"{lg_time:<8.1f} | {lg_quality}")
    click.echo(f"Strands      | {st['steps']} steps (model) | {st['tokens']:<12} | "
               f"{st_time:<8.1f} | {st_quality}")

    click.echo(f"\n--- LangGraph answer ---\n{lg_answer}")
    click.echo(f"\n--- Strands answer ---\n{st_answer}")

    # Winner: better quality, then fewer tokens.
    if lg_quality != st_quality:
        winner, why = ("LangGraph", "better quality") if lg_quality > st_quality else ("Strands", "better quality")
    else:
        winner, why = ("LangGraph", "fewer tokens") if lg.get("token_count", 0) <= st["tokens"] else ("Strands", "fewer tokens")
    click.echo(f"\nWinner: {winner} ({why})")


@cli.command(name="context-stats")
@click.argument("question")
def context_stats(question: str) -> None:
    """Show the token budget, RAG docs, and estimated context size for QUESTION."""
    from context.manager import ContextManager
    from context.rag import needs_rag, relevant_docs
    from prompts.loader import load_prompt

    cm = ContextManager()
    ctx = {"system_prompt": load_prompt("single_shot_agent"), "question": question}
    docs: list[str] = []
    if needs_rag(question):
        hits = relevant_docs(question, k=3)
        docs = [h["source"] for h in hits]
        ctx["retrieved_context"] = "\n\n".join(h["text"] for h in hits)

    ctx = cm.enforce_budget(ctx)
    click.echo(cm.budget_report(ctx))
    click.echo(f"\nRAG triggered: {needs_rag(question)}")
    if docs:
        click.echo("Docs retrieved: " + ", ".join(sorted(set(docs))))
    context_total = sum(cm.count_tokens(t) for t in ctx.values())
    reserve = cm.CONTEXT_BUDGET["response_reserve"]
    click.echo(
        f"Estimated context tokens before LLM call: {context_total} "
        f"(+{reserve} reserved for the response)"
    )


@cli.command(name="memory-clear")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt.")
def memory_clear(yes: bool) -> None:
    """Wipe all stored turns from the vector-store memory."""
    from memory.vector_store import VectorStoreMemory

    mem = VectorStoreMemory()
    turns = mem.stats()["turns"]
    if turns == 0:
        click.echo("Vector store is already empty — nothing to clear.")
        return
    if not yes:
        click.confirm(f"Delete all {turns} stored turn(s) from {mem.store_dir}?", abort=True)
    mem.clear()
    click.echo(f"Cleared {turns} turn(s) from the vector store.")


@cli.command(name="memory-decay")
@click.option("--summary-days", type=int, default=None, help="Age (days) a turn becomes a summary.")
@click.option("--marker-days", type=int, default=None, help="Age (days) a turn becomes a topic marker.")
@click.option("--archived-days", type=int, default=None, help="Age (days) a turn is archived (deleted).")
def memory_decay(summary_days: int | None, marker_days: int | None, archived_days: int | None) -> None:
    """Age out stored turns by tier (full -> summary -> marker -> archived)."""
    from memory.vector_store import VectorStoreMemory

    overrides = {
        k: v for k, v in (
            ("summary", summary_days), ("marker", marker_days), ("archived", archived_days)
        ) if v is not None
    }
    mem = VectorStoreMemory(decay_days=overrides) if overrides else VectorStoreMemory()
    counts = mem.decay_memory()
    click.echo(f"Thresholds (days): {mem.decay_days}")
    click.echo(
        f"{counts['full']} full, {counts['summary']} summary, "
        f"{counts['marker']} marker, {counts['archived']} archived"
    )


if __name__ == "__main__":
    cli()
