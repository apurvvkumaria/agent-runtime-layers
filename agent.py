"""A simple ReAct research agent: Claude + DuckDuckGo web search via LangChain.

Run with: uv run python agent.py
Requires the ANTHROPIC_API_KEY environment variable to be set.
"""

import ast
import asyncio
import operator
import os
import random
import time
from datetime import datetime
from pathlib import Path

import click
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# Conversation history is persisted here so it survives across CLI invocations
# (e.g. `agent ask` then `agent history` run in separate processes).
HISTORY_PATH = Path(__file__).parent / ".agent_history.json"

# Allowlist of AST node types -> the operator function they map to.
# Anything outside this set is rejected, so the calculator can't be used
# to execute arbitrary code (the safety goal behind the ast.literal_eval pattern).
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST, allowing only numbers and math ops.

    ast.literal_eval can't handle operators like "2 + 2", so we parse to an AST
    ourselves and walk an explicit allowlist of node types — same safety idea,
    extended to arithmetic.
    """
    if isinstance(node, ast.Constant):  # a literal number
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Use this for any math, e.g. "2 + 2", "100 * 0.15", or "150 * 1234.56".
    Supports + - * / // % ** and parentheses. Input must be a single expression.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"Error: could not evaluate {expression!r} ({exc})"


@tool
def storage_metrics(cluster_name: str) -> str:
    """Return current distributed-storage metrics for a named cluster.

    Use this to look up live operational metrics (throughput, latency, disk,
    replication, partitions) for a storage cluster by name, e.g. "prod-us-east-1".
    """
    # Fake but realistic numbers — a stand-in for a real metrics backend.
    rps = random.randint(50_000, 100_000)
    p99_latency_ms = round(random.uniform(1, 5), 2)
    disk_utilization_pct = round(random.uniform(40, 85), 1)
    replication_lag_ms = random.randint(0, 50)
    active_partitions = random.randint(100, 500)

    return (
        f"Storage metrics for cluster '{cluster_name}':\n"
        f"  requests_per_second:  {rps:,}\n"
        f"  p99_latency_ms:       {p99_latency_ms}\n"
        f"  disk_utilization_pct: {disk_utilization_pct}\n"
        f"  replication_lag_ms:   {replication_lag_ms}\n"
        f"  active_partitions:    {active_partitions}"
    )


# Shared ReAct format instructions: alternate Thought / Action / Action Input /
# Observation until the model can answer. {tools}/{tool_names} are filled by
# create_react_agent; {input}/{agent_scratchpad} per invocation.
_REACT_FORMAT = """Answer the following question as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

# WITH memory: a {chat_history} slot is injected before the question so the model
# can resolve references to earlier turns ("at that price"). Used by chat/research.
REACT_PROMPT = PromptTemplate.from_template(
    _REACT_FORMAT
    + """
Previous conversation history:
{chat_history}

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# WITHOUT memory: no {chat_history} at all, so a single-shot `ask` starts with
# zero prior context — a much smaller prompt (~370 input tokens vs. thousands once
# history accumulates). Used by ask.
REACT_PROMPT_SINGLE = PromptTemplate.from_template(
    _REACT_FORMAT
    + """
Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)


def _now() -> str:
    """Wall-clock timestamp with millisecond precision, for hook logging."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _token_counts(response) -> dict | None:
    """Best-effort extraction of token usage from an LLMResult.

    Where the counts live depends on the model/streaming path, so check both the
    aggregated llm_output and the per-generation message metadata.
    """
    usage = (response.llm_output or {}).get("usage") or (response.llm_output or {}).get(
        "token_usage"
    )
    if usage:
        return usage
    try:
        message = response.generations[0][0].message
    except (IndexError, AttributeError):
        return None
    return getattr(message, "usage_metadata", None)


class StepLogger(BaseCallbackHandler):
    """Lifecycle hooks: print every tool call and LLM call as the agent runs.

    These are LangChain callbacks — the framework invokes each on_* method at the
    matching point in the run. This is the explicit, observable version of what
    verbose=True does implicitly, plus timing and token counts.
    """

    def __init__(self) -> None:
        # run_id -> perf_counter() at tool start, so on_tool_end can time it.
        self._tool_started_at: dict = {}

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self._tool_started_at[run_id] = time.perf_counter()
        name = (serialized or {}).get("name", "unknown")
        print(f"[{_now()}] 🔧 tool start  → {name}({input_str!r})")

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        started = self._tool_started_at.pop(run_id, None)
        elapsed = f"{(time.perf_counter() - started) * 1000:.1f}ms" if started else "n/a"
        # output may be a str or a ToolMessage depending on LangChain version.
        text = output if isinstance(output, str) else getattr(output, "content", str(output))
        print(f"[{_now()}] ✅ tool end    ← ({elapsed})\n{text}")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        # Chat models fall back to on_llm_start when on_chat_model_start is unset.
        print(f"[{_now()}] 🧠 LLM thinking...")

    def on_llm_end(self, response, **kwargs) -> None:
        usage = _token_counts(response)
        if usage:
            print(f"[{_now()}] 🧠 LLM done — tokens: {usage}")
        else:
            print(f"[{_now()}] 🧠 LLM done — token counts unavailable")


def _langfuse_configured() -> bool:
    """True only if both LangFuse keys look real (set and not placeholders)."""
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    # The shipped .env uses "your-...-here" placeholders; treat those as unset.
    return bool(pub) and bool(sec) and "your-" not in pub and "your-" not in sec


def build_langfuse_handler() -> BaseCallbackHandler | None:
    """Return a LangFuse CallbackHandler if credentials are configured, else None.

    This follows the official Langfuse LangChain integration pattern: configure
    credentials via the LANGFUSE_* env vars (read by the client — no constructor
    args), get the client, and construct a bare `CallbackHandler()`. Trace
    attributes are set per-call via config metadata in `stream_answer`, not here.
    Any failure degrades gracefully to print-only hooks rather than crashing.
    """
    if not _langfuse_configured():
        print("[langfuse] keys not configured — using print hooks only.")
        return None
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        client = get_client()
        # Confirm credentials/connectivity before relying on tracing; never fatal.
        if not client.auth_check():
            print("[langfuse] auth check failed — using print hooks only.")
            return None
        print("[langfuse] tracing enabled.")
        return CallbackHandler()
    except Exception as exc:  # network, bad keys, version drift, etc.
        print(f"[langfuse] disabled ({exc}) — using print hooks only.")
        return None


def build_memory() -> ConversationBufferMemory:
    """Conversation memory backed by a JSON file so it survives across processes.

    memory_key must match the {chat_history} prompt slot. output_key pins which
    executor output to store — the streaming path also surfaces a "messages" key,
    and without this the buffer warns about the ambiguity on every save.
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        output_key="output",
        chat_memory=FileChatMessageHistory(str(HISTORY_PATH)),
    )


def _build_llm_and_tools() -> tuple[ChatAnthropic, list]:
    """Shared setup for both agent builders: the LLM and the tool list."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    # Claude as the reasoning LLM. Auth is read from ANTHROPIC_API_KEY.
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
    )

    # The tools the agent can choose from: web search, arithmetic, and a fake
    # distributed-storage metrics backend.
    search = DuckDuckGoSearchRun()
    tools = [search, calculator, storage_metrics]
    return llm, tools


# verbose=True prints every Thought/Action/Observation so the loop is visible. The
# explicit StepLogger hooks are attached per-call via config in stream_answer —
# constructor callbacks don't propagate through astream_events.


def build_chat_agent() -> AgentExecutor:
    """ReAct executor WITH file-backed conversation memory.

    Used by `chat` and `research`: each turn's input/output is saved and replayed
    into the {chat_history} prompt slot, so the model has prior context.
    """
    llm, tools = _build_llm_and_tools()
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=build_memory(),
        verbose=True,
        handle_parsing_errors=True,
    )


def build_single_shot_agent() -> AgentExecutor:
    """ReAct executor with NO memory at all.

    Used by `ask`: it neither loads nor saves history, and its prompt has no
    {chat_history} slot — so every call starts from zero prior context. This is
    what keeps a single-shot question's input token count minimal instead of
    growing with the accumulated conversation.
    """
    llm, tools = _build_llm_and_tools()
    agent = create_react_agent(llm, tools, REACT_PROMPT_SINGLE)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def _show_memory(executor: AgentExecutor) -> None:
    """Print the full conversation buffer accumulated this session."""
    print("\n=== Full conversation memory ===")
    print(executor.memory.buffer or "(empty — no questions were answered)")


# The ReAct model emits its whole turn as one stream: the reasoning
# (Thought/Action/...) and then "Final Answer: <text>". We let the reasoning
# tokens flow into the verbose trace and only stream the text *after* this
# marker as the clean answer.
_ANSWER_MARKER = "Final Answer:"
_SEPARATOR = "─" * 60


def _chunk_text(chunk) -> str:
    """Extract plain text from a streamed AIMessageChunk.

    ChatAnthropic streams content either as a string or as a list of content
    blocks (dicts like {"type": "text", "text": "..."}); handle both.
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


async def stream_answer(
    executor: AgentExecutor,
    question: str,
    extra_callbacks: list | None = None,
    session_id: str | None = None,
    echo: bool = True,
) -> str:
    """Run one turn; stream the final answer token-by-token and return it.

    `astream_events` yields a typed event for everything the executor does while
    it runs the ReAct loop. We `async for` over that stream and react to
    `on_chat_model_stream` events — each carries one LLM token. The verbose
    trace prints the reasoning steps as usual; here we suppress tokens until the
    model reaches "Final Answer:", then print every token after it the instant
    it arrives (when `echo`). The accumulated answer text is returned so callers
    like `research` can persist it.
    """
    buffer = ""          # accumulates tokens until we spot the marker
    streaming = False     # flips True once we're past "Final Answer:"
    answer: list[str] = []  # the final-answer text, collected for the return value

    # Per the official Langfuse LangChain pattern, trace attributes are passed
    # through config — not the handler constructor. `run_name` gives the trace a
    # findable name; the `langfuse_*` metadata keys group turns into one session
    # and tag them for filtering. The print-based StepLogger rides in the same
    # callbacks list; both fire under astream_events.
    callbacks = [StepLogger(), *(extra_callbacks or [])]
    config = {
        "callbacks": callbacks,
        "run_name": "research-agent-turn",
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["agent-practice", "react-agent"],
        },
    }
    async for event in executor.astream_events({"input": question}, version="v2", config=config):
        if event["event"] != "on_chat_model_stream":
            continue
        token = _chunk_text(event["data"]["chunk"])
        if not token:
            continue

        if streaming:
            # Already in the answer — collect it and print each token as it lands.
            answer.append(token)
            if echo:
                print(token, end="", flush=True)
            continue

        # Still in the reasoning portion: buffer and watch for the marker, which
        # may be split across several tokens.
        buffer += token
        marker_at = buffer.find(_ANSWER_MARKER)
        if marker_at != -1:
            streaming = True
            tail = buffer[marker_at + len(_ANSWER_MARKER):].lstrip()
            answer.append(tail)
            if echo:
                print(f"\n{_SEPARATOR}\nAnswer (streaming):\n")
                print(tail, end="", flush=True)

    if echo:
        print()  # newline after the streamed answer
        if not streaming:
            print(f"\n{_SEPARATOR}\n(no final answer was streamed)")

    return "".join(answer).strip()


async def converse(
    executor: AgentExecutor,
    extra_callbacks: list | None = None,
    session_id: str | None = None,
) -> None:
    """Interactive REPL: read a question, stream the answer, repeat."""
    print("\nAgent ready. Type your question or 'exit' to quit.")

    try:
        while True:
            question = input("\n> ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            # Same executor (and memory) every turn, so each answer is appended
            # to {chat_history} and visible to later turns.
            await stream_answer(executor, question, extra_callbacks, session_id)
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C, or end-of-stream when stdin is piped/redirected: end cleanly
        # rather than dumping a traceback.
        print("\nSession ended.")

    _show_memory(executor)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
#
# The CLI gives the agent multiple "front doors": an interactive chat, one-shot
# ask/research commands that hit the LLM, and fast direct-tool commands (calc,
# metrics) that skip the LLM entirely. LLM-backed commands carry LangFuse traces.


def _new_session_id() -> str:
    """A timestamped session id, grouping a command's traces in Langfuse."""
    return f"agent-{datetime.now():%Y%m%d-%H%M%S}"


def _start_llm_session(
    builder,
) -> tuple[AgentExecutor, BaseCallbackHandler | None, str]:
    """Build the executor (via `builder`) + LangFuse handler for an LLM command.

    `builder` is build_chat_agent (memory) or build_single_shot_agent (no memory).
    Converts a missing API key into a clean CLI error instead of a traceback.
    """
    try:
        executor = builder()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    handler = build_langfuse_handler()
    return executor, handler, _new_session_id()


def _flush_langfuse(handler: BaseCallbackHandler | None) -> None:
    """Flush buffered traces before exit so short-lived commands don't drop them."""
    if handler is not None:
        from langfuse import get_client

        get_client().flush()


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
    """
    # Load .env up front so ANTHROPIC_API_KEY and LANGFUSE_* are available to
    # every command. (Langfuse is imported lazily, after this, in its builder.)
    load_dotenv()


@cli.command()
def chat() -> None:
    """Start an interactive multi-turn conversation (memory + streaming)."""
    executor, handler, session_id = _start_llm_session(build_chat_agent)
    try:
        asyncio.run(converse(executor, [handler] if handler else [], session_id))
    finally:
        _flush_langfuse(handler)


@cli.command()
@click.argument("question")
def ask(question: str) -> None:
    """Answer a single QUESTION and exit (no interactive loop, no memory)."""
    executor, handler, session_id = _start_llm_session(build_single_shot_agent)
    try:
        asyncio.run(stream_answer(executor, question, [handler] if handler else [], session_id))
    finally:
        _flush_langfuse(handler)


@cli.command()
@click.argument("topic")
@click.option(
    "--output", "-o", default="report.md", show_default=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Markdown file to write the report to.",
)
def research(topic: str, output: str) -> None:
    """Research TOPIC on the web and save a markdown report."""
    executor, handler, session_id = _start_llm_session(build_chat_agent)
    question = (
        f"Research this topic using web search and write a thorough, well-structured "
        f"summary with key points and sources: {topic}"
    )
    try:
        answer = asyncio.run(
            stream_answer(executor, question, [handler] if handler else [], session_id)
        )
    finally:
        _flush_langfuse(handler)

    if not answer:
        raise click.ClickException("The agent produced no answer; nothing was saved.")

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
def history() -> None:
    """Show the last 10 conversation turns from saved memory."""
    messages = build_memory().chat_memory.messages

    # Pair the flat message list into (question, answer) turns.
    turns: list[tuple[str, str]] = []
    pending_q: str | None = None
    for msg in messages:
        if msg.type == "human":
            pending_q = msg.content
        elif msg.type == "ai" and pending_q is not None:
            turns.append((pending_q, msg.content))
            pending_q = None

    if not turns:
        click.echo("No conversation history yet")
        return

    recent = turns[-10:]
    click.echo(f"Last {len(recent)} conversation turn(s):\n")
    for i, (question, answer) in enumerate(recent, start=1):
        one_line = " ".join(answer.split())
        snippet = one_line if len(one_line) <= 200 else one_line[:200] + "…"
        click.echo(f"{i}. Q: {question}")
        click.echo(f"   A: {snippet}\n")


@cli.command()
@click.option("--port", default=8000, show_default=True, type=int, help="Port to listen on.")
def serve(port: int) -> None:
    """Start the REST API server (placeholder — FastAPI lands in the next layer)."""
    click.echo(f"REST API server starting on port {port}")


if __name__ == "__main__":
    cli()
