"""Core agent runtime: prompts, memory, agent builders, and the streaming loop.

This is the shared engine both front doors (CLI in agent.py, REST API in api.py)
build on. It knows nothing about Click or FastAPI — just how to construct and run
the ReAct agent.
"""

import os
from datetime import datetime
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.prompts import PromptTemplate

from hooks import _langfuse_configured, get_callbacks
from prompts.loader import load_prompt
from tools import get_tools

# Conversation history is persisted here so it survives across CLI invocations
# (e.g. `agent ask` then `agent history` run in separate processes).
HISTORY_PATH = Path(__file__).parent / ".agent_history.json"

# The single-shot prompt is also managed in LangFuse under this name; the CLI's
# `sync-prompt` pushes the local file there.
LANGFUSE_PROMPT_NAME = "react-agent-prompt"


def _prompt_from_langfuse(name: str) -> str | None:
    """Fetch a prompt's template text from LangFuse, or None if unavailable.

    Returns None when LangFuse isn't configured or the prompt doesn't exist, so
    callers can fall back to the local file. Never raises.
    """
    if not _langfuse_configured():
        return None
    try:
        from langfuse import get_client

        return get_client().get_prompt(name, type="text").prompt
    except Exception:
        return None


def _react_prompt(prompt_name: str, langfuse_name: str | None = None) -> PromptTemplate:
    """Build a ReAct PromptTemplate, preferring LangFuse then the local file.

    `prompt_name` is the local `prompts/{name}.md`. If `langfuse_name` is given,
    try LangFuse first and fall back to the local file.
    """
    text = _prompt_from_langfuse(langfuse_name) if langfuse_name else None
    if text is None:
        text = load_prompt(prompt_name)
    return PromptTemplate.from_template(text)


def build_memory(persist: bool = True) -> ConversationBufferMemory:
    """Conversation memory for the chat agent.

    memory_key must match the {chat_history} prompt slot. output_key pins which
    executor output to store — the streaming path also surfaces a "messages" key,
    and without this the buffer warns about the ambiguity on every save.

    persist=True backs the buffer with a JSON file so it survives across processes
    (used by the CLI). persist=False keeps it in-process — used by the API, which
    isolates memory per session in a dict instead of sharing one global file.
    """
    kwargs: dict = {"memory_key": "chat_history", "output_key": "output"}
    if persist:
        kwargs["chat_memory"] = FileChatMessageHistory(str(HISTORY_PATH))
    return ConversationBufferMemory(**kwargs)


def _build_llm_and_tools() -> tuple[ChatAnthropic, list]:
    """Shared setup for both agent builders: the LLM and the tool list."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")

    # Claude as the reasoning LLM. Auth is read from ANTHROPIC_API_KEY.
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    return llm, get_tools()


def build_chat_agent(
    memory=None,
    prompt_name: str = "chat_agent",
    use_vector_memory: bool = True,
    use_buffer_memory: bool = False,
) -> AgentExecutor:
    """ReAct executor WITH conversation memory, prompt loaded from prompts/.

    Used by `chat` (chat_agent prompt) and `research` (research_agent prompt): each
    turn's input/output is saved and the relevant history replayed into the
    {chat_history} slot. The chosen prompt must include a {chat_history} placeholder.

    Memory selection (when `memory` is not passed explicitly):
      - default: VectorStoreMemory — retrieves the top-k semantically similar past
        turns, keeping the prompt bounded as the conversation grows.
      - use_buffer_memory=True (or use_vector_memory=False): the file-backed
        ConversationBufferMemory, which replays the full history (for comparison).
    Pass an explicit `memory` (e.g. the API's per-session buffer) to override both.

    verbose=True prints the Thought/Action/Observation loop; the explicit hooks are
    attached per-call via config in stream_answer (constructor callbacks don't
    propagate through astream_events).
    """
    if memory is None:
        if use_buffer_memory or not use_vector_memory:
            memory = build_memory()
        else:
            from memory.vector_store import VectorStoreMemory  # lazy: heavy-ish import

            memory = VectorStoreMemory()

    llm, tools = _build_llm_and_tools()
    agent = create_react_agent(llm, tools, _react_prompt(prompt_name))
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
    )


def build_single_shot_agent(prompt_name: str = "single_shot_agent") -> AgentExecutor:
    """ReAct executor with NO memory at all, prompt loaded from prompts/.

    Used by `ask`: it neither loads nor saves history, and its prompt has no
    {chat_history} slot — so every call starts from zero prior context, keeping the
    input token count minimal. For the default prompt, the template is fetched from
    LangFuse first (`react-agent-prompt`) and falls back to the local file.
    """
    llm, tools = _build_llm_and_tools()
    langfuse_name = LANGFUSE_PROMPT_NAME if prompt_name == "single_shot_agent" else None
    agent = create_react_agent(llm, tools, _react_prompt(prompt_name, langfuse_name))
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
    )


def new_session_id() -> str:
    """A timestamped session id, grouping a run's traces in Langfuse."""
    return f"agent-{datetime.now():%Y%m%d-%H%M%S}"


# The ReAct model emits its whole turn as one stream: the reasoning
# (Thought/Action/...) and then "Final Answer: <text>". We let the reasoning
# tokens flow into the verbose trace and only stream the text *after* this marker
# as the clean answer.
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


def _prepare_input(question: str, echo: bool) -> str:
    """Apply context management before the LLM call.

    For storage/latency questions, retrieve relevant docs via RAG and prepend them
    to the question. A ContextManager bounds each source to its token budget (and
    logs usage); when `echo`, the budget report is printed. Context modules are
    imported lazily so non-agent code paths don't pay for tiktoken/chroma.
    """
    from context.manager import ContextManager
    from context.rag import needs_rag, retrieve

    cm = ContextManager()
    ctx = {"question": question}
    if needs_rag(question):
        ctx["retrieved_context"] = retrieve(question)

    ctx = cm.enforce_budget(ctx)  # truncate each source to budget + log usage
    if echo:
        print(cm.budget_report(ctx))

    retrieved = ctx.get("retrieved_context")
    if retrieved:
        return f"Relevant documentation:\n{retrieved}\n\nQuestion: {ctx['question']}"
    return ctx["question"]


async def stream_answer(
    executor: AgentExecutor,
    question: str,
    session_id: str | None = None,
    echo: bool = True,
) -> str:
    """Run one turn; stream the final answer token-by-token and return it.

    `astream_events` yields a typed event for everything the executor does while it
    runs the ReAct loop. We `async for` over that stream and react to
    `on_chat_model_stream` events — each carries one LLM token. Tokens before
    "Final Answer:" stay in the reasoning trace; tokens after it are the answer,
    printed live (when `echo`) and returned for callers that persist it.
    """
    agent_input = _prepare_input(question, echo)
    buffer = ""          # accumulates tokens until we spot the marker
    streaming = False     # flips True once we're past "Final Answer:"
    answer: list[str] = []  # the final-answer text, collected for the return value

    # Callbacks (print hooks + LangFuse) are attached via config so they fire under
    # astream_events. Trace attributes follow the official Langfuse pattern: run_name
    # for a findable trace, langfuse_* metadata to group a session and tag it.
    config = {
        "callbacks": get_callbacks(),
        "run_name": "research-agent-turn",
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_tags": ["agent-practice", "react-agent"],
        },
    }
    async for event in executor.astream_events({"input": agent_input}, version="v2", config=config):
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


def load_recent_turns(limit: int = 10) -> list[tuple[str, str]]:
    """Return the most recent (question, answer) turns from persisted memory."""
    messages = build_memory().chat_memory.messages

    turns: list[tuple[str, str]] = []
    pending_q: str | None = None
    for msg in messages:
        if msg.type == "human":
            pending_q = msg.content
        elif msg.type == "ai" and pending_q is not None:
            turns.append((pending_q, msg.content))
            pending_q = None

    return turns[-limit:]
