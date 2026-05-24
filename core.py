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

from hooks import get_callbacks
from tools import get_tools

# Conversation history is persisted here so it survives across CLI invocations
# (e.g. `agent ask` then `agent history` run in separate processes).
HISTORY_PATH = Path(__file__).parent / ".agent_history.json"

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


def build_chat_agent(memory: ConversationBufferMemory | None = None) -> AgentExecutor:
    """ReAct executor WITH conversation memory.

    Used by `chat` and `research`: each turn's input/output is saved and replayed
    into the {chat_history} prompt slot, so the model has prior context. Pass a
    `memory` to use an isolated buffer (e.g. per-session in the API); the default
    is the file-backed buffer shared across CLI runs.

    verbose=True prints the Thought/Action/Observation loop; the explicit hooks are
    attached per-call via config in stream_answer (constructor callbacks don't
    propagate through astream_events).
    """
    llm, tools = _build_llm_and_tools()
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory or build_memory(),
        verbose=True,
        handle_parsing_errors=True,
    )


def build_single_shot_agent() -> AgentExecutor:
    """ReAct executor with NO memory at all.

    Used by `ask`: it neither loads nor saves history, and its prompt has no
    {chat_history} slot — so every call starts from zero prior context. This keeps
    a single-shot question's input token count minimal instead of growing with the
    accumulated conversation.
    """
    llm, tools = _build_llm_and_tools()
    agent = create_react_agent(llm, tools, REACT_PROMPT_SINGLE)
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
