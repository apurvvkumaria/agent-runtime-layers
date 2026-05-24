# CLAUDE.md

## What this project is

A hands-on learning project for understanding **agent runtime patterns** by building a
small research agent from scratch. The goal is to build intuition for how agent frameworks
actually work under the hood, not to ship a product.

The agent has two front doors — a Click CLI (`agent.py`) and a FastAPI REST server
(`api.py`) — over one shared ReAct core (`core.py`): it reasons with Claude, calls tools
(web search, a calculator, and a fake storage-metrics backend) to gather facts, remembers
the conversation across turns, streams its final answer token-by-token, and emits both
print-based hooks and structured LangFuse traces. Every reasoning step is visible. It was
built up in eight deliberate layers (see below), each adding one runtime capability.

## How to run it

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # auth for Claude (LLM commands only)

uv run python agent.py --help              # list all commands
uv run python agent.py chat                # interactive REPL (memory + streaming)
uv run python agent.py ask "What is X?"    # one question, one answer
uv run python agent.py research "topic" -o report.md   # research → markdown file
uv run python agent.py calc "150 * 223.48"             # direct calculator, no LLM
uv run python agent.py metrics prod-us-east-1          # direct metrics, no LLM
uv run python agent.py history             # last 10 turns from saved memory
uv run python agent.py serve --port 8000   # start the FastAPI REST server
```

`chat`, `ask`, and `research` hit the LLM (need the API key) and carry LangFuse traces.
`calc`, `metrics`, and `history` call tools/memory directly — no API key, no network, instant.

The same capabilities are also exposed over HTTP. Start the server with
`agent serve` (or `uvicorn api:app`), then:

```bash
curl localhost:8000/health
curl "localhost:8000/calc?expr=150*223.48"
curl localhost:8000/metrics/prod-us-east-1
curl -X POST localhost:8000/ask  -H 'Content-Type: application/json' -d '{"question":"What is X?"}'
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"Hi","session_id":"s1"}'
```

`/chat` keeps per-session memory keyed by `session_id` (isolated, in-process — distinct
from the CLI's single shared history file). Interactive API docs at `/docs`.

In `chat` you get an interactive prompt; the agent prints its Thought / Action / Observation
loop (`verbose=True`), then streams the final answer below a separator. Follow-ups resolve
against memory. `exit`/`quit` or Ctrl+C ends the session. To drive it non-interactively,
pipe input: `printf 'What is 25 * 4?\nexit\n' | uv run python agent.py chat`.

LangFuse credentials are read from `.env` (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
`LANGFUSE_HOST`), loaded at startup via `python-dotenv`. If they're absent or left as the
`your-...` placeholders, tracing is skipped and the agent runs fine on the print hooks
alone. `.env` is git-ignored — it can hold real secrets, so never commit it.

## Stack and why each dependency exists

This project uses **uv** for dependency and environment management. Add deps with
`uv add <pkg>`; never edit `pyproject.toml` deps by hand.

| Dependency | Why it's here |
|---|---|
| `langchain` | Core abstractions (prompts, tools, runnables). |
| `langchain-classic` | Home of the classic `create_react_agent` + `AgentExecutor` in LangChain 1.x. The top-level `langchain.agents` no longer exposes these. |
| `langchain-anthropic` | `ChatAnthropic` — Claude as the reasoning LLM. |
| `langchain-community` | `DuckDuckGoSearchRun`, the web search tool. |
| `ddgs` | The actual DuckDuckGo client. Replaces the deprecated `duckduckgo-search`; the community tool requires it at runtime. |
| `langfuse` | Production observability — sends each run to LangFuse as structured spans (tool calls, LLM calls, latency, token counts). |
| `python-dotenv` | Loads `.env` at startup so credentials (incl. `LANGFUSE_*`) stay out of the source. |
| `click` | The CLI framework — command groups, arguments/options, and auto-generated `--help`. |
| `fastapi` | The REST API layer (`api.py`) — async endpoints, request validation, CORS. |
| `uvicorn` | ASGI server that runs the FastAPI app (`agent serve`). |

## Project structure

The code is split by responsibility. Dependency direction is one-way:
`tools`/`hooks` ← `core` ← (`agent`, `api`); `agent` also imports `api` for `serve`.
Nothing imports `agent`.

| File | What it does |
|---|---|
| `tools.py` | All tool definitions: `calculator` (safe AST eval), `storage_metrics` (synthetic), and `DuckDuckGoSearchRun`. Exports `get_tools()`. Tools are also imported directly for the no-LLM `calc`/`metrics` paths. |
| `hooks.py` | Observability. `StepLogger` (print-based `on_tool_*`/`on_llm_*` callbacks) and the LangFuse handler setup. Exports `get_callbacks()` (print hooks + LangFuse when configured; LangFuse handler cached per process) and `flush_traces()`. |
| `core.py` | The agent runtime, framework-agnostic. ReAct prompts, `build_memory()`, the two builders — `build_chat_agent()` (memory) and `build_single_shot_agent()` (none), `new_session_id()`, the `stream_answer()` streaming loop, and `load_recent_turns()`. Imports `tools` + `hooks`. |
| `api.py` | FastAPI app with async endpoints — `POST /ask`, `POST /chat` (per-session in-memory agents), `GET /metrics/{cluster}`, `GET /calc?expr=`, `GET /health` — plus CORS. Imports `core`. |
| `agent.py` | The Click CLI only: `chat`, `ask`, `research`, `calc`, `metrics`, `history`, `serve` (starts `api.app` via uvicorn). The REPL (`converse`) lives here. Imports `core` + `api`. |
| `.env` | Local secrets (`LANGFUSE_*`). Git-ignored; ships with `your-...` placeholders. |
| `.agent_history.json` | Persisted conversation memory (`FileChatMessageHistory`), so the CLI's memory survives across invocations. Git-ignored; created on first turn. |
| `main.py` | The uv-generated stub. Not used by the agent; safe to ignore or repurpose. |
| `pyproject.toml` / `uv.lock` | Dependencies, managed by uv. |

## The eight layers

The agent was built incrementally, each layer adding one agent-runtime capability on top of
the last. They all live in the current `agent.py`; this is the conceptual progression, not
separate files.

| Layer | Capability | What it added in code | Runtime concept |
|---|---|---|---|
| **1 — ReAct agent + tools** | Single-shot question answering with tool use | `create_react_agent` + `AgentExecutor(verbose=True)`, the inline `REACT_PROMPT`, and two tools: `DuckDuckGoSearchRun` and a custom `@tool` `calculator` (safe AST eval, not `eval`). | The agent loop, tool calling, the ReAct text protocol. The LLM picks which tool per step; the executor just runs the loop. |
| **2 — Conversation memory** | Remembers earlier turns | `ConversationBufferMemory(memory_key="chat_history", output_key="output")` passed to the executor; a `{chat_history}` slot added to the prompt. | Memory is *string concatenation into the prompt*. The LLM is stateless every call; the executor fakes continuity by replaying the buffer. A follow-up like "150 shares at that price" only resolves because the prior answer is in `{chat_history}`. |
| **3 — Interactive loop** | Multi-turn REPL | A `while True` loop reading `input()`, `exit`/`quit` handling, graceful `KeyboardInterrupt`/`EOFError` exit, and an end-of-session memory dump. The executor (and its memory) is built **once, before the loop**, so context persists. | Session lifecycle and state ownership: rebuild the agent per turn and you get amnesia. |
| **4 — Streaming output** | Final answer streams token-by-token | `stream_answer()` consumes `executor.astream_events(..., version="v2")` with `async for`, filters `on_chat_model_stream` events, and prints tokens after the `Final Answer:` marker. `converse()`/`main()` became async (`asyncio.run`). | Token streaming vs. step streaming: `.astream()` streams whole agent steps; `astream_events` exposes the underlying chat model's tokens. Streaming observes generation *in progress*; the verbose callback observes the same call *at completion* (hence the final answer can appear twice — once streamed, once in the trace). |
| **5 — Custom tool + hooks** | A domain tool, and explicit lifecycle observability | `@tool storage_metrics(cluster_name)` returns fake-but-realistic distributed-storage metrics. `StepLogger(BaseCallbackHandler)` implements `on_tool_start/end` (with timing) and `on_llm_start/end` (with token counts). | Callbacks are the framework's lifecycle hooks — the explicit version of what `verbose=True` does implicitly. Note: `on_chat_model_start` falls back to `on_llm_start` for chat models, so the `on_llm_start` hook fires for Claude. |
| **6 — Production observability** | Structured traces in LangFuse | `build_langfuse_handler()` returns a LangFuse `CallbackHandler` when `LANGFUSE_*` keys are real (else `None`, degrading to print hooks). `python-dotenv` loads `.env`; the LLM commands flush the client on exit. Trace attributes (`run_name`, `langfuse_session_id`, `langfuse_tags`) are set per-call via `config` metadata, per the official Langfuse skill pattern. | Print hooks are for *you, now*; tracing is for *operators, later* — same callback mechanism, durable structured spans (tool/LLM spans, latency, token counts) instead of stdout. Both run together; LangFuse failures never crash the agent. |
| **7 — CLI with multiple front doors** | One agent, several entry points | A Click `cli` group: `chat` (REPL), `ask` (one-shot), `research` (saves markdown, `stream_answer` now returns the answer text), `calc`/`metrics` (direct tool calls, no LLM), `history` (reads saved memory), `serve` (placeholder). Memory moved to `FileChatMessageHistory` so it persists across processes. Two agent builders: memory-backed (`chat`/`research`) vs. memory-free (`ask`). | Process boundaries force state to be externalized: `history` runs in a *different* process than `chat`, so in-process memory would always be empty — hence the on-disk store. Match the agent to the command: a single-shot `ask` must not replay accumulated history into its prompt, or input tokens balloon with every past turn. LLM commands and direct-tool commands are deliberately separate front doors over the same core. |
| **8 — Separation of concerns + REST API** | Modular package; a second front door | Split into `tools.py` / `hooks.py` / `core.py` / `api.py` / `agent.py` with one-way imports. Added a FastAPI app (`api.py`) with async `/ask`, `/chat`, `/metrics`, `/calc`, `/health` + CORS; `serve` now runs it via uvicorn. `/chat` keeps isolated per-session memory in a dict. | The runtime (`core`) is decoupled from its delivery (CLI vs. HTTP) — both front doors call the same builders and `stream_answer`. Different transports want different memory models: the CLI shares one file; the API isolates per `session_id`, so `build_chat_agent(memory=...)` takes an injected buffer. |

## Key concepts being learned

The point of the project is to internalize these, so explanations should connect code to
the underlying pattern:

- **The agent loop** — an LLM in a loop that decides an action, observes a result, and
  repeats until it can answer. `AgentExecutor` *is* that loop; reading its verbose output
  shows each iteration.
- **Tool calling** — exposing a function (web search) to the model with a name and
  description, letting the model choose when to invoke it and feeding the result back in.
- **The ReAct pattern** — interleaving **Rea**soning (Thought) and **Act**ing (Action) in
  text. The prompt in `agent.py` (`REACT_PROMPT`) defines the Thought/Action/Action
  Input/Observation format the model must follow. Worth contrasting with native
  tool-calling APIs, where the framework handles structure instead of a text prompt.
- **Conversation memory** — short-term state replayed into the prompt each turn
  (`ConversationBufferMemory` → `{chat_history}`). The model stays stateless; continuity is
  an illusion the executor maintains by re-sending history. Backed by `FileChatMessageHistory`
  so it survives across CLI processes — state has to live somewhere outside the process once
  separate commands need to share it. The flip side: replaying history isn't free — it's
  prompt tokens on every call, so a single-shot `ask` uses a memory-free agent to stay flat
  instead of paying for context it doesn't need.
- **Async streaming** — `astream_events` + `async for` to surface LLM tokens as they're
  generated, vs. blocking until the whole answer is ready. The async/await structure here is
  the part worth understanding deeply.
- **Callbacks / hooks** — the framework calls `on_tool_start/end`, `on_llm_start/end`, etc.
  at each lifecycle point. One mechanism powers `verbose=True`, the `StepLogger` print hooks,
  and LangFuse tracing alike. **Gotcha that bit us:** callbacks set on the `AgentExecutor`
  constructor do *not* propagate through `astream_events` — they must be passed per-call via
  `config={"callbacks": [...]}`. That's why `stream_answer` attaches them at invocation.
- **Observability as data, not prints** — LangFuse turns the same hook events into structured
  spans (tool/LLM, latency, tokens) you can query later. Print hooks debug the run in front of
  you; tracing serves whoever operates it next.

## Conventions for Claude Code

- **Audience:** I have a deep distributed-systems background and trying to experiment with agent frameworks further outside of my current work exposure.
  Skip generic programming explanations; lean into *why* an agent framework is designed a
  certain way and how it maps to systems concepts I already know (control loops, retries,
  orchestration, state). Don't dumb things down.
- **Explain the runtime, not just the code.** When changing `agent.py`, say what the
  framework does behind the call (e.g., what `AgentExecutor.invoke` does per iteration),
  since understanding the runtime is the whole point.
- **Use uv for everything** — `uv run python agent.py`, `uv add`, `uv remove`. Don't
  suggest `pip` or hand-editing dependency lists.
- **Keep it minimal and readable.** This is a learning artifact; favor clear, well-commented
  code over abstraction or production hardening unless I ask.
- **Flag framework-version gotchas.** LangChain 1.x and LangFuse v4 moved/renamed things —
  e.g. classic agents → `langchain-classic`, `duckduckgo-search` → `ddgs`, and LangFuse's
  handler → `langfuse.langchain.CallbackHandler` (not `langfuse.callback`) with auth via
  `LANGFUSE_*` env vars instead of constructor args. Call these out rather than silently
  working around them.
- **Never commit secrets.** `.env` holds real keys and is git-ignored; don't echo secret
  values or write them into tracked files.
- **Don't make billed API calls without asking** — running the agent live hits the
  Anthropic API. Smoke-test construction with a dummy key when verifying changes.
