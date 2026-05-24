# langchain-agent-layers

A small ReAct research agent — **Claude + LangChain** — built up in **nine deliberate
layers**, each adding one agent-runtime capability. It's a hands-on project for
understanding how agent frameworks actually work under the hood: the agent loop, tool
calling, memory, streaming, lifecycle hooks, production tracing, a CLI, a REST API, and
tests + evals.

```bash
uv run python agent.py ask "What is a Merkle tree?"
```

## The nine layers

The agent was built incrementally; each layer adds one capability on top of the last.

| Layer | Capability | Runtime concept |
|---|---|---|
| **1 — ReAct agent + tools** | Single-shot Q&A with tool use | The agent loop, tool calling, and the ReAct text protocol. The LLM picks which tool per step; the executor just runs the loop. |
| **2 — Conversation memory** | Remembers earlier turns | Memory is *string concatenation into the prompt*. The LLM is stateless every call; continuity is replayed history (`ConversationBufferMemory` → `{chat_history}`). |
| **3 — Interactive loop** | Multi-turn REPL | Session lifecycle and state ownership — the executor and its memory are built once, before the loop, so context persists. |
| **4 — Streaming output** | Final answer streams token-by-token | Token streaming vs. step streaming: `astream_events` exposes the chat model's individual tokens, surfaced via `async for`. |
| **5 — Custom tool + hooks** | A domain tool + explicit observability | Callbacks are the framework's lifecycle hooks — `on_tool_start/end`, `on_llm_start/end` — the explicit version of what `verbose=True` does implicitly. |
| **6 — Production observability** | Structured traces in LangFuse | Same callback mechanism, durable structured spans (tool/LLM, latency, tokens) instead of stdout. Degrades gracefully to print hooks when unconfigured. |
| **7 — CLI with multiple front doors** | One agent, several entry points | A Click CLI; memory persisted to disk so it survives across separate command processes. Single-shot `ask` uses a memory-free agent to keep its prompt small. |
| **8 — Separation of concerns + REST API** | Modular package; an HTTP front door | Split into `tools` / `hooks` / `core` / `api` / `agent` with one-way imports, plus a FastAPI server. The runtime is decoupled from its delivery — CLI and API share the same core; the API isolates memory per `session_id`. |
| **9 — Testing + evals** | Automated quality gate | pytest (`tests/`) for deterministic plumbing with the LLM stubbed; evals (`evals/`) for probabilistic agent behavior — tool/answer assertions and LLM-as-judge relevance scores written to LangFuse. Tests answer "does it work?"; evals answer "is the answer good?" |

## How it works

```
question ──▶ ReAct loop (AgentExecutor) ──▶ Thought → Action → Observation … → Final Answer
                  │
                  ├─ tools: DuckDuckGo search · calculator · storage_metrics
                  ├─ memory: file-backed chat history → {chat_history}
                  ├─ callbacks: StepLogger (stdout) + LangFuse (traces)
                  └─ streaming: astream_events → tokens after "Final Answer:"
```

The ReAct prompt instructs Claude to interleave reasoning (`Thought:`) and actions
(`Action:` / `Action Input:`). After each tool call, the executor feeds the result back as
an `Observation:` and re-prompts — looping until the model emits `Final Answer:`.

## Stack

| Dependency | Role |
|---|---|
| `langchain` / `langchain-classic` | Core abstractions; the classic `create_react_agent` + `AgentExecutor` (in `langchain-classic` as of LangChain 1.x). |
| `langchain-anthropic` | `ChatAnthropic` — Claude as the reasoning LLM. |
| `langchain-community` | `DuckDuckGoSearchRun` and `FileChatMessageHistory`. |
| `ddgs` | DuckDuckGo client (replaces the deprecated `duckduckgo-search`). |
| `langfuse` | Optional production tracing via the LangChain callback handler. |
| `python-dotenv` | Loads `.env` so credentials stay out of source. |
| `click` | The CLI framework and auto-generated `--help`. |
| `fastapi` / `uvicorn` | The REST API layer and the ASGI server that runs it. |
| `pytest` | Unit/integration test runner (`tests/`). |
| `deepeval` | Installed for eval tooling (its pytest plugin is disabled — see [Notes](#notes)). |

Managed with [uv](https://docs.astral.sh/uv/). Requires Python 3.13+.

## Getting started

```bash
# 1. Install dependencies
uv sync

# 2. Set up credentials
cp .env.example .env          # then edit .env and add your real keys
```

`.env` (git-ignored) is loaded at startup, so its values reach both the CLI and the API
server. At minimum set `ANTHROPIC_API_KEY` (required for the LLM-backed commands and
`/ask` / `/chat`). The `LANGFUSE_*` keys are optional — leave the placeholders and tracing
is skipped, with the agent running fine on the built-in print hooks. You can also just
`export ANTHROPIC_API_KEY=...` in your shell instead of using `.env`.

## Usage

```bash
uv run python agent.py --help              # list all commands

uv run python agent.py chat                # interactive REPL (memory + streaming)
uv run python agent.py ask "What is X?"    # one question, one answer
uv run python agent.py research "topic" -o report.md   # research → markdown file
uv run python agent.py calc "150 * 223.48"             # direct calculator, no LLM
uv run python agent.py metrics prod-us-east-1          # direct metrics tool, no LLM
uv run python agent.py history             # last 10 turns from saved memory
uv run python agent.py serve --port 8000   # start the FastAPI REST server
uv run python agent.py test                # run tests + evals, print a summary
```

- **`chat`, `ask`, `research`** hit the LLM (need `ANTHROPIC_API_KEY`) and carry LangFuse
  traces when configured.
- **`calc`, `metrics`, `history`** call tools/memory directly — no API key, no network, instant.

In `chat`, the agent prints its Thought / Action / Observation loop, then streams the final
answer. Follow-up questions resolve against memory. Exit with `exit`/`quit` or Ctrl+C.

### REST API

Start the server with `agent serve` (or `uvicorn api:app`), then:

```bash
curl localhost:8000/health
curl "localhost:8000/calc?expr=150*223.48"
curl localhost:8000/metrics/prod-us-east-1
curl -X POST localhost:8000/ask  -H 'Content-Type: application/json' -d '{"question":"What is X?"}'
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"Hi","session_id":"s1"}'
```

`/chat` keeps isolated per-session memory keyed by `session_id`. Interactive docs at `/docs`.

### Tests and evals

```bash
uv run pytest                # unit + integration tests (fast, no API key)
uv run python agent.py test  # tests + evals + summary (evals make real LLM calls)
```

Two kinds of checks: **tests** (`tests/`) assert deterministic plumbing with the LLM
stubbed; **evals** (`evals/`) assert probabilistic agent behavior — deterministic tool/answer
cases, plus LLM-as-judge relevance scores written to LangFuse.

## Project structure

| File | What it does |
|---|---|
| `tools.py` | Tool definitions (`calculator`, `storage_metrics`, web search) + `get_tools()`. |
| `hooks.py` | Print-based `StepLogger` callbacks and LangFuse setup; `get_callbacks()` / `flush_traces()`. |
| `core.py` | The framework-agnostic runtime: ReAct prompts, memory, agent builders, `stream_answer()`. |
| `api.py` | FastAPI app: `/ask`, `/chat`, `/metrics`, `/calc`, `/health` + CORS. |
| `agent.py` | The Click CLI; `serve` runs the API via uvicorn, `test` runs the quality gate. |
| `tests/` | pytest suite — tool units + API integration (LLM stubbed). |
| `evals/` | Real-agent behavioral evals: deterministic cases + LLM-as-judge scoring. |
| `.env` | Local secrets (`LANGFUSE_*`). Git-ignored. |
| `.agent_history.json` | Persisted conversation memory. Git-ignored; created on first turn. |
| `pyproject.toml` / `uv.lock` | Dependencies, managed by uv. |

Import direction is one-way: `tools`/`hooks` ← `core` ← `agent`/`api`.

## Notes

- **Observability:** print hooks (`StepLogger`) and LangFuse tracing share one callback
  mechanism. Callbacks are attached per-call via `config={"callbacks": [...]}` because
  constructor-level callbacks don't propagate through `astream_events`.
- **Memory persistence:** backed by `FileChatMessageHistory` so `history` (a separate
  process from `chat`) can read prior turns — state has to live outside the process once
  separate commands share it.
- **Version notes:** LangChain 1.x moved the classic agents to `langchain-classic`; LangFuse
  v4 exposes its handler at `langfuse.langchain.CallbackHandler` with auth via `LANGFUSE_*`
  env vars (not constructor args).
- **deepeval:** its pytest plugin imports the removed `langchain.schema`, which breaks
  collection, so it's disabled via `addopts = "-p no:plugins"` in `pyproject.toml`. deepeval
  is installed but the current evals use a lightweight custom approach rather than its API.

This is a learning project, not a production system — the `storage_metrics` tool returns
synthetic data, and answers depend on live web search.

## License

[MIT](LICENSE)
