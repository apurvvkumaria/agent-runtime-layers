# agent-runtime-layers

![Layers](https://img.shields.io/badge/layers-29-blue)

A production-grade agent runtime with a full skill ecosystem — built on **Claude, LangChain,
LangGraph, and Strands** — assembled across **twenty-nine deliberate layers**, each adding one
runtime capability: memory, observability, autonomous operation, sandboxed execution, and a
skill ecosystem with versioning, composition, a verified marketplace, dependency resolution,
and hot-reload. The core is a ReAct agent; later layers add a LangGraph multi-agent pipeline,
and the same pipeline rebuilt with Strands, as contrasting paradigms. It's a hands-on project for
understanding how agent frameworks actually work under the hood: the agent loop, tool
calling, memory, streaming, lifecycle hooks, production tracing, a CLI, a REST API,
tests + evals, MCP (Model Context Protocol) in both directions, file/LangFuse-based
prompt management, vector-store memory, a LangGraph multi-agent pipeline, token-budget
context management with RAG, a LangGraph-vs-Strands comparison, age-based memory decay,
a streaming multi-agent graph, autonomous (cron + heartbeat) operation, a dead-letter
queue for failed runs, a composed skill, running the whole agent inside an **NVIDIA
OpenShell** sandbox under a declarative network/filesystem policy, a two-tier cache that
makes re-running the LLM-as-judge evals cost zero tokens, and skills packaged in the exact
OpenClaw shape (`SKILL.md` + `SkillContext` + `ctx.call_tool`).

```bash
uv run python agent.py ask "What is a Merkle tree?"
```

## The twenty-nine layers

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
| **10 — MCP integration** | Speak MCP both ways | **Client:** wrap the official filesystem MCP server as the `filesystem` tool (sandboxed to `docs/`). **Server:** expose `ask_agent` / `get_storage_metrics` / `calculate` over MCP via FastMCP (`agent mcp-serve`). As a client the agent consumes any MCP server as a tool; as a server the whole agent becomes a tool other MCP clients can call. |
| **11 — Prompt management** | Prompts as managed assets | System prompts live in `prompts/*.md` (loaded by `load_prompt`), not string literals in code. The single-shot prompt is fetched from LangFuse first (`react-agent-prompt`) with the local file as fallback; `agent sync-prompt` pushes the local copy to LangFuse as a new version. Prompts become reviewable in diffs and versionable without a redeploy. |
| **12 — Vector-store memory** | Bounded memory via semantic retrieval | `build_chat_agent` defaults to `VectorStoreMemory`: each turn is embedded and stored, and only the top-k *similar* past turns are replayed into the prompt — so history tokens stay bounded as the conversation grows (vs. buffer memory, which re-sends everything). `agent memory-stats` and `evals/memory_comparison.py` quantify it (~65% fewer history tokens at 8+ turns). |
| **13 — Multi-agent (LangGraph)** | A graph of cooperating agents | A `StateGraph` of five agent nodes — orchestrator (routes research/calculate/both), research, calculator, writer, reviewer — with conditional edges and a reviewer→research retry loop gated on a quality score. `agent pipeline "..."` streams each node. A fixed, inspectable topology vs. the ReAct loop's per-turn tool choice. |
| **14 — Context management** | Budget the window; ground in docs | A `ContextManager` (tiktoken) allocates a token budget across sources and truncates each to its share; RAG over `docs/` (ChromaDB + sentence-transformers) is auto-injected for storage/latency questions. `agent context-stats "..."` previews the allocation; `evals/rag_comparison.py` shows token cost vs. answer quality (with vs. without RAG). |
| **15 — Strands + framework comparison** | The same pipeline, model-driven | The research pipeline rebuilt with Strands Agents — research + calculator specialists exposed to an orchestrator via `Agent.as_tool()`, with *no* explicit graph (the model decides routing). `agent pipeline --framework {langgraph,strands,both}` runs either; `agent compare "..."` tabulates nodes/steps, tokens, time, and quality. Fixed graph (cheap, predictable) vs. emergent model-driven orchestration (flexible, more LLM round-trips). |
| **16 — Memory decay** | Old context compresses, then expires | Each stored turn carries a tier that downgrades with age: `full` (<3d, verbatim) → `summary` (3-30d, one-sentence LLM summary) → `marker` (30-90d, topic tag) → `archived` (>90d, deleted). `agent memory-decay` ages turns out; `memory-stats` shows the breakdown. Keeps the retrievable store bounded without a hard cutoff — recent stays sharp, old blurs to a gist, then drops. |
| **17 — Streaming the pipeline** | Live node progress + token-by-token answer | `agent pipeline` (LangGraph) streams via multi-mode `astream(["updates","messages"])`: node names print as they complete, and the writer node's answer streams token-by-token (filtered by `langgraph_node`). Layer 4 streamed the single ReAct loop; this streams a multi-agent graph. |
| **18 — Autonomous modes** | Run without a human in the loop | `AgentScheduler` runs a question on a cron schedule (APScheduler), appending timestamped answers to a file; `HeartbeatLoop` polls `tasks.json`, runs pending tasks, and queues agent-suggested follow-ups (self-directing, bounded). CLI: `agent schedule`/`heartbeat`/`add-task`. Both are long-running blocking processes. |
| **19 — Dead-letter queue** | Failed runs are captured, not lost | `core` records each failed run to a DLQ with a reason, classified transient (retry) vs. permanent (review). `agent dlq-retry` replays transient failures with exponential backoff (promoting exhausted ones to permanent); `dlq-stats`/`dlq-clear` report and review. Failures are also flagged 0 in LangFuse. Same idea as a message-queue DLQ, for agent runs. |
| **20 — Skills (composed tools)** | One tool that orchestrates several | `research_and_summarize` is a `@tool` that internally runs web search → storage metrics → LLM summarization and returns a structured report (Research Findings / Storage Context / Summary). It's in `get_tools()`, so the agent picks it for "research and summarize" requests; `agent skill "..."` runs it directly. A skill packages a multi-tool workflow behind one tool interface (same pattern as OpenClaw skills). |
| **21 — Run inside an OpenShell sandbox** ✅ *verified* | The agent executes under a declarative sandbox policy | `agent sandbox-ask "…"` runs the whole agent inside an NVIDIA OpenShell sandbox via one `openshell sandbox create --upload … --no-keep -- … agent.py ask`, then auto-deletes it; `agent sandbox-info` shows gateway/sandboxes/policy. The policy (`openshell/policy.yaml`, real v0.0.47 schema) is **binary-keyed, default-deny** egress — the sandbox's python may reach only `api.anthropic.com` + DuckDuckGo. `openshell/agent-sandbox/` bakes the deps; setup in `openshell/setup.md` + `scripts/`. Verified end-to-end: answers `2 + 2 = 4` from inside the sandbox. The agent as an isolated, policy-constrained workload instead of a host process. |
| **22 — Eval response caching** | Re-judging unchanged prompts costs zero tokens | A judge call at temperature 0 is a pure function of `(prompt, input, model)`, so it's cacheable — memoize it (two-tier: in-process `lru_cache` over a SHA256-keyed JSON store) and a re-run spends zero tokens. Determinism is what makes the cache safe. |
| **23 — Skills as OpenClaw packages** | A skill is structurally identical to an OpenClaw skill | A skill is a self-describing unit — a capability manifest the LLM reads (`SKILL.md`) plus an impl that calls tools through a context object (`ctx.call_tool`, mirroring OpenClaw's `ctx.call_claw`). Same shape as an OpenClaw skill, so one written here would drop straight into OpenClaw. |
| **24 — Skill-level evals** | A behavioral contract per skill | A skill is a contract boundary, so it deserves its own contract test — assert the output shape each consumer relies on, separately from the end-to-end agent evals of Layer 9. Deterministic grading keeps it cheap even though the skill runs for real. |
| **25 — Skill versioning & deprecation** | Skills carry a version and can be retired gracefully | Skills are an API surface, so they need API lifecycle: a version to pin against, and a deprecation window where old callers keep working (warned, with a sunset date) while the agent and new wiring are steered to the replacement. |
| **26 — Skill composition (skills calling skills)** | A skill orchestrates *other skills* | A skill is itself a callable unit, so skills nest like functions (Layers 20/23 composed *tools*; this composes *skills*). Composition is a call graph that can recurse, so the runtime concern is bounding it — a hard `max_depth` on `ctx.call_skill`. |
| **27 — Skill marketplace / remote loading** | Discover, verify, and install skills from a registry | Loading remote code is a supply-chain surface, so trust is pinned by content hash (a SHA256 in the registry) and verified *before* anything executes — fail-closed on a mismatch. Provenance (the hash) and containment (the Layer 21 sandbox) are complementary controls. |
| **28 — Skill dependency resolution** | Installing a skill pulls in its dependencies, in order | A skill registry is a package ecosystem, so it inherits a package manager's job — a dependency graph, version constraints, a topological install order, and cycle detection — resolved across two satisfaction sources: already-installed skills vs. fetch-from-marketplace. |
| **29 — Skill hot-reloading** | Pick up edited / newly-installed skills without a restart | A running process caches imported modules, so live code changes stay invisible until you deliberately re-exec them. Hot-reload trades restart-simplicity for continuity: fingerprint each skill's source and `importlib.reload` only what changed. |

## How this maps to NemoClaw

These 29 layers are a from-scratch rebuild of the patterns in **NemoClaw** —
NVIDIA's batteries-included agent stack (**OpenShell** sandbox + **OpenClaw** agent
+ **NeMo/Nemotron** inference). [`docs/nemoclaw_mapping.md`](docs/nemoclaw_mapping.md)
maps every layer to its NemoClaw equivalent, and calls out what NemoClaw adds
(multi-channel I/O, local GPU inference, managed onboarding) vs. what this project
adds on top (eval response caching, a quality-gated retry loop, a dependency
resolver). The short version: same architectural patterns, built here to
understand them rather than install them — with Claude standing in for a local
Nemotron, and the agent actually running inside a real OpenShell sandbox (Layer 21).

## Architecture

The runtime data flow: every front door — the Click **CLI**, the FastAPI **REST API**,
and the **MCP** server — runs over one shared ReAct **core**, which orchestrates tools,
memory, hooks, and context/RAG around the Claude LLM. The twenty-nine layers stack on top
of this spine.

![Agent runtime architecture: CLI/API/MCP → core → tools/memory/hooks/context → Claude](assets/agent_runtime_layers_architecture.svg)

Two layers with non-trivial control flow have their own diagrams:

| Multi-agent pipeline (Layers 13 & 17) | Dead-letter queue (Layer 19) |
|---|---|
| ![LangGraph StateGraph with a quality-gated retry loop](assets/langgraph_pipeline_retry_loop.svg) | ![DLQ transient/permanent classification and backoff retry](assets/dlq_classification_and_retry_flow.svg) |

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

## Numbers

Rough figures from a single run each (Claude Sonnet) — enough to make the tradeoffs
concrete; they vary with the question and model load. Reproduce with `agent compare "…"`,
`agent context-stats "…"`, `uv run python -m evals.memory_comparison`, and
`uv run python -m evals.rag_comparison`.

| Comparison | Result |
|---|---|
| **Vector vs. buffer memory** (L12) | After 8 turns, replaying history costs **440 tokens** (buffer = full transcript) vs **150 tokens** (vector = top-k retrieval) — **~66% fewer** history tokens, and the gap widens as the conversation grows. |
| **RAG, on/off** (L14) | "What does OpenShell's broker do?" — **without RAG: 15 input tokens but ungrounded** (the model declines/guesses); **with RAG: 276 input tokens, grounded in `docs/`** and correct. RAG buys a sourced answer for **+261 input tokens**. |
| **Context budgeting** (L14) | A storage/latency question fills **362 / 5,700** budgeted tokens (system 175 · question 21 · retrieved 166), +1,000 reserved for the response — **6.4%** of the window, with `sla_thresholds.md` auto-injected. |
| **LangGraph vs. Strands** (L15), `150 × 223.48` | Same answer (33,522), same quality (0.7). **LangGraph: 4 nodes · 194 tokens · 2.7 s.** **Strands: 2 model steps · 3,079 tokens · 5.8 s.** The fixed graph used **~16× fewer tokens and ~2× less time** than model-driven orchestration — the cost of letting the model route. |

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
| `deepeval` | The eval framework (`evals/`) — `LLMTestCase`, `ToolCorrectnessMetric`, a custom substring metric, and `AnswerRelevancyMetric` judged by Claude. |
| `mcp` | Model Context Protocol SDK — client (consume the filesystem server) and server (expose the agent as MCP tools). |
| `numpy` | Numeric support across the ML stack. |
| `chromadb` | Persistent vector store for conversation memory (`./chroma_db/`). |
| `sentence-transformers` | Local `all-MiniLM-L6-v2` embeddings (free, no API key) for vector memory. |
| `langgraph` | The multi-agent pipeline — a `StateGraph` with conditional edges and a retry loop. |
| `tiktoken` | Token counting (cl100k_base) for the context-budget manager. |
| `strands-agents` / `strands-agents-tools` | The Strands agents-as-tools pipeline (model-driven routing). |
| `apscheduler` | Cron scheduling for autonomous mode. |

Managed with [uv](https://docs.astral.sh/uv/). Requires Python 3.13+ on a native **arm64**
mac (or linux/win) — `torch`/`onnxruntime` (for the vector-memory stack) have no
macOS-x86_64 wheels, so an x86_64/Rosetta toolchain won't `uv sync`.

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
uv run python agent.py skill "..."         # research_and_summarize skill (composed tools)
uv run python agent.py history             # last 10 turns from saved memory
uv run python agent.py serve --port 8000   # start the FastAPI REST server
uv run python agent.py mcp-serve --port 3000  # start the MCP server (SSE)
uv run python agent.py sync-prompt         # push local single-shot prompt to LangFuse
uv run python agent.py memory-stats        # vector-store turns + estimated token savings
uv run python agent.py memory-clear        # wipe all stored turns (use --yes to skip prompt)
uv run python agent.py memory-decay        # age out stored turns by tier
uv run python agent.py pipeline "..."      # multi-agent pipeline (--framework langgraph|strands|both)
uv run python agent.py compare "..."       # run LangGraph vs Strands side by side
uv run python agent.py add-task "..."      # queue a task for the heartbeat loop
uv run python agent.py heartbeat           # process tasks.json on a loop (blocks)
uv run python agent.py schedule "..." --cron "0 9 * * *" --output report.md  # cron (blocks)
uv run python agent.py dlq-stats           # failed-run counts; dlq-retry / dlq-clear too
uv run python agent.py ask "..." --timeout 1   # force a tool_timeout into the DLQ (testing)
uv run python agent.py context-stats "..." # token budget + RAG docs for a question
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

Two kinds of checks for two kinds of failure. **Tests** (`tests/`) assert deterministic
plumbing with the LLM stubbed — tool math, API request/response shapes, memory wiring — so
they're fast, free, and run anywhere. **Evals** (`evals/`) assert *probabilistic agent
behavior*, which you can't unit-test: the agent runs for real (real Claude calls) and only the
**grading** is automated.

**What the evals check:**

- **Tool correctness + substring** — `deterministic_evals.py` (deepeval, no judge): five
  `LLMTestCase`s, each a `(question, expected_tool, expected_substring)`. *"What is 25 times
  4?"* must call `calculator` and contain `100`; *"Show the storage metrics for cluster
  prod-east-1"* must call `storage_metrics` and contain `prod-east-1`; *"…what year was the
  Eiffel Tower completed?"* must call `duckduckgo_search` and contain `1889`. A case **passes
  iff the expected tool was actually invoked AND the answer contains the expected string** —
  `ToolCorrectnessMetric` (threshold 0.5) checks a callback-recorded list of tool calls against
  the expected tool, and the custom `SubstringMetric` checks the text. Both are deterministic,
  so they're cheap and repeatable even though the agent ran for real.
- **Answer relevancy (LLM-as-judge)** — `langfuse_evals.py`: open-ended questions graded by
  deepeval's `AnswerRelevancyMetric`, **judged by Claude** (a `DeepEvalBaseLLM` wrapper, so no
  OpenAI key), threshold 0.5. The 0–1 score is written back onto the question's **LangFuse
  trace** with the judge's reasoning as the comment.
- **Behavioral comparisons** — `memory_comparison.py` / `rag_comparison.py`: the
  token/grounding measurements in [Numbers](#numbers) (memory footprint is LLM-free; RAG uses
  raw LLM calls).

**What pass/fail looks like.** `agent test` runs pytest, then both eval suites, then a summary:

```
[PASS] What is 25 times 4?
        ToolCorrectness: 1.00 (used ['calculator'], want calculator)
        Substring: 1.00 (answer contains '100')
...
[trace 154ffef…] answer_relevancy=1.00  Q: What is 15 times 15?

========== SUMMARY ==========
Unit tests: 12/12 passed
Deterministic evals: 5/5 passed
LangFuse evals: avg answer_relevancy 1.00/1.0
```

A `FAIL` names the offending metric — e.g. the agent answered from memory without calling the
calculator (`ToolCorrectness: 0.00 (used [], want calculator)`), or the answer drifted off the
expected value (`Substring: 0.00`). That distinction matters: a tool-correctness failure is a
*routing* regression, a substring/relevancy failure is an *output* regression.

**In LangFuse**, each eval question is one trace: the agent's LLM and tool calls appear as
nested spans (latency + token counts per call), with the `answer_relevancy` score attached at
the trace level and the judge's reason as a comment — so you can sort/filter for low-scoring
traces and click into exactly which step went wrong. Failed runs are also flagged `0` from the
dead-letter queue (Layer 19), so a *broken* run and a *low-quality* run show up on the same
dashboard.

**Cadence in production.** Today, evals run on demand via `agent test`. For a production
deployment, the cadence would tier: deterministic plumbing tests on every commit (already in
place, ~0 cost), tool-correctness and substring assertions on every PR (deterministic, no
LLM), `AnswerRelevancyMetric` (LLM-as-judge via Claude) on every PR to `main` with response
caching keyed on `(model, prompt, input)`, and full multi-pipeline behavioral evals (the
LangGraph-vs-Strands comparison, RAG-on-vs-off) nightly. Sample 1–5% of production traffic
for the same judge metrics and feed low-scoring traces back into the eval dataset. This
matches the tiered pattern most production LLM teams converged on in 2026: deterministic at
base, classifier in the middle, LLM-judge at the top, behavioral end-to-end on the side.

### MCP

The agent speaks the Model Context Protocol both ways:

- **As a client** — the `filesystem` tool wraps the official
  `@modelcontextprotocol/server-filesystem` (run via `npx`), letting the agent read files
  from `docs/` over MCP, sandboxed to that directory. Requires Node/`npx`.
- **As a server** — `agent mcp-serve --port 3000` exposes `ask_agent`,
  `get_storage_metrics`, and `calculate` as MCP tools (FastMCP over SSE), so any
  MCP-compatible client can call the agent.

### Prompts

System prompts live in `prompts/*.md` (loaded by `prompts/loader.py`) rather than as string
literals in code — so they're reviewable in diffs. The single-shot prompt is also managed in
LangFuse: `build_single_shot_agent` fetches `react-agent-prompt` from LangFuse first and
falls back to the local file, and `agent sync-prompt` pushes the local copy up as a new
version. Editing a prompt is a behavior change — re-run `agent test` after.

## Running inside an OpenShell sandbox (Layer 21)

[OpenShell](https://github.com/NVIDIA/OpenShell) is NVIDIA's open-source runtime for
executing autonomous AI agents inside isolated, policy-constrained sandboxes. Layer 21 runs
*this* agent inside one: instead of `agent ask` executing on the host with the host's
privileges, the entire reasoning + tool-calling loop runs as a sandboxed workload whose
network egress, filesystem, and resources are governed by a declarative policy.

**How it's wired.** A **gateway** runs as a Docker container (control plane); the `openshell`
CLI talks to it over **mTLS**, the gateway talks to per-run **sandbox** containers over gRPC
with gateway-minted JWTs, and a Rust **supervisor** inside each sandbox enforces the policy
(Landlock for the filesystem, an egress proxy for the network). `scripts/setup-openshell.sh`
brings the gateway up on **macOS + Docker Desktop** with full mTLS (the only configuration
where sandboxes actually run — `openshell/setup.md` explains why the plaintext Quick Start
dead-ends); `scripts/teardown-openshell.sh` resets it.

**Our usage.**

```sh
uv run python agent.py sandbox-info                  # gateway status + running sandboxes + active policy
uv run python agent.py sandbox-ask "What is 2 + 2?"  # run `ask` inside a sandbox, then auto-delete it
```

`sandbox-ask` does the whole round trip in **one** `openshell sandbox create` call: it stages
the agent source + a generated `.env` into a `workspace/` dir, `--upload`s it, runs
`agent.py ask` under the policy, and `--no-keep`s to delete the sandbox when the command exits
(`--keep` to retain). `sandbox_runner.py` drives the CLI as a subprocess; `openshell/` holds
the policy + setup docs; `openshell/agent-sandbox/` is the deps-baked image (`FROM` the
community `base` sandbox + the project's locked deps + a pre-cached tiktoken encoding, since
the egress policy denies PyPI at run time).

**The policy.** `openshell/policy.yaml` (real OpenShell v0.0.47 schema) is **default-deny**,
and egress is **keyed to the binary** making the connection: the sandbox's python may reach
only `api.anthropic.com` (reasoning) and DuckDuckGo (the search tool) — everything else is
denied. The filesystem is narrowed to `/sandbox` + `/tmp`; CPU/memory are capped via create
flags. The split that matters for a system-design discussion: **file delivery is host-side
control plane** (not policy-governed), while the **agent process** runs under the supervisor's
Landlock + egress enforcement — the policy governs the workload, not the plumbing.

> **Verified end-to-end:** `sandbox-ask "What is 2 + 2?"` runs the agent inside the sandbox
> and returns `2 + 2 = 4`, with the only allowed egress being the Anthropic API.

**Gotchas worth knowing** (each confirmed against the gateway's own logs — building blind would
have shipped plausible-but-wrong assumptions):

- A bare `sandbox create` **hangs** on an interactive shell; the one-shot pattern is
  `create … --no-keep -- <cmd>`.
- Egress is matched on the **fully-resolved exe path** (`/proc/{pid}/exe`), so the policy
  allowlists `/sandbox/.uv/python/**`, not the venv symlink.
- **`tls: skip`, not `terminate`** — automatic TLS termination MITMs the connection with the
  gateway's cert, which the Anthropic Python SDK (httpx + certifi) won't trust; `skip` tunnels
  to the real Anthropic cert.
- **`exec` runs with a sanitized env** that drops the image's Dockerfile `ENV`, so anything the
  agent needs (e.g. `TIKTOKEN_CACHE_DIR`) is exported in the run command.

### Isolation overhead (measured)

What does enforcing the policy actually cost? Rough p50 from a few runs of `"What is 2 + 2?"`
(Claude Sonnet), comparing the agent on the host vs. inside the sandbox:

| Measurement | p50 | What it captures |
|---|---|---|
| Host `ask` (no OpenShell) | 10.83 s | agent on bare host (incl. ~8 s of heavy imports + LLM) |
| OpenShell `sandbox-ask`, end-to-end | 16.23 s | create + upload + run + teardown |
| Warm in-sandbox exec (kept sandbox) | 9.27 s | repeated `exec`, no provisioning |
| **One-time provisioning** (≈ e2e − warm) | **~7.0 s** | container create + upload + supervisor boot + teardown |
| Per-LLM-call latency, **direct** | 1.68 s | Anthropic round trip on the host |
| Per-LLM-call latency, **via egress proxy** | 1.67 s | same round trip, *through* the policy proxy |

**Takeaway: enforcement is essentially free per-request.** The Anthropic round trip through
OpenShell's egress proxy (1.67 s) is indistinguishable from a direct call (1.68 s) — the OPA
host check + proxy hop add ~0 to the request path, and Landlock adds nothing measurable to the
agent's in-sandbox work. **The entire isolation tax is one-time provisioning (~7 s)** —
container create, source upload, supervisor boot, teardown — which is amortizable by pooling /
reusing warm sandboxes (the warm number is ≤ host). So the latency optimization target is
**cold-start provisioning, not the enforcement datapath**.

> Caveats (kept so the numbers stay honest): small N (3–5); wall time is dominated by ~8 s of
> heavy Python imports (torch/chromadb) + ~3.3 s of LLM calls, so policy is a small slice
> either way; and the warm-vs-host comparison has a confound — the host path goes through
> `uv run` (env resolution) while the in-sandbox path runs `python` directly, so read it as
> "per-request overhead within noise," not "the sandbox is faster." The clean same-method
> signal is the per-LLM-call latency (1.67 vs 1.68 s).

## Project structure

| File | What it does |
|---|---|
| `tools.py` | Tool definitions (`calculator`, `storage_metrics`, web search) + `get_tools()`. |
| `hooks.py` | Print-based `StepLogger` callbacks and LangFuse setup; `get_callbacks()` / `flush_traces()`. |
| `core.py` | The framework-agnostic runtime: ReAct prompts, memory, agent builders, `stream_answer()`. |
| `api.py` | FastAPI app: `/ask`, `/chat`, `/metrics`, `/calc`, `/health` + CORS. |
| `agent.py` | The Click CLI; `serve` runs the API, `mcp-serve` runs the MCP server, `test` runs the quality gate. |
| `mcp_integration/` | MCP both ways — `client.py` (filesystem server → `filesystem` tool), `server.py` (agent → MCP tools). Named `mcp_integration` to avoid shadowing the `mcp` SDK. |
| `prompts/` | System prompts as markdown (`single_shot_agent`, `chat_agent`, `research_agent`, `storage_agent`) + `loader.py`. |
| `memory/` | `vector_store.py` — `VectorStoreMemory` with top-k semantic retrieval (sentence-transformers + ChromaDB) and age-based decay tiers. |
| `langgraph_agents/` | `pipeline.py` — a LangGraph `StateGraph` of five agent nodes with a quality-gated retry loop (`agent pipeline`). |
| `strands_agent/` | `agent.py` — the same pipeline via Strands (agents-as-tools, model-driven routing). |
| `autonomy/` | `scheduler.py` — cron `AgentScheduler` + task-driven `HeartbeatLoop` (`agent schedule`/`heartbeat`/`add-task`). |
| `dlq/` | `manager.py` — `DLQManager`: capture/classify/retry failed runs (`agent dlq-stats`/`dlq-retry`/`dlq-clear`). |
| `skills/` | OpenClaw-pattern skills (Layer 23): each `skills/<name>/` has `SKILL.md` + `skill.py` (`async def <name>(arg, ctx=None)`) + `policy.yaml`; `context.py` (`SkillContext`) and `registry.py` (`SkillRegistry`) discover them and expose each as a tool. `SkillContext` carries `call_tool` + `call_skill` (Layer 26 composition). Each `SKILL.md` carries `## Version` + `## Status` (Layer 25); the registry warns on deprecated skills and hides them from `get_tools()`. Skills: `research_and_summarize`, `storage_health_check`, `cluster_briefing` (composes the prior two), and the deprecated `storage_sla_report` (`agent skills` / `agent skill`). `marketplace.py` verifies + installs skills (Layer 27); `resolver.py` resolves `## Requires` into a deps-first install order (Layer 28); `SkillRegistry.reload()` + `reload_demo.py` hot-reload edited/installed skills (Layer 29). |
| `marketplace/` | Skill marketplace / "remote" registry: `index.json` pins a SHA256 over each installable skill — `error_budget` (standalone) and `capacity_planner` (requires error_budget + storage_health_check). `agent marketplace-install` resolves deps, verifies each hash, then installs into `skills/`. |
| `context/` | `manager.py` (tiktoken token budgeting) + `rag.py` (RAG over `docs/` via ChromaDB). |
| `docs/` | Markdown read by the `filesystem` tool; the MCP filesystem server's only allowed directory. Includes `nemoclaw_mapping.md` (the 29-layers → NemoClaw mapping). |
| `scripts/` | OpenShell local-dev tooling (not part of the agent): `setup-openshell.sh` (gateway container w/ full mTLS on macOS + Docker Desktop), `create-sandbox.sh` (one-command sandbox + health check), `teardown-openshell.sh` (full reset), `benchmark-openshell.sh` (host-vs-sandbox latency benchmark behind the "Isolation overhead" numbers). |
| `sandbox_runner.py` | Layer 21 — drives the `openshell` CLI to run `agent.py ask` inside a policy-constrained sandbox (`agent sandbox-ask`/`sandbox-info`). |
| `openshell/` | Layer 21 config + docs (not a Python package, so it can't shadow the `openshell` SDK): `policy.yaml` (sandbox network/fs policy), `setup.md` (macOS gateway setup), `agent-sandbox/` (deps-baked sandbox image). |
| `assets/` | Architecture diagrams (SVG) embedded in this README. |
| `tests/` | pytest suite — tool units + API integration (LLM stubbed). |
| `evals/` | Real-agent behavioral evals: deterministic cases + LLM-as-judge scoring, `cache.py` / `cache_demo.py` (Layer 22 — two-tier judge-response cache), and `skill_evals.py` (Layer 24 — per-skill output-contract cases). |
| `.env` | Local secrets (`LANGFUSE_*`). Git-ignored. |
| `.agent_history.json` | Persisted conversation memory. Git-ignored; created on first turn. |
| `pyproject.toml` / `uv.lock` | Dependencies, managed by uv. |
| `CLAUDE.md` | Project guide for [Claude Code](https://claude.com/claude-code) — the full layer-by-layer build log, runtime concepts, and conventions; also useful orientation for a human reader. |

Import direction is one-way: `tools`/`hooks` ← `core` ← `agent`/`api`/`mcp_integration.server`.

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
- **deepeval:** pinned to 4.x. The older 2.x imported the removed `langchain.schema` (broke
  pytest collection); 4.x is LangChain 1.x-compatible but requires `click<8.4`, so `click` is
  pinned `>=8.1,<8.4`. Metrics judge with Claude via a custom `DeepEvalBaseLLM` (no OpenAI key).
- **MCP naming:** the integration lives in `mcp_integration/`, not `mcp/` — a top-level `mcp/`
  directory would shadow the installed `mcp` SDK and break `from mcp import ...`.
- **Native arm64 required for the ML stack:** `torch` and `onnxruntime` have no
  macOS-x86_64 + Python-3.13 wheels, so `sentence-transformers` and `chromadb` only install on
  a native arm64 mac (or linux/win). On Apple Silicon, use an arm64 `uv` + arm64 Python 3.13
  (not an x86_64/Rosetta toolchain). `uv sync` will fail on x86_64 mac because those wheels
  don't exist.

This is a learning project, not a production system — the `storage_metrics` tool returns
synthetic data, and answers depend on live web search.

## Roadmap — what Layer 22+ looks like

Twenty-one layers is a waypoint, not a ceiling. Each layer added one runtime capability; the
next ones are the concerns that turn a single-tenant demo into a *service*, and they slot onto
the same spine (front door → core → tools/memory) rather than requiring rewrites:

- **Multi-tenancy & isolation** — per-tenant memory namespaces, sessions, and vector
  collections. Today memory is one shared file (CLI) or a per-`session_id` dict (API); a real
  deployment needs tenant-scoped stores and quotas.
- **AuthN / AuthZ** — API keys or OIDC on the FastAPI + MCP front doors, plus per-tenant tool
  allowlists (which tenant may call `storage_metrics`, for which cluster).
- **Rate limiting & backpressure** — per-tenant request/token limits at the front door, feeding
  back into the heartbeat loop so autonomous work yields to interactive traffic.
- **Cost caps & accounting** — the context manager already *counts* tokens (Layer 14); this
  turns counting into *enforcement* (reject/trim before the LLM call) and attributes spend
  per-tenant via the existing LangFuse traces.
- **Human-in-the-loop approvals** — gating high-impact tool calls; the OpenShell sandbox
  (Layer 21) is the isolation primitive this would build on.

## License

[MIT](LICENSE)
