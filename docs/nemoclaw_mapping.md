# NemoClaw Architecture Mapping

How the 29 layers built in this project map onto **NemoClaw**, NVIDIA's
batteries-included agent stack. This is a mapping/orientation document, not a new
layer.

> Note on numbering: the layer numbers below follow this repo's canonical
> 29-layer table (see `README.md` / `CLAUDE.md`). The model is the other
> substitution — this project reasons with **Claude (Anthropic)**; NemoClaw uses
> **NeMo / Nemotron** for local GPU inference.

## What NemoClaw Is

NemoClaw = **OpenShell** (the sandbox runtime) + **OpenClaw** (the agent) +
**NeMo inference** (the Nemotron model served locally). It bundles the three into
one managed, installable stack:

```sh
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

This project rebuilds that stack's *patterns* from scratch — the agent loop,
memory, observability, a skill ecosystem, and an actual OpenShell sandbox — using
Claude as the model instead of a local Nemotron.

## Architecture Comparison Table

| Layer (this repo) | What it is here | NemoClaw equivalent |
|---|---|---|
| 1 — ReAct agent + tools | ReAct loop, tool calling | OpenClaw agent + claws |
| 2 — Conversation memory | buffer replayed into the prompt | OpenClaw short-term memory |
| 3 — Interactive loop | multi-turn REPL / session | OpenClaw conversation/session loop |
| 4 — Streaming output | token-by-token streaming | OpenClaw streaming responses |
| 5 — Custom tool + hooks | domain tool + lifecycle callbacks | OpenClaw claws + lifecycle hooks |
| 6 — Production observability | LangFuse traces (OTEL-style) | NeMo Agent Toolkit tracing |
| 7 — CLI front door | Click CLI | OpenClaw gateway (CLI channel) |
| 8 — Separation of concerns + REST API | FastAPI HTTP front door | OpenClaw gateway — adds Telegram/Discord/Slack channels |
| 9 — Testing + evals | pytest + deepeval | NeMo Agent Toolkit evals |
| 10 — MCP integration | MCP client + server | OpenClaw MCP integration |
| 11 — Prompt management | `prompts/*.md` (+ LangFuse) | OpenClaw `SKILL.md` / managed prompt assets |
| 12 — Vector-store memory | semantic top-k retrieval | OpenClaw long-term memory |
| 13 — Multi-agent (LangGraph) | `StateGraph` pipeline | OpenClaw orchestrator |
| 14 — Context management + RAG | tiktoken budget + docs RAG | OpenClaw RAG integration |
| 15 — Strands + framework comparison | model-driven orchestration | *(no direct NemoClaw equivalent — a comparison study)* |
| 16 — Memory decay | tiered aging (full→summary→marker→archived) | OpenClaw memory lifecycle |
| 17 — Streaming the pipeline | multi-agent graph streaming | OpenClaw orchestrator streaming |
| 18 — Autonomous modes | cron + heartbeat | OpenClaw always-on daemon |
| 19 — Dead-letter queue | failure capture, classify, retry | OpenClaw failure handling *(explicit DLQ is an extra here)* |
| 20 — Skills (composed tools) | `research_and_summarize` | OpenClaw `.agents/skills/` |
| 21 — Run inside OpenShell sandbox | `sandbox-ask` under a policy | **OpenShell (identical runtime)** — OpenClaw runs inside OpenShell |
| 22 — Eval response caching | two-tier judge cache | **NOT in NemoClaw — built extra** |
| 23 — Skills as OpenClaw packages | `SKILL.md` + `SkillContext` + `ctx.call_tool` | OpenClaw skills (identical pattern) |
| 24 — Skill-level evals | per-skill output contracts | OpenClaw skill evals / NeMo Agent Toolkit |
| 25 — Skill versioning & deprecation | semver + sunset window | OpenClaw skill lifecycle |
| 26 — Skill composition | `call_skill` + depth guard | OpenClaw skill composition |
| 27 — Skill marketplace / remote loading | SHA256-pinned registry | OpenClaw skill marketplace |
| 28 — Skill dependency resolution | resolver, topological install order | OpenClaw skill resolver |
| 29 — Skill hot-reloading | `importlib.reload` on fingerprint change | OpenClaw live reload |
| *(model)* | Claude (Anthropic) | NeMo / Nemotron, served locally on GPU |

## Three Things NemoClaw Adds That This Project Doesn't Have

1. **Multi-channel front doors** — Telegram, Discord, Slack, WhatsApp built in
   (this project has CLI + REST + MCP).
2. **Local GPU inference** — Nemotron (e.g. 120B) served via Ollama, instead of a
   hosted Claude API.
3. **Managed onboarding + lifecycle** — an install wizard and managed lifecycle
   for the whole stack (here it's `uv` + manual scripts).

## Three Things This Project Has That NemoClaw Doesn't

1. **Eval response caching** — re-running unchanged LLM-as-judge prompts costs
   zero tokens (two-tier `lru_cache` over a SHA256-keyed JSON cache).
2. **A LangGraph quality-gated retry loop** — the reviewer node scores the answer
   and loops back to research below a threshold (a fixed, inspectable topology).
3. **A skill dependency resolver** — topological install order with version
   constraints and cycle detection, resolving across installed-core vs.
   marketplace sources.

## The Bridge Story

> I built a simplified version of the NemoClaw stack from scratch across 29
> layers. Starting from a bare ReAct loop, I added every production concern:
> memory with decay, observability via OTEL-style tracing, autonomous operation,
> a skill ecosystem with versioning / composition / marketplace / hot-reload, and
> finally ran the agent inside an actual OpenShell sandbox. The things NemoClaw
> adds that I'm simulating are multi-channel integration and local GPU inference
> (I reason with Claude instead of a local Nemotron). The architectural patterns
> are identical — I built them to understand them, not just install them.
