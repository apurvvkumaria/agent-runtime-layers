# Contributing

Thanks for your interest! This is a learning project that demonstrates agent-runtime
patterns (see the [README](README.md) for the eight-layer overview and
[CLAUDE.md](CLAUDE.md) for deeper architecture notes). Contributions that keep it small,
readable, and well-explained are very welcome.

## Prerequisites

- **Python 3.13+**
- **[uv](https://docs.astral.sh/uv/)** for dependency and environment management
- An **Anthropic API key** for the LLM-backed features (the direct-tool features work
  without one)

## Setup

```bash
git clone https://github.com/apurvvkumaria/agent-runtime-layers.git
cd agent-runtime-layers

uv sync                 # create the venv and install dependencies
cp .env.example .env    # then edit .env and add your keys
```

At minimum set `ANTHROPIC_API_KEY` in `.env`. The `LANGFUSE_*` keys are optional — leave
the placeholders and tracing is skipped (the agent falls back to print-based hooks).
`.env` is git-ignored; **never commit real secrets.** When you add a new environment
variable, add a placeholder line to `.env.example`.

## Running it

All commands run through `uv run` so they use the project venv.

```bash
uv run python agent.py --help              # list all CLI commands

# LLM-backed (need ANTHROPIC_API_KEY):
uv run python agent.py chat                # interactive REPL
uv run python agent.py ask "What is X?"    # single shot
uv run python agent.py research "topic" -o report.md

# Direct tools (no API key, no network):
uv run python agent.py calc "150 * 223.48"
uv run python agent.py metrics prod-us-east-1
uv run python agent.py history
```

### REST API

`serve` starts the FastAPI app via uvicorn — you do **not** need to run uvicorn yourself:

```bash
uv run python agent.py serve --port 8000   # leave running in one terminal
```

Then, from another terminal (or browse the interactive docs at `http://localhost:8000/docs`):

```bash
curl localhost:8000/health
curl "localhost:8000/calc?expr=150*223.48"
curl localhost:8000/metrics/prod-us-east-1
curl -X POST localhost:8000/ask  -H 'Content-Type: application/json' -d '{"question":"What is X?"}'
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"Hi","session_id":"s1"}'
```

For live-reload while editing, use `uv run uvicorn api:app --reload` instead of `serve`.

## Project layout

Modules have **one-way imports**: `tools` / `hooks` ← `core` ← `agent` / `api`. Keep it that
way — `core` must not import `agent` or `api`.

| File | Responsibility |
|---|---|
| `tools.py` | Tool definitions + `get_tools()` |
| `hooks.py` | Print hooks + LangFuse; `get_callbacks()` / `flush_traces()` |
| `core.py` | Framework-agnostic runtime: prompts, memory, agent builders, `stream_answer()` |
| `api.py` | FastAPI app (imports `core`) |
| `agent.py` | Click CLI (imports `core` + `api`) |

Common changes:

- **New tool:** add an `@tool` function in `tools.py` and include it in `get_tools()`. It's
  then available to the agent automatically; add a direct CLI command/endpoint only if you
  want no-LLM access to it.
- **New CLI command:** add a `@cli.command()` in `agent.py`.
- **New API endpoint:** add an `async def` route in `api.py`, reusing `core` helpers.

## Conventions

- **Use uv for everything** — `uv run ...`, `uv add <pkg>`, `uv remove <pkg>`. Don't edit
  dependency lists by hand or use `pip`.
- **Keep it minimal and readable.** This is a teaching artifact; favor clear, commented code
  over abstraction or premature optimization.
- **Match the surrounding style** — type hints, concise docstrings, and comments that explain
  *why*, not *what*.
- **No secrets in the repo.** Keep them in `.env`; add placeholders to `.env.example`.

## Verifying changes

There's no test suite yet, so verify manually:

```bash
# Syntax/import sanity for all modules:
uv run python -c "import tools, hooks, core, api, agent; print('imports OK')"

# Exercise the paths you touched (CLI commands and/or curl against `serve`).
```

When verifying agent behavior, prefer the fast no-LLM commands (`calc`, `metrics`) or a
short single-shot `ask` to keep API usage minimal.

## Submitting changes

1. Branch off `main`.
2. Make focused commits with clear messages.
3. Verify the affected commands/endpoints run.
4. Open a pull request describing what changed and how you tested it.
