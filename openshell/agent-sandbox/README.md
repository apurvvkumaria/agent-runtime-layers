# `agent-sandbox` — deps-baked sandbox image

The image that lets `agent sandbox-ask` actually run the agent end-to-end inside
an OpenShell sandbox. It's the community `base` sandbox plus the project's locked
Python dependencies, installed into `/sandbox/.venv`.

## Why this exists

`openshell/policy.yaml` allows egress only to `api.anthropic.com` + DuckDuckGo,
so the running sandbox **cannot reach PyPI**. The agent's deps therefore can't be
installed at run time — they must be baked into the image at build time (the host
Docker build has full network; the sandbox does not).

## Build

Build from the **repo root** so the Dockerfile's `COPY` can reach
`pyproject.toml` / `uv.lock`:

```sh
docker build -f openshell/agent-sandbox/Dockerfile -t agent-sandbox:latest .
```

This pulls the full stack (langchain, chromadb, sentence-transformers → torch,
tiktoken, mcp, …) via `uv sync --frozen`, so the first build is large and takes
several minutes. On Apple Silicon the build is arm64 — matching the project's
required native-arm64 toolchain.

## Run

```sh
uv run python agent.py sandbox-ask --image agent-sandbox:latest "What is 2 + 2?"
# or set it once:
export OPENSHELL_SANDBOX_IMAGE=agent-sandbox:latest
uv run python agent.py sandbox-ask "What is 2 + 2?"
```

For `What is 2 + 2?` the agent only uses the calculator + the Anthropic API, so
the run stays within the policy: the sole egress is `api.anthropic.com`.

## Notes / caveats

- **Local image resolution.** `sandbox-ask` passes `--from agent-sandbox:latest`
  to `openshell sandbox create`. OpenShell treats that as a full container image
  reference and should use the local Docker image. If it instead tries to pull
  from a registry, retag/point at a reference it resolves, or build the image
  under a name it recognizes.
- **`uv sync` and the existing venv.** The base image already created
  `/sandbox/.venv` (via `uv venv --seed`). `uv sync` with
  `UV_PROJECT_ENVIRONMENT=/sandbox/.venv` installs the locked deps into it. If a
  uv version balks at the pre-existing venv, delete it first in the Dockerfile
  (`RUN rm -rf /sandbox/.venv`) and let `uv sync` recreate it at that path.
- **The `filesystem` tool needs npm at run time.** It's only invoked for doc
  reads (not for `What is 2 + 2?`), and the MCP server is fetched via `npx` —
  which the policy blocks. To exercise that tool in-sandbox, pre-install
  `@modelcontextprotocol/server-filesystem` globally in the image and/or add its
  registry host to the policy.
- **Slimmer image (optional).** This bakes the *entire* dependency set for
  correctness. A run that only needs `ask` could install a minimal subset
  (langchain + langchain-anthropic/-community/-classic, ddgs, tiktoken, mcp,
  click, python-dotenv) for a much smaller, faster image — at the cost of
  breaking if an import path pulls something heavier.
