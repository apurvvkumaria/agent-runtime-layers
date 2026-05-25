# syntax=docker/dockerfile:1
#
# Deps-baked sandbox image for running the agent inside OpenShell (Layer 21).
#
# Builds ON TOP OF the community `base` sandbox (which already has uv-managed
# Python 3.13, Node 22, and a writable seeded venv at /sandbox/.venv that is
# first on PATH and owned by the `sandbox` user) and installs the project's
# locked Python dependencies into that venv.
#
# Why /sandbox/.venv specifically: it's exactly the interpreter our policy
# (openshell/policy.yaml) allowlists for egress, and what `python` resolves to
# inside the sandbox — so `python agent.py ask` runs with the deps present AND
# under the policy's network rules, no extra wiring.
#
# The egress policy denies PyPI at *run* time, so deps MUST be baked in here at
# *build* time (host Docker build has full network; the running sandbox does not).
#
# Build (from the REPO ROOT, so the COPY can reach pyproject.toml/uv.lock):
#   docker build -f openshell/agent-sandbox/Dockerfile -t agent-sandbox:latest .
#
# Run:
#   uv run python agent.py sandbox-ask --image agent-sandbox:latest "What is 2 + 2?"

FROM ghcr.io/nvidia/openshell-community/sandboxes/base:latest

# Target the base image's existing writable venv, and copy (don't hardlink)
# package files so the install works across the layer's filesystem boundary.
ENV UV_PROJECT_ENVIRONMENT=/sandbox/.venv \
    UV_LINK_MODE=copy

# Only the dependency manifests, so this layer caches unless the deps change.
COPY --chown=sandbox:sandbox pyproject.toml uv.lock /sandbox/agent-build/

USER sandbox
WORKDIR /sandbox/agent-build

# --frozen: honor uv.lock exactly (reproducible; no resolution at build time).
# --no-install-project: the agent isn't a package — we only want its deps in the
#   venv; the source itself is uploaded into /sandbox/workspace at run time.
# This pulls the full stack (langchain, chromadb, sentence-transformers→torch,
# tiktoken, mcp, …), so it's a large, multi-minute build on first run.
RUN uv sync --frozen --no-install-project \
    && uv cache clean

# Pre-cache the tiktoken encoding the context manager uses (cl100k_base).
# tiktoken fetches it from openaipublic.blob.core.windows.net on first use, and
# the run-time egress policy denies that host — so cache it now (build has
# network) and point TIKTOKEN_CACHE_DIR at it. Separate layer so the heavy
# uv-sync layer above stays cached when only this changes.
ENV TIKTOKEN_CACHE_DIR=/sandbox/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

WORKDIR /sandbox
ENTRYPOINT ["/bin/bash"]
