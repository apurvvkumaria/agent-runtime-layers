"""Run the agent inside an OpenShell sandbox (Layer 21).

Drives the `openshell` CLI to execute `agent.py ask` inside a policy-constrained
sandbox instead of on the host. The agent's reasoning + tool calls run under the
network / filesystem / resource limits in `openshell/policy.yaml`, enforced by
the sandbox supervisor (Landlock + the gateway's egress rules).

The whole round trip is ONE `openshell sandbox create` call — the CLI is built
for it: `--upload` stages files in, `-- <cmd>` runs the agent, `--no-keep`
deletes the sandbox after the command exits. We stage the source + a generated
`.env` (the API key + offline flags) into a temp `workspace/` dir and upload
that, because `--upload` can only be given once and the repo's own `.env` is
git-ignored (so it's filtered out of a plain source upload — good for secrets,
but then the key has to be injected deliberately).

We use the `openshell` CLI *binary* (subprocess), never `import openshell` — so
the sibling `openshell/` directory (policy + docs, no `__init__.py`) can't shadow
the real `openshell` SDK as a namespace package (the trap that put MCP code in
`mcp_integration/`).

Prerequisites (see openshell/setup.md): a running gateway, and a sandbox image
with the project's Python deps PRE-INSTALLED (openshell/agent-sandbox/) — the
policy denies PyPI egress, so deps can't be installed at run time.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "openshell" / "policy.yaml"
WORKSPACE = "/sandbox/workspace"  # matches the writable path in policy.yaml
# Where the agent-sandbox image bakes the tiktoken encoding. `openshell exec`
# runs with a sanitized env that drops the image's Dockerfile ENV, so we must
# re-export this in the run command or tiktoken tries to download (denied).
TIKTOKEN_CACHE_DIR = "/sandbox/.tiktoken"

# Project files/dirs the agent needs to answer a question. Missing ones are
# skipped, so this stays correct as the tree evolves. (.venv/.git/chroma_db are
# never staged — the image supplies the installed deps.)
_UPLOAD = [
    "agent.py", "core.py", "tools.py", "hooks.py", "api.py",
    "prompts", "skills", "mcp_integration", "memory", "context",
    "dlq", "autonomy", "langgraph_agents", "strands_agent", "docs",
    "pyproject.toml", "uv.lock",
]


class SandboxError(RuntimeError):
    """The sandbox workflow can't proceed (missing CLI, gateway down, run failed)."""


def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        raise SandboxError(f"`{binary}` not found in PATH.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout/stderr as text (never raises on nonzero)."""
    return subprocess.run(cmd, text=True, capture_output=True)


# --- Read-only queries (sandbox-info) ------------------------------------

def gateway_status() -> str:
    _require("openshell")
    p = _run(["openshell", "status"])
    if p.returncode != 0:
        raise SandboxError(p.stderr.strip() or "gateway not reachable (run scripts/setup-openshell.sh).")
    return p.stdout.strip()


def list_sandboxes() -> str:
    _require("openshell")
    p = _run(["openshell", "sandbox", "list"])
    return (p.stdout or p.stderr).strip() or "(none running)"


def policy_text() -> str:
    if not POLICY_PATH.exists():
        return "(no policy file at openshell/policy.yaml)"
    return POLICY_PATH.read_text(encoding="utf-8").strip()


# --- The run-in-sandbox workflow (sandbox-ask) ---------------------------

def _sandbox_image(image: str | None) -> str:
    return image or os.environ.get("OPENSHELL_SANDBOX_IMAGE", "agent-sandbox")


def _sandbox_python() -> str:
    return os.environ.get("OPENSHELL_SANDBOX_PYTHON", "python")


def _build_staging() -> Path:
    """Stage source + a generated .env into a temp `workspace/` dir, return its path.

    Uploading `<dir>:/sandbox` lands `<dir>`'s contents at `/sandbox/<dir-name>`,
    so the dir is named `workspace` to produce `/sandbox/workspace/...`.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SandboxError("ANTHROPIC_API_KEY is not set on the host; the sandboxed agent needs it.")

    tmp = Path(tempfile.mkdtemp(prefix="osb-"))
    stage = tmp / "workspace"
    stage.mkdir()
    for rel in _UPLOAD:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = stage / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.json"))
        else:
            shutil.copy2(src, dst)

    # The key + flags that keep a touched chromadb/transformers import from
    # stalling on egress the policy denies (PostHog telemetry / HuggingFace).
    (stage / ".env").write_text(
        f"ANTHROPIC_API_KEY={key}\n"
        "ANONYMIZED_TELEMETRY=False\n"
        "HF_HUB_OFFLINE=1\n"
        "TRANSFORMERS_OFFLINE=1\n",
        encoding="utf-8",
    )
    return stage


def run_in_sandbox(question: str, keep: bool = False, image: str | None = None) -> str:
    """Create a sandbox, run `ask` inside it under the policy, and return the answer.

    One `openshell sandbox create` call does it all: upload the staged source,
    run the agent, and (unless `keep`) delete the sandbox when the command exits.
    Raises SandboxError if the gateway is down or the run fails.
    """
    _require("openshell")
    gateway_status()  # fail fast if the gateway isn't up

    image = _sandbox_image(image)
    name = f"agent-ask-{uuid.uuid4().hex[:8]}"
    print("🔒 Running inside OpenShell sandbox...")
    print(f"   image={image}  policy={'openshell/policy.yaml' if POLICY_PATH.exists() else '(none)'}  sandbox={name}")
    print("   creating sandbox, uploading source, running `agent.py ask` (policy-constrained)...\n")

    stage = _build_staging()
    # `cd` into the workspace so the agent's relative paths + load_dotenv(.env)
    # work, and export TIKTOKEN_CACHE_DIR (the image ENV doesn't survive exec).
    inner = (
        f"cd {WORKSPACE} && export TIKTOKEN_CACHE_DIR={TIKTOKEN_CACHE_DIR} && "
        f"exec {_sandbox_python()} agent.py ask {shlex.quote(question)}"
    )
    cmd = [
        "openshell", "sandbox", "create",
        "--from", image, "--name", name,
        "--cpu", "2", "--memory", "1Gi",
        "--policy", str(POLICY_PATH),
        "--upload", f"{stage}:/sandbox",   # -> /sandbox/workspace/...
        "--no-tty",
    ]
    if not keep:
        cmd.append("--no-keep")
    cmd += ["--", "sh", "-lc", inner]

    try:
        p = _run(cmd)
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)

    if p.returncode != 0:
        raise SandboxError(f"sandboxed run failed:\n{(p.stderr or p.stdout).strip()}")
    if keep:
        print(f"   kept sandbox '{name}' (delete with: openshell sandbox delete {name})\n")
    return p.stdout.strip()
