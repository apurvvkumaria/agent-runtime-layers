"""Run the agent inside an OpenShell sandbox (Layer 21).

Orchestrates the `openshell` CLI + Docker to execute `agent.py ask` inside a
policy-constrained sandbox container instead of on the host. This is sandboxed
agent execution: the agent's reasoning and tool calls run under the network /
filesystem / resource limits declared in `openshell/policy.yaml`, enforced by
the sandbox supervisor (Landlock + the gateway's egress rules).

We drive the `openshell` CLI *binary* via subprocess — nothing here does
`import openshell`. That's deliberate: the sibling `openshell/` directory
(policy + docs, no `__init__.py`) would otherwise shadow the real `openshell`
PyPI SDK as a namespace package, the same trap that put the MCP code in
`mcp_integration/`. Driving the CLI sidesteps it entirely.

Prerequisites (see openshell/setup.md):
  - a running gateway (scripts/setup-openshell.sh)
  - a sandbox image with the project's Python deps PRE-INSTALLED — the policy
    denies PyPI egress, so deps can't be installed at run time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
POLICY_PATH = ROOT / "openshell" / "policy.yaml"
WORKSPACE = "/sandbox/workspace"  # matches the writable path in policy.yaml
SUCCESS_MARKER = "OpenShell Sandbox Supervisor success"

# Project files/dirs the agent needs to answer a question. Missing ones are
# skipped, so this stays correct as the tree evolves. (We avoid uploading
# .venv/.git/chroma_db — the sandbox image supplies the installed deps.)
_UPLOAD = [
    "agent.py", "core.py", "tools.py", "hooks.py", "api.py",
    "prompts", "skills", "mcp_integration", "memory", "context",
    "dlq", "autonomy", "langgraph_agents", "strands_agent", "docs",
    "pyproject.toml", "uv.lock",
]


class SandboxError(RuntimeError):
    """The sandbox workflow can't proceed (missing CLI/Docker, gateway down, etc.)."""


def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        raise SandboxError(f"`{binary}` not found in PATH.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout/stderr as text (never raises on nonzero)."""
    return subprocess.run(cmd, text=True, capture_output=True)


def _logs(container: str, tail: int | None = None) -> str:
    """`docker logs` with stderr folded into stdout (the supervisor logs to both)."""
    cmd = ["docker", "logs"]
    if tail is not None:
        cmd += ["--tail", str(tail)]
    cmd.append(container)
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


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


def _create_sandbox(name: str, image: str) -> None:
    """Create a sandbox with the given name from `image`, applying the policy.

    We pass `--name` ourselves so the name is deterministic (no parsing the
    "Created sandbox:" line). The sandbox is left running so we can upload the
    source and exec into it; run_in_sandbox deletes it afterward.
    """
    # Resources are create-time flags, not policy fields (the policy schema has
    # no resources block) — these encode the intended 2 cores / 1 GiB limit.
    cmd = ["openshell", "sandbox", "create", "--from", image, "--name", name,
           "--cpu", "2", "--memory", "1Gi"]
    if POLICY_PATH.exists():
        cmd += ["--policy", str(POLICY_PATH)]
    p = _run(cmd)
    if p.returncode != 0:
        raise SandboxError(f"`sandbox create` failed: {p.stderr.strip() or p.stdout.strip()}")


def _container_for(name: str) -> str | None:
    """The data-plane container is openshell-<name>-<uuid>; match by prefix."""
    p = _run(["docker", "ps", "--filter", f"name=openshell-{name}-", "--format", "{{.Names}}"])
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    return lines[0] if lines else None


def _wait_healthy(container: str, tries: int = 15) -> bool:
    """Poll the supervisor logs — "Created" != healthy (it can crashloop on cert/JWT)."""
    for _ in range(tries):
        if SUCCESS_MARKER in _logs(container):
            return True
        time.sleep(1)
    return False


def _provision_env(container: str) -> None:
    """Drop a .env with the Anthropic key into the workspace; load_dotenv() reads it."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SandboxError("ANTHROPIC_API_KEY is not set on the host; the sandboxed agent needs it.")
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write(f"ANTHROPIC_API_KEY={key}\n")
        tmp = f.name
    try:
        p = _run(["docker", "cp", tmp, f"{container}:{WORKSPACE}/.env"])
        if p.returncode != 0:
            raise SandboxError(f"failed to provision .env: {p.stderr.strip()}")
    finally:
        os.unlink(tmp)


def _upload_source(container: str) -> None:
    """Copy the project source into the sandbox's workspace via the Docker control plane.

    Note: `docker cp`/`docker exec` from the host are NOT subject to the sandbox
    policy — Landlock constrains the sandbox's own process tree (the agent run),
    not host-initiated operations on the container. So we provision inputs
    out-of-band here; the thing the policy actually governs is the agent process
    we later start through `openshell sandbox exec` (gRPC → supervisor).
    """
    _run(["docker", "exec", container, "mkdir", "-p", WORKSPACE])  # best effort
    for rel in _UPLOAD:
        src = ROOT / rel
        if not src.exists():
            continue
        p = _run(["docker", "cp", str(src), f"{container}:{WORKSPACE}/"])
        if p.returncode != 0:
            raise SandboxError(f"upload of {rel} failed: {p.stderr.strip()}")


def _exec_ask(name: str, question: str) -> str:
    """Run `agent.py ask` inside the sandbox via the CLI, so the policy is enforced.

    `exec` auto-disables the PTY when stdout isn't a terminal (we capture it),
    which keeps the streamed answer from being garbled by ANSI control codes.
    """
    cmd = [
        "openshell", "sandbox", "exec", "-n", name, "--workdir", WORKSPACE,
        "--", _sandbox_python(), "agent.py", "ask", question,
    ]
    p = _run(cmd)
    if p.returncode != 0:
        raise SandboxError(f"sandboxed run failed:\n{p.stderr.strip() or p.stdout.strip()}")
    return p.stdout.strip()


def run_in_sandbox(question: str, keep: bool = False, image: str | None = None) -> str:
    """Create a sandbox, run `ask` inside it under the policy, and return the answer.

    Deletes the sandbox afterward unless `keep` is True. Raises SandboxError if
    any stage fails (missing tooling, gateway down, unhealthy supervisor, etc.).
    """
    _require("openshell")
    _require("docker")
    gateway_status()  # fail fast if the gateway isn't up

    image = _sandbox_image(image)
    name = f"agent-ask-{uuid.uuid4().hex[:8]}"
    print("🔒 Running inside OpenShell sandbox...")
    print(f"   image={image}  policy={'openshell/policy.yaml' if POLICY_PATH.exists() else '(none)'}")

    _create_sandbox(name, image)
    print(f"   sandbox={name}")
    try:
        container = _container_for(name)
        if not container:
            raise SandboxError(f"no running container found for sandbox '{name}'.")
        if not _wait_healthy(container):
            raise SandboxError(
                f"sandbox '{name}' never reported the supervisor success marker:\n"
                f"{_logs(container, tail=30)}"
            )
        _provision_env(container)
        _upload_source(container)
        print("   executing: agent.py ask (policy-constrained)\n")
        return _exec_ask(name, question)
    finally:
        if keep:
            print(f"\n   kept sandbox '{name}' (delete with: openshell sandbox delete {name})")
        else:
            _run(["openshell", "sandbox", "delete", name])
            print(f"\n   deleted sandbox '{name}'")
