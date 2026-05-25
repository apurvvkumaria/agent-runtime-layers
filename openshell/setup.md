# Running the agent inside an OpenShell sandbox (macOS setup)

This documents what it took to get the OpenShell gateway running locally on
**macOS + Docker Desktop** so the agent can execute inside a policy-constrained
sandbox (Layer 21). The automated version of everything here lives in
`scripts/setup-openshell.sh`; this file explains the *why* behind the
non-obvious parts.

> OpenShell is alpha. The official Quick Start (plaintext, auto-bootstrap)
> does not actually work on macOS for anything past `openshell status` — see
> "Why auto-bootstrap fails" below. The full-mTLS path documented here is the
> only configuration where sandboxes actually run.

## 1. The Docker socket path issue on macOS

The gateway runs as a Docker container and talks to the Docker daemon to spawn
sandbox containers, so it needs the host Docker socket bind-mounted in. On
macOS the catch is:

- `/var/run/docker.sock` is a **symlink** to `$HOME/.docker/run/docker.sock`.
- Symlinks **do not resolve across the container boundary** — mounting the
  symlink path gives the gateway a dangling link.

Mount the **real** path:

```sh
-v "$HOME/.docker/run/docker.sock:/var/run/docker.sock"
```

Two related macOS-isms:

- The socket is `0660 root:root` inside the container and the gateway image's
  default user can't read it, so run the gateway container as `--user 0:0`
  (root-in-container — fine for local dev only).
- The gateway tells the Docker daemon to bind-mount paths *from inside its own
  container* into sandbox containers, and those paths must exist on the **host**
  (Docker Desktop validates mounts against File Sharing). Mount host paths to
  themselves (`$HOME/foo:$HOME/foo`) and set `HOME`/`XDG_DATA_HOME`/
  `XDG_STATE_HOME` so every path the gateway hands the daemon is a valid host
  path.

## 2. Manual gateway start (`docker run`)

`openshell gateway` subcommands only manage **registrations**, not lifecycle —
there is no `openshell gateway start`. You launch the gateway image yourself:

```sh
# (after generating TLS PKI with --server-san host.openshell.internal; see below)
docker run -d \
  --name openshell-gateway \
  --restart unless-stopped \
  --user 0:0 \
  -p 8080:8080 \
  -v "$HOME/.local/state/openshell:$HOME/.local/state/openshell" \
  -v "$HOME/.local/share/openshell:$HOME/.local/share/openshell" \
  -v "$HOME/.docker/run/docker.sock:/var/run/docker.sock" \
  -e HOME="$HOME" \
  -e XDG_DATA_HOME="$HOME/.local/share" \
  -e XDG_STATE_HOME="$HOME/.local/state" \
  -e OPENSHELL_DRIVERS=docker \
  -e OPENSHELL_DB_URL="sqlite:$HOME/.local/state/openshell/openshell.db" \
  -e OPENSHELL_LOCAL_TLS_DIR="$HOME/.local/state/openshell/tls" \
  -e OPENSHELL_TLS_CERT="$HOME/.local/state/openshell/tls/server/tls.crt" \
  -e OPENSHELL_TLS_KEY="$HOME/.local/state/openshell/tls/server/tls.key" \
  -e OPENSHELL_TLS_CLIENT_CA="$HOME/.local/state/openshell/tls/ca.crt" \
  -e OPENSHELL_ENABLE_MTLS_AUTH=true \
  ghcr.io/nvidia/openshell/gateway:latest
```

Two things worth calling out:

- **`-p 8080:8080`, not `-p 127.0.0.1:8080:8080`.** Sandboxes are *sibling*
  containers and reach the gateway through Docker Desktop's host bridge via
  `host.openshell.internal`; a loopback-only publish is invisible to them.
- **The TLS server cert must include the SAN `host.openshell.internal`.**
  That's the hostname the Docker driver injects into sandboxes as the gateway
  endpoint. `generate-certs` defaults to a Kubernetes-shaped SAN list that
  omits it, so without `--server-san host.openshell.internal` the
  sandbox→gateway handshake fails with a misleading "failed to connect". Run:

  ```sh
  docker run --rm --user 0:0 \
    -v "$HOME/.local/state/openshell:$HOME/.local/state/openshell" \
    -e HOME="$HOME" \
    ghcr.io/nvidia/openshell/gateway:latest \
    generate-certs \
      --output-dir "$HOME/.local/state/openshell/tls" \
      --server-san host.openshell.internal
  ```

## 3. Registering the gateway with the CLI

`gateway add` records the endpoint and the client mTLS bundle the CLI presents
on every call:

```sh
openshell gateway add "https://127.0.0.1:8080" --local --name local
openshell status   # smoke test — should report the gateway healthy
```

Regenerating the PKI invalidates the CLI's client bundle, so re-register
(`gateway remove local` then `gateway add ...`) whenever certs change.

## 4. Why auto-bootstrap fails on macOS (known alpha issue)

The docs' Quick Start implies the CLI can bootstrap a gateway for you over a
plaintext connection. On macOS this dead-ends:

- **Plaintext + working sandboxes is not a supported combination.** Creating a
  sandbox requires the sandbox supervisor to authenticate back to the gateway
  with a **JWT**. The gateway only mints/accepts those JWTs when TLS material is
  configured, which transitively requires the **CLI** to authenticate via
  **mTLS**. So "plaintext is fine for local dev" holds only until the first
  `sandbox create`, which then hangs/—rejects with an opaque error.
- The auto-bootstrap path also assumes it can find/launch the gateway and reach
  the Docker socket without the macOS-specific socket and `--user` handling
  above, so it can't bring a working gateway up on Docker Desktop.

Net: skip auto-bootstrap, start the gateway manually with full mTLS (the
`scripts/setup-openshell.sh` flow), and the CLI ↔ gateway ↔ sandbox round trip
works.

## 5. Running the agent in a sandbox

With the gateway up:

```sh
uv run python agent.py sandbox-info                 # gateway + sandboxes + active policy
uv run python agent.py sandbox-ask "What is 2 + 2?" # run `ask` inside a sandbox, then delete it
```

`sandbox-ask` does the whole round trip in **one** `openshell sandbox create`
call — the CLI is built for it:

```sh
openshell sandbox create --from <image> --name <n> --cpu 2 --memory 1Gi \
  --policy openshell/policy.yaml \
  --upload <staged-workspace>:/sandbox \
  --no-tty --no-keep \
  -- sh -lc 'cd /sandbox/workspace && export TIKTOKEN_CACHE_DIR=/sandbox/.tiktoken && exec python agent.py ask "…"'
```

`--upload` stages files in, `-- <cmd>` runs the agent, `--no-keep` deletes the
sandbox when the command exits (`--keep` to leave it running). Because `--upload`
can only be given once and nests by basename (`<dir>:/sandbox` → `/sandbox/<dir-name>`),
`sandbox_runner` stages the source **plus** a generated `.env` into one temp
`workspace/` dir and uploads that. The repo's own `.env` is git-ignored, so the
API key has to be injected deliberately.

### The real policy schema, and the gotchas that took live debugging

`openshell/policy.yaml` uses the **actual** v0.0.47 schema (`version`,
`filesystem_policy`, `landlock`, `process`, `network_policies`) — a naive
`sandbox: {resources, network: {allow/deny}, filesystem: {writable/denied}}`
shape is **rejected** by the gateway (`unknown field 'sandbox'`). The hard-won
details, all confirmed against the gateway's own logs:

- **Egress is binary-keyed and default-deny.** A `network_policies` entry binds
  *endpoints* (host/port) to the specific *binaries* allowed to reach them
  (identity via the fully-resolved `/proc/{pid}/exe`, SHA256 trust-on-first-use).
  No `deny: '*'` — anything unlisted is denied.
- **Match the binary on its resolved path.** `/sandbox/.venv/bin/python` is a
  symlink chain ending at uv's patch-versioned interpreter
  (`/sandbox/.uv/python/cpython-3.13.12-…/bin/python3.13`), and OPA matches the
  resolved path — so the policy allowlists the glob `/sandbox/.uv/python/**`.
- **`tls: skip`, not `terminate`.** Automatic TLS termination MITMs the
  connection with the gateway's cert, which the Anthropic Python SDK
  (httpx + certifi) won't trust → cert failure. `skip` tunnels to the real cert.
- **Resources aren't policy fields** — `cpu`/`memory` are `create` flags
  (`--cpu 2 --memory 1Gi`).
- **`exec`/the run command get a sanitized env** that drops the image's
  Dockerfile `ENV`, so anything the agent needs (e.g. `TIKTOKEN_CACHE_DIR`) must
  be exported in the run command.

**Verified end-to-end:** `agent sandbox-ask "What is 2 + 2?"` runs the agent
inside the sandbox under this policy and returns `2 + 2 = 4`, with the sandbox
auto-deleted afterward — sole allowed egress being `api.anthropic.com`.

**Prerequisite — the sandbox image must already contain the project's Python
deps.** The policy allows egress only to Anthropic + DuckDuckGo, so the sandbox
*cannot* reach PyPI to install anything at run time. The deps-baked image for
this is `openshell/agent-sandbox/` — build it and point `sandbox-ask` at it:

```sh
docker build -f openshell/agent-sandbox/Dockerfile -t agent-sandbox:latest .
uv run python agent.py sandbox-ask --image agent-sandbox:latest "What is 2 + 2?"
```

See `openshell/agent-sandbox/README.md` for the details. Override the in-sandbox
interpreter with `OPENSHELL_SANDBOX_PYTHON` (default `python`, which resolves to
`/sandbox/.venv/bin/python`). The Anthropic key is passed in by writing a `.env`
into the workspace, which the agent's `load_dotenv()` picks up.
