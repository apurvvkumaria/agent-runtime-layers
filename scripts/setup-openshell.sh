#!/usr/bin/env bash
#
# OpenShell local gateway setup for macOS + Docker Desktop.
#
# Brings up the OpenShell gateway as a Docker container with full mTLS,
# generates the PKI bundle, registers the gateway with the CLI, and
# verifies the round trip with a test sandbox.
#
# Idempotent — safe to re-run; will recreate the gateway container.
#
# Prereqs:
#   - Docker Desktop running (verify: `docker version` shows both Client + Server)
#   - openshell CLI installed
#   - For `sandbox create -- claude`: Claude Code installed and logged in
#     locally, so OpenShell can bootstrap the claude-code provider from creds.

set -euo pipefail

# --- Config ---------------------------------------------------------------

CONTAINER_NAME="openshell-gateway"
GATEWAY_PORT="8080"
GATEWAY_NAME="local"
GATEWAY_IMAGE="ghcr.io/nvidia/openshell/gateway:latest"

STATE_DIR="$HOME/.local/state/openshell"
SHARE_DIR="$HOME/.local/share/openshell"
CONFIG_DIR="$HOME/.config/openshell"
TLS_DIR="$STATE_DIR/tls"
DOCKER_SOCK="$HOME/.docker/run/docker.sock"  # Docker Desktop's real socket on Mac

# --- Sanity checks --------------------------------------------------------

if ! docker version >/dev/null 2>&1; then
  echo "ERROR: Docker daemon not reachable. Start Docker Desktop and retry." >&2
  exit 1
fi

if [ ! -S "$DOCKER_SOCK" ]; then
  echo "ERROR: Docker socket not found at $DOCKER_SOCK." >&2
  echo "On macOS the real socket lives here; /var/run/docker.sock is a symlink" >&2
  echo "that does not traverse the container boundary." >&2
  exit 1
fi

if ! command -v openshell >/dev/null 2>&1; then
  echo "ERROR: openshell CLI not found in PATH." >&2
  exit 1
fi

# --- Prep host directories ------------------------------------------------

mkdir -p "$STATE_DIR" "$SHARE_DIR" "$CONFIG_DIR"

# --- Remove any prior container ------------------------------------------

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

# --- Generate TLS PKI -----------------------------------------------------
#
# Critical SAN: host.openshell.internal
#   The gateway advertises itself to sandboxes at https://host.openshell.internal:8080.
#   The default generate-certs SAN list (tuned for Kubernetes) omits this name,
#   so without --server-san the TLS handshake from sandbox → gateway fails with
#   a generic "failed to connect" error.
#
# Only regenerate if certs don't already exist, so re-runs don't invalidate
# the CLI's client bundle and force a re-registration.

if [ ! -f "$TLS_DIR/server/tls.crt" ]; then
  echo "Generating TLS PKI (one-time setup)..."
  mkdir -p "$TLS_DIR"
  docker run --rm \
    --user 0:0 \
    -v "$STATE_DIR:$STATE_DIR" \
    -v "$CONFIG_DIR:$CONFIG_DIR" \
    -e HOME="$HOME" \
    "$GATEWAY_IMAGE" \
    generate-certs \
      --output-dir "$TLS_DIR" \
      --server-san host.openshell.internal
  echo "TLS PKI written to $TLS_DIR"
else
  echo "TLS PKI already present at $TLS_DIR (skipping regeneration)"
fi

# --- Start the gateway ----------------------------------------------------
#
# Key Mac/Docker-Desktop specifics:
#
#   --user 0:0
#     The gateway image's default user can't read the host Docker socket
#     (0660 root:root inside the container). Running as root inside the
#     container is the simplest fix. Fine for local dev only.
#
#   Mount $HOME/.docker/run/docker.sock
#     /var/run/docker.sock on macOS is a symlink to this path. Symlinks
#     don't resolve across the container boundary — mount the real path.
#
#   -p 8080:8080 (not -p 127.0.0.1:8080:8080)
#     Sandboxes are sibling containers on Docker's bridge network. They
#     reach the gateway via host.openshell.internal, which routes through
#     Docker Desktop's host bridge. Loopback-only binding makes the gateway
#     unreachable from sibling containers.
#
#   Symmetric host:host bind-mount paths + XDG_* env vars
#     The gateway tells the Docker daemon which paths to mount into sandbox
#     containers. Those paths must exist on the host (Docker Desktop
#     validates mounts against File Sharing). Mounting host:host (not
#     host:/root/...) and setting HOME/XDG_DATA_HOME/XDG_STATE_HOME keeps
#     the path string identical in both namespaces, so anything the gateway
#     hands to Docker is a valid host path.
#
# Auth posture:
#   TLS enabled (server cert, key, client CA)
#   mTLS user auth enabled (CLI presents a client cert)
#   Sandbox JWT minting enabled (auto-on when TLS dir present)
#   This is a coherent, fail-closed auth model. Plaintext + JWT is NOT
#   coherent — the gateway rejects requests when there's no CLI auth
#   mechanism, even though the JWT issuer is unrelated.

echo "Starting gateway..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --user 0:0 \
  -p "${GATEWAY_PORT}:8080" \
  -v "$STATE_DIR:$STATE_DIR" \
  -v "$SHARE_DIR:$SHARE_DIR" \
  -v "$DOCKER_SOCK:/var/run/docker.sock" \
  -e HOME="$HOME" \
  -e XDG_DATA_HOME="$HOME/.local/share" \
  -e XDG_STATE_HOME="$HOME/.local/state" \
  -e OPENSHELL_DRIVERS=docker \
  -e OPENSHELL_DB_URL="sqlite:$STATE_DIR/openshell.db" \
  -e OPENSHELL_LOCAL_TLS_DIR="$TLS_DIR" \
  -e OPENSHELL_TLS_CERT="$TLS_DIR/server/tls.crt" \
  -e OPENSHELL_TLS_KEY="$TLS_DIR/server/tls.key" \
  -e OPENSHELL_TLS_CLIENT_CA="$TLS_DIR/ca.crt" \
  -e OPENSHELL_ENABLE_MTLS_AUTH=true \
  "$GATEWAY_IMAGE" \
  >/dev/null

# --- Wait for readiness ---------------------------------------------------

echo "Waiting for gateway to be ready..."
for _ in {1..20}; do
  if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Server listening"; then
    break
  fi
  sleep 1
done

if ! docker logs "$CONTAINER_NAME" 2>&1 | grep -q "Server listening"; then
  echo "ERROR: Gateway did not become ready. Recent logs:" >&2
  docker logs "$CONTAINER_NAME" --tail 30 >&2
  exit 1
fi

# --- Register gateway with CLI -------------------------------------------
#
# Re-register on every run because cert regeneration would otherwise leave
# the CLI with a stale client bundle. If the gateway was already registered,
# `remove` is a no-op via the `|| true`.

openshell gateway remove "$GATEWAY_NAME" >/dev/null 2>&1 || true
openshell gateway add "https://127.0.0.1:${GATEWAY_PORT}" --local --name "$GATEWAY_NAME"

# --- Smoke test ----------------------------------------------------------

openshell status

echo
echo "✓ Gateway ready."
echo "  Try: openshell sandbox create"
echo "  Or:  openshell sandbox create -- claude   # if Claude Code is on PATH inside the sandbox image"
