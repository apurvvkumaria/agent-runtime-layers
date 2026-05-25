#!/usr/bin/env bash
#
# scripts/create-sandbox.sh — create an OpenShell sandbox the easy way.
#
# Encodes the "how do I make a sandbox" knowledge so you don't re-derive it:
#   - sandbox creation ALWAYS needs `--from <image>`. A bare `sandbox create`
#     (or `-- claude` with no --from) silently uses the `base` image, which has
#     no `claude`, and the CLI then hangs on "Requesting sandbox...".
#   - the Claude Code image is `openclaw` (a published image, not in the repo).
#   - "Created sandbox" != healthy: the supervisor can still crashloop (cert/JWT),
#     so the leave-running path below verifies the supervisor's success marker.
#
# Usage:
#   scripts/create-sandbox.sh                      # openclaw image, one-shot `claude`
#   scripts/create-sandbox.sh base                 # a community image, left running
#   scripts/create-sandbox.sh openclaw -- claude   # explicit image + command
#   scripts/create-sandbox.sh openclaw -- bash     # openclaw, but a shell instead
#
# Assumes the gateway is already up — run scripts/setup-openshell.sh first.

set -euo pipefail

SUCCESS_MARKER="OpenShell Sandbox Supervisor success"

if ! command -v openshell >/dev/null 2>&1; then
  echo "ERROR: openshell CLI not found in PATH." >&2
  exit 1
fi

# Gateway must be reachable, or `sandbox create` fails with an opaque error.
if ! openshell status >/dev/null 2>&1; then
  echo "ERROR: gateway not reachable. Run scripts/setup-openshell.sh first." >&2
  exit 1
fi

IMAGE="${1:-openclaw}"
shift || true   # drop the image arg; remaining "$@" is the optional `-- <cmd>`

# Interactive / one-shot command paths hand the terminal straight over (exec),
# so health shows up in the session itself. Only the create-and-leave-running
# path can — and should — be verified afterward.
if [ "$#" -gt 0 ]; then
  # Caller passed an explicit command (e.g. `-- claude` / `-- bash`); forward it.
  exec openshell sandbox create --from "$IMAGE" "$@"
elif [ "$IMAGE" = "openclaw" ]; then
  # openclaw with no command defaults to a one-shot Claude session.
  exec openshell sandbox create --from "$IMAGE" -- claude
fi

# --- Leave-running path: create, then verify the supervisor actually came up ---

out="$(openshell sandbox create --from "$IMAGE")"
printf '%s\n' "$out"

# Parse the sandbox name from "Created sandbox: <name>" (best effort).
name="$(printf '%s\n' "$out" \
  | sed -n 's/.*[Cc]reated sandbox:[[:space:]]*\([A-Za-z0-9_-]\{1,\}\).*/\1/p' | head -1)"

if [ -z "$name" ]; then
  echo "Note: couldn't parse the sandbox name — verify health manually:" >&2
  echo "  docker logs <openshell-...> --tail 30   # look for '$SUCCESS_MARKER'" >&2
  exit 0
fi

# The data-plane container is openshell-<name>-<uuid>; match by prefix.
cid="$(docker ps --filter "name=openshell-${name}-" --format '{{.Names}}' | head -1)"
if [ -z "$cid" ]; then
  echo "Note: '$name' created but no running container found — check 'openshell sandbox list'." >&2
  exit 0
fi

echo "Verifying supervisor health ($cid)..."
for _ in {1..15}; do
  if docker logs "$cid" 2>&1 | grep -q "$SUCCESS_MARKER"; then
    echo "✓ sandbox '$name' healthy.  Connect: openshell sandbox connect $name"
    exit 0
  fi
  sleep 1
done

echo "WARNING: never saw '$SUCCESS_MARKER' for '$cid' — supervisor may be crashlooping." >&2
echo "Recent logs:" >&2
docker logs "$cid" --tail 30 >&2
exit 1
