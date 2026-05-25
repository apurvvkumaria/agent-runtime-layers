#!/usr/bin/env bash
# scripts/teardown-openshell.sh — full reset of local OpenShell state.

set -euo pipefail

# Remove all sandboxes through the CLI first (best effort)
openshell sandbox list 2>/dev/null | awk 'NR>1 {print $1}' | while read -r name; do
  [ -n "$name" ] && openshell sandbox delete "$name" 2>/dev/null || true
done

# Stop and remove the gateway container
docker rm -f openshell-gateway >/dev/null 2>&1 || true

# Stop and remove any orphaned sandbox containers
docker ps -a --filter "name=openshell-" --format "{{.Names}}" 2>/dev/null | while read -r name; do
  [ -n "$name" ] && docker rm -f "$name" >/dev/null 2>&1 || true
done

# Remove CLI registrations
openshell gateway remove local >/dev/null 2>&1 || true
openshell gateway remove openshell >/dev/null 2>&1 || true

# Remove host state
rm -rf "$HOME/.local/state/openshell" \
       "$HOME/.local/share/openshell" \
       "$HOME/.config/openshell"

echo "OpenShell local state removed."
