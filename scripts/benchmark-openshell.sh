#!/usr/bin/env bash
#
# scripts/benchmark-openshell.sh — measure the isolation overhead of running the
# agent inside an OpenShell sandbox vs. on the host, and decompose where the cost
# goes (one-time provisioning vs. per-request policy/proxy enforcement).
#
# Produces the numbers in the README's "Isolation overhead (measured)" table:
#   - host `agent ask`                 — agent on bare host
#   - OpenShell `sandbox-ask` e2e      — create + upload + run + teardown
#   - warm in-sandbox exec             — repeated exec into one kept sandbox
#   - provisioning   ≈ e2e - warm      — one-time container/upload/boot/teardown
#   - per-request    ≈ warm - host     — recurring Landlock + egress-proxy cost
#   - per-LLM-call latency, direct vs. via the egress proxy (from StepLogger ts)
#
# Caveats baked into how you read it: small N; wall time is import- and LLM-
# dominated; and host runs via `uv run` while in-sandbox runs `python` directly,
# so the clean same-method signal is the per-LLM-call latency, not warm-vs-host.
#
# Prereqs: gateway up (scripts/setup-openshell.sh), the deps-baked image built
# (docker build -f openshell/agent-sandbox/Dockerfile -t agent-sandbox:latest .),
# and ANTHROPIC_API_KEY in .env. NOTE: makes real (billed) LLM calls and creates
# and deletes sandboxes.
#
# Usage: scripts/benchmark-openshell.sh ["question"] [host_runs] [e2e_runs] [warm_runs]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

Q="${1:-What is 2 + 2?}"
HOST_N="${2:-5}"
E2E_N="${3:-3}"
WARM_N="${4:-3}"
IMG="${OPENSHELL_SANDBOX_IMAGE:-agent-sandbox:latest}"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

now() { python3 -c 'import time;print(f"{time.time():.3f}")'; }
elapsed() { python3 -c "print(f'{$2-$1:.2f}')"; }

command -v openshell >/dev/null 2>&1 || { echo "ERROR: openshell CLI not in PATH." >&2; exit 1; }
"$UV" run python -c "import sys" >/dev/null 2>&1 || { echo "ERROR: uv not usable at $UV." >&2; exit 1; }

: > /tmp/bench_host.txt; : > /tmp/bench_e2e.txt; : > /tmp/bench_warm.txt

echo "### [1/5] HOST  agent ask  (N=$HOST_N)"
for ((i=1; i<=HOST_N; i++)); do
  s=$(now); "$UV" run python agent.py ask "$Q" >/dev/null 2>&1; e=$(now)
  d=$(elapsed "$s" "$e"); echo "$d" >> /tmp/bench_host.txt; echo "  run $i: ${d}s"
done

echo "### [2/5] OPENSHELL  sandbox-ask end-to-end  (N=$E2E_N)"
for ((i=1; i<=E2E_N; i++)); do
  s=$(now); OPENSHELL_SANDBOX_IMAGE=$IMG "$UV" run python agent.py sandbox-ask --image "$IMG" "$Q" >/dev/null 2>&1; e=$(now)
  d=$(elapsed "$s" "$e"); echo "$d" >> /tmp/bench_e2e.txt; echo "  run $i: ${d}s"
done

echo "### [3/5] WARM in-sandbox exec  (keep 1 sandbox, exec N=$WARM_N)"
OUT=$(OPENSHELL_SANDBOX_IMAGE=$IMG "$UV" run python agent.py sandbox-ask --keep --image "$IMG" "$Q" 2>&1)
NAME=$(echo "$OUT" | grep -oE 'agent-ask-[0-9a-f]+' | head -1)
echo "  kept sandbox: $NAME"
for ((i=1; i<=WARM_N; i++)); do
  s=$(now)
  openshell sandbox exec -n "$NAME" --workdir /sandbox/workspace -- \
    sh -lc "export TIKTOKEN_CACHE_DIR=/sandbox/.tiktoken && exec python agent.py ask \"$Q\"" >/dev/null 2>&1
  e=$(now); d=$(elapsed "$s" "$e"); echo "$d" >> /tmp/bench_warm.txt; echo "  run $i: ${d}s"
done
[ -n "$NAME" ] && openshell sandbox delete "$NAME" >/dev/null 2>&1 && echo "  deleted $NAME"

echo "### [4/5] verbose runs to extract per-LLM-call latency"
"$UV" run python agent.py ask "$Q" > /tmp/bench_host_verbose.txt 2>&1
OPENSHELL_SANDBOX_IMAGE=$IMG "$UV" run python agent.py sandbox-ask --image "$IMG" "$Q" > /tmp/bench_sbx_verbose.txt 2>&1

echo "### [5/5] SUMMARY"
python3 <<'PY'
import re, statistics, pathlib
def load(f):
    p = pathlib.Path(f)
    return [float(x) for x in p.read_text().split()] if p.exists() else []
def fmt(xs):
    return f"p50={statistics.median(xs):.2f}s  min={min(xs):.2f}  max={max(xs):.2f}  (n={len(xs)})" if xs else "n/a"
host, e2e, warm = load('/tmp/bench_host.txt'), load('/tmp/bench_e2e.txt'), load('/tmp/bench_warm.txt')
print(f"Host ask (no OpenShell):     {fmt(host)}")
print(f"OpenShell sandbox-ask e2e:   {fmt(e2e)}")
print(f"Warm in-sandbox exec:        {fmt(warm)}")
if host and e2e and warm:
    m = statistics.median
    print(f"\n  one-time provisioning   ~ e2e - warm  = {m(e2e)-m(warm):.2f}s")
    print(f"  per-request enforcement ~ warm - host = {m(warm)-m(host):.2f}s (read as ~0; see caveats)")

def llm_calls(f):
    p = pathlib.Path(f)
    if not p.exists(): return []
    times = []
    for mt in re.finditer(r'\[(\d\d):(\d\d):(\d\d\.\d+)\]\s+\S*\s*LLM (thinking|done)', p.read_text()):
        h, mm, ss, kind = mt.groups()
        times.append((int(h)*3600 + int(mm)*60 + float(ss), kind))
    calls, pend = [], None
    for t, kind in times:
        if kind == 'thinking': pend = t
        elif kind == 'done' and pend is not None: calls.append(t - pend); pend = None
    return calls
def avg(x): return f"{sum(x)/len(x):.2f}s avg over {len(x)} calls" if x else "n/a"
hc, sc = llm_calls('/tmp/bench_host_verbose.txt'), llm_calls('/tmp/bench_sbx_verbose.txt')
print(f"\nPer-LLM-call latency (Anthropic round trip):")
print(f"  host (direct):        {avg(hc)}")
print(f"  sandbox (via proxy):  {avg(sc)}")
PY
echo "### DONE"
