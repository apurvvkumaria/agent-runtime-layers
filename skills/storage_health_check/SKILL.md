# storage_health_check

## What it does
Checks a storage cluster's live metrics against the SLA thresholds and returns a health assessment.

## When to use it
When the user asks whether a cluster is healthy, in breach, or degraded, or wants an SLA /
health check for a named cluster (e.g. "is prod-us-east-1 healthy?"). Not for raw metric
dumps (use `storage_metrics` directly) or unrelated research.

## Inputs
- `cluster` (str) — the cluster name to assess (default: `prod-us-east-1`).

## Outputs
A health assessment (str): an overall verdict (Excellent / Normal / Degraded / Critical), the
p99 latency vs. its SLA band, and any disk-utilization or replication-lag flags.

## Tools used internally
- `storage_metrics` — live metrics for the cluster
- `filesystem` — reads `docs/sla_thresholds.md` for the SLA bands

## Example invocation
`storage_health_check("prod-us-east-1")`
