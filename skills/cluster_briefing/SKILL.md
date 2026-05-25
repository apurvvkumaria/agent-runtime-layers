# cluster_briefing

## Version
1.0.0

## Status
active

## What it does
Produces a full briefing on a storage cluster by composing two other skills: a live health check and background research.

## When to use it
When the user wants a complete briefing, overview, or "deep dive" on a cluster — not when
they only want the health check (`storage_health_check`) or only research
(`research_and_summarize`) on their own.

## Inputs
- `cluster` (str) — the cluster name to brief on (default: `prod-us-east-1`).

## Outputs
A combined markdown briefing with a **Health** section (the live SLA assessment) and a
**Background** section (research on storage reliability/SLA practices).

## Tools used internally
- `storage_health_check` — sub-skill: live SLA health assessment for the cluster
- `research_and_summarize` — sub-skill: background research report

## Example invocation
`cluster_briefing("prod-us-east-1")`
