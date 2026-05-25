# storage_sla_report

## Version
0.9.0

## Status
deprecated

## Deprecation
- Replaced by: storage_health_check
- Remove after: 2026-09-30
- Reason: superseded by the SLA-aware `storage_health_check`, which reads the
  thresholds doc and emits a structured rating + verdict. This older report is kept
  only so existing callers keep working through the deprecation window.

## What it does
(Legacy) Reports a storage cluster's metrics against SLA thresholds.

## When to use it
Do not use for new work — call `storage_health_check` instead. Retained only for
backward compatibility with callers that still reference this name.

## Inputs
- `cluster` (str) — the cluster name (default: `prod-us-east-1`).

## Outputs
The same assessment `storage_health_check` returns (this skill delegates to it).

## Tools used internally
- `storage_health_check` — delegates entirely to the replacement skill

## Example invocation
`storage_sla_report("prod-us-east-1")`
