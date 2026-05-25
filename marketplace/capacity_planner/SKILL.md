# capacity_planner

## Version
1.0.0

## Status
active

## Requires
- error_budget >= 1.0.0
- storage_health_check >= 1.0.0

## What it does
Produces a capacity plan for a cluster: its current SLA health plus the monthly error budget implied by a 99.9% availability target.

## When to use it
When the user wants a capacity plan, an error-budget-aware view of a cluster, or to combine a
cluster's live health with its SLA downtime allowance.

## Inputs
- `cluster` (str) — the cluster name (default: `prod-us-east-1`).

## Outputs
A markdown capacity plan: an SLA-target error-budget section + a current-health section.

## Tools used internally
- error_budget — sub-skill (marketplace): monthly downtime allowance for the target
- storage_health_check — sub-skill (core): live SLA health assessment

## Example invocation
`capacity_planner("prod-us-east-1")`
