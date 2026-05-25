# error_budget

## Version
1.0.0

## Status
active

## What it does
Computes the monthly error budget (allowed downtime) for a target availability percentage.

## When to use it
When the user asks about error budgets, allowed downtime, or what an availability target
(e.g. "99.9%") works out to in minutes or hours per month.

## Inputs
- `availability` (str) — target availability percent, e.g. `99.9` or `99.95%`.

## Outputs
A sentence stating the allowed downtime per 30-day month, in minutes and hours.

## Tools used internally
None — pure arithmetic, no tools or network.

## Example invocation
`error_budget("99.9")`
