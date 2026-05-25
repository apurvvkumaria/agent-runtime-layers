# research_and_summarize

## Version
1.0.0

## Status
active

## What it does
Researches a topic on the web and returns a structured markdown report, tied to live storage context.

## When to use it
When the user asks to "research and summarize" a topic, or wants a full report / research
write-up on something (a technology, system, or concept) — not for simple factual lookups,
math, or a single metric query.

## Inputs
- `topic` (str) — the subject to research.

## Outputs
A markdown report (str) with three sections: `## Research Findings`, `## Storage Context`,
and `## Summary`.

## Tools used internally
- `web_search` — DuckDuckGo search for the topic
- `storage_metrics` — live metrics for the `prod-us-east-1` cluster
- `llm_summarize` — Claude synthesizes the findings into the report

## Example invocation
`research_and_summarize("OpenShell sandbox runtime")`
