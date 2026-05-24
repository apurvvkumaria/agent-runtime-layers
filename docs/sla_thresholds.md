# Storage SLA Thresholds

These thresholds define how to interpret a storage cluster's p99 read latency. Use them
when assessing whether a cluster is healthy.

| p99 latency | Rating | Meaning |
|---|---|---|
| < 5 ms | **Excellent** | Well within target; no action needed. |
| 5 – 20 ms | **Normal** | Acceptable operating range; keep monitoring. |
| 20 – 50 ms | **Degraded** | Elevated; investigate load, hot partitions, or replication lag. |
| > 50 ms | **Critical** | Breaching SLA; page on-call and shed load if needed. |

Supporting guidance:

- **p99 < 5 ms is excellent** — the cluster is comfortably meeting its latency objective.
- **5–20 ms is normal** — expected under typical production load.
- **> 50 ms is critical** — treat as an incident.

Other signals to weigh alongside latency: disk utilization above ~85% and replication lag
above ~50 ms both warrant attention even when p99 still looks acceptable.
