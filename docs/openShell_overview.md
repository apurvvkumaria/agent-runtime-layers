# OpenShell Overview

OpenShell is a sandbox runtime that gives autonomous agents a controlled, isolated
execution environment. Rather than letting an agent touch the host directly, OpenShell
mediates every action — filesystem reads and writes, network egress, and system calls —
through a declarative policy layer. Each agent runs inside its own shell with a least-
privilege capability set, so a misbehaving or compromised agent cannot escape the boundary
it was granted. Policies are expressed as YAML and evaluated per request, which makes the
trust boundary auditable and easy to reason about.

Architecturally, OpenShell sits between the agent loop and the operating system as a thin
broker process. The broker exposes a small, stable API (open, read, write, exec, connect)
and enforces the active policy on each call before forwarding it to the kernel. Because the
broker is the single choke point, observability and rate limiting are centralized: every
mediated action emits a structured event that downstream tracing can consume. This design
trades a small amount of per-call overhead for strong, uniform isolation across every agent
in the fleet.
