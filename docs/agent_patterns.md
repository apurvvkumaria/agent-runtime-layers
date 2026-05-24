# Agent Patterns

**ReAct** interleaves reasoning and acting in a single loop. The model emits a *Thought*,
chooses an *Action* (a tool call) with an *Action Input*, reads the *Observation* returned
by the tool, and repeats until it can produce a *Final Answer*. Its strength is simplicity
and transparency — the whole trajectory is plain text you can read — which makes it ideal
for tool-using assistants where each step depends on the last observation.

**Plan-and-Execute** separates planning from doing. A planner first decomposes the task
into an explicit multi-step plan, then an executor carries out each step (often with its own
sub-tool-calls), optionally re-planning if a step fails. Compared to ReAct it reduces
wasted LLM calls on long tasks and makes the intended approach inspectable up front, at the
cost of being less reactive to surprises discovered mid-execution.

**Multi-agent** decomposes a problem across several specialized agents that coordinate — for
example a researcher, a writer, and a critic, or a supervisor that routes work to workers.
Each agent has a focused role, prompt, and toolset, and they communicate via messages or a
shared scratchpad. This scales to complex workflows and enables specialization, but adds
orchestration overhead and new failure modes around communication and termination.
