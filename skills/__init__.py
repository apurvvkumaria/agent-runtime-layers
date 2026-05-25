"""Skills: OpenClaw-style packages that compose tools behind one interface.

Each skill is a directory `skills/<name>/` with `SKILL.md` (capability declaration),
`skill.py` (`async def <name>(arg, ctx=None)`), and `policy.yaml` (capability policy).
`SkillRegistry.auto_discover()` finds them and exposes each as a LangChain tool.
"""
