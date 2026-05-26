"""Hot-reload demo (Layer 29): prove a skill's behavior swaps live, no restart.

In ONE process: create a throwaway skill, discover + invoke it (v1), rewrite its
`skill.py`, call `SkillRegistry.reload()`, invoke again (v2) — and show v1 != v2.
Without the reload, `importlib.import_module` would keep handing back the cached
module and the edit would be invisible. Cleans up the throwaway skill afterward.

Run directly:  uv run python skills/reload_demo.py   (or `agent skill-reload-demo`)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent
_NAME = "_hotreload_demo"

_SKILL_MD = """# _hotreload_demo

## Version
1.0.0

## What it does
Throwaway skill used by the hot-reload demo.

## Tools used internally
None.
"""

_POLICY = "network:\n  egress: []\nresources:\n  cpu_cores: 1\n  memory_mb: 64\n  timeout_seconds: 5\n"

_SKILL_PY = '''\
from __future__ import annotations
from skills.context import SkillContext

async def {name}(arg: str = "", ctx: SkillContext | None = None) -> str:
    return "{version}: hello from the hot-reload demo skill"
'''


def _write(version: str) -> None:
    d = _SKILLS_DIR / _NAME
    d.mkdir(exist_ok=True)
    (d / "__init__.py").write_text("", encoding="utf-8")
    (d / "SKILL.md").write_text(_SKILL_MD, encoding="utf-8")
    (d / "policy.yaml").write_text(_POLICY, encoding="utf-8")
    (d / "skill.py").write_text(_SKILL_PY.format(name=_NAME, version=version), encoding="utf-8")


def run() -> None:
    from skills.registry import SkillRegistry

    d = _SKILLS_DIR / _NAME
    try:
        _write("v1 (original)")
        reg = SkillRegistry().auto_discover()
        before = reg.get_skill(_NAME).invoke("x")
        print(f"v1 invoke -> {before}")

        print("\n(editing the skill's source on disk...)")
        _write("v2 (hot-reloaded, no restart)")

        changes = reg.reload()
        print(f"reload() -> {{'reloaded': {changes['reloaded']}, 'added': {changes['added']}, "
              f"'removed': {changes['removed']}}}")

        after = reg.get_skill(_NAME).invoke("x")
        print(f"v2 invoke -> {after}")

        print(f"\nhot-reload {'WORKED' if before != after else 'FAILED'}: "
              f"live behavior {'changed' if before != after else 'did NOT change'} without a restart.")
    finally:
        shutil.rmtree(d, ignore_errors=True)
        # drop the throwaway module objects so nothing stale lingers in this process
        for mod in [m for m in sys.modules if m.startswith(f"skills.{_NAME}")]:
            del sys.modules[mod]


if __name__ == "__main__":
    run()
