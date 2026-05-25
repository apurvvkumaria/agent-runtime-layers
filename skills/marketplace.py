"""Skill marketplace / remote skill loading (Layer 27).

Loading skill *code* from a remote source is an arbitrary-code-execution +
supply-chain surface, so this is built integrity-first:

  - A registry (`marketplace/index.json`) lists available skills and **pins a
    SHA256** over each skill package (SKILL.md + skill.py + policy.yaml).
  - `install()` recomputes that hash and **refuses to materialize the code on any
    mismatch** (tamper / supply-chain protection) — only a verified package is
    copied into `skills/`, where the SkillRegistry then discovers it.

The "remote" source is a local directory by default, but is configurable
(`SKILL_MARKETPLACE` / `source=`) so an `http(s)://` registry slots in with the
same contract: fetch bytes → verify hash → install. Provenance is one control;
running an untrusted skill under the OpenShell sandbox (Layer 21) is the
complementary one — verify *what* the code is, then bound *what it can do*.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

_SKILLS_DIR = Path(__file__).resolve().parent
_DEFAULT_SOURCE = _SKILLS_DIR.parent / "marketplace"
_PKG_FILES = ("SKILL.md", "skill.py", "policy.yaml")


class IntegrityError(RuntimeError):
    """A package's contents don't match the SHA256 pinned in the marketplace index."""


def package_hash(pkg_dir: Path) -> str:
    """Deterministic SHA256 over a skill package's files (name + bytes, sorted)."""
    h = hashlib.sha256()
    for name in _PKG_FILES:  # fixed order
        path = pkg_dir / name
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes() if path.exists() else b"")
        h.update(b"\0")
    return h.hexdigest()


class SkillMarketplace:
    """Lists, verifies, and installs skills from a (local-by-default) registry."""

    def __init__(self, source: str | os.PathLike | None = None) -> None:
        self.source = Path(source or os.environ.get("SKILL_MARKETPLACE", _DEFAULT_SOURCE))
        self.index_path = self.source / "index.json"

    def _index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8")).get("skills", [])

    def _entry(self, name: str) -> dict:
        for e in self._index():
            if e["name"] == name:
                return e
        raise KeyError(f"skill {name!r} is not in the marketplace index")

    def available(self) -> list[dict]:
        """Catalog entries, each annotated with whether it's already installed."""
        return [{**e, "installed": (_SKILLS_DIR / e["name"]).exists()} for e in self._index()]

    def verify(self, name: str) -> str:
        """Recompute the package hash and compare to the index; raise on mismatch."""
        entry = self._entry(name)
        actual = package_hash(self.source / entry["path"])
        if actual != entry["sha256"]:
            raise IntegrityError(
                f"checksum mismatch for {name!r}: index pins {entry['sha256'][:12]}…, "
                f"package is {actual[:12]}… — refusing to install"
            )
        return actual

    def install(self, name: str) -> dict:
        """Verify the package, then copy it into `skills/` so the registry loads it."""
        entry = self._entry(name)
        pkg = self.source / entry["path"]
        if not pkg.exists():
            raise FileNotFoundError(f"marketplace package directory missing: {pkg}")
        self.verify(name)  # fail closed BEFORE any code lands in skills/
        dest = _SKILLS_DIR / entry["name"]
        shutil.copytree(pkg, dest, dirs_exist_ok=True)
        return {"name": name, "version": entry["version"], "dest": str(dest)}

    def uninstall(self, name: str) -> bool:
        dest = _SKILLS_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
            return True
        return False
