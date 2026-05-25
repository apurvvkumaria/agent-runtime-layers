"""Dead-letter queue (DLQ) for failed agent runs.

When a run fails, core records it here with its reason. Failures are classified:

- **transient** (retryable): tool_timeout, api_error_5xx, sandbox_crash
- **permanent** (needs review): budget_exceeded, api_error_4xx, max_retries_exceeded

Transient failures land in `failed_tasks.json` and can be replayed with exponential
backoff (1s, 2s, 4s); exhausting the retries reclassifies them as
`max_retries_exceeded` and moves them to `permanent_failures.json`. Unknown reasons
default to permanent (don't silently retry something we don't understand).
"""

import json
import time
from datetime import datetime
from pathlib import Path

DLQ_DIR = Path(__file__).resolve().parent
FAILED_PATH = DLQ_DIR / "failed_tasks.json"          # active (transient) failures
PERMANENT_PATH = DLQ_DIR / "permanent_failures.json"  # permanent failures

TRANSIENT_REASONS = {"tool_timeout", "api_error_5xx", "sandbox_crash"}
PERMANENT_REASONS = {"budget_exceeded", "api_error_4xx", "max_retries_exceeded"}
_BACKOFFS = (1, 2, 4)  # seconds to wait before each retry attempt


def classify(failure_reason: str) -> str:
    """Map a failure reason to 'transient' or 'permanent' (unknown -> permanent)."""
    if failure_reason in TRANSIENT_REASONS:
        return "transient"
    return "permanent"


def classify_exception(exc: Exception) -> str:
    """Map a caught exception to one of the known failure reasons."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        if 500 <= status < 600:
            return "api_error_5xx"
        if 400 <= status < 500:
            return "api_error_4xx"
    text = f"{type(exc).__name__} {exc}".lower()
    if "timeout" in text:
        return "tool_timeout"
    if "budget" in text or "context length" in text:
        return "budget_exceeded"
    if "anthropic_api_key" in text or "api key" in text or "unauthorized" in text:
        return "api_error_4xx"
    return "sandbox_crash"  # default unknown crash -> transient, gets a bounded retry


def flag_in_langfuse(session_id: str | None, reason: str) -> None:
    """Best-effort: score the failed run 0 in LangFuse so it surfaces in the dashboard."""
    from hooks import _langfuse_configured

    if not session_id or not _langfuse_configured():
        return
    try:
        from langfuse import get_client

        get_client().create_score(
            name="run_failed", value=0, session_id=session_id,
            data_type="NUMERIC", comment=reason,
        )
    except Exception:  # noqa: BLE001 — never let observability break the failure path
        pass


class DLQManager:
    """Records, classifies, retries, and reports failed agent runs."""

    def __init__(self, failed_path: Path = FAILED_PATH, permanent_path: Path = PERMANENT_PATH) -> None:
        self.failed_path = Path(failed_path)
        self.permanent_path = Path(permanent_path)

    # --- storage -------------------------------------------------------------
    @staticmethod
    def _load(path: Path) -> list[dict]:
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _save(path: Path, items: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def _append(self, path: Path, item: dict) -> None:
        items = self._load(path)
        items.append(item)
        self._save(path, items)

    # --- API -----------------------------------------------------------------
    def add_failure(
        self, task_id: str, question: str, reason: str,
        partial_state: dict | None = None, retry_count: int = 0,
    ) -> dict:
        """Record a failure, routed to the transient or permanent store by reason."""
        classification = classify(reason)
        entry = {
            "task_id": task_id,
            "question": question,
            "reason": reason,
            "classification": classification,
            "partial_state": partial_state or {},
            "retry_count": retry_count,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        target = self.permanent_path if classification == "permanent" else self.failed_path
        self._append(target, entry)
        return entry

    def _rerun(self, question: str) -> str:
        # Use the executor directly (not stream_answer) so a retry failure doesn't
        # re-enter the DLQ recursively.
        from core import build_single_shot_agent

        return build_single_shot_agent().invoke({"input": question})["output"]

    def retry_transient(self) -> dict:
        """Replay every transient failure with exponential backoff (1s, 2s, 4s)."""
        failures = self._load(self.failed_path)
        summary = {"retried": 0, "succeeded": 0, "moved_to_permanent": 0}
        for entry in failures:
            summary["retried"] += 1
            succeeded = False
            for delay in _BACKOFFS:
                time.sleep(delay)
                try:
                    self._rerun(entry["question"])
                    succeeded = True
                    break
                except Exception:  # noqa: BLE001 — try the next backoff
                    continue
            if succeeded:
                summary["succeeded"] += 1
            else:
                entry["reason"] = "max_retries_exceeded"
                entry["classification"] = "permanent"
                entry["retry_count"] = entry.get("retry_count", 0) + len(_BACKOFFS)
                self._append(self.permanent_path, entry)
                summary["moved_to_permanent"] += 1
        self._save(self.failed_path, [])  # all transient entries processed this pass
        return summary

    def clear_permanent(self) -> int:
        """Clear permanent failures (after human review). Returns how many were cleared."""
        count = len(self._load(self.permanent_path))
        self._save(self.permanent_path, [])
        return count

    def stats(self) -> dict:
        """Counts by failure reason and by classification, across both stores."""
        entries = self._load(self.failed_path) + self._load(self.permanent_path)
        by_reason: dict[str, int] = {}
        by_classification = {"transient": 0, "permanent": 0}
        for e in entries:
            by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
            by_classification[e.get("classification", classify(e["reason"]))] += 1
        return {
            "total": len(entries),
            "active_transient": len(self._load(self.failed_path)),
            "permanent": len(self._load(self.permanent_path)),
            "by_reason": by_reason,
            "by_classification": by_classification,
        }
