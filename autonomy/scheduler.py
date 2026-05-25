"""Autonomous operation modes for the agent.

Two ways to run the agent without a human in the loop:

- **Scheduled (cron):** `AgentScheduler` runs a question on a cron schedule via
  APScheduler, appending each timestamped answer to an output file. It runs once
  immediately to verify, then on the schedule.
- **Heartbeat:** `HeartbeatLoop` polls `tasks.json` every N seconds, runs the
  agent on pending tasks, marks them complete, and can queue follow-up tasks the
  agent suggests (self-directing, bounded to one level so it can't run away).

The blocking run loops (`AgentScheduler.schedule`, `HeartbeatLoop.run`) are thin
wrappers over the testable one-shot methods (`run_once`, `process_pending`).
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

TASKS_PATH = Path(__file__).resolve().parent.parent / "tasks.json"


def _answer(question: str) -> str:
    """Run the single-shot agent on a question and return its answer."""
    from core import build_single_shot_agent, new_session_id, stream_answer

    executor = build_single_shot_agent()
    return asyncio.run(stream_answer(executor, question, new_session_id(), echo=False))


# --- tasks.json helpers ----------------------------------------------------- #
def _new_task(question: str, source: str = "user") -> dict:
    return {
        "id": uuid.uuid4().hex,
        "question": question,
        "status": "pending",
        "source": source,  # "user" or "auto" (agent-generated)
        "created": datetime.now().isoformat(timespec="seconds"),
        "answer": None,
    }


def load_tasks() -> list[dict]:
    if not TASKS_PATH.exists():
        return []
    try:
        return json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_tasks(tasks: list[dict]) -> None:
    TASKS_PATH.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def add_task(question: str, source: str = "user") -> dict:
    """Append a pending task to tasks.json and return it."""
    tasks = load_tasks()
    task = _new_task(question, source)
    tasks.append(task)
    save_tasks(tasks)
    return task


# --- scheduled (cron) mode -------------------------------------------------- #
class AgentScheduler:
    """Run a question on a cron schedule, appending answers to a file."""

    def __init__(self, on_log=print) -> None:
        self._scheduler = BlockingScheduler()
        self._log = on_log

    def run_once(self, question: str, output_file: str) -> str | None:
        """Run the agent once and append the timestamped answer to output_file."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path = Path(output_file)
        try:
            answer = _answer(question)
        except Exception as exc:  # noqa: BLE001 — keep the scheduler alive; DLQ (18b) will hook here
            self._log(f"[{ts}] run FAILED: {exc}")
            with path.open("a", encoding="utf-8") as f:
                f.write(f"## {ts}\n\nQ: {question}\n\nERROR: {exc}\n\n")
            return None
        with path.open("a", encoding="utf-8") as f:
            f.write(f"## {ts}\n\nQ: {question}\n\n{answer}\n\n")
        self._log(f"[{ts}] ran {question[:50]!r} -> {output_file}")
        return answer

    def schedule(self, question: str, cron_expr: str, output_file: str) -> None:
        """Run once immediately to verify, then on the cron schedule (blocks)."""
        self.run_once(question, output_file)  # immediate verification run
        self._scheduler.add_job(
            self.run_once,
            CronTrigger.from_crontab(cron_expr),
            args=[question, output_file],
            id="scheduled-agent",
        )
        self._log(f"Scheduled on cron {cron_expr!r}; appending to {output_file}. Ctrl+C to stop.")
        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self._scheduler.shutdown(wait=False)
            self._log("Scheduler stopped.")


# --- heartbeat mode --------------------------------------------------------- #
class HeartbeatLoop:
    """Poll tasks.json, run pending tasks, and self-direct follow-ups."""

    def __init__(self, interval: int = 60, on_log=print) -> None:
        self.interval = interval
        self._log = on_log

    def process_pending(self) -> dict:
        """One pass: run each pending task, mark it done, maybe queue a follow-up."""
        tasks = load_tasks()
        completed = added = failed = 0
        for task in [t for t in tasks if t.get("status") == "pending"]:
            try:
                answer = _answer(task["question"])
            except Exception as exc:  # noqa: BLE001 — DLQ (18b) will hook here
                task["status"] = "failed"
                task["error"] = str(exc)
                failed += 1
                self._log(f"task {task['id'][:8]} FAILED: {exc}")
                continue
            task["status"] = "complete"
            task["answer"] = answer
            task["completed"] = datetime.now().isoformat(timespec="seconds")
            completed += 1
            self._log(f"completed {task['id'][:8]}: {task['question'][:50]}")
            # Self-direct: only user tasks spawn follow-ups, so auto tasks can't chain.
            if task.get("source") != "auto":
                follow = self._followup(task["question"], answer)
                if follow:
                    tasks.append(_new_task(follow, source="auto"))
                    added += 1
                    self._log(f"  + queued follow-up: {follow[:50]}")
        save_tasks(tasks)
        pending = sum(1 for t in tasks if t.get("status") == "pending")
        return {"completed": completed, "added": added, "failed": failed, "pending": pending}

    def run(self) -> None:
        """Loop forever, processing pending tasks every `interval` seconds (blocks)."""
        self._log(f"Heartbeat every {self.interval}s. Ctrl+C to stop.")
        try:
            while True:
                result = self.process_pending()
                if result["completed"] or result["added"] or result["failed"]:
                    self._log(f"tick: {result}")
                time.sleep(self.interval)
        except KeyboardInterrupt:
            self._log("Heartbeat stopped.")

    @staticmethod
    def _followup(question: str, answer: str) -> str | None:
        """Ask the model for ONE concrete follow-up task (or None)."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
            prompt = (
                "Given this Q&A, suggest ONE concrete follow-up research task that would "
                "deepen it, or reply exactly NONE.\n"
                f"Q: {question}\nA: {answer}\nFollow-up task:"
            )
            content = llm.invoke(prompt).content
            text = content if isinstance(content, str) else "".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
            text = text.strip().splitlines()[0].strip() if text.strip() else ""
            return None if (not text or text.upper().startswith("NONE")) else text
        except Exception:  # noqa: BLE001
            return None
