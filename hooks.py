"""Observability hooks: print-based lifecycle logging + optional LangFuse tracing.

`get_callbacks()` returns the active callback list to attach to a run — always the
print-based StepLogger, plus a LangFuse handler when credentials are configured.
The LangFuse handler is built once per process and cached (auth check is not free).
"""

import os
import time
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler


def _now() -> str:
    """Wall-clock timestamp with millisecond precision, for hook logging."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _token_counts(response) -> dict | None:
    """Best-effort extraction of token usage from an LLMResult.

    Where the counts live depends on the model/streaming path, so check both the
    aggregated llm_output and the per-generation message metadata.
    """
    usage = (response.llm_output or {}).get("usage") or (response.llm_output or {}).get(
        "token_usage"
    )
    if usage:
        return usage
    try:
        message = response.generations[0][0].message
    except (IndexError, AttributeError):
        return None
    return getattr(message, "usage_metadata", None)


class StepLogger(BaseCallbackHandler):
    """Lifecycle hooks: print every tool call and LLM call as the agent runs.

    These are LangChain callbacks — the framework invokes each on_* method at the
    matching point in the run. This is the explicit, observable version of what
    verbose=True does implicitly, plus timing and token counts.
    """

    def __init__(self) -> None:
        # run_id -> perf_counter() at tool start, so on_tool_end can time it.
        self._tool_started_at: dict = {}

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self._tool_started_at[run_id] = time.perf_counter()
        name = (serialized or {}).get("name", "unknown")
        print(f"[{_now()}] 🔧 tool start  → {name}({input_str!r})")

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        started = self._tool_started_at.pop(run_id, None)
        elapsed = f"{(time.perf_counter() - started) * 1000:.1f}ms" if started else "n/a"
        # output may be a str or a ToolMessage depending on LangChain version.
        text = output if isinstance(output, str) else getattr(output, "content", str(output))
        print(f"[{_now()}] ✅ tool end    ← ({elapsed})\n{text}")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        # Chat models fall back to on_llm_start when on_chat_model_start is unset.
        print(f"[{_now()}] 🧠 LLM thinking...")

    def on_llm_end(self, response, **kwargs) -> None:
        usage = _token_counts(response)
        if usage:
            print(f"[{_now()}] 🧠 LLM done — tokens: {usage}")
        else:
            print(f"[{_now()}] 🧠 LLM done — token counts unavailable")


def _langfuse_configured() -> bool:
    """True only if both LangFuse keys look real (set and not placeholders)."""
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "")
    # The shipped .env uses "your-...-here" placeholders; treat those as unset.
    return bool(pub) and bool(sec) and "your-" not in pub and "your-" not in sec


def build_langfuse_handler() -> BaseCallbackHandler | None:
    """Return a LangFuse CallbackHandler if credentials are configured, else None.

    Follows the official Langfuse LangChain pattern: credentials via the LANGFUSE_*
    env vars (read by the client — no constructor args), then a bare
    `CallbackHandler()`. Trace attributes are set per-call via config metadata in
    `core.stream_answer`, not here. Any failure degrades gracefully to print-only
    hooks rather than crashing.
    """
    if not _langfuse_configured():
        print("[langfuse] keys not configured — using print hooks only.")
        return None
    try:
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        client = get_client()
        # Confirm credentials/connectivity before relying on tracing; never fatal.
        if not client.auth_check():
            print("[langfuse] auth check failed — using print hooks only.")
            return None
        print("[langfuse] tracing enabled.")
        return CallbackHandler()
    except Exception as exc:  # network, bad keys, version drift, etc.
        print(f"[langfuse] disabled ({exc}) — using print hooks only.")
        return None


# The LangFuse handler is created once per process (auth check is a network call)
# and reused across every run.
_langfuse_handler: BaseCallbackHandler | None = None
_langfuse_ready = False


def _get_langfuse_handler() -> BaseCallbackHandler | None:
    global _langfuse_handler, _langfuse_ready
    if not _langfuse_ready:
        _langfuse_handler = build_langfuse_handler()
        _langfuse_ready = True
    return _langfuse_handler


def get_callbacks() -> list:
    """Active callbacks for a run: the print hooks, plus LangFuse when configured.

    A fresh StepLogger per call keeps its per-run timing state isolated; the
    LangFuse handler is the shared, cached one.
    """
    callbacks: list = [StepLogger()]
    handler = _get_langfuse_handler()
    if handler is not None:
        callbacks.append(handler)
    return callbacks


def flush_traces() -> None:
    """Flush buffered LangFuse traces before exit so short-lived runs don't drop them."""
    if _get_langfuse_handler() is not None:
        from langfuse import get_client

        get_client().flush()
