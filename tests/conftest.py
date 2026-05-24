"""Shared pytest fixtures.

Puts the repo root on sys.path so the top-level modules (`api`, `core`, `tools`)
import cleanly, and provides a FastAPI TestClient plus an LLM stub so endpoint
tests run fast and free (no real Claude calls).
"""

import os
import sys
import pathlib

# Repo root on the path so `import api` / `import tools` work under pytest.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# A dummy key so building an agent never raises during tests; real LLM calls are
# stubbed out, so the value is never used.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """A FastAPI TestClient bound to the app."""
    return TestClient(api.app)


@pytest.fixture
def stub_llm(monkeypatch):
    """Stub the LLM path so /ask and /chat don't make real Claude calls.

    Patches the agent builders and the streaming function in the `api` module's
    namespace, and silences trace flushing. Endpoint wiring (validation, session
    handling, response shape) is still exercised for real.
    """
    async def fake_stream_answer(executor, question, session_id=None, echo=True):
        return f"stubbed answer to: {question}"

    monkeypatch.setattr(api, "stream_answer", fake_stream_answer)
    monkeypatch.setattr(api, "build_single_shot_agent", lambda: object())
    monkeypatch.setattr(api, "_session_agent", lambda session_id: object())
    monkeypatch.setattr(api, "flush_traces", lambda: None)
    return fake_stream_answer
