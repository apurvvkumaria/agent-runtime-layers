"""FastAPI REST interface over the agent core.

Run standalone with `uvicorn api:app` or via the CLI's `agent serve`. All endpoints
are async. LLM-backed endpoints (/ask, /chat) reuse the same core builders and
streaming loop as the CLI; /metrics and /calc hit the tools directly with no LLM.
"""

from dotenv import load_dotenv

# Load .env before anything reads ANTHROPIC_API_KEY / LANGFUSE_* (e.g. when run via
# `uvicorn api:app` without going through the CLI).
load_dotenv()

from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core import (  # noqa: E402
    build_chat_agent,
    build_memory,
    build_single_shot_agent,
    new_session_id,
    stream_answer,
)
from hooks import flush_traces  # noqa: E402
from tools import calculator, storage_metrics  # noqa: E402

app = FastAPI(
    title="Agent API",
    version="1.0.0",
    description="REST front door over a Claude + LangChain ReAct agent.",
)

# Permissive CORS for a learning/demo project — any origin may call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


# Per-session chat agents, each with its own in-process memory so conversations
# stay isolated (unlike the CLI's single shared history file). Lives for the
# lifetime of the server process.
_sessions: dict[str, object] = {}


def _session_agent(session_id: str):
    """Get (or lazily create) the chat agent for a session, with isolated memory."""
    if session_id not in _sessions:
        _sessions[session_id] = build_chat_agent(memory=build_memory(persist=False))
    return _sessions[session_id]


@app.get("/health")
async def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    """Single-shot question: no memory, fresh context every call."""
    try:
        executor = build_single_shot_agent()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        answer = await stream_answer(executor, req.question, new_session_id(), echo=False)
    finally:
        flush_traces()
    return {"question": req.question, "answer": answer}


@app.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Session-based conversation: replies are remembered within a session_id."""
    session_id = req.session_id or new_session_id()
    try:
        executor = _session_agent(session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        answer = await stream_answer(executor, req.message, session_id, echo=False)
    finally:
        flush_traces()
    return {"session_id": session_id, "answer": answer}


@app.get("/metrics/{cluster_name}")
async def metrics(cluster_name: str) -> dict:
    """Direct storage-metrics tool access (no LLM)."""
    return {"cluster": cluster_name, "metrics": storage_metrics.invoke(cluster_name)}


@app.get("/calc")
async def calc(expr: str = Query(..., description="Arithmetic expression, e.g. 150*223.48")) -> dict:
    """Direct calculator tool access (no LLM)."""
    return {"expression": expr, "result": calculator.invoke(expr)}
