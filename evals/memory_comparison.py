"""Compare prompt-token footprint: buffer memory vs. vector memory.

Both memory types are seeded with the same background conversation, then asked the
same 3 questions. For each question we measure the tokens the memory injects into
the {chat_history} slot (via load_memory_variables) and sum across the 3 turns.

Buffer memory replays *every* past turn, so its footprint grows with history;
vector memory retrieves only the top-k similar turns, so it stays bounded. This is
free — no LLM calls, just local hashing embeddings.

Run directly:  uv run python evals/memory_comparison.py
"""

import pathlib
import sys
import tempfile

# Repo root on the path so the top-level modules import regardless of cwd.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from langchain_classic.memory import ConversationBufferMemory  # noqa: E402

from memory.vector_store import VectorStoreMemory  # noqa: E402

# A background conversation, so there's real history for the buffer to accumulate.
BACKGROUND = [
    ("What is the capital of France?", "Paris."),
    ("What is 15 times 15?", "225."),
    ("Explain what a hash table is.", "A key-value structure with O(1) average lookup."),
    ("Who wrote Pride and Prejudice?", "Jane Austen."),
    ("What is the boiling point of water at sea level?", "100 degrees Celsius."),
    ("Define idempotent.", "An operation that yields the same result when applied repeatedly."),
    ("What is the speed of light?", "About 299,792 km/s."),
    ("What does CPU stand for?", "Central Processing Unit."),
]

# The 3 questions whose chat_history footprint we measure under each memory type.
QUESTIONS = [
    "Remind me, what is the capital of France?",
    "What was that multiplication result again?",
    "Can you re-explain hash tables?",
]


def _est_tokens(text: str) -> int:
    return len(text) // 4  # ~4 chars per token


def _seed(memory) -> None:
    for q, a in BACKGROUND:
        memory.save_context({"input": q}, {"output": a})


def _footprint(memory) -> int:
    """Sum the chat_history tokens the memory injects across the 3 questions."""
    total = 0
    for q in QUESTIONS:
        history = memory.load_memory_variables({"input": q})["chat_history"]
        total += _est_tokens(history)
        memory.save_context({"input": q}, {"output": "(answer)"})
    return total


def run() -> tuple[int, int]:
    """Return (buffer_tokens, vector_tokens) and print the comparison."""
    buffer_mem = ConversationBufferMemory(memory_key="chat_history", output_key="output")
    _seed(buffer_mem)
    buffer_tokens = _footprint(buffer_mem)

    with tempfile.TemporaryDirectory() as tmp:
        vector_mem = VectorStoreMemory(store_dir=tmp, k=3)
        _seed(vector_mem)
        vector_tokens = _footprint(vector_mem)

    saved = buffer_tokens - vector_tokens
    pct = round(100 * saved / buffer_tokens, 1) if buffer_tokens else 0.0
    print(
        f"Buffer memory: {buffer_tokens} tokens, Vector memory: {vector_tokens} tokens, "
        f"Savings: {saved} tokens ({pct}%)"
    )
    print(
        f"(measured over {len(QUESTIONS)} questions after {len(BACKGROUND)} background turns; "
        "estimated at ~4 chars/token)"
    )
    return buffer_tokens, vector_tokens


if __name__ == "__main__":
    run()
