"""A deepeval judge model backed by Claude, with two-tier response caching.

deepeval's LLM-judged metrics default to OpenAI; this wraps `ChatAnthropic` in
deepeval's `DeepEvalBaseLLM` interface so the metrics judge with Claude instead
(no OpenAI key needed). Supports schema-based structured output, which deepeval's
newer metrics use for their statement/verdict generation.

Caching (Layer 22): every judge call is keyed on `SHA256(prompt, schema, model)`
and served from a two-tier cache — an in-process `functools.lru_cache` (memory
tier) over the JSON disk cache in `cache.py`. A re-run on unchanged prompts is
all cache hits and spends zero tokens.
"""

import functools
import os
import pathlib
import sys

# Repo root on the path so `import core`-style imports resolve from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Quiet deepeval's telemetry/network chatter during evals.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402

from evals.cache import get_cache, make_key  # noqa: E402

_MODEL = "claude-sonnet-4-6"


def _as_text(content) -> str:
    """Anthropic content may be a string or a list of blocks; normalize to text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


@functools.lru_cache(maxsize=8)
def _client_for(model: str) -> ChatAnthropic:
    return ChatAnthropic(model=model, temperature=0)


def _call_claude(prompt: str, model: str, schema):
    """One real judge call. Returns (result, token_count); result is text or a schema obj."""
    client = _client_for(model)
    if schema is not None:
        # include_raw so we keep the parsed object AND the token usage.
        out = client.with_structured_output(schema, include_raw=True).invoke(prompt)
        tokens = (getattr(out["raw"], "usage_metadata", None) or {}).get("total_tokens", 0)
        return out["parsed"], tokens
    msg = client.invoke(prompt)
    tokens = (getattr(msg, "usage_metadata", None) or {}).get("total_tokens", 0)
    return _as_text(msg.content), tokens


@functools.lru_cache(maxsize=1024)
def _judge_call(prompt: str, model: str, schema):
    """Memory tier (lru) over the disk tier. `schema` is a hashable class or None.

    Within one process, an identical (prompt, model, schema) returns from lru
    without touching disk. Across processes the lru is cold, so the JSON disk
    cache serves the hit. On a true miss we call Claude and persist the result.
    """
    cache = get_cache()
    schema_name = schema.__name__ if schema is not None else ""
    key = make_key(prompt, schema_name, model)

    cached = cache.get(key)  # disk tier (counts hit/miss)
    if cached is not None:
        print("        [cache hit]")
        return schema.model_validate_json(cached) if schema is not None else cached

    result, tokens = _call_claude(prompt, model, schema)
    print(f"        [cache miss] {tokens} tokens used")
    value = result.model_dump_json() if schema is not None else result
    cache.set(key, value, prompt_hash=key, model=model, token_count=tokens)
    return result


def clear_memory_tier() -> None:
    """Drop the in-process lru so the next call falls through to the disk tier.

    Used by the cache demo to show the JSON cache serving hits within one process
    (the lru would otherwise short-circuit before disk).
    """
    _judge_call.cache_clear()


class ClaudeJudge(DeepEvalBaseLLM):
    """deepeval model that delegates to Claude, through the two-tier cache."""

    def __init__(self, model: str = _MODEL) -> None:
        self.model_name = model

    def load_model(self):
        return _client_for(self.model_name)

    def generate(self, prompt: str, schema=None):
        return _judge_call(prompt, self.model_name, schema)

    async def a_generate(self, prompt: str, schema=None):
        # async_mode is False for our metrics, but route through the cache anyway.
        return _judge_call(prompt, self.model_name, schema)

    def get_model_name(self) -> str:
        return self.model_name


def claude_judge() -> ClaudeJudge:
    return ClaudeJudge()
