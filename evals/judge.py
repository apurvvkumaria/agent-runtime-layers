"""A deepeval judge model backed by Claude.

deepeval's LLM-judged metrics default to OpenAI; this wraps `ChatAnthropic` in
deepeval's `DeepEvalBaseLLM` interface so the metrics judge with Claude instead
(no OpenAI key needed). Supports schema-based structured output, which deepeval's
newer metrics use for their statement/verdict generation.
"""

import os
import pathlib
import sys

# Repo root on the path so `import core`-style imports resolve from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Quiet deepeval's telemetry/network chatter during evals.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from langchain_anthropic import ChatAnthropic  # noqa: E402

_MODEL = "claude-sonnet-4-6"


def _as_text(content) -> str:
    """Anthropic content may be a string or a list of blocks; normalize to text."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


class ClaudeJudge(DeepEvalBaseLLM):
    """deepeval model that delegates to Claude via LangChain's ChatAnthropic."""

    def __init__(self, model: str = _MODEL) -> None:
        self.model_name = model
        self._client = ChatAnthropic(model=model, temperature=0)

    def load_model(self):
        return self._client

    def generate(self, prompt: str, schema=None):
        # When deepeval asks for structured output, return the schema instance.
        if schema is not None:
            return self._client.with_structured_output(schema).invoke(prompt)
        return _as_text(self._client.invoke(prompt).content)

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self._client.with_structured_output(schema).ainvoke(prompt)
        result = await self._client.ainvoke(prompt)
        return _as_text(result.content)

    def get_model_name(self) -> str:
        return self.model_name


def claude_judge() -> ClaudeJudge:
    return ClaudeJudge()
