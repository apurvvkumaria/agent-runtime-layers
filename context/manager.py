"""Token-budget management for the agent's context window.

A context window is a fixed budget; this allocates it across the sources that make
up a prompt (system prompt, history, retrieved docs, tool results, question) and
truncates each source to its share so the whole stays bounded. Tokens are counted
with tiktoken's cl100k_base encoding.
"""

import logging

import tiktoken

logger = logging.getLogger("context.manager")


class ContextManager:
    """Counts tokens and enforces a per-source budget over the context window."""

    CONTEXT_BUDGET = {
        "system_prompt": 500,
        "conversation_history": 2000,
        "retrieved_context": 1500,
        "tool_results": 500,
        "question": 200,
        "response_reserve": 1000,
    }
    _DEFAULT_BUDGET = 500

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._enc = None

    @property
    def _encoding(self):
        if self._enc is None:
            self._enc = tiktoken.get_encoding(self._encoding_name)
        return self._enc

    def count_tokens(self, text: str) -> int:
        """Number of cl100k_base tokens in `text`."""
        return len(self._encoding.encode(text or ""))

    def _truncate(self, text: str, budget: int, keep: str = "head") -> str:
        tokens = self._encoding.encode(text or "")
        if len(tokens) <= budget:
            return text or ""
        kept = tokens[:budget] if keep == "head" else tokens[-budget:]
        return self._encoding.decode(kept)

    def enforce_budget(self, context_dict: dict) -> dict:
        """Truncate each source to its budget; log tokens used per source.

        Keeps the *most recent* tokens for conversation_history (tail), the head
        for everything else.
        """
        truncated = {}
        for source, text in context_dict.items():
            budget = self.CONTEXT_BUDGET.get(source, self._DEFAULT_BUDGET)
            keep = "tail" if source == "conversation_history" else "head"
            before = self.count_tokens(text)
            result = self._truncate(text, budget, keep)
            after = self.count_tokens(result)
            logger.info(
                "source=%s tokens=%d->%d budget=%d%s",
                source, before, after, budget, " (truncated)" if after < before else "",
            )
            truncated[source] = result
        return truncated

    def truncate_history(self, turns: list[str], budget: int) -> str:
        """Keep the most recent turns that fit within `budget`, dropping oldest first."""
        kept: list[str] = []
        total = 0
        for turn in reversed(turns):
            cost = self.count_tokens(turn)
            if total + cost > budget:
                break
            kept.append(turn)
            total += cost
        kept.reverse()
        return "\n".join(kept)

    def budget_report(self, context_dict: dict) -> str:
        """Human-readable token usage per source vs. budget, plus the total."""
        window = sum(self.CONTEXT_BUDGET.values())
        lines = ["Context budget report:"]
        total = 0
        for source, text in context_dict.items():
            used = self.count_tokens(text)
            total += used
            budget = self.CONTEXT_BUDGET.get(source, self._DEFAULT_BUDGET)
            flag = "  OVER" if used > budget else ""
            lines.append(f"  {source:<22} {used:>5} / {budget:<5} tokens{flag}")
        pct = round(100 * total / window, 1) if window else 0.0
        lines.append(f"  {'TOTAL':<22} {total:>5} / {window:<5} tokens ({pct}% of window)")
        return "\n".join(lines)
