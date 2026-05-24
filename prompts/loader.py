"""Load prompt templates from markdown files in this directory.

`load_prompt(name, **kwargs)` reads `prompts/{name}.md` and substitutes any
provided kwargs as `{placeholder}` replacements. Crucially, it only replaces the
kwargs you pass — the ReAct template placeholders (`{tools}`, `{tool_names}`,
`{input}`, `{agent_scratchpad}`, `{chat_history}`) are left intact for
`create_react_agent` / `PromptTemplate` to fill later.
"""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str, **kwargs) -> str:
    """Read prompts/{name}.md and apply {placeholder} substitutions from kwargs.

    Raises FileNotFoundError (listing available prompts) if the file is missing.
    """
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))) or "(none)"
        raise FileNotFoundError(
            f"Prompt {name!r} not found at {path}. Available prompts: {available}"
        )

    text = path.read_text(encoding="utf-8")
    # Replace only the placeholders the caller supplied; leave everything else
    # (e.g. the ReAct {tools}/{input} slots) untouched.
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text
