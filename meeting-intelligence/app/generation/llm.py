"""LLM client implementations.

`OpenAILLM` / `AnthropicLLM` are the real backends (lazy imports, keys from
config). `EchoLLM` is a deterministic, key-free client used by the demo profile
and tests: it emits an extractive answer built from the first context block and
cites it, so the full request->retrieve->cite->render path is exercisable
end-to-end with no network. It is not meant to reason — just to keep the plumbing
runnable and testable offline.
"""

from __future__ import annotations

import re

from app.interfaces import LLMClient


class EchoLLM(LLMClient):
    can_generate = False  # extractive only — no rewriting, no abstractive text

    def complete(self, system: str, user: str) -> str:
        # Echo back the first *hit* excerpt with its citation, mimicking a
        # grounded extractive answer. Skip excerpts marked "(context)" (the
        # neighbouring turns added for context) so the answer lands on a real
        # hit rather than a surrounding line.
        first = None
        for m in re.finditer(r"\[(\d+)\]\s*\([^)]*\)\s*(.+)", user):
            text = m.group(2).strip()
            if first is None:
                first = (m.group(1), text)
            if not text.startswith("(context)"):
                return f"{text} [{m.group(1)}]"
        if first is None:
            return "Not discussed in the transcript."
        return f"{first[1]} [{first[0]}]"


class OpenAILLM(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # lazy

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        res = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return res.choices[0].message.content or ""


class AnthropicLLM(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        import anthropic  # lazy

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> str:
        res = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in res.content if block.type == "text")
