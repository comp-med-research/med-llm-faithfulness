from __future__ import annotations

import os
from typing import Optional

try:
    import anthropic  # type: ignore
except Exception:  # pragma: no cover
    anthropic = None  # type: ignore

from .types import ModelClient


class AnthropicChatClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if anthropic is None:
            raise ImportError("anthropic package not installed. Add 'anthropic' to requirements.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        return "".join(part.text for part in msg.content)


