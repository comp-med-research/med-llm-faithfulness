from __future__ import annotations

import os
from typing import Optional

import anthropic
from dotenv import load_dotenv, find_dotenv

from .types import ModelClient


class AnthropicChatClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        # Load .env if present
        _dotenv_path = find_dotenv(usecwd=True)
        if _dotenv_path:
            load_dotenv(_dotenv_path)
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        # Anthropic expects system as separate param and content as a list of blocks
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or None,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(part.text for part in msg.content)


