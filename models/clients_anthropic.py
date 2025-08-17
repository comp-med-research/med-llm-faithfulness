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
        self.model = model or os.getenv("ANTHROPIC_MODEL")
        print(f"Using model: {self.model}")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        # Anthropic expects system as separate param and content as a list of blocks
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(part.text for part in msg.content)


