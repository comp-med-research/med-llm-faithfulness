from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from .types import ModelClient


class OpenAIChatClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        # Load .env if present
        _dotenv_path = find_dotenv(usecwd=True)
        if _dotenv_path:
            load_dotenv(_dotenv_path)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


