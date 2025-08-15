from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

from .types import ModelClient


class OpenAIChatClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI is None:
            raise ImportError("openai package not installed. Add 'openai' to requirements.")
        self.client = OpenAI(api_key=self.api_key)  # type: ignore

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


