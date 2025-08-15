from __future__ import annotations

import os
from typing import Optional

try:
    import openai  # xAI Grok compatible OpenAI client interface
except Exception:  # pragma: no cover
    openai = None  # type: ignore

from .types import ModelClient


class XaiGrokClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        # Grok uses OpenAI-compatible API at different base URL
        self.model = model or os.getenv("XAI_MODEL", "grok-2-latest")
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        if openai is None:
            raise ImportError("openai package not installed. Add 'openai' to requirements.")
        openai.api_key = self.api_key
        openai.base_url = self.base_url

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp["choices"][0]["message"]["content"]


