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
        self.model = model or os.getenv("OPENAI_MODEL")
        print(f"Using model: {self.model}")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"instructions": system_prompt})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.responses.create(
            model=self.model,
            input=messages,
        )
        return resp.output_text or ""


