from __future__ import annotations

import os
from typing import Optional

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore

from .types import ModelClient


class GoogleGeminiClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = model or os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-pro")
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if genai is None:
            raise ImportError("google-generativeai package not installed. Add 'google-generativeai' to requirements.")
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt: str, *, temperature: float = 0.0, max_tokens: int = 512, system_prompt: Optional[str] = None) -> str:
        # google SDK uses different params; we pass temperature and rely on defaults for token limits
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = self.client.generate_content(full_prompt, generation_config={"temperature": temperature})
        return getattr(resp, "text", "")


