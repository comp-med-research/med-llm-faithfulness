from __future__ import annotations

import os
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

from .types import ModelClient

# Load environment variables from a .env file if present (searches upwards)
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path)


class GoogleGeminiClient:
    """Google Gemini client compatible with experiment interface.

    Uses the modern `google-genai` SDK.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self.model = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-2.0-flash-001") or model
        print(f"Using model: {self.model}") 
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        # v1beta currently provides the text generation surface
        self.client = genai.Client(api_key=self.api_key, http_options={"api_version": "v1beta"})

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from Gemini.

        Args:
            prompt: User prompt content.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            system_prompt: Optional system instruction; will be prepended.

        Returns:
            Model response text.
        """
        content = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = self.client.models.generate_content(
            model=self.model,
            contents=content
        )
        
        return (getattr(resp, "text", None) or "").strip()

