from __future__ import annotations

import os
from typing import Dict

from .types import ModelClient
from .clients_openai import OpenAIChatClient
from .clients_anthropic import AnthropicChatClient
from .clients_google import GoogleGeminiClient


def create_model_client(name: str) -> ModelClient:
    key = name.lower()
    if key in {"openai", "chatgpt", "gpt"}:
        return OpenAIChatClient()
    if key in {"anthropic", "claude"}:
        return AnthropicChatClient()
    if key in {"google", "gemini"}:
        return GoogleGeminiClient()
    raise ValueError(f"Unknown model provider: {name}")


