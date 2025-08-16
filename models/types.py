from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol


class ModelClient(Protocol):
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        ...


@dataclass
class ModelConfig:
    name: str
    api_key_env: str
    model_id: str


