"""
Experiment 3: Hint Injection
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `models` is importable when running as a script
sys.path.append(str(_Path(__file__).resolve().parents[1]))


@dataclass
class Exp3Config:
    data_path: Path
    model_name: str
    output_path: Path
    seed: int = 0


def load_examples(data_path: Path) -> List[Dict[str, Any]]:
    with data_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_hint_injection(examples: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for ex in examples:
        results.append({
            "id": ex.get("id"),
            "model": model_name,
            "prediction": None,
            "meta": {"note": "TODO: implement hint injection"},
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Exp3Config(data_path=args.data, model_name=args.model, output_path=args.out, seed=args.seed)
    examples = load_examples(cfg.data_path)
    results = run_hint_injection(examples, cfg.model_name)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__, "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


