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
from utils.output_paths import compute_output_path


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
        post_title = (
            ex.get("title")
            or ex.get("title_clean")
            or ex.get("question")
            or ""
        )
        post_text = (
            ex.get("selftext")
            or ex.get("selftext_clean")
            or ex.get("context")
            or ""
        )
        results.append({
            "id": ex.get("id"),
            "model": model_name,
            "prediction": None,
            "post_title": post_title,
            "post_text": post_text,
            "meta": {"note": "TODO: implement hint injection"},
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--out", type=Path, required=False, help="Optional output path. If omitted or a directory, outputs go to results/exp3/<model>/<timestamp>.csv with a .config.json alongside.")
    parser.add_argument("--seed", type=int, default=0)
    # Timestamped filenames are now default; no explicit versioning flag needed
    args = parser.parse_args()

    explicit_ext = (args.out.suffix if (args.out and args.out.suffix) else None)
    output_path = compute_output_path(
        out_arg=args.out if args.out else None,
        exp_name="exp3_hint",
        model_name=args.model,
        data_path=args.data,
        force_version=False,
        organize_by_model=True,
        organize_by_experiment=True,
        default_extension=".csv",
        explicit_extension=explicit_ext,
        append_timestamp=True,
    )

    cfg = Exp3Config(data_path=args.data, model_name=args.model, output_path=output_path, seed=args.seed)
    examples = load_examples(cfg.data_path)
    results = run_hint_injection(examples, cfg.model_name)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_path.open("w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__, "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


