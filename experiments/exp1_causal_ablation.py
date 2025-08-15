"""
Experiment 1: Causal Ablation

Scaffold script to run causal ablation experiments on processed datasets.

Supports input and output as either JSON (.json) or CSV (.csv).

Usage:
  python experiments/exp1_causal_ablation.py --data data/processed/medqa.csv --model gpt-4o --out experiments/outputs/exp1_medqa.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from models import create_model_client


@dataclass
class Exp1Config:
    data_path: Path
    model_name: str
    output_path: Path
    seed: int = 0


def load_examples(data_path: Path) -> List[Dict[str, Any]]:
    suffix = data_path.suffix.lower()
    if suffix == ".json":
        with data_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".csv":
        df = pd.read_csv(data_path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported data format: {suffix}. Use .json or .csv")


def run_causal_ablation(examples: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    client = create_model_client(model_name)
    for ex in examples:
        # TODO: implement proper ablation. For now, simple prompt over question/context/options
        question = ex.get("question")
        context = ex.get("context", "")
        options = ex.get("options")
        prompt_parts = [
            "You are a careful medical assistant. Answer faithfully.",
            f"Question: {question}",
        ]
        if context:
            prompt_parts.append(f"Context: {context}")
        if options:
            prompt_parts.append(f"Options: {options}")
        prompt = "\n".join(prompt_parts)
        try:
            completion = client.generate(prompt, temperature=0.0, max_tokens=256)
        except Exception as e:  # keep experiments running
            completion = f"ERROR: {e}"
        results.append({
            "id": ex.get("id"),
            "model": model_name,
            "prediction": completion,
            "meta": {"note": "baseline prompt", "ablation": None},
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = Exp1Config(data_path=args.data, model_name=args.model, output_path=args.out, seed=args.seed)
    examples = load_examples(cfg.data_path)
    results = run_causal_ablation(examples, cfg.model_name)

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suffix = cfg.output_path.suffix.lower()
    if out_suffix == ".json":
        with cfg.output_path.open("w", encoding="utf-8") as f:
            json.dump({"config": cfg.__dict__, "results": results}, f, ensure_ascii=False, indent=2)
    elif out_suffix == ".csv":
        # Flatten results for CSV; serialize nested fields as JSON strings
        flat_rows: List[Dict[str, Any]] = []
        for r in results:
            flat_rows.append({
                "id": r.get("id"),
                "model": r.get("model"),
                "prediction": r.get("prediction"),
                "meta": json.dumps(r.get("meta", {}), ensure_ascii=False),
            })
        pd.DataFrame(flat_rows).to_csv(cfg.output_path, index=False)
        # Save config alongside CSV
        with cfg.output_path.with_suffix(".config.json").open("w", encoding="utf-8") as f:
            json.dump({"config": cfg.__dict__}, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {out_suffix}. Use .json or .csv")


if __name__ == "__main__":
    main()


