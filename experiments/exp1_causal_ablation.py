"""
Experiment 1: Causal Ablation

Scaffold script to run causal ablation experiments on processed datasets.

Supports input as JSON (.json), CSV (.csv), or Parquet (.parquet), and
output as JSON (.json) or CSV (.csv).

Usage:
  python experiments/exp1_causal_ablation.py --data data/processed/medqa.csv --model gpt-5 --out results/exp1_medqa.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so `models` is importable when running as a script
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from models import create_model_client
from utils.output_paths import compute_output_path


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
    if suffix in {".parquet", ".parq"}:
        # Requires pyarrow or fastparquet
        df = pd.read_parquet(data_path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported data format: {suffix}. Use .json, .csv, or .parquet")


def run_causal_ablation(examples: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    client = create_model_client(model_name)
    for ex in examples:
        # Generic prompt over structured QA fields
        question = ex.get("question")
        context = ex.get("context", "")
        options = ex.get("options")
        system_prompt = "You are a careful medical assistant. Answer faithfully."
        prompt_parts = [f"Question: {question}"]
        if context:
            prompt_parts.append(f"Context: {context}")
        if options:
            prompt_parts.append(f"Options: {options}")
        prompt = "\n".join(prompt_parts)

        try:
            completion = client.generate(prompt, temperature=0.0, max_tokens=256, system_prompt=system_prompt)
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
    parser.add_argument("--model", type=str, default="gemini")
    parser.add_argument("--out", type=Path, required=False, help="Optional output path. If omitted or a directory, outputs go to results/exp1/<model>/<timestamp>.csv with a .config.json alongside.")
    parser.add_argument("--seed", type=int, default=0)
    # Timestamped filenames are now default; no explicit versioning flag needed
    args = parser.parse_args()

    explicit_ext = (args.out.suffix if (args.out and args.out.suffix) else None)
    output_path = compute_output_path(
        out_arg=args.out if args.out else None,
        exp_name="exp1_causal",
        model_name=args.model,
        data_path=args.data,
        force_version=False,
        organize_by_model=True,
        organize_by_experiment=True,
        default_extension=".csv",
        explicit_extension=explicit_ext,
        append_timestamp=True,
    )

    cfg = Exp1Config(data_path=args.data, model_name=args.model, output_path=output_path, seed=args.seed)
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


