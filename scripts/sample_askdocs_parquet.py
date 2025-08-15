"""
Sample a small subset of AskDocs data from a Parquet file.

Usage examples:
  python scripts/sample_askdocs_parquet.py --input data/raw/askdocs/askdocs_train_en.parquet
  python scripts/sample_askdocs_parquet.py --input data/raw/askdocs/askdocs_train_en.parquet --n 5 --seed 123 --out data/raw/askdocs/askdocs_sample.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample N rows from an AskDocs Parquet file")
    parser.add_argument("--input", type=Path, required=True, help="Path to source Parquet file")
    parser.add_argument("--out", type=Path, default=Path("data/raw/askdocs/askdocs_sample.parquet"), help="Output Parquet path")
    parser.add_argument("--n", type=int, default=5, help="Number of rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input Parquet not found: {args.input}")

    df = pd.read_parquet(args.input)
    if df.empty:
        raise ValueError("Input Parquet is empty; nothing to sample")

    n = min(args.n, len(df))
    sampled = df.sample(n=n, random_state=args.seed) if len(df) > n else df.copy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.out, index=False)
    print(f"Wrote sample of {len(sampled)} rows to {args.out}")


if __name__ == "__main__":
    main()


