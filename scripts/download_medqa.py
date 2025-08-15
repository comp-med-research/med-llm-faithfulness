"""
Download and normalize the MedQA dataset into the project layout.

Raw CSVs will be saved under `data/raw/medqa/medqa_<split>.csv`.
Processed CSVs (schema: id,question,context,options,answer) under
`data/processed/medqa/medqa_<split>.csv`.

Usage:
  python scripts/download_medqa.py --splits train validation test
  python scripts/download_medqa.py --splits train --max-rows 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize MedQA dataset")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Dataset splits to download (e.g., train validation test)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/medqa"),
        help="Directory to write raw CSV files",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/medqa"),
        help="Directory to write processed CSV files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openlifescienceai/medqa",
        help="Hugging Face dataset id",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If > 0, limit rows per split for a smaller sample",
    )
    return parser.parse_args()


def load_medqa_split(dataset_id: str, split: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset  # lazy import
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_id, split=split)
    return pd.DataFrame(ds)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stringify_options(options_value: Any) -> str:
    if options_value is None:
        return ""
    if isinstance(options_value, str):
        # If already a JSON array string, keep it; else try to split by a common delimiter
        s = options_value.strip()
        if s.startswith("[") and s.endswith("]"):
            return s
        parts = [p.strip() for p in s.split("||") if p.strip()]
        return json.dumps(parts, ensure_ascii=False)
    if isinstance(options_value, (list, tuple)):
        return json.dumps(list(options_value), ensure_ascii=False)
    # Fallback to string
    return json.dumps([str(options_value)], ensure_ascii=False)


def to_label_from_index(idx_value: Any) -> Optional[str]:
    if idx_value is None:
        return None
    try:
        idx = int(idx_value)
        if idx < 0:
            return None
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[idx]
    except Exception:
        return None


def normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    def normalize_row(row: pd.Series) -> Dict[str, Any]:
        question = row.get("question") or row.get("question_text") or row.get("question_str")
        context = row.get("context") or row.get("passage") or ""
        options_raw = row.get("options") or row.get("choices") or row.get("options_str")
        options = stringify_options(options_raw)

        # gold answer can be a label (e.g., "A") or index (e.g., 1)
        answer_label = row.get("answer") or row.get("answer_label")
        if not answer_label:
            answer_label = to_label_from_index(row.get("answer_idx"))

        return {
            "id": row.get("id", row.name),
            "question": question,
            "context": context,
            "options": options,
            "answer": answer_label,
        }

    records = [normalize_row(r) for _, r in df.iterrows()]
    return pd.DataFrame.from_records(records, columns=["id", "question", "context", "options", "answer"])


def process_split(dataset_id: str, split: str, raw_dir: Path, processed_dir: Path, max_rows: int = 0) -> None:
    print(f"Downloading split: {split}")
    df = load_medqa_split(dataset_id, split)
    if max_rows and max_rows > 0:
        df = df.head(max_rows)

    ensure_dir(raw_dir)
    raw_path = raw_dir / f"medqa_{split}.csv"
    df.to_csv(raw_path, index=False)
    print(f"Saved raw: {raw_path}")

    ensure_dir(processed_dir)
    norm_df = normalize_rows(df)
    proc_path = processed_dir / f"medqa_{split}.csv"
    norm_df.to_csv(proc_path, index=False)
    print(f"Saved processed: {proc_path}")


def main() -> None:
    args = parse_args()
    for split in args.splits:
        process_split(
            dataset_id=args.dataset,
            split=split,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            max_rows=args.max_rows,
        )


if __name__ == "__main__":
    main()


