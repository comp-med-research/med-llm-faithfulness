"""
Experiment 4: Real-World Evaluation

Evaluate models on real-world forum-style data (e.g., AskDocs).
Requires each example to include non-empty 'system_prompt', 'title', and 'selftext'.
Supports JSON/CSV/Parquet input; JSON/CSV output.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterator
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path as _Path
import re
import datetime as _dt

# Ensure project root is on sys.path so `models` is importable when running as a script
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from models import create_model_client
from utils.output_paths import compute_output_path


@dataclass
class Exp4Config:
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
        df = pd.read_parquet(data_path)
        return df.to_dict(orient="records")
    raise ValueError(f"Unsupported data format: {suffix}. Use .json, .csv, or .parquet")


def _serialize_config(cfg: "Exp4Config") -> Dict[str, Any]:
    """Convert config to a JSON-serializable dict (e.g., stringify Paths)."""
    return {
        "data_path": str(cfg.data_path),
        "model_name": cfg.model_name,
        "output_path": str(cfg.output_path),
        "seed": cfg.seed,
    }


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


# def _filter_valid_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     valid: List[Dict[str, Any]] = []
#     skipped = 0
#     for ex in examples:
#         if all(_is_non_empty_string(ex.get(k)) for k in ("title_clean", "selftext_clean")):
#             valid.append(ex)
#         else:
#             skipped += 1
#     if skipped:
#         print(f"[exp4] Skipped {skipped} examples missing required fields (title/selftext)")
#     return valid


def iter_realworld_eval(examples: List[Dict[str, Any]], model_name: str) -> Iterator[Dict[str, Any]]:
    """Yield results one-by-one to enable chunked/streamed output writing."""
    client = create_model_client(model_name)
    valid_examples = examples
    for ex in tqdm(valid_examples, total=len(valid_examples), desc="Exp4 generation"):
        title = ex["title_clean"]
        selftext = ex["selftext_clean"]

        user_parts = [
            "The user posted the following question on a medical forum.",
            f"Title: {title}",
            f"Post: {selftext}",
            """Please provide an answer in plain text (no lists, no markdown).
            Response should be concise, and to the point; do not provide any
            unnecessary details."""
        ]
        prompt = "\n".join(user_parts)

        try:
            completion = client.generate(prompt)
        except Exception as e:
            completion = f"ERROR: {e}"

        completion = clean_prediction_text(completion)

        yield {
            "id": ex.get("q_id"),
            "model": model_name,
            "prediction": completion,
            "post_title": title,
            "post_text": selftext,
            "meta": {"note": "real-world eval"},
        }



def clean_prediction_text(text: str) -> str:
    """Normalize model output to a single paragraph without artifacts."""
    if not isinstance(text, str):
        return ""   
    s = text.strip()
    # Remove accidental pandas Series footer if present
    s = re.sub(r"Name:\s*prediction,\s*dtype:\s*object\s*$", "", s, flags=re.I | re.M)
    # Drop line-number prefixes like "0    " at line starts
    s = re.sub(r"^\s*\d+\s+", "", s, flags=re.M)
    # Remove markdown bullets/emphasis
    s = re.sub(r"^[\-\*•]\s+", "", s, flags=re.M)
    s = s.replace("**", "")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=str, default="gemini")
    parser.add_argument("--out", type=Path, required=False, help="Optional output path. If omitted or a directory, outputs go to results/exp4/<model>/exp4_askdocs_<data>_<model>[_vNNN].csv with a .config.json alongside.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=200, help="Number of rows per CSV part file for chunked writing (only for CSV outputs).")
    # Timestamped filenames are now default; no flag needed
    # Numeric versioning retained for backward compat via utility's force flag (not exposed here)
    # Always organize by experiment and model to keep results separated
    args = parser.parse_args()

    ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    # Determine explicit extension from user-provided --out (if any)
    explicit_ext = (args.out.suffix if (args.out and args.out.suffix) else None)
    # Compute a non-overwriting output path. If user provided a filename, we keep it.
    # If a directory or omitted, we build results/exp4/<model>/<timestamp>.csv
    output_path = compute_output_path(
        out_arg=args.out if args.out else None,
        exp_name="exp4_askdocs",
        model_name=args.model,
        data_path=args.data,
        force_version=False,
        organize_by_model=True,
        organize_by_experiment=True,
        default_extension=".csv",
        explicit_extension=explicit_ext,
        append_timestamp=True,
    )
    # Timestamp already included by compute_output_path

    cfg = Exp4Config(data_path=args.data, model_name=args.model, output_path=output_path, seed=args.seed)
    examples = load_examples(cfg.data_path)
    meta = {"timestamp_utc": ts, "num_input": len(examples)}

    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    out_suffix = cfg.output_path.suffix.lower()
    if out_suffix == ".json":
        # For JSON, we still write a single file to preserve structure
        results_list = list(iter_realworld_eval(examples, cfg.model_name))
        meta["num_output"] = len(results_list)
        with cfg.output_path.open("w", encoding="utf-8") as f:
            json.dump({"config": _serialize_config(cfg), "meta": meta, "results": results_list}, f, ensure_ascii=False, indent=2)
    elif out_suffix == ".csv":
        # Chunked CSV writing: writes part files like <stem>.part0001.csv
        part_files: List[str] = []
        chunk_buffer: List[Dict[str, Any]] = []
        part_index = 1
        stem = cfg.output_path.with_suffix("")

        def write_part(rows: List[Dict[str, Any]], idx: int) -> None:
            part_path = stem.parent / f"{stem.name}.part{idx:04d}.csv"
            df_part = pd.DataFrame(rows)
            df_part.to_csv(part_path, index=False)
            part_files.append(str(part_path))

        num_output = 0
        for r in iter_realworld_eval(examples, cfg.model_name):
            flat_row = {
                "id": r.get("id"),
                "model": r.get("model"),
                "prediction": r.get("prediction"),
                "post_title": r.get("post_title", ""),
                "post_text": r.get("post_text", ""),
                "meta": json.dumps(r.get("meta", {}), ensure_ascii=False),
            }
            chunk_buffer.append(flat_row)
            num_output += 1
            if len(chunk_buffer) >= args.chunk_size:
                write_part(chunk_buffer, part_index)
                chunk_buffer = []
                part_index += 1

        if chunk_buffer:
            write_part(chunk_buffer, part_index)

        meta["num_output"] = num_output
        meta["part_files"] = part_files
        with cfg.output_path.with_suffix(".config.json").open("w", encoding="utf-8") as f:
            json.dump({"config": _serialize_config(cfg), "meta": meta}, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {out_suffix}. Use .json or .csv")


if __name__ == "__main__":
    main()


