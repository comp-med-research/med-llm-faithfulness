"""
Experiment 4: Map survey responses to model options using mapping CSVs.

Inputs (in a directory):
  - Four mapping CSVs (Doctors 1–15, Doctors 16–30, Layperson 1–15, Layperson 16–30)
    schema: case_num, case_id, answer_label, model, role
  - Four form response CSVs with columns like:
    Timestamp, Name, Role, "Case 1 — Answer A — Logical consistency", ...

Outputs:
  - doctors_mapped.csv (analysis-ready)
  - laypeople_mapped.csv (analysis-ready)

Each output row corresponds to: cohort × respondent × case × answer_label
and contains the per-answer question ratings plus the mapped model.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


HEADER_RE = re.compile(r"^Case\s*(?P<case>\d+)\s*—\s*Answer\s*(?P<label>[A-Z])\s*—\s*(?P<question>.+)$")


def _detect_batch_from_name(path: str) -> str:
    name = os.path.basename(path)
    if "1-15" in name:
        return "1-15"
    if "16-30" in name or "16–30" in name:
        return "16-30"
    return "unknown"


def _detect_cohort_from_name(path: str) -> str:
    name = os.path.basename(path).lower()
    if "doctor" in name:
        return "Doctor"
    if "layperson" in name or "laypeople" in name:
        return "Layperson"
    return "Unknown"


def _load_mapping(mapping_csv: str) -> pd.DataFrame:
    df = pd.read_csv(mapping_csv)
    # Normalize columns and types
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "case_num" in df.columns:
        df["case_num"] = pd.to_numeric(df["case_num"], errors="coerce").astype("Int64")
    if "answer_label" in df.columns:
        df["answer_label"] = df["answer_label"].astype(str).str.strip().str.upper()
    # Pretty display model names
    def _pretty(m: str) -> str:
        low = str(m).lower()
        if "chatgpt" in low:
            return "ChatGPT-5"
        if "claude" in low:
            return "Claude 4.1 Opus"
        if "gemini" in low:
            return "Gemini Pro 2.5"
        return str(m)
    df["model_display"] = df["model"].apply(_pretty) if "model" in df.columns else pd.NA
    return df


def _reshape_responses(form_csv: str, cohort: str, batch: str) -> pd.DataFrame:
    raw = pd.read_csv(form_csv)
    raw = raw.rename(columns={c: c.strip() for c in raw.columns})

    # Identify respondent metadata columns (keep if present)
    meta_cols = [c for c in ["Timestamp", "Name", "Role", "Email", "ID"] if c in raw.columns]

    # Collect triplets: (case_num, label, question) for all matching columns
    parsed_cols: List[Tuple[str, str, str, str]] = []  # (col_name, case, label, question)
    for col in raw.columns:
        m = HEADER_RE.match(str(col))
        if m:
            parsed_cols.append((col, m.group("case"), m.group("label").upper(), m.group("question").strip()))

    # Build long-format rows per respondent, case, label
    records: List[Dict] = []
    for idx, row in raw.iterrows():
        respondent_id = row.get("Name", idx)
        timestamp = row.get("Timestamp", pd.NA)
        role = row.get("Role", pd.NA)
        # We accumulate values per (case, label)
        grouped: Dict[Tuple[int, str], Dict[str, object]] = {}
        for col, case_str, label, question in parsed_cols:
            val = row[col]
            try:
                case_num = int(case_str)
            except Exception:
                continue
            key = (case_num, label)
            if key not in grouped:
                grouped[key] = {}
            grouped[key][question] = val
        # Emit rows
        for (case_num, label), qvals in grouped.items():
            rec = {
                "cohort": cohort,
                "batch": batch,
                "respondent_id": respondent_id,
                "timestamp": timestamp,
                "role": role,
                "case_num": case_num,
                "answer_label": label,
            }
            rec.update(qvals)
            records.append(rec)

    return pd.DataFrame(records)


def _map_cohort(mapping_files: List[str], response_files: List[str]) -> pd.DataFrame:
    # Load and concat mappings
    maps = [_load_mapping(p) for p in mapping_files]
    map_df = pd.concat(maps, ignore_index=True) if maps else pd.DataFrame()
    # Prepare responses
    resp_frames: List[pd.DataFrame] = []
    for p in response_files:
        cohort = _detect_cohort_from_name(p)
        batch = _detect_batch_from_name(p)
        df = _reshape_responses(p, cohort=cohort, batch=batch)
        df["answer_label"] = df["answer_label"].astype(str).str.upper()
        resp_frames.append(df)
    resp_df = pd.concat(resp_frames, ignore_index=True) if resp_frames else pd.DataFrame()

    if resp_df.empty:
        return resp_df

    # Merge mapping → add model + model_display + case_id if present
    on_cols = ["case_num", "answer_label"]
    keep_cols = [c for c in ["model", "model_display", "case_id"] if c in map_df.columns]
    mapped = resp_df.merge(map_df[on_cols + keep_cols], on=on_cols, how="left")

    # Convert Yes/No columns to 1/0 only for pure Yes/No fields
    def _convert_yes_no_columns(df: pd.DataFrame) -> pd.DataFrame:
        yesno = {"yes", "no"}
        for col in df.columns:
            if df[col].dtype == object:
                norm = df[col].astype(str).str.strip().str.lower()
                uniq = set(u for u in norm.unique() if u not in {"", "nan", "none", "na"})
                if uniq and uniq.issubset(yesno):
                    df[col] = norm.map({"yes": 1, "no": 0}).astype("Int64")
        return df

    mapped = _convert_yes_no_columns(mapped)

    # Drop columns per request and rename model_display -> model
    drop_cols = ["cohort", "batch", "timestamp", "case_id", "model", "answer_label"]
    mapped = mapped.drop(columns=[c for c in drop_cols if c in mapped.columns], errors="ignore")
    if "model_display" in mapped.columns:
        mapped = mapped.rename(columns={"model_display": "model"})

    return mapped


def _find_files(in_dir: str, cohort_keyword: str, kind_keyword: str) -> List[str]:
    out: List[str] = []
    for name in os.listdir(in_dir):
        if cohort_keyword.lower() in name.lower() and kind_keyword.lower() in name.lower() and name.lower().endswith(".csv"):
            out.append(os.path.join(in_dir, name))
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Map Exp4 responses to model options using mapping CSVs.")
    parser.add_argument("--indir", required=True, help="Directory containing mapping and form response CSVs")
    parser.add_argument("--outdir", required=True, help="Directory to write mapped CSVs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Doctors
    doc_maps = _find_files(args.indir, cohort_keyword="Doctor", kind_keyword="Mapping")
    doc_forms = _find_files(args.indir, cohort_keyword="Doctor", kind_keyword="Form Responses")
    doc_mapped = _map_cohort(doc_maps, doc_forms)
    doctors_out = os.path.join(args.outdir, "doctors_mapped.csv")
    if not doc_mapped.empty:
        doc_mapped.to_csv(doctors_out, index=False)

    # Laypeople
    lay_maps = _find_files(args.indir, cohort_keyword="Layperson", kind_keyword="Mapping")
    lay_forms = _find_files(args.indir, cohort_keyword="Layperson", kind_keyword="Form Responses")
    lay_mapped = _map_cohort(lay_maps, lay_forms)
    lay_out = os.path.join(args.outdir, "laypeople_mapped.csv")
    if not lay_mapped.empty:
        lay_mapped.to_csv(lay_out, index=False)

    # Print paths for convenience
    existing = [p for p in [doctors_out, lay_out] if os.path.exists(p)]
    if existing:
        print("Wrote:\n" + "\n".join(existing))
    else:
        print("No outputs were generated. Check input directory contents.")


if __name__ == "__main__":
    main()


