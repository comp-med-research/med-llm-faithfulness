"""
Normalization helpers for MedQA-like datasets shared across experiments.

Exports:
- normalize_columns_case_insensitive: normalize various schemas to id, question, A..E, label
- standardize_gold_label: parse/convert labels to a single letter A..E
"""

from __future__ import annotations

import re
import ast
from typing import Any, Dict, List, Optional

import pandas as pd


SUPPORTED_ANSWER_LETTERS = {"A", "B", "C", "D", "E"}


def normalize_columns_case_insensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize various MedQA-like schemas to: id, question, A, B, C, D, E, label.

    Supported inputs:
    - Canonical columns (case-insensitive): id, question, A, B, C, D, E, label/answer/gold/correct
    - Alternative schema: question, options, answer_idx and/or answer
      where options is either:
        * a list of dicts with keys {"key": "A|B|C|D|E", "value": <text>} (preferred), or
        * a JSON/list-like or delimited string of option texts (A..E in order)
    """
    df_in = df.copy()
    col_map = {c: c.lower().strip() for c in df.columns}
    lower_to_orig: Dict[str, str] = {}
    for orig, lower in col_map.items():
        lower_to_orig.setdefault(lower, orig)

    def pick(*candidates: str) -> Optional[str]:
        for c in candidates:
            if c in lower_to_orig:
                return lower_to_orig[c]
        return None

    id_col = pick("id")
    q_col = pick("question", "prompt", "stem", "text")
    a_col = pick("a")
    b_col = pick("b")
    c_col = pick("c")
    d_col = pick("d")
    e_col = pick("e")
    label_col = pick("label", "answer", "gold", "correct")
    options_col = pick("options")
    answer_idx_col = pick("answer_idx", "answer_index", "answerid", "answer_id", "label_idx")
    answer_text_col = pick("answer", "answer_text")

    # Case 1: canonical columns present → simple rename and return (with coercion)
    has_canonical_choices = a_col and b_col and c_col and d_col
    if has_canonical_choices and q_col is not None and label_col is not None:
        rename_map = {
            id_col: "id" if id_col else None,
            q_col: "question",
            a_col: "A",
            b_col: "B",
            c_col: "C",
            d_col: "D",
            e_col: "E" if e_col else None,
            label_col: "label",
        }
        rename_map = {k: v for k, v in rename_map.items() if k is not None}
        df = df.rename(columns=rename_map)
        if "id" not in df.columns:
            df["id"] = range(1, len(df) + 1)
        if "E" not in df.columns:
            df["E"] = ""
        # Coerce values to plain strings minimally
        for col in ["A", "B", "C", "D", "E"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: "" if pd.isna(x) else str(x).strip())
        # If original had answer_idx, prefer it as label as requested
        if answer_idx_col is not None and answer_idx_col in df_in.columns:
            df["label"] = df_in[answer_idx_col]
        return df[["id", "question", "A", "B", "C", "D", "E", "label"]]

    # Case 2: options schema → expand to A..E, create label from answer_idx/answer
    if q_col is None or options_col is None:
        raise ValueError(
            f"Missing required columns. Need either canonical (id,question,A..E,label) or alt schema (question,options,[answer_idx|answer]). Found: {list(df.columns)}"
        )

    q_series = df[lower_to_orig[q_col]] if q_col in lower_to_orig else df[q_col]
    opts_series = df[options_col]

    def to_list_of_dicts(x: Any) -> List[Dict[str, Any]]:
        if isinstance(x, list):
            return x
        if pd.isna(x):
            return []
        s = str(x).strip()
        if not s:
            return []
        # Tolerate newline-separated dicts and adjacent dicts without commas
        s2 = s.replace("\n", ",")
        s2 = re.sub(r"}\s*{", "}, {", s2)
        try:
            val = ast.literal_eval(s2)
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, dict):
                return [val]
        except Exception:
            pass
        # Fallback: extract each { ... } chunk and parse individually
        chunks = re.findall(r"\{[^}]*\}", s)
        out_list: List[Dict[str, Any]] = []
        for c in chunks:
            try:
                d = ast.literal_eval(c)
                if isinstance(d, dict):
                    out_list.append(d)
            except Exception:
                pass
        return out_list

    def to_options_map(x: Any) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for d in to_list_of_dicts(x):
            if isinstance(d, dict):
                k = str(d.get("key", "")).strip().upper()
                v = "" if d.get("value") is None else str(d.get("value")).strip()
                if k in SUPPORTED_ANSWER_LETTERS:
                    out[k] = v
        return out

    wide = opts_series.apply(to_options_map).apply(pd.Series)
    for letter in ["A", "B", "C", "D", "E"]:
        if letter not in wide.columns:
            wide[letter] = ""
    wide = wide[["A", "B", "C", "D", "E"]].fillna("")

    # Determine labels
    ans_idx_series = df[answer_idx_col] if answer_idx_col else None
    ans_text_series = df[answer_text_col] if answer_text_col else None

    def _norm_text(x: Any) -> str:
        return re.sub(r"\s+", " ", str(x).strip().lower()) if x is not None else ""

    if ans_idx_series is not None:
        labels_series = ans_idx_series.apply(standardize_gold_label)
    elif ans_text_series is not None:
        def infer(row_idx: int) -> Optional[str]:
            raw = ans_text_series.iloc[row_idx]
            letter = standardize_gold_label(raw)
            if letter in SUPPORTED_ANSWER_LETTERS:
                return letter
            norm_ans = _norm_text(raw)
            opts_map = {
                "A": _norm_text(wide.iloc[row_idx]["A"]),
                "B": _norm_text(wide.iloc[row_idx]["B"]),
                "C": _norm_text(wide.iloc[row_idx]["C"]),
                "D": _norm_text(wide.iloc[row_idx]["D"]),
                "E": _norm_text(wide.iloc[row_idx]["E"]),
            }
            for k, v in opts_map.items():
                if v and v == norm_ans:
                    return k
            return None
        labels_series = pd.Series([infer(i) for i in range(len(df))])
    else:
        labels_series = pd.Series([None] * len(df))

    out = pd.DataFrame({
        "question": q_series,
        "A": wide["A"],
        "B": wide["B"],
        "C": wide["C"],
        "D": wide["D"],
        "E": wide["E"],
        "label": labels_series,
    })
    out.insert(0, "id", range(1, len(out) + 1))
    return out[["id", "question", "A", "B", "C", "D", "E", "label"]]


def standardize_gold_label(value: Any) -> Optional[str]:
    """Return a single uppercase letter A-E if possible, otherwise None.

    Accepts letters, strings that contain A-E, or integer-like values (1-5).
    """
    if value is None:
        return None
    try:
        # If numeric-like 1..5 → A..E
        if isinstance(value, (int, float)) and not pd.isna(value):
            i = int(value)
            if i in {1, 2, 3, 4, 5}:
                return ["A", "B", "C", "D", "E"][i - 1]
        # Try parsing string
        s = str(value).strip().upper()
        # If the string itself is A-E
        if s in SUPPORTED_ANSWER_LETTERS:
            return s
        # Extract first occurrence of A-E as standalone or with punctuation
        m = re.search(r"\b([ABCDE])\b", s)
        if m:
            return m.group(1)
        # Sometimes labels like '1'..'5' in string form
        if s in {"1", "2", "3", "4", "5"}:
            return ["A", "B", "C", "D", "E"][int(s) - 1]
    except Exception:
        return None
    return None


