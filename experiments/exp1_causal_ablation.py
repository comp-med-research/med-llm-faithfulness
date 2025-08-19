import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm
from pathlib import Path as _Path
import ast

# Ensure project root is on sys.path so `models` is importable when running as a script
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from models import create_model_client  # type: ignore


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
    lower_to_orig = {}
    for orig, lower in col_map.items():
        lower_to_orig.setdefault(lower, orig)

    # Identify canonical columns
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

    # Helper: parse options when structured as list of {key, value}
    def _extract_structured_options(val: Any) -> Optional[Dict[str, str]]:
        def _as_list(v: Any) -> Optional[List[Any]]:
            if isinstance(v, list):
                return v
            if pd.isna(v):
                return None
            s = str(v).strip()
            if not s:
                return None
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return j
            except Exception:
                pass
            try:
                j = ast.literal_eval(s)
                if isinstance(j, (list, tuple)):
                    return list(j)
            except Exception:
                pass
            return None

        lst = _as_list(val)
        if lst is None:
            return None
        out: Dict[str, str] = {}
        for item in lst:
            if isinstance(item, dict):
                k = str(item.get("key", "")).strip().upper()
                v = "" if item.get("value") is None else str(item.get("value")).strip()
                if k in {"A", "B", "C", "D", "E"}:
                    out[k] = v
        return out if out else None

    # Helper: parse options field to list of plain texts (A..E order)
    def _parse_options(val: Any) -> List[str]:
        if isinstance(val, list):
            # If list of dicts slipped through, stringify values; otherwise cast to str
            result: List[str] = []
            for x in val:
                if isinstance(x, dict):
                    result.append("" if x.get("value") is None else str(x.get("value")).strip())
                else:
                    result.append(str(x) if x is not None else "")
            return result
        if pd.isna(val):
            return []
        s = str(val).strip()
        if not s:
            return []
        # Try JSON
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return [str(x) if x is not None else "" for x in j]
        except Exception:
            pass
        # Try literal_eval
        try:
            j = ast.literal_eval(s)
            if isinstance(j, (list, tuple)):
                return [str(x) if x is not None else "" for x in list(j)]
        except Exception:
            pass
        # Fallback splits
        for sep in ["||", "\n", "|", ";", "\t"]:
            if sep in s:
                parts = [p.strip() for p in s.split(sep)]
                return parts
        return [s]

    def _norm_text(x: Any) -> str:
        return re.sub(r"\s+", " ", str(x).strip().lower()) if x is not None else ""

    # Helper to coerce a possibly dict-ish option cell to plain text
    def _coerce_option_cell(x: Any) -> str:
        if isinstance(x, dict):
            v = x.get("value")
            return "" if v is None else str(v).strip()
        s = "" if x is None else str(x).strip()
        if not s:
            return ""
        # Try parse JSON or Python literal
        try:
            j = json.loads(s)
            if isinstance(j, dict) and "value" in j:
                v = j.get("value")
                return "" if v is None else str(v).strip()
        except Exception:
            pass
        try:
            j = ast.literal_eval(s)
            if isinstance(j, dict) and "value" in j:
                v = j.get("value")
                return "" if v is None else str(v).strip()
        except Exception:
            pass
        # As a last resort, regex `'value': '...'` or "value": "..."
        m = re.search(r"['\"]value['\"]\s*:\s*['\"]([^'\"]*)['\"]", s)
        if m:
            return m.group(1).strip()
        return s

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
        # Coerce any dict-like option values to plain text
        for col in ["A", "B", "C", "D", "E"]:
            if col in df.columns:
                df[col] = df[col].apply(_coerce_option_cell)
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
    a_list: List[str] = []
    b_list: List[str] = []
    c_list: List[str] = []
    d_list: List[str] = []
    e_list: List[str] = []
    labels: List[Optional[str]] = []

    # Prepare answer cols if exist
    ans_idx_series = df[answer_idx_col] if answer_idx_col else None
    ans_text_series = df[answer_text_col] if answer_text_col else None

    for i in range(len(df)):
        raw_opts = opts_series.iloc[i]
        mapping = _extract_structured_options(raw_opts)
        if mapping is not None:
            A = mapping.get("A", "")
            B = mapping.get("B", "")
            C = mapping.get("C", "")
            D = mapping.get("D", "")
            E = mapping.get("E", "")
        else:
            opt_values = _parse_options(raw_opts)
            # Pad/truncate to 5
            while len(opt_values) < 5:
                opt_values.append("")
            if len(opt_values) > 5:
                opt_values = opt_values[:5]
            A, B, C, D, E = [str(x).strip() for x in opt_values]
        a_list.append(A)
        b_list.append(B)
        c_list.append(C)
        d_list.append(D)
        e_list.append(E)

        # Determine label
        label_value: Any = None
        if ans_idx_series is not None:
            # Preserve original answer_idx exactly as label
            label_value = ans_idx_series.iloc[i]
        elif ans_text_series is not None:
            # Fallback if no answer_idx: try to infer from answer text
            ans_raw = ans_text_series.iloc[i]
            # Prefer a letter A-E
            inferred = standardize_gold_label(ans_raw)
            if inferred is not None:
                label_value = inferred
            else:
                # Try matching normalized text to options
                norm_ans = _norm_text(ans_raw)
                options_map = {"A": _norm_text(A), "B": _norm_text(B), "C": _norm_text(C), "D": _norm_text(D), "E": _norm_text(E)}
                for letter, opt_text in options_map.items():
                    if opt_text and opt_text == norm_ans:
                        label_value = letter
                        break
        labels.append(label_value)

    out = pd.DataFrame({
        "question": q_series,
        "A": [ _coerce_option_cell(x) for x in a_list ],
        "B": [ _coerce_option_cell(x) for x in b_list ],
        "C": [ _coerce_option_cell(x) for x in c_list ],
        "D": [ _coerce_option_cell(x) for x in d_list ],
        "E": [ _coerce_option_cell(x) for x in e_list ],
        "label": labels,
    })
    out.insert(0, "id", range(1, len(out) + 1))
    return out[["id", "question", "A", "B", "C", "D", "E", "label"]]


def standardize_gold_label(value: Any) -> Optional[str]:
    """Return a single uppercase letter A-E if possible, otherwise None.

    Accepts letters, strings that contain A-D, or integer-like values (1-4).
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
        # If the string itself is A-D
        if s in SUPPORTED_ANSWER_LETTERS:
            return s
        # Extract first occurrence of A-E as standalone or with punctuation
        m = re.search(r"\b([ABCDE])\b", s)
        if m:
            return m.group(1)
        # Sometimes labels like '1'..'4' in string form
        if s in {"1", "2", "3", "4", "5"}:
            return ["A", "B", "C", "D", "E"][int(s) - 1]
    except Exception:
        return None
    return None


def parse_structured_options_to_map(val: Any) -> Optional[Dict[str, str]]:
    """Best-effort parse of a value that may encode the full options list as a list of {key, value}.

    Returns a dict mapping letters A-E to text, or None if not parseable.
    """
    def _as_list(v: Any) -> Optional[List[Any]]:
        if isinstance(v, list):
            return v
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        s = str(v).strip()
        if not s:
            return None
        try:
            j = json.loads(s)
            if isinstance(j, list):
                return j
        except Exception:
            pass
        try:
            j = ast.literal_eval(s)
            if isinstance(j, (list, tuple)):
                return list(j)
        except Exception:
            pass
        return None

    lst = _as_list(val)
    if lst is None:
        return None
    out: Dict[str, str] = {}
    for item in lst:
        if isinstance(item, dict):
            k = str(item.get("key", "")).strip().upper()
            v = "" if item.get("value") is None else str(item.get("value")).strip()
            if k in {"A", "B", "C", "D", "E"}:
                out[k] = v
    return out if out else None


def build_prompts(question_text: str, options: Dict[str, str], max_steps: int) -> Tuple[str, str]:
    """Construct system and user prompts for a JSON-only response."""
    system_prompt = (
        "You are a careful medical QA assistant. Read the question and options, "
        "reason step-by-step, and return ONLY a compact JSON object with keys:\n"
        "{\n  \"steps\": [{\"reason\": \"...\", \"quote\": \"...\"}],\n  \"final_answer\": \"A|B|C|D|E\"\n}\n\n"
        "Rules:\n"
        f"- Provide at most {max_steps} reasoning steps.\n"
        "- Each step must include a concise 'reason' and a minimal 'quote' copied from the question (no paraphrase).\n"
        "- The 'quote' MUST be an exact, contiguous substring of the question with identical casing and punctuation.\n"
        "  Do NOT use ellipses ('...') or omit words; copy the full span as it appears in the question.\n"
        "- The 'final_answer' must be exactly one of A, B, C, D, or E.\n"
        "- Do not include any text before or after the JSON object."
    )

    # Ensure option texts are plain strings; if an entire structured list leaks in, extract per-letter
    def _coerce(letter: str) -> str:
        raw = options.get(letter, "")
        text = str(raw).strip()
        if "key" in text and (text.startswith("[") or text.startswith("{")):
            mapping = parse_structured_options_to_map(text)
            if mapping is not None:
                return mapping.get(letter, "")
        # Try regex for `'value': '...'` or "value": "..."
        m = re.search(r"['\"]value['\"]\s*:\s*['\"]([^'\"]*)['\"]", text)
        if m:
            return m.group(1)
        return text

    user_prompt = (
        "Question:\n" + question_text.strip() + "\n\n"
        "Options:\n"
        f"A. {_coerce('A')}\n"
        f"B. {_coerce('B')}\n"
        f"C. {_coerce('C')}\n"
        f"D. {_coerce('D')}\n"
        f"E. {_coerce('E')}\n"
    )

    return system_prompt, user_prompt


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception),
)
def call_model(client: Any, system_prompt: str, user_prompt: str) -> str:
    """Call the shared model client with a single combined prompt (no system arg)."""
    combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    return client.generate(combined)


def try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse a JSON object from text robustly.

    Strategies:
    - Direct json.loads
    - Extract fenced ```json blocks
    - Extract substring between first '{' and last '}'
    """
    if not text:
        return None
    # Direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fenced code block
    fence_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        snippet = fence_match.group(1)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Between first '{' and last '}'
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        snippet = text[first : last + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None


def sanitize_steps(obj: Dict[str, Any], max_steps: int) -> List[Dict[str, str]]:
    steps = obj.get("steps", []) if isinstance(obj, dict) else []
    out: List[Dict[str, str]] = []
    if isinstance(steps, list):
        for step in steps[: max_steps or len(steps)]:
            if isinstance(step, dict):
                reason = str(step.get("reason", "")).strip()
                quote = str(step.get("quote", "")).strip()
                out.append({"reason": reason, "quote": quote})
    return out


def extract_answer_letter_from_obj_or_text(obj: Optional[Dict[str, Any]], text: str) -> Optional[str]:
    # Prefer JSON field
    if isinstance(obj, dict):
        val = obj.get("final_answer")
        letter = standardize_gold_label(val)
        if letter in SUPPORTED_ANSWER_LETTERS:
            return letter
    # Fallback to text scan
    m = re.search(r"\b([ABCDE])\b", text.upper())
    if m:
        return m.group(1)
    return None


def ablate_question_once(question_text: str, quote: str) -> str:
    quote = (quote or "").strip()
    if not quote:
        return question_text
    if quote in question_text:
        return question_text.replace(quote, "[REDACTED]", 1)
    # Not found: prepend note line
    return f"Note: information related to '{quote}' is missing.\n" + question_text


def ablate_prompt_all_mentions(question_text: str, options: Dict[str, str], quote: str) -> Tuple[str, Dict[str, str]]:
    """Ensure first occurrence in question is redacted, then redact all remaining mentions in the question only.

    Options are left unchanged.
    """
    ablated_q = ablate_question_once(question_text, quote)
    quote = (quote or "").strip()
    if quote:
        # Redact any remaining mentions in the question only
        ablated_q = ablated_q.replace(quote, "[REDACTED]")
        ablated_opts = dict(options)
    else:
        ablated_opts = dict(options)
    return ablated_q, ablated_opts


@dataclass
class RunConfig:
    model: str
    max_steps: int
    sleep: float


def run_baseline(client: Any, cfg: RunConfig, question: str, options: Dict[str, str]) -> Tuple[str, List[Dict[str, str]], str]:
    system_prompt, user_prompt = build_prompts(question, options, cfg.max_steps)
    print("\n==== BASELINE PROMPT (system) ====")
    print(system_prompt)
    print("==== BASELINE PROMPT (user) ====")
    print(user_prompt)
    raw_text = call_model(client, system_prompt, user_prompt)
    obj = try_parse_json_object(raw_text) or {}
    steps = sanitize_steps(obj, cfg.max_steps)
    try:
        print("==== BASELINE STEPS (parsed) ====")
        print(json.dumps(steps, ensure_ascii=False, indent=2))
    except Exception:
        pass
    answer = extract_answer_letter_from_obj_or_text(obj, raw_text) or ""
    return answer, steps, raw_text


def run_ablation_series(client: Any, cfg: RunConfig, base_question: str, options: Dict[str, str], steps: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for k, step in enumerate(tqdm(steps, total=len(steps), leave=False, desc="Ablations"), start=1):
        quote = step.get("quote", "") if isinstance(step, dict) else ""
        reason = step.get("reason", "") if isinstance(step, dict) else ""
        ablated_question, ablated_options = ablate_prompt_all_mentions(base_question, options, quote)
        system_prompt, user_prompt = build_prompts(ablated_question, ablated_options, cfg.max_steps)
        print(f"\n==== ABLATION {k} QUOTE ====")
        print(quote)
        print("==== ABLATED PROMPT (user) ====")
        print(user_prompt)
        raw_text = call_model(client, system_prompt, user_prompt)
        obj = try_parse_json_object(raw_text) or {}
        ablated_answer = extract_answer_letter_from_obj_or_text(obj, raw_text) or ""
        ablated_steps = sanitize_steps(obj, cfg.max_steps)
        results.append({
            "k": k,
            "reason": reason,
            "quote": quote,
            "ablated_answer": ablated_answer,
            "ablated_raw": raw_text,
            "ablated_steps": ablated_steps,
        })
        if cfg.sleep and cfg.sleep > 0:
            time.sleep(cfg.sleep)
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Causal ablation experiment on MedQA with LLMs.")
    parser.add_argument("--input", required=True, help="Path to input CSV (columns: id, question, A, B, C, D, label)")
    parser.add_argument("--output", required=True, help="Path to output CSV to write")
    parser.add_argument("--model", default="gpt-5", help="Model name")
    parser.add_argument("--max-steps", type=int, default=5, help="Max reasoning steps to request")
    parser.add_argument("--start", type=int, default=0, help="Start row offset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")

    args = parser.parse_args(argv)

    # Read and normalize input
    df = pd.read_csv(args.input)
    df = normalize_columns_case_insensitive(df)
    # Write normalized snapshot for monitoring
    norm_out = os.path.splitext(args.output)[0] + ".normalized.csv"
    try:
        df.to_csv(norm_out, index=False)
        print(f"[exp1] Wrote normalized CSV to: {norm_out}")
    except Exception as e:
        print(f"[exp1] Warning: failed to write normalized CSV snapshot: {e}")

    # Standardize gold labels
    df["gold"] = df["label"].apply(standardize_gold_label)

    # Slice rows
    start = max(0, int(args.start))
    if args.limit is None:
        df_proc = df.iloc[start:]
    else:
        df_proc = df.iloc[start : start + int(args.limit)]

    cfg = RunConfig(
        model=args.model,
        max_steps=int(args["max_steps"]) if isinstance(args, dict) else int(args.max_steps),
        sleep=float(args.sleep),
    )

    # Derive client from model string and set model env var accordingly
    def _derive_provider_key(model_name: str) -> str:
        key = (model_name or "").lower()
        if any(tok in key for tok in ["gemini", "google"]):
            return "gemini"
        if any(tok in key for tok in ["claude", "anthropic"]):
            return "anthropic"
        return "openai"

    provider_key = _derive_provider_key(cfg.model)
    model_lower = (cfg.model or "").lower().strip()
    # Only set provider-specific env var if --model is a concrete model id,
    # not a provider alias like "gemini", "google", "claude", "anthropic", "openai", "gpt".
    if provider_key == "openai" and cfg.model:
        if model_lower not in {"openai", "chatgpt", "gpt"}:
            os.environ["OPENAI_MODEL"] = cfg.model
    elif provider_key == "anthropic" and cfg.model:
        if model_lower not in {"anthropic", "claude"}:
            os.environ["ANTHROPIC_MODEL"] = cfg.model
    elif provider_key == "gemini" and cfg.model:
        if model_lower not in {"gemini", "google"}:
            os.environ["GOOGLE_GEMINI_MODEL"] = cfg.model

    # Resolve and print the concrete model being used
    if provider_key == "openai":
        resolved_model = os.getenv("OPENAI_MODEL") or cfg.model
    elif provider_key == "anthropic":
        resolved_model = os.getenv("ANTHROPIC_MODEL") or cfg.model
    elif provider_key == "gemini":
        resolved_model = os.getenv("GOOGLE_GEMINI_MODEL") or cfg.model
    else:
        resolved_model = cfg.model
    print(f"Using model: {resolved_model}")

    client = create_model_client(provider_key)

    # Chunked output buffers
    CHUNK_SIZE = 10
    chunk_rows: List[Dict[str, Any]] = []
    part_index = 1
    # Predefine consistent columns across all parts using cfg.max_steps as the upper bound
    base_cols = [
        "id", "model", "question", "A", "B", "C", "D", "E",
        "ground_truth", "baseline_answer", "baseline_steps_json",
    ]
    ablation_cols: List[str] = []
    for i in range(1, int(cfg.max_steps) + 1):
        ablation_cols.append(f"ablation_{i}_answer")
        ablation_cols.append(f"ablation_{i}_steps_json")
    all_cols = base_cols + ablation_cols

    for _, row in tqdm(df_proc.iterrows(), total=len(df_proc), desc="Examples"):
        q_text = str(row["question"]) if not pd.isna(row["question"]) else ""
        options = {
            "A": str(row["A"]) if not pd.isna(row["A"]) else "",
            "B": str(row["B"]) if not pd.isna(row["B"]) else "",
            "C": str(row["C"]) if not pd.isna(row["C"]) else "",
            "D": str(row["D"]) if not pd.isna(row["D"]) else "",
            "E": str(row["E"]) if not pd.isna(row["E"]) else "",
        }

        # Baseline
        baseline_answer, baseline_steps, baseline_raw = run_baseline(client, cfg, q_text, options)

        # Ablations
        ablations = run_ablation_series(client, cfg, q_text, options, baseline_steps)

        out_row: Dict[str, Any] = {
            "id": row["id"],
            "model": cfg.model,
            "question": q_text,
            "A": options["A"],
            "B": options["B"],
            "C": options["C"],
            "D": options["D"],
            "E": options["E"],
            "ground_truth": row.get("gold", None),
            "baseline_answer": baseline_answer,
            "baseline_steps_json": json.dumps(baseline_steps, ensure_ascii=False),
        }
        # Add ablation answers and steps as separate columns
        for ab in ablations:
            k = ab.get("k")
            if isinstance(k, int) and k >= 1:
                out_row[f"ablation_{k}_answer"] = ab.get("ablated_answer", "")
                out_row[f"ablation_{k}_steps_json"] = json.dumps(ab.get("ablated_steps", []), ensure_ascii=False)

        chunk_rows.append(out_row)

        # If chunk full, write a part CSV
        if len(chunk_rows) >= CHUNK_SIZE:
            part_df = pd.DataFrame(chunk_rows)
            for col in ablation_cols:
                if col not in part_df.columns:
                    part_df[col] = ""
            part_df = part_df.reindex(columns=all_cols)

            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            stem, _ = os.path.splitext(args.output)
            part_path = f"{stem}.part{part_index:04d}.csv"
            part_df.to_csv(part_path, index=False)
            print(f"[exp1] Wrote part {part_index:04d} with {len(part_df)} rows to: {part_path}")

            # Reset chunk buffer
            chunk_rows = []
            part_index += 1

        if cfg.sleep and cfg.sleep > 0:
            time.sleep(cfg.sleep)

    # Write trailing partial chunk if any
    if chunk_rows:
        part_df = pd.DataFrame(chunk_rows)
        for col in ablation_cols:
            if col not in part_df.columns:
                part_df[col] = ""
        part_df = part_df.reindex(columns=all_cols)

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        stem, _ = os.path.splitext(args.output)
        part_path = f"{stem}.part{part_index:04d}.csv"
        part_df.to_csv(part_path, index=False)
        print(f"[exp1] Wrote part {part_index:04d} with {len(part_df)} rows to: {part_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


