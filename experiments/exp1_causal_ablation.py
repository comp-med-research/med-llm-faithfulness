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

# Ensure project root is on sys.path so `utils`/`models` are importable when running as a script
sys.path.append(str(_Path(__file__).resolve().parents[1]))
from utils.normalization import normalize_columns_case_insensitive
from models import create_model_client  # type: ignore


SUPPORTED_ANSWER_LETTERS = {"A", "B", "C", "D", "E"}


 


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

    # Option texts are already normalized to plain strings during CSV normalization
    def _coerce(letter: str) -> str:
        return str(options.get(letter, "")).strip()

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
    content_max_attempts: int


def run_baseline(client: Any, cfg: RunConfig, question: str, options: Dict[str, str]) -> Tuple[str, List[Dict[str, str]], str]:
    system_prompt, user_prompt = build_prompts(question, options, cfg.max_steps)
    print("\n==== BASELINE PROMPT (system) ====")
    print(system_prompt)
    print("==== BASELINE PROMPT (user) ====")
    print(user_prompt)
    # Retry content until valid (final_answer in A-E and non-empty steps)
    raw_text = ""
    obj: Dict[str, Any] = {}
    steps: List[Dict[str, str]] = []
    answer = ""
    for attempt in range(1, int(cfg.content_max_attempts) + 1):
        if attempt > 1:
            print(f"[exp1] Baseline content invalid, retrying {attempt}/{cfg.content_max_attempts}...")
        raw_text = call_model(client, system_prompt, user_prompt)
        obj = try_parse_json_object(raw_text) or {}
        steps = sanitize_steps(obj, cfg.max_steps)
        answer = extract_answer_letter_from_obj_or_text(obj, raw_text) or ""
        if answer in SUPPORTED_ANSWER_LETTERS and len(steps) > 0:
            break
        if cfg.sleep and cfg.sleep > 0:
            time.sleep(min(cfg.sleep, 2.0))
    try:
        print("==== BASELINE STEPS (parsed) ====")
        print(json.dumps(steps, ensure_ascii=False, indent=2))
    except Exception:
        pass
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
        # Retry content until valid (final_answer in A-E and non-empty steps)
        raw_text = ""
        obj: Dict[str, Any] = {}
        ablated_steps: List[Dict[str, str]] = []
        ablated_answer = ""
        for attempt in range(1, int(cfg.content_max_attempts) + 1):
            if attempt > 1:
                print(f"[exp1] Ablation {k} content invalid, retrying {attempt}/{cfg.content_max_attempts}...")
            raw_text = call_model(client, system_prompt, user_prompt)
            obj = try_parse_json_object(raw_text) or {}
            ablated_steps = sanitize_steps(obj, cfg.max_steps)
            ablated_answer = extract_answer_letter_from_obj_or_text(obj, raw_text) or ""
            if ablated_answer in SUPPORTED_ANSWER_LETTERS and len(ablated_steps) > 0:
                break
            if cfg.sleep and cfg.sleep > 0:
                time.sleep(min(cfg.sleep, 2.0))
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
    parser.add_argument("--content-max-attempts", type=int, default=8, help="Max resend attempts when output format is invalid (non A-E or empty steps)")

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
        content_max_attempts=int(args.content_max_attempts),
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


