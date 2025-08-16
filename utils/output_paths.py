from __future__ import annotations

from pathlib import Path
import re
import datetime as _dt
from typing import Optional


_NON_ALNUM = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_token(token: str) -> str:
    s = _NON_ALNUM.sub("_", token).strip("_")
    return s or "data"


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_model_dir_name(model_name: str) -> str:
    """Map any provided model identifier to one of the three canonical dirs.

    - Any name containing 'claude' -> 'claude'
    - Any name containing 'gemini' -> 'gemini'
    - Otherwise -> 'chatgpt' (covers GPT family and unknowns by default)
    """
    m = model_name.lower()
    if "claude" in m:
        return "claude"
    if "gemini" in m:
        return "gemini"
    if "gpt" in m or "chatgpt" in m:
        return "chatgpt"
    return "chatgpt"


def build_versioned_file_path(
    directory: Path,
    base_stem: str,
    extension: str,
    start_version: int = 1,
) -> Path:
    """Return the first non-existing file path with an incrementing version suffix.

    Example: base_stem='exp4_askdocs_chatgpt', extension='.csv' ->
    directory/exp4_askdocs_chatgpt_v001.csv (or next available).
    """
    directory.mkdir(parents=True, exist_ok=True)
    version = max(1, int(start_version))
    while True:
        candidate = directory / f"{base_stem}_v{version:03d}{extension}"
        if not candidate.exists():
            return candidate
        version += 1


def compute_output_path(
    out_arg: Optional[Path],
    exp_name: str,
    model_name: str,
    data_path: Path,
    *,
    force_version: bool = False,
    organize_by_model: bool = False,
    organize_by_experiment: bool = False,
    default_extension: str = ".csv",
    explicit_extension: Optional[str] = None,
    append_timestamp: bool = True,
) -> Path:
    """Compute a non-overwriting output path.

    Behavior:
    - If `out_arg` has a suffix, treat it as a file path. If `force_version` is True
      or the file already exists, append an incrementing `_vNNN` before the suffix.
    - If `out_arg` is a directory (no suffix), construct a filename of the form
      `{exp_name}_{data_id}_{model_name}` and place it in that directory. If
      `organize_by_model` is True, create a `{model_name}` subdir. Always version.
    - Extension priority: `explicit_extension` > suffix of `out_arg` > `default_extension`.
    """
    exp_token = _sanitize_token(exp_name)
    # Use only the leading experiment identifier (e.g., 'exp4') for directory names
    exp_dir_token = exp_token.split("_")[0] if "_" in exp_token else exp_token
    model_token = _sanitize_token(model_name)
    model_dir = _normalize_model_dir_name(model_name)
    data_token = _sanitize_token(Path(data_path).stem)

    # Default root if not provided
    if out_arg is None:
        out_arg = Path("results")

    # Determine if out_arg is a file (has suffix) or a directory
    suffix = out_arg.suffix
    is_file_like = suffix != ""

    # Precompute timestamp once per call
    # Local time in the format YYYY_MM_DD_HHMM (e.g., 2025_08_16_1745)
    ts = _dt.datetime.now().strftime("%Y_%m_%d_%H%M") if append_timestamp else ""

    if is_file_like:
        extension = explicit_extension or suffix
        parent_dir = out_arg.parent
        stem = out_arg.stem

        # Optionally organize by experiment and model under the provided parent directory
        if organize_by_experiment:
            parent_dir = parent_dir / exp_dir_token
        if organize_by_model:
            parent_dir = parent_dir / model_dir

        # Existing numeric version suffix respected; still append timestamp if requested
        final_stem = stem
        if append_timestamp:
            final_stem = f"{final_stem}_{ts}"

        # Apply numeric versioning only if explicitly requested (kept for backwards-compat)
        if force_version and not append_timestamp:
            candidate = build_versioned_file_path(parent_dir, final_stem, extension)
            _ensure_parent_dir(candidate)
            return candidate

        final_path = parent_dir / f"{final_stem}{extension}"
        _ensure_parent_dir(final_path)
        return final_path

    # Directory case: build structured, versioned filename
    base_dir = out_arg
    if organize_by_experiment:
        base_dir = base_dir / exp_dir_token
    if organize_by_model:
        base_dir = base_dir / model_dir

    extension = explicit_extension or default_extension
    # Minimal filename when a directory (or default root) is provided: timestamp only
    base_stem = ts if append_timestamp else exp_token

    # Apply numeric versioning only if explicitly requested (kept for backwards-compat)
    if force_version and not append_timestamp:
        candidate = build_versioned_file_path(base_dir, base_stem, extension)
        _ensure_parent_dir(candidate)
        return candidate

    final_path = base_dir / f"{base_stem}{extension}"
    _ensure_parent_dir(final_path)
    return final_path


