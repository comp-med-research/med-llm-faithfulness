"""
Experiment 2 analysis CLI: aggregates positional-bias experiments and produces
summary tables and figures with 95% confidence intervals.

Input CSV schema (per-item rows):
    model (str)
    condition (str) in {unbiased, biased_to_gold, biased_to_wrong}
    pred (str) — model prediction (A|B|C|D|E)
    gold (str)
    biased_pos (str)
    pred_unbiased (str, optional)
    flip_from_unbiased (0/1, optional)
    reveal (0/1, optional)

Optional pre-aggregated rows:
    scope="global", metric, value, se?, ci_low?, ci_high?, rows?, model

Usage:
    python -m analysis.exp2_analyze \
      --input /path/to/exp2_results.csv \
      --outdir /path/to/outputs/exp2 \
      --models "gpt-4o,claude-3,gemini-pro"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import ast
import matplotlib as _mpl

from analysis.metrics import (
    compute_accuracy,
    compute_delta,
    compute_damage_rescue_netflip,
    position_pick_rate_df,
    reveal_rate_on_flip,
    add_wilson_ci,
    bar_with_ci,
    grouped_bars_with_ci,
)


MANDATORY_COLS = ["model", "condition", "pred", "gold", "biased_pos"]


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_letters(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip().str.upper()
    return out


def _filter_models(df: pd.DataFrame, models: Optional[List[str]]) -> pd.DataFrame:
    if not models:
        return df
    return df[df["model"].isin(models)].copy()


def _has_cols(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


def _prefer_preaggregated(df: pd.DataFrame, metric: str) -> Optional[pd.DataFrame]:
    if not {"scope", "metric"}.issubset(df.columns):
        return None
    mask = (df["scope"] == "global") & (df["metric"] == metric)
    if mask.any():
        cols = ["model", "value", "ci_low", "ci_high", "se"]
        found = df.loc[mask, [c for c in cols if c in df.columns]].copy()
        # ensure all expected columns present
        for c in ["ci_low", "ci_high", "se"]:
            if c not in found.columns:
                found[c] = pd.NA
        return found
    return None


def _prettify_model_label(raw: str) -> str:
    low = str(raw).lower()
    if "chatgpt" in low:
        return "ChatGPT"
    if "claude" in low:
        return "Claude"
    if "gemini" in low:
        return "Gemini"
    if "llama" in low and ("405" in low or "405b" in low):
        return "Llama-405b"
    return str(raw)


def _plot_accuracy_slopegraph(
    acc_unb: Optional[pd.DataFrame],
    acc_bg: Optional[pd.DataFrame],
    acc_bw: Optional[pd.DataFrame],
    models_list: List[str],
    outfile: str,
) -> None:
    """Slopegraph of accuracy by condition with 95% CIs for each model."""
    if acc_unb is None or acc_bg is None or acc_bw is None:
        return
    import numpy as _np
    import matplotlib.pyplot as _plt
    # Ensure vector-friendly fonts in PDFs for LaTeX inclusion and global sizes
    _mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
        "savefig.bbox": "tight",
        "figure.autolayout": False,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })

    x = _np.array([0, 1, 2])
    x_labels = ["Unbiased", "Bias→Gold", "Bias→Wrong"]

    color_map = {
        "Claude": "#4C78A8",
        "ChatGPT": "#F58518",
        "Gemini": "#54A24B",
    }
    model_offsets = {"Claude": -0.02, "ChatGPT": 0.0, "Gemini": 0.02}

    fig, ax = _plt.subplots(figsize=(10, 5))

    all_lows = []
    all_highs = []

    for m in models_list:
        pretty = _prettify_model_label(m)
        col = color_map.get(pretty, None)

        def _vals(df: pd.DataFrame) -> Tuple[float, float, float]:
            row = df[df["model"] == m].iloc[0]
            v = float(row["value"])
            lo = row.get("ci_low", v)
            hi = row.get("ci_high", v)
            lo = v if lo is None or pd.isna(lo) else float(lo)
            hi = v if hi is None or pd.isna(hi) else float(hi)
            if lo > hi:
                lo, hi = hi, lo
            return v, lo, hi

        v0, lo0, hi0 = _vals(acc_unb)
        v1, lo1, hi1 = _vals(acc_bg)
        v2, lo2, hi2 = _vals(acc_bw)
        y = _np.array([v0, v1, v2], dtype=float)
        lo = _np.array([lo0, lo1, lo2], dtype=float)
        hi = _np.array([hi0, hi1, hi2], dtype=float)

        # Keep decimals (0–1) for plotting, per request

        # Slight constant horizontal offset per model to avoid error bar overlap
        mo = model_offsets.get(pretty, 0.0)
        xi = x + mo
        ax.plot(xi, y, marker="o", linewidth=2.0, markersize=6, label=pretty, color=col, zorder=4)

        all_lows.append(lo)
        all_highs.append(hi)

    # Set requested y-scale
    ax.set_ylim(0.75, 1.00)

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Accuracy")
    import math as _math
    y0, y1 = ax.get_ylim()
    tick_start = 0.05 * _math.floor(y0 / 0.05)
    ticks = _np.arange(tick_start, y1 + 0.0001, 0.05)
    ax.set_yticks(ticks)
    ax.set_title("Experiment 2: Accuracy by Condition")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title=None)

    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    _plt.close(fig)


def _parse_ci_tuple(s: object) -> Tuple[Optional[float], Optional[float]]:
    """Parse CI tuple stored as a string like '(np.float64(0.86), np.float64(0.96))'.

    Uses ast.literal_eval after stripping 'np.float64' wrappers; no regex fallbacks.
    """
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return (None, None)
    text = str(s).strip()
    try:
        cleaned = text.replace("np.float64", "")
        val = ast.literal_eval(cleaned)
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            lo, hi = float(val[0]), float(val[1])
            if lo > hi:
                lo, hi = hi, lo
            return (lo, hi)
    except Exception:
        return (None, None)
    return (None, None)


def analyze_from_metrics(metrics_csvs: List[str], outdir: str) -> Dict[str, Dict]:
    """Build Exp2 figures from pre-aggregated exp2_metrics.csv files from each model."""
    rows: List[Dict] = []
    for path in metrics_csvs:
        df = pd.read_csv(path)
        if df.empty:
            continue
        r = df.iloc[0]
        model_raw = r.get("model", "")
        model = _prettify_model_label(model_raw)
        # Parse CI strings directly from each model's exp2_metrics.csv (ground truth)
        unb_ci = _parse_ci_tuple(r.get("Acc_unbiased_ci"))
        bg_ci = _parse_ci_tuple(r.get("Acc_bias_gold_ci"))
        bw_ci = _parse_ci_tuple(r.get("Acc_bias_wrong_ci"))
        lo_unb, hi_unb = unb_ci
        lo_bg, hi_bg = bg_ci
        lo_bw, hi_bw = bw_ci
        rows.append({
            "model": model,
            "Acc_unbiased": float(r.get("Acc_unbiased", 0.0)),
            "Acc_unbiased_ci_low": lo_unb,
            "Acc_unbiased_ci_high": hi_unb,
            "Acc_bias_gold": float(r.get("Acc_bias_gold", 0.0)),
            "Acc_bias_gold_ci_low": lo_bg,
            "Acc_bias_gold_ci_high": hi_bg,
            "Acc_bias_wrong": float(r.get("Acc_bias_wrong", 0.0)),
            "Acc_bias_wrong_ci_low": lo_bw,
            "Acc_bias_wrong_ci_high": hi_bw,
            "Damage_rate_wrong": float(r.get("Damage_rate_wrong", 0.0)),
            "Rescue_rate_wrong": float(r.get("Rescue_rate_wrong", 0.0)),
            "NetFlip_wrong": float(r.get("NetFlip_wrong", 0.0)),
            "PositionPick_wrongB": float(r.get("PositionPick_wrongB", 0.0)),
        })

    if not rows:
        return {}

    os.makedirs(outdir, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(outdir, "exp2_metrics_merged.csv"), index=False)

    # Prepare DataFrames for slopegraph API
    acc_unb_raw = metrics[["model", "Acc_unbiased", "Acc_unbiased_ci_low", "Acc_unbiased_ci_high"]].rename(
        columns={"Acc_unbiased": "value", "Acc_unbiased_ci_low": "ci_low", "Acc_unbiased_ci_high": "ci_high"}
    )
    acc_bg_raw = metrics[["model", "Acc_bias_gold", "Acc_bias_gold_ci_low", "Acc_bias_gold_ci_high"]].rename(
        columns={"Acc_bias_gold": "value", "Acc_bias_gold_ci_low": "ci_low", "Acc_bias_gold_ci_high": "ci_high"}
    )
    acc_bw_raw = metrics[["model", "Acc_bias_wrong", "Acc_bias_wrong_ci_low", "Acc_bias_wrong_ci_high"]].rename(
        columns={"Acc_bias_wrong": "value", "Acc_bias_wrong_ci_low": "ci_low", "Acc_bias_wrong_ci_high": "ci_high"}
    )

    # If CI columns are entirely missing/empty due to parsing, fall back to the raw strings
    def _ensure_ci(df_num: pd.DataFrame, src_df: pd.DataFrame, src_col: str) -> pd.DataFrame:
        if df_num["ci_low"].isna().all() and "{}".format(src_col) in src_df.columns:
            parsed = src_df[["model", src_col]].copy()
            lows: List[Optional[float]] = []
            highs: List[Optional[float]] = []
            for _, r in parsed.iterrows():
                lo, hi = _parse_ci_tuple(r[src_col])
                lows.append(lo)
                highs.append(hi)
            m = df_num.merge(pd.DataFrame({"model": parsed["model"], "_lo": lows, "_hi": highs}), on="model", how="left")
            m["ci_low"] = m["ci_low"].fillna(m["_lo"])
            m["ci_high"] = m["ci_high"].fillna(m["_hi"])
            m = m.drop(columns=["_lo", "_hi"])
            return m
        return df_num

    acc_unb_raw = _ensure_ci(acc_unb_raw, metrics, "Acc_unbiased_ci")
    acc_bg_raw = _ensure_ci(acc_bg_raw, metrics, "Acc_bias_gold_ci")
    acc_bw_raw = _ensure_ci(acc_bw_raw, metrics, "Acc_bias_wrong_ci")

    models_list = metrics["model"].tolist()
    _plot_accuracy_slopegraph(
        acc_unb_raw, acc_bg_raw, acc_bw_raw, models_list,
        outfile=os.path.join(outdir, "exp2_fig2a_accuracy_slopegraph.png"),
    )
    # Also save a PDF version for LaTeX
    _plot_accuracy_slopegraph(
        acc_unb_raw, acc_bg_raw, acc_bw_raw, models_list,
        outfile=os.path.join(outdir, "exp2_fig2a_accuracy_slopegraph.pdf"),
    )

    # Grouped accuracy per model and condition -> accuracy_plot.png
    def _vals_in_order(df_num: pd.DataFrame, models: List[str]) -> List[float]:
        out: List[float] = []
        for m in models:
            row = df_num[df_num["model"] == m]
            out.append(float(row["value"].values[0]) if not row.empty else 0.0)
        return out

    def _ci_in_order(df_num: pd.DataFrame, models: List[str], bound: str) -> List[float]:
        out: List[float] = []
        for m in models:
            row = df_num[df_num["model"] == m]
            if not row.empty and pd.notna(row[bound].values[0]):
                out.append(float(row[bound].values[0]))
            else:
                # If CI missing, fall back to the mean (zero-length whisker)
                v = float(row["value"].values[0]) if not row.empty else 0.0
                out.append(v)
        return out

    series = {
        "Unbiased": _vals_in_order(acc_unb_raw, models_list),
        "Bias→Gold": _vals_in_order(acc_bg_raw, models_list),
        "Bias→Wrong": _vals_in_order(acc_bw_raw, models_list),
    }
    series_ci = {
        "Unbiased": (
            _ci_in_order(acc_unb_raw, models_list, "ci_low"),
            _ci_in_order(acc_unb_raw, models_list, "ci_high"),
        ),
        "Bias→Gold": (
            _ci_in_order(acc_bg_raw, models_list, "ci_low"),
            _ci_in_order(acc_bg_raw, models_list, "ci_high"),
        ),
        "Bias→Wrong": (
            _ci_in_order(acc_bw_raw, models_list, "ci_low"),
            _ci_in_order(acc_bw_raw, models_list, "ci_high"),
        ),
    }

    grouped_bars_with_ci(
        x_labels=models_list,
        series=series,
        series_ci=series_ci,
        outfile=os.path.join(outdir, "accuracy_plot.png"),
        ylabel="Accuracy",
        title="Experiment 2: Accuracy by Condition",
    )

    # New: Heatmap of position pick rate for letter 'B' per condition, per model
    # We need per-item CSVs to compute this precisely per user's conditioning
    def _find_item_csv_for_metrics_path(metrics_path: str) -> Optional[str]:
        # metrics_path .../<model>/analysis/exp2_metrics.csv → look in parent dir for exp2_results*.csv
        model_dir = os.path.dirname(os.path.dirname(os.path.abspath(metrics_path)))
        if not os.path.isdir(model_dir):
            return None
        try:
            for name in os.listdir(model_dir):
                if name.lower().startswith("exp2_results") and name.lower().endswith(".csv"):
                    return os.path.join(model_dir, name)
        except Exception:
            return None
        return None

    per_item_paths: List[Tuple[str, str]] = []  # (pretty_model, path)
    for p in metrics_csvs:
        item_csv = _find_item_csv_for_metrics_path(p)
        if item_csv and os.path.exists(item_csv):
            # Try to infer pretty model label from the folder name
            model_folder = os.path.basename(os.path.dirname(item_csv))
            pretty = _prettify_model_label(model_folder)
            per_item_paths.append((pretty, item_csv))

    if per_item_paths:
        import numpy as _np
        import matplotlib.pyplot as _plt
        import matplotlib as _mpl

        _mpl.rcParams["pdf.fonttype"] = 42
        _mpl.rcParams["ps.fonttype"] = 42

        # Maintain preferred model order where applicable
        preferred_order = ["Claude", "ChatGPT", "Gemini"]
        # Load and compute PPR for 'B' under each condition per user's rules
        conds = ["unbiased", "biased_to_gold", "biased_to_wrong"]
        cond_labels = ["Unbiased", "Bias→Gold", "Bias→Wrong"]

        model_to_rates: Dict[str, List[float]] = {}
        for pretty, path in per_item_paths:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            # Normalize casing
            for col in ["pred", "gold", "biased_pos"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.upper()
            rates: List[float] = []
            # Unbiased: P(pred=='B' | gold=='B', condition=='unbiased')
            sub_unb = df[(df.get("condition") == "unbiased") & (df.get("gold") == "B")]
            r0 = float(sub_unb["pred"].eq("B").mean()) if not sub_unb.empty else 0.0
            # Biased→Gold: condition is biased_to_gold, compute P(pred=='B')
            sub_bg = df[(df.get("condition") == "biased_to_gold")]
            r1 = float(sub_bg["pred"].eq("B").mean()) if not sub_bg.empty else 0.0
            # Biased→Wrong: condition is biased_to_wrong, compute P(pred=='B')
            sub_bw = df[(df.get("condition") == "biased_to_wrong")]
            r2 = float(sub_bw["pred"].eq("B").mean()) if not sub_bw.empty else 0.0
            rates = [r0, r1, r2]
            model_to_rates[pretty] = rates

        if model_to_rates:
            # Order models: preferred first then any remaining
            ordered_models: List[str] = [m for m in preferred_order if m in model_to_rates.keys()]
            for m in sorted(model_to_rates.keys()):
                if m not in ordered_models:
                    ordered_models.append(m)

            mat = _np.array([model_to_rates[m] for m in ordered_models], dtype=float)
            fig, ax = _plt.subplots(figsize=(9, 5.2))
            im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

            ax.set_xticks(list(range(len(cond_labels))))
            ax.set_xticklabels(cond_labels, rotation=0)
            ax.set_yticks(list(range(len(ordered_models))))
            ax.set_yticklabels(ordered_models)
            ax.set_title("Experiment 2: Position Pick Rate @ Position B")

            # Annotate cells with bold white/black depending on value (larger font)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    txt_color = "white" if val >= 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        va="center",
                        ha="center",
                        color=txt_color,
                        fontweight="bold",
                        fontsize=22,
                    )

            # Minor gridlines between cells
            ax.set_xticks(_np.arange(-0.5, len(cond_labels), 1), minor=True)
            ax.set_yticks(_np.arange(-0.5, len(ordered_models), 1), minor=True)
            ax.grid(which="minor", color="#CCCCCC", linestyle=":", linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)

            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel("Proportion of Picks")

            fig.tight_layout()
            os.makedirs(outdir, exist_ok=True)
            png_path = os.path.join(outdir, "exp2_fig2b_pprB_heatmap.png")
            pdf_path = os.path.join(outdir, "exp2_fig2b_pprB_heatmap.pdf")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
            _plt.close(fig)

    return {"merged_metrics_path": os.path.join(outdir, "exp2_metrics_merged.csv")}

def analyze(input_csv: str, outdir: str, models: Optional[List[str]] = None) -> Dict[str, Dict]:
    _safe_mkdir(outdir)
    raw = pd.read_csv(input_csv)

    # Split pre-aggregated vs per-item rows
    if {"scope", "metric"}.issubset(raw.columns):
        is_agg = (raw["scope"] == "global") & raw["metric"].notna()
    else:
        is_agg = pd.Series([False] * len(raw), index=raw.index)
    agg_rows = raw[is_agg].copy()
    item_rows = raw[~is_agg].copy()

    if not _has_cols(item_rows, MANDATORY_COLS) and agg_rows.empty:
        raise ValueError("Input CSV missing per-item mandatory columns and no pre-aggregated metrics present.")

    item_rows = _normalize_letters(item_rows, ["pred", "gold", "biased_pos", "pred_unbiased"]) if not item_rows.empty else item_rows
    item_rows = _filter_models(item_rows, models)
    agg_rows = _filter_models(agg_rows, models)

    models_list = sorted(item_rows["model"].unique().tolist()) if not item_rows.empty else sorted(agg_rows["model"].unique().tolist())

    outputs: Dict[str, Dict] = {}

    # Accuracies per model and condition
    acc = None
    pre_acc = _prefer_preaggregated(agg_rows, "macro_ablation_accuracy")  # not directly used, but example
    # We'll compute per-condition accuracies from items
    if not item_rows.empty and _has_cols(item_rows, ["model", "condition", "pred", "gold"]):
        acc = compute_accuracy(item_rows, ["model", "condition"], pred_col="pred", gold_col="gold")
        acc = add_wilson_ci(acc)

    # Extract per-condition accuracy tables
    acc_unb_raw = acc[acc["condition"] == "unbiased"].copy() if acc is not None else None
    acc_bg_raw = acc[acc["condition"] == "biased_to_gold"].copy() if acc is not None else None
    acc_bw_raw = acc[acc["condition"] == "biased_to_wrong"].copy() if acc is not None else None
    acc_unb = acc_unb_raw.rename(columns={"value": "Acc_unbiased"}) if acc_unb_raw is not None else None
    acc_bg = acc_bg_raw.rename(columns={"value": "Acc_bias_gold"}) if acc_bg_raw is not None else None
    acc_bw = acc_bw_raw.rename(columns={"value": "Acc_bias_wrong"}) if acc_bw_raw is not None else None

    # Deltas
    acc_dg = None
    acc_dw = None
    if acc_unb_raw is not None and not acc_unb_raw.empty and acc_bg_raw is not None and not acc_bg_raw.empty:
        acc_dg = compute_delta(acc_unb_raw[["model", "value"]], acc_bg_raw[["model", "value"]], on=["model"], name="Delta_gold")
    if acc_unb_raw is not None and not acc_unb_raw.empty and acc_bw_raw is not None and not acc_bw_raw.empty:
        acc_dw = compute_delta(acc_unb_raw[["model", "value"]], acc_bw_raw[["model", "value"]], on=["model"], name="Delta_wrong")

    # Damage/Rescue/Netflip using unbiased as base vs each biased condition
    def _drn_for_condition(cond: str, label: str) -> Optional[pd.DataFrame]:
        if item_rows.empty or not _has_cols(item_rows, ["model", "condition", "pred", "gold"]):
            return None
        # Join per item: base = unbiased pred for same model+item, pert = pred under given condition
        sub = item_rows[item_rows["condition"].isin(["unbiased", cond])].copy()
        # Assume there's an item identifier to join; if not, use row order within (model, condition)
        if "item_id" not in sub.columns:
            sub = sub.assign(_row_idx=sub.groupby(["model", "condition"]).cumcount())
            join_keys = ["model", "_row_idx"]
        else:
            join_keys = ["model", "item_id"]
        base = sub[sub["condition"] == "unbiased"][join_keys + ["pred", "gold"]].rename(columns={"pred": "pred_unbiased"})
        pert = sub[sub["condition"] == cond][join_keys + ["pred", "gold"]].rename(columns={"pred": "pred_pert"})
        joined = base.merge(pert, on=join_keys + ["gold"], how="inner")
        if joined.empty:
            return None
        drn = compute_damage_rescue_netflip(joined, ["model"], base_col="pred_unbiased", pert_col="pred_pert", gold_col="gold")
        return drn

    drn_wrong = _drn_for_condition("biased_to_wrong", "wrong")
    drn_gold = _drn_for_condition("biased_to_gold", "gold")

    # Position-Pick (wrong@biased_pos) — only in biased_to_wrong
    ppr = None
    if not item_rows.empty and _has_cols(item_rows, ["model", "condition", "pred", "gold", "biased_pos"]):
        wr = item_rows[item_rows["condition"] == "biased_to_wrong"].copy()
        ppr = position_pick_rate_df(wr, ["model"], pred_col="pred", biased_pos_col="biased_pos", gold_col="gold")
        ppr = add_wilson_ci(ppr)

    # Reveal on flip (disabled for now)
    rev = None
    # if not item_rows.empty and _has_cols(item_rows, ["model", "condition", "flip_from_unbiased", "reveal"]):
    #     wr = item_rows[item_rows["condition"] != "unbiased"].copy()
    #     rev = reveal_rate_on_flip(wr, ["model", "condition"], flip_col="flip_from_unbiased", reveal_col="reveal")
    #     rev = add_wilson_ci(rev)

    # Build summary table per model
    summary_rows: List[Dict] = []
    for m in models_list:
        row: Dict = {"model": m}
        # Accuracies
        if acc_unb is not None:
            r = acc_unb[acc_unb["model"] == m]
            if not r.empty:
                row["Acc_unbiased"] = r["Acc_unbiased"].values[0]
                row["Acc_unbiased_ci"] = (r.get("ci_low", pd.Series([pd.NA])).values[0], r.get("ci_high", pd.Series([pd.NA])).values[0])
        if acc_bg is not None:
            r = acc_bg[acc_bg["model"] == m]
            if not r.empty:
                row["Acc_bias_gold"] = r["Acc_bias_gold"].values[0]
                row["Acc_bias_gold_ci"] = (r.get("ci_low", pd.Series([pd.NA])).values[0], r.get("ci_high", pd.Series([pd.NA])).values[0])
        if acc_bw is not None:
            r = acc_bw[acc_bw["model"] == m]
            if not r.empty:
                row["Acc_bias_wrong"] = r["Acc_bias_wrong"].values[0]
                row["Acc_bias_wrong_ci"] = (r.get("ci_low", pd.Series([pd.NA])).values[0], r.get("ci_high", pd.Series([pd.NA])).values[0])
        # Deltas
        if acc_dw is not None:
            r = acc_dw[acc_dw["model"] == m]
            if not r.empty:
                row["Delta_wrong"] = r["Delta_wrong"].values[0]
        if acc_dg is not None:
            r = acc_dg[acc_dg["model"] == m]
            if not r.empty:
                row["Delta_gold"] = r["Delta_gold"].values[0]
        # Damage/Rescue/NetFlip (wrong)
        if drn_wrong is not None:
            r = drn_wrong[drn_wrong["model"] == m]
            if not r.empty:
                row["Damage_rate_wrong"] = r["damage"].values[0]
                row["Rescue_rate_wrong"] = r["rescue"].values[0]
                row["NetFlip_wrong"] = r["netflip"].values[0]
        # Position pick rate
        if ppr is not None:
            r = ppr[ppr["model"] == m]
            if not r.empty:
                row["PositionPick_wrongB"] = r["value"].values[0]
        # Reveal on flip (disabled)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save metrics as CSV instead of JSON
    metrics_csv_path = os.path.join(outdir, "exp2_metrics.csv")
    pd.DataFrame(summary_rows).to_csv(metrics_csv_path, index=False)

    # Build figures
    # Figure 1: Accuracies grouped bars
    if acc_unb is not None and acc_bg is not None and acc_bw is not None:
        cats = models_list
        series = {
            "Acc_unbiased": [acc_unb.loc[acc_unb["model"] == m, "Acc_unbiased"].values[0] for m in models_list],
            "Acc_bias_gold": [acc_bg.loc[acc_bg["model"] == m, "Acc_bias_gold"].values[0] for m in models_list],
            "Acc_bias_wrong": [acc_bw.loc[acc_bw["model"] == m, "Acc_bias_wrong"].values[0] for m in models_list],
        }
        series_ci = {
            "Acc_unbiased": (
                [acc_unb.loc[acc_unb["model"] == m, "ci_low"].values[0] for m in models_list],
                [acc_unb.loc[acc_unb["model"] == m, "ci_high"].values[0] for m in models_list],
            ),
            "Acc_bias_gold": (
                [acc_bg.loc[acc_bg["model"] == m, "ci_low"].values[0] for m in models_list],
                [acc_bg.loc[acc_bg["model"] == m, "ci_high"].values[0] for m in models_list],
            ),
            "Acc_bias_wrong": (
                [acc_bw.loc[acc_bw["model"] == m, "ci_low"].values[0] for m in models_list],
                [acc_bw.loc[acc_bw["model"] == m, "ci_high"].values[0] for m in models_list],
            ),
        }
        grouped_bars_with_ci(
            x_labels=cats,
            series=series,
            series_ci=series_ci,
            outfile=os.path.join(outdir, "exp2_fig1_accuracy_with_ci.png"),
            ylabel="Accuracy",
            title="Experiment 2: Accuracies with 95% CI",
        )

    # Figure 2: Damage/Rescue (wrong)
    if drn_wrong is not None:
        drn_wrong_ci = drn_wrong.copy()
        drn_wrong_ci_damage = add_wilson_ci(
            drn_wrong_ci.rename(columns={"damage": "value", "n_correct_base": "n"})[["model", "value", "n"]]
        ).rename(columns={"value": "damage", "ci_low": "damage_ci_low", "ci_high": "damage_ci_high"})
        drn_wrong_ci_rescue = add_wilson_ci(
            drn_wrong_ci.rename(columns={"rescue": "value", "n_incorrect_base": "n"})[["model", "value", "n"]]
        ).rename(columns={"value": "rescue", "ci_low": "rescue_ci_low", "ci_high": "rescue_ci_high"})
        merged = drn_wrong.merge(drn_wrong_ci_damage[["model", "damage_ci_low", "damage_ci_high"]], on="model")
        merged = merged.merge(drn_wrong_ci_rescue[["model", "rescue_ci_low", "rescue_ci_high"]], on="model")

        cats = models_list
        series = {
            "Damage_rate_wrong": [merged.loc[merged["model"] == m, "damage"].values[0] for m in models_list],
            "Rescue_rate_wrong": [merged.loc[merged["model"] == m, "rescue"].values[0] for m in models_list],
        }
        series_ci = {
            "Damage_rate_wrong": (
                [merged.loc[merged["model"] == m, "damage_ci_low"].values[0] for m in models_list],
                [merged.loc[merged["model"] == m, "damage_ci_high"].values[0] for m in models_list],
            ),
            "Rescue_rate_wrong": (
                [merged.loc[merged["model"] == m, "rescue_ci_low"].values[0] for m in models_list],
                [merged.loc[merged["model"] == m, "rescue_ci_high"].values[0] for m in models_list],
            ),
        }
        grouped_bars_with_ci(
            x_labels=cats,
            series=series,
            series_ci=series_ci,
            outfile=os.path.join(outdir, "exp2_fig2_damage_rescue_with_ci.png"),
            ylabel="Rate",
            title="Experiment 2: Damage vs Rescue (wrong) with 95% CI",
        )

    # Figure 2a: Accuracy slopegraph across conditions
    # If slopegraph CI parsing proves brittle for some inputs, also emit a grouped bar chart using merged CI
    if acc_unb_raw is not None and acc_bg_raw is not None and acc_bw_raw is not None:
        _plot_accuracy_slopegraph(
            acc_unb_raw, acc_bg_raw, acc_bw_raw, models_list,
            outfile=os.path.join(outdir, "exp2_fig2a_accuracy_slopegraph.png"),
        )
        # Also emit a PDF version
        _plot_accuracy_slopegraph(
            acc_unb_raw, acc_bg_raw, acc_bw_raw, models_list,
            outfile=os.path.join(outdir, "exp2_fig2a_accuracy_slopegraph.pdf"),
        )

        # Grouped bars with CI from the same data
        cats = ["Unbiased", "Bias→Gold", "Bias→Wrong"]
        # Build per-model grouped series is messy; instead plot model-averaged bars with CIs per condition
        import numpy as _np
        def _mean_ci(df: pd.DataFrame) -> Tuple[float, float, float]:
            vals = df["value"].astype(float).to_list()
            los = df["ci_low"].astype(float).to_list()
            his = df["ci_high"].astype(float).to_list()
            m = float(_np.mean(vals)) if vals else 0.0
            lo = float(_np.mean(los)) if los else m
            hi = float(_np.mean(his)) if his else m
            return m, lo, hi
        m0, lo0, hi0 = _mean_ci(acc_unb_raw)
        m1, lo1, hi1 = _mean_ci(acc_bg_raw)
        m2, lo2, hi2 = _mean_ci(acc_bw_raw)
        bar_with_ci(
            categories=cats,
            means=[m0, m1, m2],
            ci_lows=[lo0, lo1, lo2],
            ci_highs=[hi0, hi1, hi2],
            outfile=os.path.join(outdir, "exp2_fig2a_accuracy_bars_ci.png"),
            ylabel="Accuracy",
            title="Experiment 2: Accuracy by Condition (Bars with 95% CI)",
        )

    # Figure 2b: Position Pick Rate (option B) heatmap per model × condition
    if not item_rows.empty and _has_cols(item_rows, ["model", "condition", "pred"]):
        import numpy as _np
        import matplotlib.pyplot as _plt

        # Compute P(pred == 'B') per (model, condition)
        work = item_rows.copy()
        work["_picked_B"] = (work["pred"].astype(str).str.upper() == "B").astype(int)
        grouped = work.groupby(["model", "condition"], dropna=False)["_picked_B"].mean().reset_index()

        cond_order = ["unbiased", "biased_to_gold", "biased_to_wrong"]
        cond_labels = ["Unbiased", "Bias→Gold", "Bias→Wrong"]
        # Prefer a readable model order
        pretty_map = {m: _prettify_model_label(m) for m in models_list}
        preferred_order = ["Claude", "ChatGPT", "Gemini"]
        y_models_pretty = [m for m in preferred_order if m in {pretty_map[x] for x in models_list}]
        # Include any remaining models not in the preferred list
        for m in models_list:
            p = pretty_map[m]
            if p not in y_models_pretty:
                y_models_pretty.append(p)

        # Build matrix values aligned with y_models_pretty × cond_order
        def _lookup(pretty_name: str, cond: str) -> float:
            # Find original raw model name that maps to pretty_name
            raw_names = [k for k, v in pretty_map.items() if v == pretty_name]
            if not raw_names:
                return 0.0
            sub = grouped[(grouped["model"].isin(raw_names)) & (grouped["condition"] == cond)]
            if sub.empty:
                return 0.0
            return float(sub["_picked_B"].values[0])

        mat = _np.array([[ _lookup(m, c) for c in cond_order ] for m in y_models_pretty], dtype=float)

        fig, ax = _plt.subplots(figsize=(8, 4.5))
        im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        # Axes labels and ticks
        ax.set_xticks(list(range(len(cond_labels))))
        ax.set_xticklabels(cond_labels, rotation=30, ha="right")
        ax.set_yticks(list(range(len(y_models_pretty))))
        ax.set_yticklabels(y_models_pretty)
        ax.set_title("Fig 2b — Position Pick Rate (heatmap)")

        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                txt_color = "black" if val < 0.6 else "white"
                ax.text(j, i, f"{val:.2f}", va="center", ha="center", color=txt_color)

        # Gridlines
        ax.set_xticks(_np.arange(-0.5, len(cond_labels), 1), minor=True)
        ax.set_yticks(_np.arange(-0.5, len(y_models_pretty), 1), minor=True)
        ax.grid(which="minor", color="#CCCCCC", linestyle=":", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Position Pick Rate (PPR for option B)")
        fig.tight_layout()
        _safe_mkdir(outdir)
        fig.savefig(os.path.join(outdir, "exp2_fig2b_ppr_heatmap.png"), dpi=200)
        _plt.close(fig)

    # Figure 3: NetFlip_wrong bars
    if drn_wrong is not None:
        cats = models_list
        means = [drn_wrong.loc[drn_wrong["model"] == m, "netflip"].values[0] for m in models_list]
        # For CI of difference (damage-rescue), approximate by quadrature of SEs
        dmg_ci = add_wilson_ci(
            drn_wrong.rename(columns={"damage": "value", "n_correct_base": "n"})[["model", "value", "n"]]
        ).rename(columns={"value": "damage", "ci_low": "d_lo", "ci_high": "d_hi", "se": "d_se"})
        res_ci = add_wilson_ci(
            drn_wrong.rename(columns={"rescue": "value", "n_incorrect_base": "n"})[["model", "value", "n"]]
        ).rename(columns={"value": "rescue", "ci_low": "r_lo", "ci_high": "r_hi", "se": "r_se"})
        joined = dmg_ci.merge(res_ci, on="model")
        # SE of difference
        import math as _math

        se = [float(_math.sqrt(row["d_se"] ** 2 + row["r_se"] ** 2)) for _, row in joined.iterrows()]
        # 95% CI approx
        ci_lows = [m - 1.96 * s for m, s in zip(means, se)]
        ci_highs = [m + 1.96 * s for m, s in zip(means, se)]
        bar_with_ci(
            categories=cats,
            means=means,
            ci_lows=ci_lows,
            ci_highs=ci_highs,
            outfile=os.path.join(outdir, "exp2_fig3_netflip_with_ci.png"),
            ylabel="NetFlip (damage - rescue)",
            title="Experiment 2: NetFlip (wrong) with 95% CI",
        )

    # Figure 4: Position pick (wrong@B)
    if ppr is not None:
        cats = models_list
        means = [ppr.loc[ppr["model"] == m, "value"].values[0] for m in models_list]
        ci_lows = [ppr.loc[ppr["model"] == m, "ci_low"].values[0] for m in models_list]
        ci_highs = [ppr.loc[ppr["model"] == m, "ci_high"].values[0] for m in models_list]
        bar_with_ci(
            categories=cats,
            means=means,
            ci_lows=ci_lows,
            ci_highs=ci_highs,
            outfile=os.path.join(outdir, "exp2_fig4_position_pick_wrongB.png"),
            ylabel="Position pick (wrong@biased)",
            title="Experiment 2: Position-Pick (wrong@B) with 95% CI",
        )

    # Figure 5: Reveal on flip (disabled)

    # Summary table with CI text (also written as CSV)
    def _fmt_ci(lo: Optional[float], hi: Optional[float]) -> str:
        if pd.isna(lo) or pd.isna(hi):
            return ""
        return f" [{lo:.3f}, {hi:.3f}]"

    table_cols: List[str] = ["model", "Acc_unbiased", "Acc_bias_gold", "Acc_bias_wrong", "Delta_gold", "Delta_wrong", "Damage_rate_wrong", "Rescue_rate_wrong", "NetFlip_wrong", "PositionPick_wrongB", "RevealRate_on_flip_wrong"]
    table = summary_df.reindex(columns=table_cols)
    # Append CI text to accuracy columns if available
    if acc_unb is not None:
        ci_map = {r["model"]: (r.get("ci_low"), r.get("ci_high")) for _, r in acc_unb.iterrows()}
        table["Acc_unbiased"] = table.apply(lambda r: f"{r['Acc_unbiased']:.3f}" + _fmt_ci(*(ci_map.get(r['model'], (pd.NA, pd.NA)))), axis=1)
    if acc_bg is not None:
        ci_map = {r["model"]: (r.get("ci_low"), r.get("ci_high")) for _, r in acc_bg.iterrows()}
        table["Acc_bias_gold"] = table.apply(lambda r: f"{r['Acc_bias_gold']:.3f}" + _fmt_ci(*(ci_map.get(r['model'], (pd.NA, pd.NA)))), axis=1)
    if acc_bw is not None:
        ci_map = {r["model"]: (r.get("ci_low"), r.get("ci_high")) for _, r in acc_bw.iterrows()}
        table["Acc_bias_wrong"] = table.apply(lambda r: f"{r['Acc_bias_wrong']:.3f}" + _fmt_ci(*(ci_map.get(r['model'], (pd.NA, pd.NA)))), axis=1)

    table_csv = os.path.join(outdir, "exp2_summary_table_CI.csv")
    table.to_csv(table_csv, index=False)

    # Print outputs
    outputs_paths = [
        table_csv,
        os.path.join(outdir, "exp2_fig1_accuracy_with_ci.png"),
        os.path.join(outdir, "exp2_fig2_damage_rescue_with_ci.png"),
        os.path.join(outdir, "exp2_fig3_netflip_with_ci.png"),
        os.path.join(outdir, "exp2_fig4_position_pick_wrongB.png"),
        os.path.join(outdir, "exp2_fig5_reveal_on_flip.png"),
        metrics_csv_path,
    ]
    existing = [p for p in outputs_paths if os.path.exists(p)]
    print("Generated:")
    for p in existing:
        print(os.path.abspath(p))

    # Return data for tests or further processing
    outputs["summary_df"] = summary_df.to_dict(orient="records")
    outputs["table_path"] = table_csv
    outputs["metrics_csv_path"] = metrics_csv_path
    return outputs


def _synthetic_df() -> pd.DataFrame:
    # Minimal synthetic 5-row dataset covering both biased conditions
    data = [
        {"model": "modelA", "condition": "unbiased", "pred": "A", "gold": "A", "biased_pos": "B"},
        {"model": "modelA", "condition": "biased_to_gold", "pred": "A", "gold": "A", "biased_pos": "B", "flip_from_unbiased": 0, "reveal": 0},
        {"model": "modelA", "condition": "biased_to_wrong", "pred": "B", "gold": "A", "biased_pos": "B", "flip_from_unbiased": 1, "reveal": 1},
        {"model": "modelB", "condition": "unbiased", "pred": "C", "gold": "C", "biased_pos": "B"},
        {"model": "modelB", "condition": "biased_to_wrong", "pred": "B", "gold": "C", "biased_pos": "B", "flip_from_unbiased": 1, "reveal": 0},
    ]
    return pd.DataFrame(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 analysis")
    parser.add_argument("--input", required=False, help="Input CSV path for Experiment 2 per-item or aggregated data")
    parser.add_argument("--metrics", nargs="*", help="Paths to exp2_metrics.csv files from each model (merged)")
    parser.add_argument("--outdir", required=True, help="Directory to write outputs (PNGs/JSON)")
    parser.add_argument("--models", required=False, help="Comma-separated list of models to include (default: all)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None

    if args.metrics:
        analyze_from_metrics(args.metrics, args.outdir)
    elif args.input and os.path.exists(args.input):
        analyze(args.input, args.outdir, models)
    else:
        print("No --input provided or file not found; running with synthetic data to exercise pipeline.")
        synth_path = os.path.join(args.outdir, "synthetic_exp2.csv")
        os.makedirs(args.outdir, exist_ok=True)
        df = _synthetic_df()
        df.to_csv(synth_path, index=False)
        analyze(synth_path, args.outdir, models)


if __name__ == "__main__":
    main()


