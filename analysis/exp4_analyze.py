"""
Experiment 4 Analysis: Summarize mapped survey responses per cohort and model.

Inputs (mapped CSVs produced by analysis.exp4_map):
  - doctors_mapped.csv
  - laypeople_mapped.csv

Outputs:
  - doctors_summary_exp4.csv
  - laypeople_summary_exp4.csv
  - doctors_irr_exp4.csv (Cohen's kappa for binary, ICC(2,1) for numeric)
  - laypeople_irr_exp4.csv (Cohen's kappa for binary, ICC(2,1) for numeric)

For each cohort separately, and for each model, compute per-metric summaries:
  - Numeric metrics (e.g., 1–5 Likert): mean with 95% CI (normal approx), n
  - Binary metrics (0/1): yes_count, no_count, n
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


NUMERIC_EXCLUDE_COLS = {
    "respondent_id",
    "role",
    "case_num",
    "model",
}


def _list_metric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if c in NUMERIC_EXCLUDE_COLS:
            continue
        cols.append(c)
    return cols


def _classify_metric_series(s: pd.Series) -> Tuple[str, pd.Series]:
    """Classify a metric column as 'binary' or 'numeric' and return cleaned series.

    - Binary: all non-null values are in {0,1}
    - Numeric: convertible to float; will return series of dtype float
    """
    # Drop NA for classification
    nonnull = s.dropna()
    # Try direct binary check
    uniq = set(pd.Series(nonnull).astype(str).str.strip().str.lower().tolist())
    if uniq.issubset({"0", "1"}) and len(uniq) > 0:
        return "binary", pd.to_numeric(s, errors="coerce")
    # Try numeric
    as_num = pd.to_numeric(s, errors="coerce")
    if as_num.notna().any():
        # Still treat columns with only 0/1 as binary
        vals = set(as_num.dropna().unique().tolist())
        if vals.issubset({0.0, 1.0}) and len(vals) > 0:
            return "binary", as_num
        return "numeric", as_num
    # Fallback: treat as binary with no data
    return "binary", as_num


def _mean_ci(x: pd.Series, z: float = 1.96) -> Tuple[float, float, float, int]:
    data = pd.to_numeric(x, errors="coerce").dropna().astype(float)
    n = int(data.shape[0])
    if n <= 1:
        m = float(data.mean()) if n == 1 else float("nan")
        return m, float("nan"), float("nan"), n
    m = float(data.mean())
    sd = float(data.std(ddof=1))
    se = sd / math.sqrt(n)
    ci_low = m - z * se
    ci_high = m + z * se
    return m, float(ci_low), float(ci_high), n


def _summarize_by_model(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    metric_cols = _list_metric_columns(df)
    if "model" not in df.columns:
        return pd.DataFrame()

    for model, sub in df.groupby("model", dropna=False):
        for metric in metric_cols:
            mtype, series = _classify_metric_series(sub[metric])
            if mtype == "binary":
                cleaned = pd.to_numeric(series, errors="coerce")
                yes = int((cleaned == 1).sum())
                no = int((cleaned == 0).sum())
                n = int(cleaned.notna().sum())
                rows.append({
                    "model": model,
                    "metric": metric,
                    "type": "binary",
                    "n": n,
                    "yes": yes,
                    "no": no,
                })
            else:
                mean, lo, hi, n = _mean_ci(series)
                rows.append({
                    "model": model,
                    "metric": metric,
                    "type": "numeric",
                    "n": n,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                })

    return pd.DataFrame(rows)


def _cohen_kappa_binary(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cohen's kappa for two binary rater vectors with values in {0,1}."""
    mask = ~(np.isnan(y1) | np.isnan(y2))
    a = y1[mask]
    b = y2[mask]
    if a.size == 0:
        return float("nan")
    # Observed agreement
    po = float((a == b).mean())
    # Expected agreement
    p1_1 = float((a == 1).mean())
    p1_0 = 1.0 - p1_1
    p2_1 = float((b == 1).mean())
    p2_0 = 1.0 - p2_1
    pe = p1_1 * p2_1 + p1_0 * p2_0
    denom = (1.0 - pe)
    if denom == 0.0:
        return float("nan")
    return (po - pe) / denom


def _icc_2_1(mat: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.

    mat: shape (n_items, n_raters), no NaNs, n_items>=2, n_raters>=2
    """
    n, k = mat.shape
    if n < 2 or k < 2:
        return float("nan")
    mean_rows = mat.mean(axis=1, keepdims=True)
    mean_cols = mat.mean(axis=0, keepdims=True)
    grand = mat.mean()
    # Mean squares
    msr = k * float(((mean_rows - grand) ** 2).sum()) / (n - 1)
    msc = n * float(((mean_cols - grand) ** 2).sum()) / (k - 1)
    # Residual sum of squares
    residual = mat - mean_rows - mean_cols + grand
    mse = float((residual ** 2).sum()) / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
    if denom == 0.0:
        return float("nan")
    return (msr - mse) / denom


def _compute_irr_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IRR per model and metric.

    - Binary metrics: average pairwise Cohen's kappa across raters
    - Numeric metrics: ICC(2,1) using items=case_num, raters=respondent_id
    """
    rows: List[Dict] = []
    metric_cols = _list_metric_columns(df)
    if not {"model", "respondent_id", "case_num"}.issubset(df.columns):
        return pd.DataFrame()

    for model, sub in df.groupby("model", dropna=False):
        raters = [str(x) for x in sorted(sub["respondent_id"].astype(str).unique().tolist())]
        for metric in metric_cols:
            mtype, series_type = _classify_metric_series(sub[metric])
            if mtype == "binary":
                # Build aligned series per rater by case
                pair_kappas: List[float] = []
                # pivot by case_num x respondent_id
                pivot = sub.pivot_table(index="case_num", columns="respondent_id", values=metric, aggfunc="first")
                pivot = pivot.reindex(sorted(pivot.index))
                cols = list(pivot.columns)
                # compute pairwise kappa
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        a = pd.to_numeric(pivot[cols[i]], errors="coerce").to_numpy(dtype=float)
                        b = pd.to_numeric(pivot[cols[j]], errors="coerce").to_numpy(dtype=float)
                        kappa = _cohen_kappa_binary(a, b)
                        if not np.isnan(kappa):
                            pair_kappas.append(float(kappa))
                val = float(np.mean(pair_kappas)) if pair_kappas else float("nan")
                rows.append({
                    "model": model,
                    "metric": metric,
                    "irr_type": "cohen_kappa_avg",
                    "value": val,
                    "n_raters": len(cols),
                    "n_items": int(pivot.shape[0]),
                })
            else:
                # Numeric: ICC(2,1)
                pivot = sub.pivot_table(index="case_num", columns="respondent_id", values=metric, aggfunc="first")
                # Require complete cases across raters; cast entire DataFrame to numeric
                pivot_num = pivot.apply(pd.to_numeric, errors="coerce")
                mat = pivot_num.to_numpy(dtype=float)
                # Drop rows with NaNs
                mask_rows = ~np.isnan(mat).any(axis=1)
                mat2 = mat[mask_rows]
                val = _icc_2_1(mat2) if mat2.size and mat2.shape[1] >= 2 and mat2.shape[0] >= 2 else float("nan")
                rows.append({
                    "model": model,
                    "metric": metric,
                    "irr_type": "icc_2_1",
                    "value": val,
                    "n_raters": int(pivot.shape[1]),
                    "n_items": int(mat2.shape[0]) if mat2.size else 0,
                })

    return pd.DataFrame(rows)


def _analyze_one(input_csv: str, out_csv: str) -> None:
    if not input_csv or not os.path.exists(input_csv):
        return
    df = pd.read_csv(input_csv)
    summary = _summarize_by_model(df)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    summary.to_csv(out_csv, index=False)


def _irr_one(input_csv: str, out_csv: str) -> None:
    if not input_csv or not os.path.exists(input_csv):
        return
    df = pd.read_csv(input_csv)
    irr = _compute_irr_by_model(df)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    irr.to_csv(out_csv, index=False)


def _set_publication_rc() -> None:
    mpl.rcParams.update({
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


def _continuous_metric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in _list_metric_columns(df):
        mt, series = _classify_metric_series(df[c])
        if mt == "numeric":
            # Require at least 3 distinct numeric values to exclude binary/degenerate
            uniq = set(pd.to_numeric(series, errors="coerce").dropna().unique().tolist())
            if len(uniq) >= 3:
                cols.append(c)
    return cols


def _means_by_model(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, List[float]]:
    models = [m for m in ["Claude", "ChatGPT", "Gemini"] if m in df["model"].unique().tolist()]
    for m in df["model"].unique().tolist():
        if m not in models:
            models.append(m)
    series_by_model: Dict[str, List[float]] = {}
    for m in models:
        sub = df[df["model"] == m]
        vals: List[float] = []
        for c in metric_cols:
            v = pd.to_numeric(sub[c], errors="coerce").dropna()
            vals.append(float(v.mean()) if not v.empty else np.nan)
        series_by_model[m] = vals
    return series_by_model


def _plot_radar(categories: List[str], model_to_vals: Dict[str, List[float]], outfile: str, title: str, labels: List[str] | None = None) -> None:
    if not categories or not model_to_vals:
        return
    _set_publication_rc()
    # Angles
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Radial grid: try to infer scale from data; default 1..5
    all_vals = np.array([v for vals in model_to_vals.values() for v in vals if not np.isnan(v)])
    rmin = 0.0
    rmax = 5.0 if all_vals.size == 0 else float(np.nanmax(all_vals))
    rmax = max(1.0, round(rmax))
    ax.set_rlabel_position(0)
    ax.set_ylim(0, rmax)

    # Labels around circle
    ax.set_xticks(angles[:-1])
    tick_labels = labels if labels is not None else categories
    ax.set_xticklabels(tick_labels)
    # Enforce requested font sizes explicitly
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(30)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(30)
    # Push tick labels outward for readability
    ax.tick_params(axis='x', pad=28)
    ax.tick_params(axis='y', pad=24)

    colors = {
        "Claude": "#4C78A8",
        "ChatGPT": "#F58518",
        "Gemini": "#54A24B",
    }

    for model, vals in model_to_vals.items():
        if not vals:
            continue
        data = vals + vals[:1]
        ax.plot(angles, data, linewidth=2, label=model, color=colors.get(model))
        ax.fill(angles, data, alpha=0.12, color=colors.get(model))

    ax.set_title(title, pad=28, fontsize=32)
    ax.legend(loc="upper right", bbox_to_anchor=(1.6, 1.16), prop={"size": 30})
    outdir = os.path.dirname(outfile) or "."
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    root, _ = os.path.splitext(outfile)
    fig.savefig(f"{root}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp4 mapped CSVs by model and metric.")
    parser.add_argument("--doctors", required=False, help="Path to doctors_mapped.csv")
    parser.add_argument("--laypeople", required=False, help="Path to laypeople_mapped.csv")
    parser.add_argument("--outdir", required=True, help="Directory to write summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.doctors and os.path.exists(args.doctors):
        _analyze_one(args.doctors, os.path.join(args.outdir, "doctors_summary_exp4.csv"))
        _irr_one(args.doctors, os.path.join(args.outdir, "doctors_irr_exp4.csv"))
        # Radar for doctors: only continuous metrics
        df_doc = pd.read_csv(args.doctors)
        doc_metrics = _continuous_metric_columns(df_doc)
        # Reorder to avoid overlapping labels: separate key labels around the circle
        apr_label = "Appropriateness of urgency"
        comp_label = "Completeness of response"
        harm_label = None
        for m in doc_metrics:
            if m.strip().lower().startswith("potential harm"):
                harm_label = m
                break
        if apr_label in doc_metrics or comp_label in doc_metrics or harm_label in doc_metrics:
            others = [m for m in doc_metrics if m not in {apr_label, comp_label} and m != harm_label]
            new_order = []
            if apr_label in doc_metrics:
                new_order.append(apr_label)
            # split others into two halves
            mid = len(others) // 2
            first_half = others[:mid]
            second_half = others[mid:]
            new_order.extend(first_half)
            if harm_label:
                new_order.append(harm_label)
            new_order.extend(second_half)
            if comp_label in doc_metrics:
                new_order.append(comp_label)
            # Keep any remaining not included (safety)
            for m in doc_metrics:
                if m not in new_order:
                    new_order.append(m)
            doc_metrics = new_order
        if doc_metrics:
            doc_vals = _means_by_model(df_doc, doc_metrics)
            # Pretty labels: shorten "Potential harm if followed" if present
            pretty_labels = [
                ("Potential harm" if m.strip().lower().startswith("potential harm") else m)
                for m in doc_metrics
            ]
            _plot_radar(doc_metrics, doc_vals, os.path.join(args.outdir, "exp4_radar_clinicians.pdf"), title="Clinician Ratings", labels=pretty_labels)
    if args.laypeople and os.path.exists(args.laypeople):
        _analyze_one(args.laypeople, os.path.join(args.outdir, "laypeople_summary_exp4.csv"))
        _irr_one(args.laypeople, os.path.join(args.outdir, "laypeople_irr_exp4.csv"))
        # Radar for laypeople: use all numeric metrics (treat 1–5 Likert)
        df_lay = pd.read_csv(args.laypeople)
        lay_metrics = _continuous_metric_columns(df_lay)
        if lay_metrics:
            lay_vals = _means_by_model(df_lay, lay_metrics)
            _plot_radar(lay_metrics, lay_vals, os.path.join(args.outdir, "exp4_radar_laypeople.pdf"), title="Layperson Ratings")

    print("Wrote summaries (if inputs existed) to:")
    print(os.path.join(args.outdir, "doctors_summary_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_summary_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_irr_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_irr_exp4.csv"))


if __name__ == "__main__":
    main()


