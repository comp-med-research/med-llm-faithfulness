"""
Experiment 4 Analysis: Summarize mapped survey responses per cohort and model.

Inputs (mapped CSVs produced by analysis.exp4_map):
  - doctors_mapped.csv
  - laypeople_mapped.csv

Outputs:
  - doctors_summary_exp4.csv
  - laypeople_summary_exp4.csv
  - doctors_irr_exp4.csv (ICC(2,1) for numeric)
  - laypeople_irr_exp4.csv (ICC(2,1) for numeric)

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


def _icc_2_k(mat: np.ndarray) -> float:
    """ICC(2,k): two-way random effects, absolute agreement, mean of k raters.

    mat: shape (n_items, n_raters), no NaNs, n_items>=2, n_raters>=2
    """
    n, k = mat.shape
    if n < 2 or k < 2:
        return float("nan")
    mean_rows = mat.mean(axis=1, keepdims=True)
    mean_cols = mat.mean(axis=0, keepdims=True)
    grand = mat.mean()
    msr = k * float(((mean_rows - grand) ** 2).sum()) / (n - 1)
    msc = n * float(((mean_cols - grand) ** 2).sum()) / (k - 1)
    residual = mat - mean_rows - mean_cols + grand
    mse = float((residual ** 2).sum()) / ((n - 1) * (k - 1))
    denom = msr + (msc - mse) / n
    if denom == 0.0:
        return float("nan")
    return (msr - mse) / denom

def _bootstrap_ci(stat_fn, mat: np.ndarray, num_bootstrap: int = 5000, alpha: float = 0.05, random_state: int = 0) -> Tuple[float, float]:
    """Bootstrap CI for a statistic computed on rows of mat (items×raters).

    Resamples items with replacement and recomputes the statistic each time.
    Returns (ci_low, ci_high).
    """
    if mat.size == 0 or mat.shape[0] < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    n = mat.shape[0]
    samples: List[float] = []
    for _ in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        v = float(stat_fn(mat[idx, :]))
        if not np.isnan(v):
            samples.append(v)
    if not samples:
        return float("nan"), float("nan")
    lo = float(np.quantile(samples, alpha / 2.0))
    hi = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return lo, hi

def _compute_irr_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Compute IRR per model and metric.

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
            if mtype != "binary":
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


def _compute_overall_icc(df: pd.DataFrame, item_mode: str = "model_case") -> Dict[str, float]:
    """Compute overall ICCs across all models using numeric questions.

    item_mode:
      - "model_case": items are (model×case_num); each rater's value is the
        mean of all numeric metrics for that model×case.
      - "metric_items": items are (model×case_num×metric); each numeric metric
        is treated as a separate item.
    Returns a dict with keys: value (ICC(2,1)), value_icc2k (ICC(2,k)), n_raters, n_items.
    """
    metric_cols = _continuous_metric_columns(df)
    if not metric_cols:
        return {"value": float("nan"), "n_raters": 0, "n_items": 0}

    if item_mode == "metric_items":
        # Long format over metrics
        frames: List[pd.DataFrame] = []
        for m in metric_cols:
            sub = df[["model", "case_num", "respondent_id", m]].copy()
            sub = sub.rename(columns={m: "value"})
            sub["metric"] = m
            frames.append(sub)
        work = pd.concat(frames, ignore_index=True)
        if not {"model", "case_num", "respondent_id", "metric"}.issubset(work.columns):
            return {"value": float("nan"), "n_raters": 0, "n_items": 0}
        pivot = work.pivot_table(index=["model", "case_num", "metric"], columns="respondent_id", values="value", aggfunc="first")
    else:
        # Compute per-row mean across numeric metrics, then pivot by model×case
        work = df.copy()
        for c in metric_cols:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work["_overall_mean"] = work[metric_cols].mean(axis=1, skipna=True)
        if not {"model", "case_num", "respondent_id"}.issubset(work.columns):
            return {"value": float("nan"), "n_raters": 0, "n_items": 0}
        pivot = work.pivot_table(index=["model", "case_num"], columns="respondent_id", values="_overall_mean", aggfunc="first")

    pivot_num = pivot.apply(pd.to_numeric, errors="coerce")
    mat = pivot_num.to_numpy(dtype=float)
    # Drop rows with any NaNs to ensure complete cases across raters
    mask_rows = ~np.isnan(mat).any(axis=1)
    mat2 = mat[mask_rows]
    icc2_1 = _icc_2_1(mat2) if mat2.size and mat2.shape[1] >= 2 and mat2.shape[0] >= 2 else float("nan")
    icc2_k = _icc_2_k(mat2) if mat2.size and mat2.shape[1] >= 2 and mat2.shape[0] >= 2 else float("nan")
    icc2_1_lo, icc2_1_hi = _bootstrap_ci(_icc_2_1, mat2) if mat2.size else (float("nan"), float("nan"))
    icc2_k_lo, icc2_k_hi = _bootstrap_ci(_icc_2_k, mat2) if mat2.size else (float("nan"), float("nan"))
    return {
        "value": float(icc2_1),
        "n_raters": int(pivot.shape[1]),
        "n_items": int(mat2.shape[0]) if mat2.size else 0,
        "value_icc2k": float(icc2_k),
        "ci_low": float(icc2_1_lo),
        "ci_high": float(icc2_1_hi),
        "icc2k_ci_low": float(icc2_k_lo),
        "icc2k_ci_high": float(icc2_k_hi),
    }

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
    models = [m for m in ["Claude 4.1 Opus", "ChatGPT-5", "Gemini Pro 2.5"] if m in df["model"].unique().tolist()]
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


def _plot_radar(
    categories: List[str],
    model_to_vals: Dict[str, List[float]],
    outfile: str,
    title: str,
    labels: List[str] | None = None,
) -> None:
    if not categories or not model_to_vals:
        return
    _set_publication_rc()
    import textwrap as _tw
    # Angles
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    # Larger canvas and reserved right margin for legend
    fig = plt.figure(figsize=(10.4, 10.4))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    # Leave extra space on the right for legend, and a bit on bottom/top for labels
    # Reserve generous right margin for legend
    fig.subplots_adjust(left=0.13, right=0.90, top=0.88, bottom=0.12)

    # Radial grid: try to infer scale from data; default 1..5
    all_vals = np.array([v for vals in model_to_vals.values() for v in vals if not np.isnan(v)])
    rmin = 0.0
    rmax = 5.0 if all_vals.size == 0 else float(np.nanmax(all_vals))
    rmax = max(1.0, round(rmax))
    ax.set_rlabel_position(0)
    ax.set_ylim(0, rmax)

    # Labels around circle (wrap long labels; keep horizontal for readability)
    ax.set_xticks(angles[:-1])
    raw_labels = labels if labels is not None else categories
    def _wrap(s: str, width: int = 18) -> str:
        # Prefer explicit wrap for known long labels
        s2 = s.replace("Completeness of response", "Completeness\nof response\n")
        s2 = s2.replace("Logical consistency", "Logical\nconsistency\n")
        # Generic wrap as a fallback
        return "\n".join(_tw.wrap(s2, width=width)) if "\n" not in s2 else s2
    tick_labels = [_wrap(str(s)) for s in raw_labels]
    ax.set_xticklabels(tick_labels)
    # Enforce requested font sizes explicitly
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(30)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(30)
    # Push tick labels outward for readability (less pad with multiline)
    # Push labels farther away from the circle to avoid overlap
    ax.tick_params(axis='x', pad=85)
    ax.tick_params(axis='y', pad=24)

    colors = {
        "Claude 4.1 Opus": "#4C78A8",
        "ChatGPT-5": "#F58518",
        "Gemini Pro 2.5": "#54A24B",
    }

    for model, vals in model_to_vals.items():
        if not vals:
            continue
        data = vals + vals[:1]
        ax.plot(angles, data, linewidth=2, label=model, color=colors.get(model))
        ax.fill(angles, data, alpha=0.12, color=colors.get(model))

    ax.set_title(title, pad=32, fontsize=32)
    # Place legend at the very top-right (tight to top)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.25), prop={"size":33}, frameon=False)
    # No per-label overrides; spacing is handled via tick_params and layout
    outdir = os.path.dirname(outfile) or "."
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    root, _ = os.path.splitext(outfile)
    fig.savefig(f"{root}.png", dpi=300)
    plt.close(fig)


def _per_item_means(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Compute mean per (model, case_num) for the given numeric metric columns."""
    work = df.copy()
    for c in metric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    grouped = work.groupby(["model", "case_num"], dropna=False)[metric_cols].mean().reset_index()
    return grouped


def _plot_correlations_by_metric(
    doc_means: pd.DataFrame,
    lay_means: pd.DataFrame,
    doc_metrics: List[str],
    lay_metrics: List[str],
    outdir: str,
) -> None:
    if doc_means is None or lay_means is None or not doc_metrics or not lay_metrics:
        return
    _set_publication_rc()
    merged = pd.merge(doc_means, lay_means, on=["model", "case_num"], suffixes=("_doc", "_lay"))
    if merged.empty:
        return
    import math as _math

    # Collect correlation stats to CSV
    corr_rows: List[Dict[str, float | str | int]] = []

    def _pearson_r_p(xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
        r_val = float(np.corrcoef(xv, yv)[0, 1])
        p_val: float
        try:
            import scipy.stats as _stats  # type: ignore
            r_val2, p_val = _stats.pearsonr(xv, yv)
            return float(r_val2), float(p_val)
        except Exception:
            # Permutation test fallback (two-sided)
            rng = np.random.default_rng(0)
            B = 5000
            count = 0
            for _ in range(B):
                perm = rng.permutation(yv)
                r_perm = float(np.corrcoef(xv, perm)[0, 1])
                if not np.isnan(r_perm) and abs(r_perm) >= abs(r_val):
                    count += 1
            p_val = (count + 1.0) / (B + 1.0)
            return r_val, float(p_val)

    for lmet in lay_metrics:
        # Grid layout for doctor metrics
        n = len(doc_metrics)
        # Use a 3×2-style grid (rows×cols) with the last (bottom-right) cell reserved for a legend.
        # Compute rows based on n plots + 1 legend cell.
        cols = 2
        rows = int(_math.ceil((n + 1) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)

        # Reserve bottom-right as a legend pane the same size as a subplot
        legend_ax = axes[rows - 1, cols - 1]
        legend_ax.axis("off")
        from matplotlib.lines import Line2D as _Line2D
        _legend_handles = [
            _Line2D([0], [0], marker='o', color='w', markerfacecolor="#4C78A8", label="Claude 4.1 Opus", markersize=25),
            _Line2D([0], [0], marker='o', color='w', markerfacecolor="#F58518", label="ChatGPT-5", markersize=25),
            _Line2D([0], [0], marker='o', color='w', markerfacecolor="#54A24B", label="Gemini Pro 2.5", markersize=25),
        ]
        legend_ax.legend(handles=_legend_handles, loc="center", frameon=False, fontsize=25)

        for idx, dmet in enumerate(doc_metrics):
            # Map metrics into grid; last cell is reserved for legend
            cell = idx
            r = cell // cols
            c = cell % cols
            ax = axes[r, c]
            # Columns may or may not have suffixes depending on overlap
            dcol = dmet + "_doc" if (dmet + "_doc") in merged.columns else dmet
            lcol = lmet + "_lay" if (lmet + "_lay") in merged.columns else lmet
            if dcol not in merged.columns or lcol not in merged.columns:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(dmet)
                ax.set_ylabel(lmet)
                ax.grid(True, linestyle=":", alpha=0.4)
                continue
            x = pd.to_numeric(merged[dcol], errors="coerce")
            y = pd.to_numeric(merged[lcol], errors="coerce")
            mask = x.notna() & y.notna()
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            if xv.size == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel(dmet)
                ax.set_ylabel(lmet)
                ax.grid(True, linestyle=":", alpha=0.4)
                continue
            # Per-model scatter and fits
            colors = {"Claude 4.1 Opus": "#4C78A8", "ChatGPT-5": "#F58518", "Gemini Pro 2.5": "#54A24B"}
            models_present = [m for m in ["Claude 4.1 Opus", "ChatGPT-5", "Gemini Pro 2.5"] if m in merged["model"].unique().tolist()]
            legends = []
            for m in models_present:
                sub = merged[merged["model"] == m]
                x_m = pd.to_numeric(sub[dcol], errors="coerce")
                y_m = pd.to_numeric(sub[lcol], errors="coerce")
                mask_m = x_m.notna() & y_m.notna()
                xv_m = x_m[mask_m].to_numpy(dtype=float)
                yv_m = y_m[mask_m].to_numpy(dtype=float)
                if xv_m.size == 0:
                    continue
                ax.scatter(xv_m, yv_m, s=20, alpha=0.7, color=colors.get(m), label=m)
                try:
                    coef_m = np.polyfit(xv_m, yv_m, 1)
                    xx_m = np.linspace(np.nanmin(xv_m), np.nanmax(xv_m), 30)
                    yy_m = coef_m[0] * xx_m + coef_m[1]
                    ax.plot(xx_m, yy_m, color=colors.get(m), linewidth=1.4)
                except Exception:
                    pass
                r_m, p_m = _pearson_r_p(xv_m, yv_m)
                corr_rows.append({
                    "lay_metric": lmet,
                    "clinician_metric": dmet,
                    "model": m,
                    "n": int(xv_m.size),
                    "pearson_r": float(r_m),
                    "p_value": float(p_m),
                })
                legends.append(f"{m}: r={r_m:.2f}, p={p_m:.3f}")

            # Overall r and p still recorded in CSV (no line drawn)
            r_all, p_all = _pearson_r_p(xv, yv)
            corr_rows.append({
                "lay_metric": lmet,
                "clinician_metric": dmet,
                "model": "All",
                "n": int(xv.size),
                "pearson_r": float(r_all),
                "p_value": float(p_all),
            })

            # Remove per-panel title; keep x-axis label
            ax.set_xlabel(dmet)
            ax.set_ylabel(lmet)
            ax.grid(True, linestyle=":", alpha=0.4)
            # No per-axes legends; legend is rendered in the [0,0] pane

        # Hide any leftover axes after placing n plots + 1 legend cell
        used = n + 1
        total = rows * cols
        if used < total:
            for j in range(used, total):
                rr = j // cols
                cc = j % cols
                axes[rr, cc].set_visible(False)

        fig.suptitle(f"Correlation: Layperson {lmet} vs Clinician metrics", y=0.98, fontsize=16)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        os.makedirs(outdir, exist_ok=True)
        base = os.path.join(outdir, f"exp4_corr_{lmet.replace(' ', '_').lower()}")
        fig.savefig(base + ".pdf", dpi=300, bbox_inches="tight")
        fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Landscape version (2 rows × 3 cols) with legend in bottom-right cell
        cols_ls = 3
        rows_ls = int(_math.ceil((n + 1) / cols_ls))
        fig2, axes2 = plt.subplots(rows_ls, cols_ls, figsize=(6.2 * cols_ls, 3.8 * rows_ls))
        if not isinstance(axes2, np.ndarray):
            axes2 = np.array([axes2])
        axes2 = axes2.reshape(rows_ls, cols_ls)

        legend_ax2 = axes2[rows_ls - 1, cols_ls - 1]
        legend_ax2.axis("off")
        from matplotlib.lines import Line2D as _Line2D
        _legend_handles2 = [
            _Line2D([0], [0], marker='o', color='w', markerfacecolor="#4C78A8", label="Claude 4.1 Opus", markersize=25),
            _Line2D([0], [0], marker='o', color='w', markerfacecolor="#F58518", label="ChatGPT-5", markersize=25),
            _Line2D([0], [0], marker='o', color='w', markerfacecolor="#54A24B", label="Gemini Pro 2.5", markersize=25),
        ]
        legend_ax2.legend(handles=_legend_handles2, loc="center", frameon=False, fontsize=25)

        # positions excluding legend cell
        plot_positions_ls = [(rr, cc) for rr in range(rows_ls) for cc in range(cols_ls) if not (rr == rows_ls - 1 and cc == cols_ls - 1)]

        for idx, dmet in enumerate(doc_metrics):
            if idx >= len(plot_positions_ls):
                break
            rr, cc = plot_positions_ls[idx]
            ax2 = axes2[rr, cc]
            dcol = dmet + "_doc" if (dmet + "_doc") in merged.columns else dmet
            lcol = lmet + "_lay" if (lmet + "_lay") in merged.columns else lmet
            if dcol not in merged.columns or lcol not in merged.columns:
                ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_xlabel(dmet)
                ax2.set_ylabel(lmet)
                ax2.grid(True, linestyle=":", alpha=0.4)
                continue
            x = pd.to_numeric(merged[dcol], errors="coerce")
            y = pd.to_numeric(merged[lcol], errors="coerce")
            mask = x.notna() & y.notna()
            xv = x[mask].to_numpy(dtype=float)
            yv = y[mask].to_numpy(dtype=float)
            if xv.size == 0:
                ax2.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax2.transAxes)
                ax2.set_xlabel(dmet)
                ax2.set_ylabel(lmet)
                ax2.grid(True, linestyle=":", alpha=0.4)
                continue
            colors = {"Claude 4.1 Opus": "#4C78A8", "ChatGPT-5": "#F58518", "Gemini Pro 2.5": "#54A24B"}
            models_present = [m for m in ["Claude 4.1 Opus", "ChatGPT-5", "Gemini Pro 2.5"] if m in merged["model"].unique().tolist()]
            for m in models_present:
                sub = merged[merged["model"] == m]
                x_m = pd.to_numeric(sub[dcol], errors="coerce")
                y_m = pd.to_numeric(sub[lcol], errors="coerce")
                mask_m = x_m.notna() & y_m.notna()
                xv_m = x_m[mask_m].to_numpy(dtype=float)
                yv_m = y_m[mask_m].to_numpy(dtype=float)
                if xv_m.size == 0:
                    continue
                ax2.scatter(xv_m, yv_m, s=20, alpha=0.7, color=colors.get(m), label=m)
                try:
                    coef_m = np.polyfit(xv_m, yv_m, 1)
                    xx_m = np.linspace(np.nanmin(xv_m), np.nanmax(xv_m), 30)
                    yy_m = coef_m[0] * xx_m + coef_m[1]
                    ax2.plot(xx_m, yy_m, color=colors.get(m), linewidth=1.4)
                except Exception:
                    pass
            # overall r/p saved already above (no need to add twice)
            ax2.set_xlabel(dmet)
            ax2.set_ylabel(lmet)
            ax2.grid(True, linestyle=":", alpha=0.4)

        used_ls = min(n, len(plot_positions_ls)) + 1
        total_ls = rows_ls * cols_ls
        if used_ls < total_ls:
            for j in range(used_ls, total_ls):
                rls = j // cols_ls
                cls = j % cols_ls
                axes2[rls, cls].set_visible(False)

        fig2.suptitle(f"Correlation: Layperson {lmet} vs Clinician metrics", y=0.98, fontsize=16)
        fig2.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        base_ls = os.path.join(outdir, f"exp4_corr_{lmet.replace(' ', '_').lower()}_landscape")
        fig2.savefig(base_ls + ".pdf", dpi=300, bbox_inches="tight")
        fig2.savefig(base_ls + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

    # Write correlation CSV (per model and overall)
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        os.makedirs(outdir, exist_ok=True)
        corr_df.to_csv(os.path.join(outdir, "exp4_correlations_by_model.csv"), index=False)

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp4 mapped CSVs by model and metric.")
    parser.add_argument("--doctors", required=False, help="Path to doctors_mapped.csv")
    parser.add_argument("--laypeople", required=False, help="Path to laypeople_mapped.csv")
    parser.add_argument("--outdir", required=True, help="Directory to write summary CSVs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # For cross-cohort correlation plotting
    doc_means_df: pd.DataFrame = pd.DataFrame()
    _doc_metrics_for_corr: List[str] = []

    if args.doctors and os.path.exists(args.doctors):
        _analyze_one(args.doctors, os.path.join(args.outdir, "doctors_summary_exp4.csv"))
        _irr_one(args.doctors, os.path.join(args.outdir, "doctors_irr_exp4.csv"))
        # Radar for doctors: only continuous metrics
        df_doc = pd.read_csv(args.doctors)
        # Overall ICC across all models and questions
        overall_doc = _compute_overall_icc(df_doc, item_mode="model_case")
        pd.DataFrame([{**overall_doc, "group": "doctors", "irr_type": "icc_2_1_overall"}]).to_csv(
            os.path.join(args.outdir, "doctors_overall_icc_exp4.csv"), index=False
        )
        # Also write ICC(2,k)
        pd.DataFrame([{"value": overall_doc.get("value_icc2k", float("nan")), "n_raters": overall_doc.get("n_raters", 0), "n_items": overall_doc.get("n_items", 0), "group": "doctors", "irr_type": "icc_2_k_overall"}]).to_csv(
            os.path.join(args.outdir, "doctors_overall_icc2k_exp4.csv"), index=False
        )
        # Metric-as-item mode (e.g., 3×30×5 = 450 for clinicians)
        overall_doc_items = _compute_overall_icc(df_doc, item_mode="metric_items")
        pd.DataFrame([{**overall_doc_items, "group": "doctors", "irr_type": "icc_2_1_overall_metric_items"}]).to_csv(
            os.path.join(args.outdir, "doctors_overall_icc_metric_items_exp4.csv"), index=False
        )
        pd.DataFrame([{"value": overall_doc_items.get("value_icc2k", float("nan")), "n_raters": overall_doc_items.get("n_raters", 0), "n_items": overall_doc_items.get("n_items", 0), "group": "doctors", "irr_type": "icc_2_k_overall_metric_items"}]).to_csv(
            os.path.join(args.outdir, "doctors_overall_icc2k_metric_items_exp4.csv"), index=False
        )
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
            # Per-item means for clinician continuous metrics (for correlations)
            try:
                doc_means_df = _per_item_means(df_doc, doc_metrics)
                # Write CSV snapshot
                out_csv = os.path.join(args.outdir, "doctors_per_item_means_exp4.csv")
                doc_means_df.to_csv(out_csv, index=False)
                _doc_metrics_for_corr = list(doc_metrics)
            except Exception as _e:
                print(f"[exp4] Warning: clinician per-item means failed: {_e}")
        # Skip overall binary agreement (Cohen's kappa) due to low positives
    if args.laypeople and os.path.exists(args.laypeople):
        _analyze_one(args.laypeople, os.path.join(args.outdir, "laypeople_summary_exp4.csv"))
        _irr_one(args.laypeople, os.path.join(args.outdir, "laypeople_irr_exp4.csv"))
        # Radar for laypeople: use all numeric metrics (treat 1–5 Likert)
        df_lay = pd.read_csv(args.laypeople)
        # Overall ICC across all models and questions
        overall_lay = _compute_overall_icc(df_lay, item_mode="model_case")
        pd.DataFrame([{**overall_lay, "group": "laypeople", "irr_type": "icc_2_1_overall"}]).to_csv(
            os.path.join(args.outdir, "laypeople_overall_icc_exp4.csv"), index=False
        )
        # Also write ICC(2,k)
        pd.DataFrame([{"value": overall_lay.get("value_icc2k", float("nan")), "n_raters": overall_lay.get("n_raters", 0), "n_items": overall_lay.get("n_items", 0), "group": "laypeople", "irr_type": "icc_2_k_overall"}]).to_csv(
            os.path.join(args.outdir, "laypeople_overall_icc2k_exp4.csv"), index=False
        )
        # Metric-as-item mode (e.g., 3×30×3 = 270 for laypeople)
        overall_lay_items = _compute_overall_icc(df_lay, item_mode="metric_items")
        pd.DataFrame([{**overall_lay_items, "group": "laypeople", "irr_type": "icc_2_1_overall_metric_items"}]).to_csv(
            os.path.join(args.outdir, "laypeople_overall_icc_metric_items_exp4.csv"), index=False
        )
        pd.DataFrame([{"value": overall_lay_items.get("value_icc2k", float("nan")), "n_raters": overall_lay_items.get("n_raters", 0), "n_items": overall_lay_items.get("n_items", 0), "group": "laypeople", "irr_type": "icc_2_k_overall_metric_items"}]).to_csv(
            os.path.join(args.outdir, "laypeople_overall_icc2k_metric_items_exp4.csv"), index=False
        )
        lay_metrics = _continuous_metric_columns(df_lay)
        if lay_metrics:
            lay_vals = _means_by_model(df_lay, lay_metrics)
            _plot_radar(
                lay_metrics,
                lay_vals,
                os.path.join(args.outdir, "exp4_radar_laypeople.pdf"),
                title="Layperson Ratings",
            )
            # Correlations between clinician and layperson per-item means
            try:
                if not doc_means_df.empty:
                    lay_means_df = _per_item_means(df_lay, lay_metrics)
                    # Write CSV snapshot
                    lay_csv = os.path.join(args.outdir, "laypeople_per_item_means_exp4.csv")
                    lay_means_df.to_csv(lay_csv, index=False)
                    _plot_correlations_by_metric(
                        doc_means=doc_means_df,
                        lay_means=lay_means_df,
                        doc_metrics=_doc_metrics_for_corr,
                        lay_metrics=list(lay_metrics),
                        outdir=args.outdir,
                    )
            except Exception as _e:
                print(f"[exp4] Warning: correlation plotting failed: {_e}")
        # Skip overall binary agreement (Cohen's kappa) due to low positives

    print("Wrote summaries (if inputs existed) to:")
    print(os.path.join(args.outdir, "doctors_summary_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_summary_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_irr_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_irr_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_overall_icc_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_overall_icc_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_overall_icc2k_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_overall_icc2k_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_overall_icc_metric_items_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_overall_icc_metric_items_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_overall_icc2k_metric_items_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_overall_icc2k_metric_items_exp4.csv"))
    print(os.path.join(args.outdir, "doctors_per_item_means_exp4.csv"))
    print(os.path.join(args.outdir, "laypeople_per_item_means_exp4.csv"))


if __name__ == "__main__":
    main()


