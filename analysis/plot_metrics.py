"""
Plot metrics with 95% confidence intervals from metrics CSVs produced by
analysis.compute_ablation_metrics.

Usage examples:
    # Per-ablation plots for one model
    python -m analysis.plot_metrics \
        --metrics /home/gcp_dev/med-llm-faithfulness/results/final/exp1/claude/claude.metrics.csv \
        --outdir /home/gcp_dev/med-llm-faithfulness/results/final/exp1/claude/plots \
        --per-ablation

    # Compare macro metrics across multiple models
    python -m analysis.plot_metrics \
        --metrics \
          /home/gcp_dev/med-llm-faithfulness/results/final/exp1/claude/claude.metrics.csv \
          /home/gcp_dev/med-llm-faithfulness/results/final/exp1/chatgpt/chatgpt.metrics.csv \
          /home/gcp_dev/med-llm-faithfulness/results/final/exp1/gemini/gemini.metrics.csv \
        --outdir /home/gcp_dev/med-llm-faithfulness/results/final/exp1/plots
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import numpy as _np
import matplotlib as _mpl


GLOBAL_METRICS_TO_PLOT = [
    ("baseline", "accuracy", "Baseline accuracy"),
    ("global", "macro_ablation_accuracy", "Macro ablation accuracy"),
    ("global", "macro_ablation_netflip_vs_baseline", "Macro ablation netflip"),
    ("global", "macro_ablation_damage_rate", "Macro damage rate"),
    ("global", "macro_ablation_rescue_rate", "Macro rescue rate"),
    ("global", "delta_accuracy", "Delta accuracy (baseline - macro)"),
    ("global", "netflip_damage_minus_rescue", "NetFlip (damage - rescue)"),
    ("global", "causal_density_mean", "Causal density (per-example mean)"),
]


def _set_publication_rc() -> None:
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


def _label_for_csv(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    parent = os.path.basename(os.path.dirname(path))
    # Prefer the stem if it contains the model name; include parent for clarity
    return f"{parent}-{stem}" if parent and parent not in stem else stem


def _ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def _plot_global_comparison(metrics_paths: List[str], outdir: str) -> None:
    frames: List[pd.DataFrame] = []
    for p in metrics_paths:
        df = pd.read_csv(p)
        df = df.copy()
        df["model"] = _label_for_csv(p)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    for scope, metric, title in GLOBAL_METRICS_TO_PLOT:
        sel = all_df[(all_df["scope"] == scope) & (all_df["metric"] == metric)].copy()
        if sel.empty:
            continue
        # yerr from CI
        sel["yerr_low"] = (sel["value"] - sel["ci_low"]).clip(lower=0)
        sel["yerr_high"] = (sel["ci_high"] - sel["value"]).clip(lower=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = range(len(sel))
        ax.bar(x, sel["value"], yerr=[sel["yerr_low"], sel["yerr_high"]], capsize=4, color="#4C78A8")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sel["model"], rotation=30, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        fig.tight_layout()
        fname = f"global_{metric}.png"
        out_path = os.path.join(outdir, fname)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def _prettify_model_label(raw: str) -> str:
    """Map file-derived labels to clean model display names.

    Special-cases common models; otherwise drops suffixes and title-cases.
    """
    low = raw.lower()
    if "chatgpt" in low:
        return "ChatGPT-5"
    if "claude" in low:
        return "Claude 4.1 Opus"
    if "gemini" in low:
        return "Gemini Pro 2.5"
    if "llama" in low and ("405" in low or "405b" in low):
        return "Llama-405b"
    base = raw.split(".")[0]
    base = base.replace("_", " ").replace("-", " ")
    pretty = base.title().replace("Gpt", "GPT")
    return pretty


def _plot_paired_dots_accuracy(metrics_paths: List[str], outdir: str) -> None:
    """Paired-dots plot comparing baseline vs macro ablation accuracy per model.

    Expects each metrics CSV to include rows:
      - scope == "baseline", metric == "accuracy"
      - scope == "global", metric == "macro_ablation_accuracy"
    with columns: value, ci_low, ci_high
    """
    import numpy as np

    records = []
    for p in metrics_paths:
        df = pd.read_csv(p)
        model = _label_for_csv(p)

        base = df[(df["scope"] == "baseline") & (df["metric"] == "accuracy")]
        macro = df[(df["scope"] == "global") & (df["metric"] == "macro_ablation_accuracy")]
        if base.empty or macro.empty:
            continue

        b = base.iloc[0]
        m = macro.iloc[0]
        records.append({
            "model": model,
            "baseline": float(b["value"]),
            "baseline_lo": float(max(0.0, b["value"] - b["ci_low"])),
            "baseline_hi": float(max(0.0, b["ci_high"] - b["value"])) ,
            "macro": float(m["value"]),
            "macro_lo": float(max(0.0, m["value"] - m["ci_low"])) ,
            "macro_hi": float(max(0.0, m["ci_high"] - m["value"])) ,
        })

    if not records:
        return

    models = [_prettify_model_label(r["model"]) for r in records]
    baseline = np.array([r["baseline"] for r in records])
    baseline_err = [np.array([r["baseline_lo"] for r in records]), np.array([r["baseline_hi"] for r in records])]
    macro = np.array([r["macro"] for r in records])
    macro_err = [np.array([r["macro_lo"] for r in records]), np.array([r["macro_hi"] for r in records])]

    x = np.arange(len(models))
    offset = 0.18

    fig, ax = plt.subplots(figsize=(10, 4))

    # Baseline points (blue)
    ax.errorbar(x - offset, baseline, yerr=baseline_err, fmt="x", color="#4C78A8", capsize=3, markersize=8, label="Baseline")
    # Macro ablation points (orange)
    ax.errorbar(x + offset, macro, yerr=macro_err, fmt="x", color="#F58518", capsize=3, markersize=8, label="Macro Ablation")

    # Connecting segments
    for xi, y0, y1 in zip(x, baseline, macro):
        ax.plot([xi - offset, xi + offset], [y0, y1], color="#B0B0B0", linewidth=1.5, alpha=0.9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=0)
    # Ensure full CI is visible
    base_low = baseline - baseline_err[0]
    base_high = baseline + baseline_err[1]
    macro_low = macro - macro_err[0]
    macro_high = macro + macro_err[1]
    ymin = float(min(base_low.min(), macro_low.min()))
    ymax = float(max(base_high.max(), macro_high.max()))
    pad = max(0.02, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("Accuracy")
    ax.set_title("Experiment 1: Baseline vs Macro Ablation Accuracy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left")

    fig.tight_layout()
    out_path_png = os.path.join(outdir, "exp1_paired_dots_accuracy.png")
    fig.savefig(out_path_png, dpi=200)
    # Also save a vector PDF for LaTeX inclusion
    out_path_pdf = os.path.join(outdir, "exp1_paired_dots_accuracy.pdf")
    fig.savefig(out_path_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_causal_density(metrics_paths: List[str], outdir: str) -> None:
    """Bar plot of causal density (per-example mean) with 95% CI across models."""
    import numpy as np

    rows = []
    for p in metrics_paths:
        df = pd.read_csv(p)
        sel = df[(df["scope"] == "global") & (df["metric"] == "causal_density_mean")]
        if sel.empty:
            continue
        r = sel.iloc[0]
        rows.append({
            "model": _prettify_model_label(_label_for_csv(p)),
            "value": float(r["value"]),
            "lo": float(max(0.0, r["value"] - r["ci_low"])) ,
            "hi": float(max(0.0, r["ci_high"] - r["value"])) ,
        })

    if not rows:
        return

    models = [r["model"] for r in rows]
    vals = np.array([r["value"] for r in rows])
    yerr = [np.array([r["lo"] for r in rows]), np.array([r["hi"] for r in rows])]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(models))
    ax.bar(x, vals, yerr=yerr, capsize=4, color="#F58518", edgecolor="#C66A12")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ymin = float((vals - yerr[0]).min())
    ymax = float((vals + yerr[1]).max())
    pad = max(0.01, 0.1 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    ax.set_ylabel("Causal Density")
    ax.set_title("Experiment 1: Causal Density")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(outdir, "exp1_causal_density.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_exp1_causal_composite(metrics_paths: List[str], outdir: str) -> None:
    """Composite figure (2x2 with bottom row spanning) for Experiment 1:
    - Top-left: Damage, Rescue, NetFlip per model (with 95% CIs; Damage plotted downward)
    - Top-right: Causal density per model (with 95% CIs)
    - Bottom (span): Paired-dots baseline vs macro ablation accuracy (with 95% CIs)
    """
    import numpy as np
    from matplotlib import gridspec as _gridspec

    rows_for_dr: List[dict] = []
    rows_for_cd: List[dict] = []
    rows_for_acc: List[dict] = []

    for p in metrics_paths:
        df = pd.read_csv(p)
        model_raw = _label_for_csv(p)
        model = _prettify_model_label(model_raw)

        # Global rows for damage/rescue/netflip/causal density
        g = df[df["scope"] == "global"]
        if not g.empty:
            dmg = g[g["metric"] == "macro_ablation_damage_rate"]
            rsc = g[g["metric"] == "macro_ablation_rescue_rate"]
            nfl = g[g["metric"] == "netflip_damage_minus_rescue"]
            cden = g[g["metric"] == "causal_density_mean"]

            if not dmg.empty and not rsc.empty and not nfl.empty:
                d = dmg.iloc[0]
                r = rsc.iloc[0]
                n = nfl.iloc[0]
                rows_for_dr.append({
                    "model": model,
                    "damage": float(d.get("value", np.nan)),
                    "damage_lo": float(max(0.0, float(d.get("value", np.nan)) - float(d.get("ci_low", np.nan))) if pd.notna(d.get("ci_low", np.nan)) else 0.0),
                    "damage_hi": float(max(0.0, float(d.get("ci_high", np.nan)) - float(d.get("value", np.nan))) if pd.notna(d.get("ci_high", np.nan)) else 0.0),
                    "rescue": float(r.get("value", np.nan)),
                    "rescue_lo": float(max(0.0, float(r.get("value", np.nan)) - float(r.get("ci_low", np.nan))) if pd.notna(r.get("ci_low", np.nan)) else 0.0),
                    "rescue_hi": float(max(0.0, float(r.get("ci_high", np.nan)) - float(r.get("value", np.nan))) if pd.notna(r.get("ci_high", np.nan)) else 0.0),
                    "netflip": float(n.get("value", np.nan)),
                    "netflip_lo": float(max(0.0, float(n.get("value", np.nan)) - float(n.get("ci_low", np.nan))) if pd.notna(n.get("ci_low", np.nan)) else 0.0),
                    "netflip_hi": float(max(0.0, float(n.get("ci_high", np.nan)) - float(n.get("value", np.nan))) if pd.notna(n.get("ci_high", np.nan)) else 0.0),
                })
            if not cden.empty:
                c = cden.iloc[0]
                rows_for_cd.append({
                    "model": model,
                    "value": float(c.get("value", np.nan)),
                    "lo": float(max(0.0, float(c.get("value", np.nan)) - float(c.get("ci_low", np.nan))) if pd.notna(c.get("ci_low", np.nan)) else 0.0),
                    "hi": float(max(0.0, float(c.get("ci_high", np.nan)) - float(c.get("value", np.nan))) if pd.notna(c.get("ci_high", np.nan)) else 0.0),
                })

        # Baseline vs macro accuracy
        base = df[(df["scope"] == "baseline") & (df["metric"] == "accuracy")]
        macr = df[(df["scope"] == "global") & (df["metric"] == "macro_ablation_accuracy")]
        if not base.empty and not macr.empty:
            b = base.iloc[0]
            m = macr.iloc[0]
            rows_for_acc.append({
                "model": model,
                "baseline": float(b.get("value", np.nan)),
                "baseline_lo": float(max(0.0, float(b.get("value", np.nan)) - float(b.get("ci_low", np.nan))) if pd.notna(b.get("ci_low", np.nan)) else 0.0),
                "baseline_hi": float(max(0.0, float(b.get("ci_high", np.nan)) - float(b.get("value", np.nan))) if pd.notna(b.get("ci_high", np.nan)) else 0.0),
                "macro": float(m.get("value", np.nan)),
                "macro_lo": float(max(0.0, float(m.get("value", np.nan)) - float(m.get("ci_low", np.nan))) if pd.notna(m.get("ci_low", np.nan)) else 0.0),
                "macro_hi": float(max(0.0, float(m.get("ci_high", np.nan)) - float(m.get("value", np.nan))) if pd.notna(m.get("ci_high", np.nan)) else 0.0),
            })

    if not rows_for_dr and not rows_for_cd and not rows_for_acc:
        return

    # Keep consistent model order across panels
    models = []
    for pref in ["Claude 4.1 Opus", "ChatGPT-5", "Gemini Pro 2.5"]:
        if any(r["model"] == pref for r in rows_for_dr + rows_for_cd + rows_for_acc):
            models.append(pref)
    for r in rows_for_dr + rows_for_cd + rows_for_acc:
        if r["model"] not in models:
            models.append(r["model"])

    fig = plt.figure(figsize=(12, 8))
    gs = _gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.25)
    ax_dr = fig.add_subplot(gs[0, 0])
    ax_cd = fig.add_subplot(gs[0, 1])
    ax_pair = fig.add_subplot(gs[1, :])

    # Top-left: Damage/Rescue/NetFlip
    if rows_for_dr:
        rows_for_dr_sorted = [next(r for r in rows_for_dr if r["model"] == m) for m in models if any(rr["model"] == m for rr in rows_for_dr)]
        x = np.arange(len(rows_for_dr_sorted))
        width = 0.22
        damage = -np.array([r["damage"] for r in rows_for_dr_sorted], dtype=float)
        damage_err = [
            np.array([r["damage_hi"] for r in rows_for_dr_sorted], dtype=float),
            np.array([r["damage_lo"] for r in rows_for_dr_sorted], dtype=float),
        ]
        rescue = np.array([r["rescue"] for r in rows_for_dr_sorted], dtype=float)
        rescue_err = [
            np.array([r["rescue_lo"] for r in rows_for_dr_sorted], dtype=float),
            np.array([r["rescue_hi"] for r in rows_for_dr_sorted], dtype=float),
        ]
        # Display NetFlip in the same direction as the larger of (rescue, damage).
        # Our stored metric is (damage - rescue); flip sign so positive means rescue > damage.
        netflip = -np.array([r["netflip"] for r in rows_for_dr_sorted], dtype=float)
        netflip_err = [
            np.array([r["netflip_lo"] for r in rows_for_dr_sorted], dtype=float),
            np.array([r["netflip_hi"] for r in rows_for_dr_sorted], dtype=float),
        ]
        ax_dr.bar(x - width, damage, width=width, yerr=damage_err, capsize=4, color="#4C78A8", label="Damage")
        ax_dr.bar(x, rescue, width=width, yerr=rescue_err, capsize=4, color="#F58518", label="Rescue")
        ax_dr.bar(x + width, netflip, width=width, yerr=netflip_err, capsize=4, color="#54A24B", label="NetFlip")
        ax_dr.set_xticks(list(x))
        ax_dr.set_xticklabels([r["model"] for r in rows_for_dr_sorted], rotation=0)
        ymin = float(min((damage - damage_err[0]).min(), (rescue - rescue_err[0]).min(), (netflip - netflip_err[0]).min()))
        ymax = float(max((damage + damage_err[1]).max(), (rescue + rescue_err[1]).max(), (netflip + netflip_err[1]).max()))
        pad = max(0.02, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
        ax_dr.set_ylim(ymin - pad, ymax + pad)
        ax_dr.set_ylabel("Rate")
        ax_dr.set_title("Damage / Rescue / NetFlip")
        ax_dr.grid(axis="y", linestyle=":", alpha=0.5)
        # Smaller legend in upper-left to avoid overlapping data
        ax_dr.legend(loc="upper left", fontsize=11, frameon=False, borderaxespad=0.2, handlelength=1.2)

    # Top-right: Causal density
    if rows_for_cd:
        rows_for_cd_sorted = [next(r for r in rows_for_cd if r["model"] == m) for m in models if any(rr["model"] == m for rr in rows_for_cd)]
        xc = np.arange(len(rows_for_cd_sorted))
        vals = np.array([r["value"] for r in rows_for_cd_sorted], dtype=float)
        yerr = [
            np.array([r["lo"] for r in rows_for_cd_sorted], dtype=float),
            np.array([r["hi"] for r in rows_for_cd_sorted], dtype=float),
        ]
        ax_cd.bar(xc, vals, yerr=yerr, capsize=4, color="#F58518", edgecolor="#C66A12")
        ax_cd.set_xticks(list(xc))
        ax_cd.set_xticklabels([r["model"] for r in rows_for_cd_sorted], rotation=0)
        ymin = float((vals - yerr[0]).min())
        ymax = float((vals + yerr[1]).max())
        pad = max(0.01, 0.1 * (ymax - ymin if ymax > ymin else 1.0))
        ax_cd.set_ylim(max(0.0, ymin - pad), ymax + pad)
        # Clarify via title; keep y-axis as the metric name
        ax_cd.set_ylabel("Rate")
        ax_cd.set_title("Causal Density")
        ax_cd.grid(axis="y", linestyle=":", alpha=0.5)

    # Bottom span: Paired dots accuracy
    if rows_for_acc:
        rows_for_acc_sorted = [next(r for r in rows_for_acc if r["model"] == m) for m in models if any(rr["model"] == m for rr in rows_for_acc)]
        baseline = np.array([r["baseline"] for r in rows_for_acc_sorted], dtype=float)
        baseline_err = [
            np.array([r["baseline_lo"] for r in rows_for_acc_sorted], dtype=float),
            np.array([r["baseline_hi"] for r in rows_for_acc_sorted], dtype=float),
        ]
        macro = np.array([r["macro"] for r in rows_for_acc_sorted], dtype=float)
        macro_err = [
            np.array([r["macro_lo"] for r in rows_for_acc_sorted], dtype=float),
            np.array([r["macro_hi"] for r in rows_for_acc_sorted], dtype=float),
        ]
        x = np.arange(len(rows_for_acc_sorted))
        offset = 0.18
        ax_pair.errorbar(x - offset, baseline, yerr=baseline_err, fmt="x", color="#4C78A8", capsize=3, markersize=8, label="Baseline")
        ax_pair.errorbar(x + offset, macro, yerr=macro_err, fmt="x", color="#F58518", capsize=3, markersize=8, label="Macro Ablation")
        for xi, y0, y1 in zip(x, baseline, macro):
            ax_pair.plot([xi - offset, xi + offset], [y0, y1], color="#B0B0B0", linewidth=1.5, alpha=0.9)
        ax_pair.set_xticks(list(x))
        ax_pair.set_xticklabels([r["model"] for r in rows_for_acc_sorted], rotation=0)
        base_low = baseline - baseline_err[0]
        base_high = baseline + baseline_err[1]
        macro_low = macro - macro_err[0]
        macro_high = macro + macro_err[1]
        ymin = float(min(base_low.min(), macro_low.min()))
        ymax = float(max(base_high.max(), macro_high.max()))
        pad = max(0.02, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
        ax_pair.set_ylim(ymin - pad, ymax + pad)
        ax_pair.set_ylabel("Accuracy")
        ax_pair.set_title("Baseline vs Macro Ablation Accuracy")
        ax_pair.grid(axis="y", linestyle=":", alpha=0.5)
        ax_pair.legend(loc="upper left")

    fig.suptitle("Experiment 1: Causal Metrics and Accuracy", y=0.99, fontsize=16)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    os.makedirs(outdir, exist_ok=True)
    out_pdf = os.path.join(outdir, "exp1_causal_composite.pdf")
    out_png = os.path.join(outdir, "exp1_causal_composite.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def _plot_damage_rescue_netflip_grouped(metrics_paths: List[str], outdir: str) -> None:
    """Grouped bars per model: Damage rate, Rescue rate, Netflip vs baseline."""
    import numpy as np

    rows = []
    for p in metrics_paths:
        df = pd.read_csv(p)
        model = _label_for_csv(p)
        g = df[df["scope"] == "global"]
        dmg = g[g["metric"] == "macro_ablation_damage_rate"]
        rsc = g[g["metric"] == "macro_ablation_rescue_rate"]
        nfl = g[g["metric"] == "macro_ablation_netflip_vs_baseline"]
        if dmg.empty or rsc.empty or nfl.empty:
            continue
        d = dmg.iloc[0]
        r = rsc.iloc[0]
        n = nfl.iloc[0]
        rows.append({
            "model": _prettify_model_label(model),
            "damage": float(d["value"]),
            "damage_lo": float(max(0.0, d["value"] - d["ci_low"])) ,
            "damage_hi": float(max(0.0, d["ci_high"] - d["value"])) ,
            "rescue": float(r["value"]),
            "rescue_lo": float(max(0.0, r["value"] - r["ci_low"])) ,
            "rescue_hi": float(max(0.0, r["ci_high"] - r["value"])) ,
            "netflip": float(n["value"]),
            "netflip_lo": float(max(0.0, n["value"] - n["ci_low"])) ,
            "netflip_hi": float(max(0.0, n["ci_high"] - n["value"])) ,
        })

    if not rows:
        return

    models = [r["model"] for r in rows]
    x = np.arange(len(models))
    width = 0.22

    # Plot damage downward by negating the value. Swap CI sides accordingly.
    damage = -np.array([r["damage"] for r in rows])
    damage_err = [
        np.array([r["damage_hi"] for r in rows]),  # distance below the bar
        np.array([r["damage_lo"] for r in rows]),  # distance above the bar
    ]
    rescue = np.array([r["rescue"] for r in rows])
    rescue_err = [np.array([r["rescue_lo"] for r in rows]), np.array([r["rescue_hi"] for r in rows])]
    netflip = np.array([r["netflip"] for r in rows])
    netflip_err = [np.array([r["netflip_lo"] for r in rows]), np.array([r["netflip_hi"] for r in rows])]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, damage, width=width, yerr=damage_err, capsize=4, color="#4C78A8", label="Damage")
    ax.bar(x, rescue, width=width, yerr=rescue_err, capsize=4, color="#F58518", label="Rescue")
    ax.bar(x + width, netflip, width=width, yerr=netflip_err, capsize=4, color="#54A24B", label="Netflip")

    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    # Ensure full CI is visible for all bars (including negative damage)
    dmg_low = damage - damage_err[0]
    dmg_high = damage + damage_err[1]
    rsc_low = rescue - rescue_err[0]
    rsc_high = rescue + rescue_err[1]
    nfl_low = netflip - netflip_err[0]
    nfl_high = netflip + netflip_err[1]
    ymin = float(min(dmg_low.min(), rsc_low.min(), nfl_low.min()))
    ymax = float(max(dmg_high.max(), rsc_high.max(), nfl_high.max()))
    pad = max(0.02, 0.05 * (ymax - ymin if ymax > ymin else 1.0))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("Rate")
    ax.set_title("Experiment 1: Damage, Rescue, and Netflip")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    out_path = os.path.join(outdir, "exp1_damage_rescue_netflip_grouped.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

def _plot_per_ablation(metrics_path: str, outdir: str) -> None:
    df = pd.read_csv(metrics_path)
    model_label = _label_for_csv(metrics_path)

    # Accuracy per ablation
    acc = df[(df["scope"] == "ablation") & (df["metric"] == "accuracy")].copy()
    if not acc.empty:
        acc = acc.sort_values("ablation_index")
        yerr_low = (acc["value"] - acc["ci_low"]).clip(lower=0)
        yerr_high = (acc["ci_high"] - acc["value"]).clip(lower=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(acc["ablation_index"], acc["value"], yerr=[yerr_low, yerr_high], fmt="-o", capsize=3)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Ablation index")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Per-ablation accuracy — {model_label}")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        out_path = os.path.join(outdir, f"per_ablation_accuracy_{model_label}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    # Netflip per ablation
    nf = df[(df["scope"] == "ablation") & (df["metric"] == "netflip_vs_baseline")].copy()
    if not nf.empty:
        nf = nf.sort_values("ablation_index")
        yerr_low = (nf["value"] - nf["ci_low"]).clip(lower=0)
        yerr_high = (nf["ci_high"] - nf["value"]).clip(lower=0)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.errorbar(nf["ablation_index"], nf["value"], yerr=[yerr_low, yerr_high], fmt="-o", capsize=3)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Ablation index")
        ax.set_ylabel("Netflip vs baseline")
        ax.set_title(f"Per-ablation netflip — {model_label}")
        ax.grid(True, linestyle=":", alpha=0.5)
        fig.tight_layout()
        out_path = os.path.join(outdir, f"per_ablation_netflip_{model_label}.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def _write_exp1_summary_table(metrics_paths: List[str], outdir: str) -> None:
    """Aggregate key Exp1 metrics across models and write CSV and LaTeX tables.

    Collects, per model, the following metrics (with 95% CI if available):
      - baseline accuracy
      - macro ablation accuracy
      - delta accuracy (baseline - macro)
      - macro damage rate
      - macro rescue rate
      - NetFlip (damage - rescue)
      - causal density mean
    """
    cols_spec = [
        ("baseline", "accuracy", "baseline_accuracy"),
        ("global", "macro_ablation_accuracy", "macro_accuracy"),
        ("global", "delta_accuracy", "delta_accuracy"),
        ("global", "macro_ablation_damage_rate", "damage_rate"),
        ("global", "macro_ablation_rescue_rate", "rescue_rate"),
        ("global", "netflip_damage_minus_rescue", "netflip"),
        ("global", "causal_density_mean", "causal_density_mean"),
    ]

    rows = []
    for p in metrics_paths:
        df = pd.read_csv(p)
        model_raw = _label_for_csv(p)
        model = _prettify_model_label(model_raw)
        rec = {"model": model}
        for scope, metric, out_name in cols_spec:
            sel = df[(df["scope"] == scope) & (df["metric"] == metric)]
            if sel.empty:
                # leave NaNs
                rec[out_name] = _np.nan
                rec[f"{out_name}_ci_low"] = _np.nan
                rec[f"{out_name}_ci_high"] = _np.nan
                continue
            r = sel.iloc[0]
            v = float(r.get("value", _np.nan))
            lo = r.get("ci_low", _np.nan)
            hi = r.get("ci_high", _np.nan)
            lo = float(lo) if pd.notna(lo) else _np.nan
            hi = float(hi) if pd.notna(hi) else _np.nan
            rec[out_name] = v
            rec[f"{out_name}_ci_low"] = lo
            rec[f"{out_name}_ci_high"] = hi
        rows.append(rec)

    if not rows:
        return

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("model")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "exp1_summary_table.csv")
    out_df.to_csv(csv_path, index=False)

    # Also write a compact LaTeX table with value and CI formatted in one cell
    def _fmt(v: float, lo: float, hi: float) -> str:
        if pd.isna(v):
            return ""
        if pd.isna(lo) or pd.isna(hi):
            return f"{v:.3f}"
        return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"

    disp_cols = [
        ("baseline_accuracy", "Baseline Acc."),
        ("macro_accuracy", "Macro Acc."),
        ("delta_accuracy", "Delta Acc."),
        ("damage_rate", "Damage"),
        ("rescue_rate", "Rescue"),
        ("netflip", "NetFlip"),
        ("causal_density_mean", "Causal Density"),
    ]
    disp = pd.DataFrame({"Model": out_df["model"]})
    for base, header in disp_cols:
        disp[header] = [
            _fmt(v, lo, hi)
            for v, lo, hi in zip(out_df[base], out_df[f"{base}_ci_low"], out_df[f"{base}_ci_high"])
        ]
    tex_path = os.path.join(outdir, "exp1_summary_table.tex")
    with open(tex_path, "w") as f:
        f.write(disp.to_latex(index=False, escape=True))

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot metrics with 95% CIs from metrics CSVs.")
    parser.add_argument("--metrics", nargs="+", required=True, help="Path(s) to metrics CSV(s)")
    parser.add_argument("--outdir", required=False, help="Directory to write plots (default: <csv_dir>/plots)")
    parser.add_argument("--per-ablation", action="store_true", help="Also plot per-ablation accuracy/netflip for each CSV")
    args = parser.parse_args()

    # If outdir omitted and exactly one input, default to sibling 'plots' folder
    if args.outdir:
        outdir = args.outdir
        _ensure_outdir(outdir)
    else:
        if len(args.metrics) == 1:
            csv_dir = os.path.dirname(os.path.abspath(args.metrics[0]))
            outdir = os.path.join(csv_dir, "plots")
            _ensure_outdir(outdir)
        else:
            # Multiple inputs: make a common 'plots' directory in their nearest common dir
            first_dir = os.path.dirname(os.path.abspath(args.metrics[0]))
            outdir = os.path.join(first_dir, "plots")
            _ensure_outdir(outdir)

    # Apply consistent publication rc defaults
    _set_publication_rc()

    # Global comparison plots (bars across files)
    _plot_global_comparison(args.metrics, outdir)

    # Paired-dots accuracy (baseline vs macro ablation)
    _plot_paired_dots_accuracy(args.metrics, outdir)

    # Grouped bars: Damage, Rescue, Netflip per model
    _plot_damage_rescue_netflip_grouped(args.metrics, outdir)

    # Causal density across models
    _plot_causal_density(args.metrics, outdir)

    # Per-ablation plots for each file if requested
    if args.per_ablation:
        for p in args.metrics:
            subdir = os.path.join(outdir, _label_for_csv(p))
            _ensure_outdir(subdir)
            _plot_per_ablation(p, subdir)

    # Summary table across provided models
    _write_exp1_summary_table(args.metrics, outdir)

    # Composite causal metrics + accuracy figure
    _plot_exp1_causal_composite(args.metrics, outdir)


if __name__ == "__main__":
    main()


