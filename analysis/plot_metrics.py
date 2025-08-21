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

    # Global comparison plots (bars across files)
    _plot_global_comparison(args.metrics, outdir)

    # Per-ablation plots for each file if requested
    if args.per_ablation:
        for p in args.metrics:
            subdir = os.path.join(outdir, _label_for_csv(p))
            _ensure_outdir(subdir)
            _plot_per_ablation(p, subdir)


if __name__ == "__main__":
    main()


