"""
Experiment 3 analysis: accuracy, deltas vs unbiased, flip rates with Wilson CIs.

Outputs:
- {outdir}/summary_metrics_exp3.csv
- {outdir}/figs/accuracy_ci.png, delta_ci.png, flip_ci.png
- {outdir}/REPORT.md
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint


CONDITIONS = ["unbiased", "hint_to_gold", "hint_to_wrong"]


def wilson_ci(p: float, n: int) -> (float, float):
    low, high = proportion_confint(count=int(round(p * n)), nobs=n, method="wilson") if n > 0 else (0.0, 0.0)
    return float(low), float(high)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    # Accuracy by condition
    for cond in CONDITIONS:
        sub = df[df["condition"] == cond].copy()
        n = len(sub)
        acc = float((sub["pred"].astype(str).str.upper() == sub["gold"].astype(str).str.upper()).mean()) if n > 0 else 0.0
        lo, hi = wilson_ci(acc, n)
        # flip rate: defined vs unbiased
        flip_rate = 0.0 if cond == "unbiased" else float(sub.get("flip_from_unbiased", pd.Series([0]*n)).astype(int).mean()) if n > 0 else 0.0
        flo, fhi = (0.0, 0.0) if cond == "unbiased" else wilson_ci(flip_rate, n)
        rows.append({
            "condition": cond,
            "n": n,
            "accuracy": acc,
            "acc_ci_low": lo,
            "acc_ci_high": hi,
            "flip_rate": flip_rate,
            "flip_ci_low": flo,
            "flip_ci_high": fhi,
        })

    metrics = pd.DataFrame(rows)
    # Deltas vs unbiased
    base_acc = float(metrics.loc[metrics["condition"] == "unbiased", "accuracy"].values[0]) if not metrics.empty else 0.0
    metrics["delta_vs_unbiased"] = metrics["accuracy"] - base_acc
    return metrics


def plot_bars_with_ci(categories: List[str], means: List[float], lows: List[float], highs: List[float], outfile: str, ylabel: str, title: str) -> None:
    x = list(range(len(categories)))
    yerr_low = [max(0.0, m - lo) for m, lo in zip(means, lows)]
    yerr_high = [max(0.0, hi - m) for m, hi in zip(means, highs)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, means, yerr=[yerr_low, yerr_high], capsize=4, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def write_report(md_path: str, metrics: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(md_path) or ".", exist_ok=True)
    lines: List[str] = []
    lines.append("# Exp3 Hint Injection Summary\n")
    for cond in CONDITIONS:
        r = metrics[metrics["condition"] == cond]
        if r.empty:
            continue
        acc = r["accuracy"].values[0]
        lo = r["acc_ci_low"].values[0]
        hi = r["acc_ci_high"].values[0]
        flip = r["flip_rate"].values[0]
        flo = r["flip_ci_low"].values[0]
        fhi = r["flip_ci_high"].values[0]
        lines.append(f"- {cond}: accuracy {acc:.3f} [{lo:.3f}, {hi:.3f}] | flip {flip:.3f} [{flo:.3f}, {fhi:.3f}]\n")
    # Deltas
    for cond in ["hint_to_gold", "hint_to_wrong"]:
        r = metrics[metrics["condition"] == cond]
        if r.empty:
            continue
        d = r["delta_vs_unbiased"].values[0]
        lines.append(f"- Delta vs unbiased ({cond}): {d:+.3f}\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 analysis")
    parser.add_argument("--input", required=True, help="Path to exp3_results.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for summary and figs")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)

    metrics = compute_metrics(df)
    metrics_path = os.path.join(args.outdir, "summary_metrics_exp3.csv")
    metrics.to_csv(metrics_path, index=False)

    # Acknowledgment rates per model (commented out for now)
    # if "ack_hint" in df.columns and "model" in df.columns:
    #     sub = df[df["condition"].isin(["hint_to_gold", "hint_to_wrong"])].copy()
    #     # Treat ack_hint as boolean-like; accept 1/true/yes (case-insensitive)
    #     sub["_ack_bool"] = sub["ack_hint"].astype(str).str.strip().str.lower().isin(["1", "true", "yes"])
    #     grp = sub.groupby(["model", "condition"], dropna=False)
    #     ack_n = grp.size().rename("n").reset_index()
    #     ack_k = grp["_ack_bool"].sum().rename("k").reset_index()
    #     ack = ack_n.merge(ack_k, on=["model", "condition"])  # columns: model, condition, n, k
    #     ack["rate"] = ack.apply(lambda r: (float(r["k"]) / float(r["n"])) if r["n"] > 0 else 0.0, axis=1)
    #     ack_ci = ack.apply(lambda r: proportion_confint(int(r["k"]), int(r["n"]), method="wilson") if r["n"] > 0 else (0.0, 0.0), axis=1)
    #     ack["ci_low"] = [float(lo) for (lo, hi) in ack_ci]
    #     ack["ci_high"] = [float(hi) for (lo, hi) in ack_ci]
    #     ack_out = os.path.join(args.outdir, "acknowledgment_rates_exp3.csv")
    #     ack.to_csv(ack_out, index=False)

    figs_dir = os.path.join(args.outdir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Accuracy
    cats = metrics["condition"].tolist()
    plot_bars_with_ci(
        categories=cats,
        means=metrics["accuracy"].tolist(),
        lows=metrics["acc_ci_low"].tolist(),
        highs=metrics["acc_ci_high"].tolist(),
        outfile=os.path.join(figs_dir, "accuracy_ci.png"),
        ylabel="Accuracy",
        title="Exp3: Accuracy by condition with 95% CI",
    )

    # Delta vs unbiased (only hints)
    hints = metrics[metrics["condition"].isin(["hint_to_gold", "hint_to_wrong"])].copy()
    plot_bars_with_ci(
        categories=hints["condition"].tolist(),
        means=hints["delta_vs_unbiased"].tolist(),
        lows=hints["delta_vs_unbiased"].tolist(),  # No CI provided for delta (could derive via bootstrap)
        highs=hints["delta_vs_unbiased"].tolist(),
        outfile=os.path.join(figs_dir, "delta_ci.png"),
        ylabel="Δ Accuracy vs unbiased",
        title="Exp3: Δ Accuracy vs unbiased",
    )

    # Flip rates
    plot_bars_with_ci(
        categories=cats,
        means=metrics["flip_rate"].tolist(),
        lows=metrics["flip_ci_low"].tolist(),
        highs=metrics["flip_ci_high"].tolist(),
        outfile=os.path.join(figs_dir, "flip_ci.png"),
        ylabel="Flip rate vs unbiased",
        title="Exp3: Flip rate by condition with 95% CI",
    )

    # REPORT.md
    write_report(os.path.join(args.outdir, "REPORT.md"), metrics)

    print(f"Wrote: {metrics_path}\nFigs in: {figs_dir}\nReport: {os.path.join(args.outdir, 'REPORT.md')}")


if __name__ == "__main__":
    main()


