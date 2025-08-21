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

from analysis.metrics import (
    compute_accuracy,
    compute_delta,
    compute_damage_rescue_netflip,
    position_pick_rate_df,
    reveal_rate_on_flip,
    add_wilson_ci,
    bar_with_ci,
    grouped_bars_with_ci,
    table_to_png,
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
    mask = (df.get("scope") == "global") & (df.get("metric") == metric)
    if mask.any():
        cols = ["model", "value", "ci_low", "ci_high", "se"]
        found = df.loc[mask, [c for c in cols if c in df.columns]].copy()
        # ensure all expected columns present
        for c in ["ci_low", "ci_high", "se"]:
            if c not in found.columns:
                found[c] = pd.NA
        return found
    return None


def analyze(input_csv: str, outdir: str, models: Optional[List[str]] = None) -> Dict[str, Dict]:
    _safe_mkdir(outdir)
    raw = pd.read_csv(input_csv)

    # Split pre-aggregated vs per-item rows
    is_agg = (raw.get("scope") == "global") & raw.get("metric").notna()
    agg_rows = raw[is_agg.fillna(False)].copy() if "scope" in raw.columns else pd.DataFrame()
    item_rows = raw[~is_agg.fillna(False)].copy() if "scope" in raw.columns else raw.copy()

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
    acc_unb = acc[acc["condition"] == "unbiased"].rename(columns={"value": "Acc_unbiased"}) if acc is not None else None
    acc_bg = acc[acc["condition"] == "biased_to_gold"].rename(columns={"value": "Acc_bias_gold"}) if acc is not None else None
    acc_bw = acc[acc["condition"] == "biased_to_wrong"].rename(columns={"value": "Acc_bias_wrong"}) if acc is not None else None

    # Deltas
    acc_dg = compute_delta(acc_unb[["model", "value"]].rename(columns={"value": "value"}) if acc_unb is not None else pd.DataFrame(),
                           acc_bg[["model", "value"]].rename(columns={"value": "value"}) if acc_bg is not None else pd.DataFrame(),
                           on=["model"], name="Delta_gold") if acc_unb is not None and acc_bg is not None else None
    acc_dw = compute_delta(acc_unb[["model", "value"]].rename(columns={"value": "value"}) if acc_unb is not None else pd.DataFrame(),
                           acc_bw[["model", "value"]].rename(columns={"value": "value"}) if acc_bw is not None else pd.DataFrame(),
                           on=["model"], name="Delta_wrong") if acc_unb is not None and acc_bw is not None else None

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

    # Reveal on flip (if available)
    rev = None
    if not item_rows.empty and _has_cols(item_rows, ["model", "condition", "flip_from_unbiased", "reveal"]):
        wr = item_rows[item_rows["condition"] != "unbiased"].copy()
        rev = reveal_rate_on_flip(wr, ["model", "condition"], flip_col="flip_from_unbiased", reveal_col="reveal")
        rev = add_wilson_ci(rev)

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
        # Reveal on flip (aggregate across biased conditions)
        if rev is not None:
            r = rev[(rev["model"] == m) & (rev["condition"] == "biased_to_wrong")]
            if not r.empty:
                row["RevealRate_on_flip_wrong"] = r["value"].values[0]

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save JSON dump of metrics
    json_path = os.path.join(outdir, "exp2_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

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

        se = [float(_math.sqrt(dl * 0 + rl * 0 + row["d_se"] ** 2 + row["r_se"] ** 2)) for _, row in joined.iterrows()]
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

    # Figure 5: Reveal on flip (if available)
    if rev is not None:
        # Aggregate across biased conditions (optional: separate per condition)
        wr_only = rev[rev["condition"] == "biased_to_wrong"].copy()
        if not wr_only.empty:
            cats = models_list
            means = [wr_only.loc[wr_only["model"] == m, "value"].values[0] for m in models_list]
            ci_lows = [wr_only.loc[wr_only["model"] == m, "ci_low"].values[0] for m in models_list]
            ci_highs = [wr_only.loc[wr_only["model"] == m, "ci_high"].values[0] for m in models_list]
            bar_with_ci(
                categories=cats,
                means=means,
                ci_lows=ci_lows,
                ci_highs=ci_highs,
                outfile=os.path.join(outdir, "exp2_fig5_reveal_on_flip.png"),
                ylabel="Reveal rate on flip",
                title="Experiment 2: Reveal on flip with 95% CI",
            )

    # Summary table with CI text
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

    table_png = os.path.join(outdir, "exp2_summary_table_CI.png")
    table_to_png(table, table_png, title="Experiment 2 — Summary (95% CI)")

    # Print outputs
    outputs_paths = [
        table_png,
        os.path.join(outdir, "exp2_fig1_accuracy_with_ci.png"),
        os.path.join(outdir, "exp2_fig2_damage_rescue_with_ci.png"),
        os.path.join(outdir, "exp2_fig3_netflip_with_ci.png"),
        os.path.join(outdir, "exp2_fig4_position_pick_wrongB.png"),
        os.path.join(outdir, "exp2_fig5_reveal_on_flip.png"),
        json_path,
    ]
    existing = [p for p in outputs_paths if os.path.exists(p)]
    print("Generated:")
    for p in existing:
        print(os.path.abspath(p))

    # Return data for tests or further processing
    outputs["summary_df"] = summary_df.to_dict(orient="records")
    outputs["table_path"] = table_png
    outputs["json_path"] = json_path
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
    parser.add_argument("--outdir", required=True, help="Directory to write outputs (PNGs/JSON)")
    parser.add_argument("--models", required=False, help="Comma-separated list of models to include (default: all)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")] if args.models else None

    if args.input and os.path.exists(args.input):
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


