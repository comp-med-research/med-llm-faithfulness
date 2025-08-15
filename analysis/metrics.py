"""
Metrics: accuracy, causal density, netflip, etc.
"""

from __future__ import annotations

from typing import Iterable, List, Optional


def accuracy(golds: Iterable[int], preds: Iterable[int]) -> float:
    gold_list: List[int] = list(golds)
    pred_list: List[int] = list(preds)
    if not gold_list:
        return 0.0
    correct = sum(int(g == p) for g, p in zip(gold_list, pred_list))
    return correct / len(gold_list)


def netflip(before: Iterable[int], after: Iterable[int]) -> float:
    before_list: List[int] = list(before)
    after_list: List[int] = list(after)
    if not before_list:
        return 0.0
    flips = sum(int(b != a) for b, a in zip(before_list, after_list))
    return flips / len(before_list)


def causal_density(effects: Iterable[Optional[float]]) -> float:
    vals = [v for v in effects if v is not None]
    if not vals:
        return 0.0
    # Placeholder: density as fraction of non-zero effects
    non_zero = sum(1 for v in vals if abs(v) > 0.0)
    return non_zero / len(vals)


