"""
Statistical tests, correlations, and confidence intervals.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import stats as scipy_stats


def bootstrap_mean_ci(values: Iterable[float], num_bootstrap: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(0)
    means = np.empty(num_bootstrap, dtype=float)
    n = arr.size
    for i in range(num_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means[i] = sample.mean()
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)
    return float(lower), float(upper)


def pearson_r(x: Iterable[float], y: Iterable[float]) -> float:
    r, _ = scipy_stats.pearsonr(list(x), list(y))
    return float(r)


def t_test_independent(a: Iterable[float], b: Iterable[float]) -> float:
    t, p = scipy_stats.ttest_ind(list(a), list(b), equal_var=False)
    return float(p)


