"""
Plotting helpers for tables and figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


def save_lineplot(series: Dict[str, Iterable[float]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in series.items():
        ax.plot(list(values), label=label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def to_latex_table(df: pd.DataFrame) -> str:
    return df.to_latex(index=False, escape=True)


