"""Visualize efficiency-trajectory knee curves (see trigger.efficiency_trajectory)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONDITION_COLORS = {
    "dummy": "#4C72B0",
    "gemini": "#C44E52",
    "openai": "#8172B2",
}


def plot_knee_curve(
    summary: pd.DataFrame,
    fractions: list[float],
    prefix: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Median normalized settling step vs. threshold fraction, one line per condition."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for trial, group in summary.groupby("trial"):
        color = CONDITION_COLORS.get(trial, "gray")
        medians, lo, hi = [], [], []
        for f in fractions:
            col = f"{prefix}{int(f * 100)}"
            vals = group[col].dropna()
            medians.append(vals.median() if not vals.empty else np.nan)
            lo.append(vals.quantile(0.25) if not vals.empty else np.nan)
            hi.append(vals.quantile(0.75) if not vals.empty else np.nan)
        ax.plot(fractions, medians, color=color, linewidth=2, label=trial, marker="o", markersize=4)
        ax.fill_between(fractions, lo, hi, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("Threshold fraction of run's final value")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Condition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"knee plot -> {output_path}")
