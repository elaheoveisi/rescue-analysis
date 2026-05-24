"""Victim-AOI fixation-duration visualization in screen pixel space.

The dynamic labeling model (victim_aoi.py) assigns each fixation an AOI type
by looking up which game object the camera was showing at that screen pixel at
that moment.  The natural coordinate space is therefore screen pixels (x, y),
not the abstract 56×56 global grid.

Default view: OpenAI trial, run 1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

ROOT    = Path(__file__).resolve().parent.parent.parent
FIX_CSV = ROOT / "data" / "processed" / "object_aoi_fixations.csv"
OUT     = ROOT / "analysis" / "visualization" / "figures"
OUT.mkdir(exist_ok=True)

# ── Screen-space AOI boundaries (pixels) ──────────────────────────────────────
SCREEN_W, SCREEN_H = 1920, 1080

AOI_BOUNDS = {
    "game_area":  ((150,  1230), (0,    1080)),
    "info_panel": ((1230, 1770), (0,    540)),
    "chat_panel": ((1230, 1770), (540,  1080)),
}

AOI_BG = {
    "game_area":  "#eef2ff",
    "info_panel": "#fff8ee",
    "chat_panel": "#eefff3",
}

# ── AOI groups ─────────────────────────────────────────────────────────────────
GAME_TYPES  = ["victim", "fake_victim", "door", "key"]
PANEL_TYPES = ["info_panel", "chat_panel"]

ALL_TYPES = GAME_TYPES + PANEL_TYPES

AOI_LABELS = {
    "victim":      "Victim",
    "fake_victim": "Fake Victim",
    "door":        "Door",
    "key":         "Key",
    "info_panel":  "Info Panel",
    "chat_panel":  "Chat Panel",
}

# ── One cubehelix colormap per AOI type (duration axis) ───────────────────────
AOI_CMAPS = {
    "victim":      sns.cubehelix_palette(start=2.0, rot=0.0,  dark=0.05, light=0.90, as_cmap=True),
    "fake_victim": sns.cubehelix_palette(start=2.6, rot=0.2,  dark=0.05, light=0.90, as_cmap=True),
    "door":        sns.cubehelix_palette(start=0.5, rot=-0.5, dark=0.10, light=0.92, as_cmap=True),
    "key":         sns.cubehelix_palette(start=1.0, rot=0.3,  dark=0.10, light=0.92, as_cmap=True),
    "info_panel":  sns.cubehelix_palette(start=0.3, rot=-0.3, dark=0.15, light=0.85, as_cmap=True),
    "chat_panel":  sns.cubehelix_palette(start=1.5, rot=0.1,  dark=0.15, light=0.85, as_cmap=True),
}

# Representative single color per type (midpoint of its cubehelix)
def _rep_color(aoi_type: str) -> tuple:
    cmap = AOI_CMAPS[aoi_type]
    return cmap(0.65)


# ── AOI boundary drawing ───────────────────────────────────────────────────────

def _draw_aoi_rects(ax: plt.Axes) -> None:
    """Draw the three AOI boundary rectangles with labeled backgrounds."""
    for name, ((x0, x1), (y0, y1)) in AOI_BOUNDS.items():
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="square,pad=0",
            facecolor=AOI_BG[name], edgecolor="#888888",
            linewidth=1.2, zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            (x0 + x1) / 2, y0 + 18, name.replace("_", " ").title(),
            ha="center", va="top", fontsize=8, color="#555555", style="italic",
        )


# ── Per-AOI-type screen-space subplot ─────────────────────────────────────────

def _draw_screen_scatter(
    ax:       plt.Axes,
    fix_df:   pd.DataFrame,
    aoi_type: str,
) -> None:
    """Scatter fixations for one AOI type in screen pixel coordinates.

    Color and size both encode fixation duration (ms) via the type's
    cubehelix palette.
    """
    sub = fix_df[fix_df["obj_type"] == aoi_type].copy()

    # AOI bounds to crop the view
    if aoi_type in GAME_TYPES:
        (x0, x1), (y0, y1) = AOI_BOUNDS["game_area"]
    else:
        (x0, x1), (y0, y1) = AOI_BOUNDS[aoi_type]

    # Background rect
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="square,pad=0",
        facecolor=AOI_BG.get("game_area" if aoi_type in GAME_TYPES else aoi_type, "#f5f5f5"),
        edgecolor="#888888", linewidth=1.0, zorder=0,
    ))

    cmap = AOI_CMAPS[aoi_type]

    if not sub.empty:
        dur_min, dur_max = sub["duration_ms"].min(), sub["duration_ms"].max()
        dur_norm = (sub["duration_ms"] - dur_min) / ((dur_max - dur_min) or 1.0)

        sc = ax.scatter(
            sub["x"], sub["y"],
            c=dur_norm, cmap=cmap,
            s=sub["duration_ms"] / 8,
            alpha=0.80, edgecolors="white", linewidths=0.35, zorder=2,
        )
        cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("Duration (norm.)", fontsize=7)
        cb.ax.tick_params(labelsize=7)

        n_fix   = len(sub)
        med_dur = sub["duration_ms"].median()
        tot_dur = sub["duration_ms"].sum() / 1000
        ax.set_title(
            f"{AOI_LABELS[aoi_type]}\n"
            f"{n_fix} fix  ·  med {med_dur:.0f} ms  ·  total {tot_dur:.1f} s",
            fontsize=9, pad=4,
        )
    else:
        ax.set_title(f"{AOI_LABELS[aoi_type]}\n(no fixations)", fontsize=9, pad=4)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)           # screen: y increases downward
    ax.set_xlabel("Screen X (px)", fontsize=8)
    ax.set_ylabel("Screen Y (px)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


# ── Duration summary bar chart ─────────────────────────────────────────────────

def _draw_duration_bars(ax: plt.Axes, fix_df: pd.DataFrame) -> None:
    """Horizontal bars: total fixation duration per AOI type, cubehelix colors."""
    order  = [t for t in ALL_TYPES if t in fix_df["obj_type"].values]
    totals = {t: fix_df[fix_df["obj_type"] == t]["duration_ms"].sum() / 1000 for t in order}
    labels = [AOI_LABELS[t] for t in order]
    values = [totals[t] for t in order]
    colors = [_rep_color(t) for t in order]

    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8)
    max_v = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            val + max_v * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} s",
            va="center", ha="left", fontsize=8,
        )

    ax.set_xlabel("Total fixation duration (s)", fontsize=9)
    ax.set_title("Total Dwell Time by AOI Type", fontsize=10, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_xlim(0, max_v * 1.20 if values else 1)


# ── Main figure ────────────────────────────────────────────────────────────────

def plot_openai_run1(trial: str = "openai", run: int = 1) -> None:
    """Screen-space fixation duration by AOI type (dynamic labeling).

    Layout:
      Row 0: [victim] [fake_victim] [door] [key]   — game-area scatter (cubehelix / duration)
      Row 1: [info_panel] [chat_panel] [duration bar chart]
    """
    fix_all = pd.read_csv(FIX_CSV)
    fix     = fix_all[(fix_all["trial"] == trial) & (fix_all["run"] == run)].copy()

    if fix.empty:
        print(f"No fixations found for trial={trial}, run={run}.")
        return

    label = {"openai": "OpenAI", "gemini": "Gemini", "dummy": "Dummy AI"}.get(trial, trial)

    fig = plt.figure(figsize=(22, 12))
    gs  = gridspec.GridSpec(
        2, 4,
        height_ratios=[1.6, 1],
        hspace=0.42, wspace=0.32,
    )

    # Row 0: game-area AOI types
    for col, aoi_type in enumerate(GAME_TYPES):
        ax = fig.add_subplot(gs[0, col])
        _draw_screen_scatter(ax, fix, aoi_type)

    # Row 1, col 0: info_panel
    ax_info = fig.add_subplot(gs[1, 0])
    _draw_screen_scatter(ax_info, fix, "info_panel")

    # Row 1, col 1: chat_panel
    ax_chat = fig.add_subplot(gs[1, 1])
    _draw_screen_scatter(ax_chat, fix, "chat_panel")

    # Row 1, cols 2-3: duration bar chart
    ax_bar = fig.add_subplot(gs[1, 2:])
    _draw_duration_bars(ax_bar, fix)

    fig.suptitle(
        f"Fixation Duration by AOI Type  ·  {label}  ·  Run {run}\n"
        "Screen-space scatter (dynamic AOI labeling)  —  "
        "color & size = fixation duration (cubehelix per AOI type)",
        fontsize=13, y=1.01,
    )

    path = OUT / f"victim_aoi_{trial}_run{run}.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_openai_run1(trial="openai", run=1)
    print(f"\nFigure saved to: {OUT}")
