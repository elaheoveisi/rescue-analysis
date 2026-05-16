"""Visualization of dynamic object AOI fixations.

Reads object_aoi_fixations.csv, object_aoi_features.csv, and
object_aoi_transitions.csv from the processed data directory and produces:

  Fixation scatter plots:
    - Per subject / trial / run
    - Averaged per condition (all subjects combined)

  Summary charts:
    - Mean fixation counts per object type by condition
    - Mean dwell time % per object type by condition
    - Aggregated transition matrix heatmap

Run:
    python visualize_object_aoi.py [--scatter] [--summary] [--transitions]
    (no flags = run everything)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# One color per AOI object type — defined once, shared across all plots.
TYPE_COLORS: dict[str, str] = {
    "victim":      "#2ca02c",
    "fake_victim": "#bcbd22",
    "door":        "#1f77b4",
    "key":         "#9467bd",
    "lava":        "#d62728",
    "info_panel":  "#aec7e8",
    "chat_panel":  "#ffbb78",
    "other":       "#c7c7c7",
    "offscreen":   "#eeeeee",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_cfg() -> dict:
    with open(ROOT / "configs" / "analysis.yml") as f:
        return yaml.safe_load(f)


def _add_metadata(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    df["condition"] = df["trial"].map(cfg["analysis"]["condition_by_category"])
    df["expertise"] = df["subject"].map(cfg.get("expertise", {})).fillna("unknown")
    return df


def _load_fixations(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / cfg["paths"]["processed"] / "object_aoi_fixations.csv")
    return _add_metadata(df, cfg)


def _load_features(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(ROOT / cfg["paths"]["processed"] / "object_aoi_features.csv")
    return _add_metadata(df, cfg)


def _load_transitions(cfg: dict) -> pd.DataFrame:
    return pd.read_csv(ROOT / cfg["paths"]["processed"] / "object_aoi_transitions.csv")


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def _setup_screen_axes(ax: plt.Axes, cfg: dict) -> None:
    w, h = cfg["eyetracker"]["screen_w"], cfg["eyetracker"]["screen_h"]
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # flip so y=0 is top, matching screen convention
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_aoi_boundaries(ax: plt.Axes, cfg: dict) -> None:
    """Draw AOI region outlines from config — no hardcoded names."""
    for aoi in cfg["aoi"]:
        color = TYPE_COLORS.get(aoi["name"], "white")
        linestyle = "-" if aoi["name"] == "game_area" else "--"
        rect = mpatches.Rectangle(
            (aoi["x_min"], aoi["y_min"]),
            aoi["x_max"] - aoi["x_min"],
            aoi["y_max"] - aoi["y_min"],
            linewidth=2, edgecolor=color, facecolor="none", linestyle=linestyle,
        )
        ax.add_patch(rect)
        ax.text(aoi["x_min"] + 6, aoi["y_min"] + 20, aoi["name"],
                color=color, fontsize=8, fontweight="bold")


def _scatter_fixations(ax: plt.Axes, fix_df: pd.DataFrame) -> None:
    """Scatter fixation dots colored by obj_type, sized by duration."""
    for obj_type, group in fix_df[fix_df["obj_type"] != "offscreen"].groupby("obj_type"):
        color = TYPE_COLORS.get(obj_type, "#c7c7c7")
        sizes = np.clip(group["duration_ms"] / 8, 4, 120)
        ax.scatter(group["x"], group["y"], c=color, s=sizes,
                   alpha=0.55, linewidths=0, label=obj_type)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Fixation scatter plots
# ---------------------------------------------------------------------------

def plot_per_run(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """One scatter plot per subject / trial / run."""
    for (sid, trial, run), group in fix_df.groupby(["subject", "trial", "run"]):
        fig, ax = plt.subplots(figsize=(12, 7))
        _setup_screen_axes(ax, cfg)
        _draw_aoi_boundaries(ax, cfg)
        _scatter_fixations(ax, group)
        n = len(group)
        dur = group["duration_ms"].sum() / 1000
        ax.set_title(f"{sid}  —  {trial}  —  run {run}   (n={n}, {dur:.1f}s)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5,
                  framealpha=0.7, title="AOI type")
        _save(fig, out_dir / f"fixations_{sid}_{trial}_run{int(run)}.png")


def plot_averaged(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """One scatter plot per condition with all subjects combined."""
    for cond, group in fix_df.groupby("condition"):
        fig, ax = plt.subplots(figsize=(12, 7))
        _setup_screen_axes(ax, cfg)
        _draw_aoi_boundaries(ax, cfg)
        _scatter_fixations(ax, group)
        n_subs = group["subject"].nunique()
        n = len(group)
        ax.set_title(
            f"All subjects  —  {cond}   ({n_subs} subjects, n={n} fixations)", fontsize=11
        )
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5,
                  framealpha=0.7, title="AOI type")
        _save(fig, out_dir / f"fixations_averaged_{cond}.png")


# ---------------------------------------------------------------------------
# Summary bar charts
# ---------------------------------------------------------------------------

def _object_types(feat_df: pd.DataFrame) -> list[str]:
    return [c.replace("n_fixations_on_", "") for c in feat_df.columns
            if c.startswith("n_fixations_on_")]


def _grouped_bars(ax: plt.Axes, df: pd.DataFrame, col_template: str,
                  labels: list[str], cfg: dict) -> None:
    """Grouped bars for col_template.format(t) split by condition."""
    conditions = sorted(df["condition"].dropna().unique())
    x = np.arange(len(labels))
    width = 0.8 / len(conditions)
    for i, cond in enumerate(conditions):
        sub = df[df["condition"] == cond]
        means = [sub[col_template.format(t)].mean() for t in labels]
        sems  = [sub[col_template.format(t)].sem()  for t in labels]
        offset = (i - len(conditions) / 2 + 0.5) * width
        ax.bar(x + offset, means, width, yerr=sems, capsize=3,
               label=cond, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend()


def plot_fixation_counts(feat_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    types = _object_types(feat_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bars(ax, feat_df, "n_fixations_on_{}", types, cfg)
    ax.set_ylabel("Mean fixation count")
    ax.set_title("Fixations per object type by condition")
    _save(fig, out_dir / "object_aoi_fixation_counts.png")


def plot_dwell_pct(feat_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    types = _object_types(feat_df)
    fig, ax = plt.subplots(figsize=(10, 5))
    _grouped_bars(ax, feat_df, "pct_dur_on_{}", types, cfg)
    ax.set_ylabel("Mean dwell time (%)")
    ax.set_title("Dwell time per object type by condition")
    _save(fig, out_dir / "object_aoi_dwell_pct.png")


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def _trans_types(trans_df: pd.DataFrame) -> list[str]:
    seen, types = set(), []
    for c in trans_df.columns:
        if c.startswith("trans_") and "_to_" in c:
            t = c.replace("trans_", "", 1).split("_to_", 1)[0]
            if t not in seen:
                seen.add(t)
                types.append(t)
    return types


def plot_transition_matrix(trans_df: pd.DataFrame, out_dir: Path,
                           normalize: bool = True) -> None:
    types = _trans_types(trans_df)
    matrix = pd.DataFrame(0.0, index=types, columns=types)
    for _, row in trans_df.iterrows():
        for s in types:
            for d in types:
                col = f"trans_{s}_to_{d}"
                if col in row:
                    matrix.loc[s, d] += row[col]

    if normalize:
        matrix = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
        fmt, vmax, cbar_label = ".2f", 1.0, "Transition probability"
    else:
        fmt, vmax, cbar_label = "d", None, "Transition count"

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix.values, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(types)))
    ax.set_yticks(range(len(types)))
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_yticklabels(types)
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    ax.set_title("Object AOI transition matrix")
    for i in range(len(types)):
        for j in range(len(types)):
            val = matrix.values[i, j]
            txt = f"{val:{fmt}}" if fmt == ".2f" else str(int(val))
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if val < 0.6 else "white", fontsize=8)
    plt.colorbar(im, ax=ax, label=cbar_label)
    _save(fig, out_dir / "object_aoi_transitions.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_visualizations(cfg: dict | None = None, scatter: bool = True,
                       summary: bool = True, normalize: bool = True) -> None:
    if cfg is None:
        cfg = _load_cfg()
    out_dir = ROOT / "analysis" / "heatmaps"
    out_dir.mkdir(exist_ok=True)

    if scatter:
        print("--- Fixation scatter plots ---")
        fix_df = _load_fixations(cfg)
        plot_per_run(fix_df, cfg, out_dir)
        plot_averaged(fix_df, cfg, out_dir)

    if summary:
        print("--- Summary charts ---")
        feat_df = _load_features(cfg)
        trans_df = _load_transitions(cfg)
        plot_fixation_counts(feat_df, cfg, out_dir)
        plot_dwell_pct(feat_df, cfg, out_dir)
        plot_transition_matrix(trans_df, out_dir, normalize=normalize)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatter",     action="store_true", help="Only scatter plots")
    parser.add_argument("--summary",     action="store_true", help="Only summary charts")
    parser.add_argument("--no-normalize", action="store_true", help="Raw counts in transition matrix")
    args = parser.parse_args()

    run_all = not args.scatter and not args.summary
    run_visualizations(
        scatter=run_all or args.scatter,
        summary=run_all or args.summary,
        normalize=not args.no_normalize,
    )
