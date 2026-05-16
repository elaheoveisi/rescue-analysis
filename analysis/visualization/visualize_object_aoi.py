"""Visualization of dynamic object AOI fixations on the full grid map.

Fixations are plotted in grid-tile coordinates so that each dot appears on
the actual room and object it corresponded to — not collapsed onto the screen
game area. The grid background is rendered from the saved grid_{trial}.json.

Outputs:
  Fixation scatter on full grid map:
    - Per subject / trial / run
    - Averaged per trial (all subjects combined)

  Summary charts:
    - Mean fixation counts per object type by condition
    - Mean dwell time % per object type by condition
    - Aggregated transition matrix heatmap

Run:
    python visualize_object_aoi.py [--scatter] [--summary] [--no-normalize]
    (no flags = run everything)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]

# Tile ID constants — mirrors game observations.py and victim_aoi.py
_WALL, _LAVA, _VICTIM, _FAKE_VICTIM = 1, 4, 5, 6
_DOOR_BASE, _DOOR_END = 10, 28
_KEY_BASE, _KEY_END = 30, 36

# Colors for rendering the grid background tiles
_TILE_RGB: dict[str, tuple] = {
    "empty": (220, 220, 220),
    "wall":  (40,  40,  40),
    "lava":  (230, 80,  10),
    "victim":      (30,  180, 30),
    "fake_victim": (180, 160, 30),
    "door":  (50,  50,  200),
    "key":   (150, 50,  180),
}

# Colors for fixation dot overlays — one per obj_type
TYPE_COLORS: dict[str, str] = {
    "victim":      "#2ca02c",
    "fake_victim": "#bcbd22",
    "door":        "#1f77b4",
    "key":         "#9467bd",
    "lava":        "#d62728",
    "info_panel":  "#aec7e8",
    "chat_panel":  "#ffbb78",
    "other":       "#c7c7c7",
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


def _load_grid(cfg: dict, trial: str) -> list[list[int]] | None:
    path = ROOT / cfg["paths"]["processed"] / "grids" / f"grid_{trial}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Grid rendering helpers
# ---------------------------------------------------------------------------

def _tile_rgb(tile: int) -> tuple:
    if tile == _WALL:
        return _TILE_RGB["wall"]
    if tile == _LAVA:
        return _TILE_RGB["lava"]
    if tile == _VICTIM:
        return _TILE_RGB["victim"]
    if tile == _FAKE_VICTIM:
        return _TILE_RGB["fake_victim"]
    if _DOOR_BASE <= tile < _DOOR_END:
        return _TILE_RGB["door"]
    if _KEY_BASE <= tile < _KEY_END:
        return _TILE_RGB["key"]
    return _TILE_RGB["empty"]


def _render_grid_bg(grid: list[list[int]], ax: plt.Axes) -> None:
    """Render the full grid as a coloured background image on ax."""
    h, w = len(grid), len(grid[0])
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            img[y, x] = _tile_rgb(int(tile))
    ax.imshow(img, origin="upper", extent=[0, w, h, 0], aspect="equal", zorder=0)


def _setup_grid_axes(ax: plt.Axes, grid: list[list[int]]) -> None:
    h, w = len(grid), len(grid[0])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)   # y=0 at top, matching grid convention
    ax.set_aspect("equal")
    ax.set_xlabel("grid x (tile)")
    ax.set_ylabel("grid y (tile)")


def _scatter_on_grid(ax: plt.Axes, fix_df: pd.DataFrame) -> None:
    """Plot fixation dots at grid coordinates, colored by obj_type."""
    on_grid = fix_df.dropna(subset=["grid_x", "grid_y"])
    for obj_type, group in on_grid[on_grid["obj_type"] != "offscreen"].groupby("obj_type"):
        color = TYPE_COLORS.get(obj_type, "#c7c7c7")
        sizes = np.clip(group["duration_ms"] / 8, 4, 120)
        ax.scatter(group["grid_x"] + 0.5, group["grid_y"] + 0.5,
                   c=color, s=sizes, alpha=0.6, linewidths=0, label=obj_type, zorder=2)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Fixation scatter plots on full grid
# ---------------------------------------------------------------------------

def plot_per_run(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """One scatter plot per subject / trial / run on the full grid map."""
    for (sid, trial, run), group in fix_df.groupby(["subject", "trial", "run"]):
        grid = _load_grid(cfg, trial)
        fig, ax = plt.subplots(figsize=(10, 10))
        if grid:
            _render_grid_bg(grid, ax)
            _setup_grid_axes(ax, grid)
        _scatter_on_grid(ax, group)
        n = len(group.dropna(subset=["grid_x"]))
        dur = group["duration_ms"].sum() / 1000
        ax.set_title(
            f"{sid}  —  {trial}  —  run {int(run)}   (n={n} on-grid fixations, {dur:.1f}s)",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5,
                  framealpha=0.8, title="AOI type")
        _save(fig, out_dir / f"fixations_{sid}_{trial}_run{int(run)}.png")


def plot_averaged(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """One scatter plot per trial (all subjects combined) on the full grid map."""
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        fig, ax = plt.subplots(figsize=(10, 10))
        if grid:
            _render_grid_bg(grid, ax)
            _setup_grid_axes(ax, grid)
        _scatter_on_grid(ax, group)
        n_subs = group["subject"].nunique()
        n = len(group.dropna(subset=["grid_x"]))
        ax.set_title(
            f"All subjects  —  {trial}   ({n_subs} subjects, n={n} on-grid fixations)",
            fontsize=10,
        )
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5,
                  framealpha=0.8, title="AOI type")
        _save(fig, out_dir / f"fixations_averaged_{trial}.png")


# ---------------------------------------------------------------------------
# Summary bar charts
# ---------------------------------------------------------------------------

def _object_types(feat_df: pd.DataFrame) -> list[str]:
    return [c.replace("n_fixations_on_", "") for c in feat_df.columns
            if c.startswith("n_fixations_on_")]


def _grouped_bars(ax: plt.Axes, df: pd.DataFrame, col_template: str,
                  labels: list[str], cfg: dict) -> None:
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
# Fixation intensity heatmap
# ---------------------------------------------------------------------------

def _build_heatmap(fix_df: pd.DataFrame, grid_h: int, grid_w: int,
                   sigma: float = 1.5) -> np.ndarray:
    """Accumulate fixation duration per grid cell, then apply Gaussian blur."""
    heat = np.zeros((grid_h, grid_w), dtype=float)
    valid = fix_df.dropna(subset=["grid_x", "grid_y"])
    for _, row in valid.iterrows():
        gx, gy = int(row["grid_x"]), int(row["grid_y"])
        if 0 <= gx < grid_w and 0 <= gy < grid_h:
            heat[gy, gx] += float(row["duration_ms"])
    return gaussian_filter(heat, sigma=sigma)


def _overlay_heatmap(ax: plt.Axes, heat: np.ndarray, grid_w: int, grid_h: int,
                     alpha: float = 0.65) -> None:
    """Overlay the heatmap on ax using a hot colormap with transparency."""
    masked = np.ma.masked_where(heat == 0, heat)
    cmap = plt.cm.jet.copy()
    cmap.set_bad(alpha=0)          # transparent where no fixations
    ax.imshow(
        masked,
        cmap=cmap,
        origin="upper",
        extent=[0, grid_w, grid_h, 0],
        aspect="equal",
        alpha=alpha,
        vmin=0,
        vmax=np.percentile(heat[heat > 0], 95) if heat.max() > 0 else 1,
        zorder=1,
    )


def plot_heatmap_per_run(fix_df: pd.DataFrame, cfg: dict, out_dir: Path,
                         sigma: float = 1.5) -> None:
    """Fixation intensity heatmap per subject / trial / run."""
    for (sid, trial, run), group in fix_df.groupby(["subject", "trial", "run"]):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        grid_h, grid_w = len(grid), len(grid[0])
        heat = _build_heatmap(group, grid_h, grid_w, sigma)

        fig, ax = plt.subplots(figsize=(10, 10))
        _render_grid_bg(grid, ax)
        _setup_grid_axes(ax, grid)
        _overlay_heatmap(ax, heat, grid_w, grid_h)

        plt.colorbar(
            plt.cm.ScalarMappable(cmap="jet",
                norm=mcolors.Normalize(vmin=0, vmax=heat.max())),
            ax=ax, label="Dwell time (ms)", shrink=0.6,
        )
        ax.set_title(f"{sid}  —  {trial}  —  run {int(run)}", fontsize=10)
        _save(fig, out_dir / f"heatmap_{sid}_{trial}_run{int(run)}.png")


def plot_heatmap_averaged(fix_df: pd.DataFrame, cfg: dict, out_dir: Path,
                          sigma: float = 1.5) -> None:
    """Fixation intensity heatmap per trial, averaged across all subjects."""
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        grid_h, grid_w = len(grid), len(grid[0])
        heat = _build_heatmap(group, grid_h, grid_w, sigma)

        fig, ax = plt.subplots(figsize=(10, 10))
        _render_grid_bg(grid, ax)
        _setup_grid_axes(ax, grid)
        _overlay_heatmap(ax, heat, grid_w, grid_h)

        plt.colorbar(
            plt.cm.ScalarMappable(cmap="jet",
                norm=mcolors.Normalize(vmin=0, vmax=heat.max())),
            ax=ax, label="Total dwell time (ms)", shrink=0.6,
        )
        n_subs = group["subject"].nunique()
        ax.set_title(f"All subjects  —  {trial}   ({n_subs} subjects)", fontsize=10)
        _save(fig, out_dir / f"heatmap_averaged_{trial}.png")


def plot_heatmap_by_expertise(fix_df: pd.DataFrame, cfg: dict, out_dir: Path,
                               sigma: float = 1.5) -> None:
    """Side-by-side heatmaps expert vs novice, one figure per trial."""
    fix_df = _add_metadata(fix_df, cfg)
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        grid_h, grid_w = len(grid), len(grid[0])

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        for ax, exp in zip(axes, ["expert", "novice"]):
            sub = group[group["expertise"] == exp]
            heat = _build_heatmap(sub, grid_h, grid_w, sigma)
            _render_grid_bg(grid, ax)
            _setup_grid_axes(ax, grid)
            _overlay_heatmap(ax, heat, grid_w, grid_h)
            ax.set_title(f"{exp.capitalize()}  —  {trial}  (n={sub['subject'].nunique()})",
                         fontsize=11)

        fig.suptitle(f"Fixation intensity by expertise — {trial}", fontsize=13)
        _save(fig, out_dir / f"heatmap_expertise_{trial}.png")


# ---------------------------------------------------------------------------
# Bubble map — each AOI as a circle sized by dwell time
# ---------------------------------------------------------------------------

def _plot_bubbles(ax: plt.Axes, fix_df: pd.DataFrame, obj_types: list[str]) -> None:
    """Draw one circle per AOI position, sized by total dwell time."""
    on_grid = fix_df.dropna(subset=["grid_x", "grid_y"])
    grouped = (
        on_grid[on_grid["obj_type"].isin(obj_types)]
        .groupby(["grid_x", "grid_y", "obj_type"], as_index=False)["duration_ms"]
        .sum()
    )
    if grouped.empty:
        return

    max_dur = grouped["duration_ms"].max()
    for t in obj_types:
        sub = grouped[grouped["obj_type"] == t]
        if sub.empty:
            continue
        sizes = ((sub["duration_ms"] / max_dur) * 300).clip(lower=10)
        ax.scatter(
            sub["grid_x"] + 0.5, sub["grid_y"] + 0.5,
            s=sizes, c=TYPE_COLORS.get(t, "#aaaaaa"),
            alpha=0.75, linewidths=0.4, edgecolors="white",
            label=t, zorder=3,
        )


def plot_bubble_map(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """Bubble map averaged across all subjects, one figure per trial."""
    obj_types = [t for t in TYPE_COLORS if t not in ("other",)]
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        fig, ax = plt.subplots(figsize=(11, 11))
        _render_grid_bg(grid, ax)
        _setup_grid_axes(ax, grid)
        _plot_bubbles(ax, group, obj_types)
        ax.legend(loc="upper right", fontsize=9, title="AOI type",
                  markerscale=1.2, framealpha=0.85)
        ax.set_title(
            f"AOI attention — {trial}  ({group['subject'].nunique()} subjects)\n"
            "Circle size ∝ total dwell time", fontsize=10,
        )
        _save(fig, out_dir / f"bubbles_{trial}.png")


def plot_bubble_map_expertise(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """Expert vs novice bubble maps side by side, one figure per trial."""
    fix_df = _add_metadata(fix_df, cfg)
    obj_types = [t for t in TYPE_COLORS if t not in ("other",)]
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(22, 11))
        for ax, exp in zip(axes, ["expert", "novice"]):
            sub = group[group["expertise"] == exp]
            _render_grid_bg(grid, ax)
            _setup_grid_axes(ax, grid)
            _plot_bubbles(ax, sub, obj_types)
            ax.set_title(
                f"{exp.capitalize()}  —  {trial}  (n={sub['subject'].nunique()})",
                fontsize=11,
            )
        axes[0].legend(loc="upper right", fontsize=8, title="AOI type", framealpha=0.85)
        fig.suptitle(f"AOI attention by expertise — {trial}", fontsize=13)
        _save(fig, out_dir / f"bubbles_expertise_{trial}.png")


# ---------------------------------------------------------------------------
# Per-type heatmap subplots
# ---------------------------------------------------------------------------

def plot_pertype_heatmaps(fix_df: pd.DataFrame, cfg: dict, out_dir: Path,
                           sigma: float = 1.5) -> None:
    """One subplot per object type showing fixation density, one figure per trial."""
    obj_types = ["victim", "door", "key", "lava", "info_panel", "chat_panel"]
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        grid_h, grid_w = len(grid), len(grid[0])

        n = len(obj_types)
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        axes = axes.flatten()

        for ax, t in zip(axes, obj_types):
            sub = group[group["obj_type"] == t]
            heat = _build_heatmap(sub, grid_h, grid_w, sigma)
            _render_grid_bg(grid, ax)
            _setup_grid_axes(ax, grid)
            if heat.max() > 0:
                _overlay_heatmap(ax, heat, grid_w, grid_h)
            ax.set_title(f"{t}  (n={len(sub)} fixations)", fontsize=10)

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"Fixation intensity by object type — {trial}", fontsize=13)
        fig.tight_layout()
        _save(fig, out_dir / f"pertype_heatmap_{trial}.png")


def plot_pertype_heatmaps_expertise(fix_df: pd.DataFrame, cfg: dict, out_dir: Path,
                                     sigma: float = 1.5) -> None:
    """Per-type heatmaps split by expertise — rows=types, cols=expert/novice."""
    fix_df = _add_metadata(fix_df, cfg)
    obj_types = ["victim", "door", "key", "lava"]
    groups_exp = ["expert", "novice"]

    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        grid_h, grid_w = len(grid), len(grid[0])

        fig, axes = plt.subplots(len(obj_types), 2, figsize=(18, len(obj_types) * 8))
        for row, t in enumerate(obj_types):
            for col, exp in enumerate(groups_exp):
                ax = axes[row, col]
                sub = group[(group["obj_type"] == t) & (group["expertise"] == exp)]
                heat = _build_heatmap(sub, grid_h, grid_w, sigma)
                _render_grid_bg(grid, ax)
                _setup_grid_axes(ax, grid)
                if heat.max() > 0:
                    _overlay_heatmap(ax, heat, grid_w, grid_h)
                ax.set_title(f"{t}  ·  {exp}  (n={len(sub)})", fontsize=10)

        fig.suptitle(f"Per-type fixation intensity: expert vs novice — {trial}", fontsize=13)
        fig.tight_layout()
        _save(fig, out_dir / f"pertype_expertise_{trial}.png")


# ---------------------------------------------------------------------------
# AOI circle map — one circle per AOI, sized by dwell time, panels included
# ---------------------------------------------------------------------------

def _compute_aoi_totals(fix_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total dwell time per AOI position.

    Returns a DataFrame with columns:
      grid_x, grid_y, obj_type, total_dur_ms
    grid_x/grid_y are NaN for panel AOIs.
    """
    game = fix_df.dropna(subset=["grid_x", "grid_y"])
    panels = fix_df[fix_df["obj_type"].isin(("info_panel", "chat_panel"))]

    game_totals = (
        game.groupby(["grid_x", "grid_y", "obj_type"], as_index=False)["duration_ms"]
        .sum()
        .rename(columns={"duration_ms": "total_dur_ms"})
    )
    panel_totals = (
        panels.groupby("obj_type", as_index=False)["duration_ms"]
        .sum()
        .rename(columns={"duration_ms": "total_dur_ms"})
    )
    panel_totals["grid_x"] = np.nan
    panel_totals["grid_y"] = np.nan

    return pd.concat([game_totals, panel_totals], ignore_index=True)


def _draw_game_circles(ax: plt.Axes, totals: pd.DataFrame, scale: float) -> None:
    """Plot one circle per game-object AOI on the grid axis."""
    game = totals.dropna(subset=["grid_x", "grid_y"])
    for t, grp in game.groupby("obj_type"):
        sizes = (np.sqrt(grp["total_dur_ms"]) * scale).clip(lower=8)
        ax.scatter(
            grp["grid_x"] + 0.5, grp["grid_y"] + 0.5,
            s=sizes,
            c=TYPE_COLORS.get(t, "#aaaaaa"),
            alpha=0.80,
            linewidths=0.3,
            edgecolors="white",
            label=t,
            zorder=3,
        )


def _draw_panel_circles(ax: plt.Axes, totals: pd.DataFrame, scale: float) -> None:
    """Draw info_panel and chat_panel as stacked circles on a side axis."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = {"info_panel": 0.75, "chat_panel": 0.25}
    for pname, ypos in positions.items():
        row = totals[totals["obj_type"] == pname]
        dur = float(row["total_dur_ms"].iloc[0]) if not row.empty else 0.0
        size = float(np.sqrt(dur) * scale)
        ax.scatter(
            [0.5], [ypos],
            s=max(size, 20),
            c=TYPE_COLORS.get(pname, "#cccccc"),
            alpha=0.85,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
        )
        ax.text(
            0.5, ypos - 0.12,
            f"{pname.replace('_', ' ')}\n{dur / 1000:.1f}s",
            ha="center", va="top", fontsize=8,
        )


def _circle_scale(totals: pd.DataFrame) -> float:
    """Compute a scale factor so the largest circle is ~600 pt²."""
    max_dur = totals["total_dur_ms"].max()
    return 600.0 / np.sqrt(max_dur) if max_dur > 0 else 1.0


def _make_legend_patches() -> list:
    return [
        mpatches.Patch(color=c, label=t)
        for t, c in TYPE_COLORS.items()
        if t != "other"
    ]


def plot_aoi_circles_full(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """One circle per AOI, sized by total dwell time — all subjects per trial."""
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue
        totals = _compute_aoi_totals(group)
        scale  = _circle_scale(totals)

        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax_grid  = fig.add_subplot(gs[0])
        ax_panel = fig.add_subplot(gs[1])

        _render_grid_bg(grid, ax_grid)
        _setup_grid_axes(ax_grid, grid)
        _draw_game_circles(ax_grid, totals, scale)
        _draw_panel_circles(ax_panel, totals, scale)

        ax_grid.legend(
            handles=_make_legend_patches(),
            loc="upper right", fontsize=8, title="AOI type", framealpha=0.85,
        )
        ax_grid.set_title(
            f"AOI attention — {trial}  ({group['subject'].nunique()} subjects)\n"
            "Circle size ∝ √(total dwell time)",
            fontsize=10,
        )
        ax_panel.set_title("Panels", fontsize=9)
        _save(fig, out_dir / f"circles_{trial}.png")


def plot_aoi_circles_expertise(fix_df: pd.DataFrame, cfg: dict, out_dir: Path) -> None:
    """Expert vs novice circle maps side by side, one figure per trial."""
    fix_df = _add_metadata(fix_df, cfg)
    for trial, group in fix_df.groupby("trial"):
        grid = _load_grid(cfg, trial)
        if not grid:
            continue

        all_totals = _compute_aoi_totals(group)
        scale = _circle_scale(all_totals)

        fig = plt.figure(figsize=(26, 10))
        gs  = fig.add_gridspec(1, 4, width_ratios=[4, 1, 4, 1], wspace=0.05)

        for col_base, exp in zip([0, 2], ["expert", "novice"]):
            sub = group[group["expertise"] == exp]
            totals = _compute_aoi_totals(sub)

            ax_grid  = fig.add_subplot(gs[col_base])
            ax_panel = fig.add_subplot(gs[col_base + 1])

            _render_grid_bg(grid, ax_grid)
            _setup_grid_axes(ax_grid, grid)
            _draw_game_circles(ax_grid, totals, scale)
            _draw_panel_circles(ax_panel, totals, scale)

            ax_grid.set_title(
                f"{exp.capitalize()}  —  {trial}  (n={sub['subject'].nunique()})",
                fontsize=11,
            )
            ax_panel.set_title("Panels", fontsize=9)

        fig.legend(
            handles=_make_legend_patches(),
            loc="lower center", ncol=6, fontsize=9,
            title="AOI type", framealpha=0.85, bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(f"AOI attention by expertise — {trial}", fontsize=13, y=1.01)
        _save(fig, out_dir / f"circles_expertise_{trial}.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_visualizations(cfg: dict | None = None, scatter: bool = True,
                       heatmap: bool = True, bubble: bool = True,
                       pertype: bool = True, circles: bool = True,
                       summary: bool = True, normalize: bool = True) -> None:
    if cfg is None:
        cfg = _load_cfg()
    out_dir = ROOT / "analysis" / "heatmaps"
    out_dir.mkdir(exist_ok=True)

    fix_df = None
    def _fix():
        nonlocal fix_df
        if fix_df is None:
            fix_df = _load_fixations(cfg)
        return fix_df

    if scatter:
        print("--- Fixation scatter plots ---")
        plot_per_run(_fix(), cfg, out_dir)
        plot_averaged(_fix(), cfg, out_dir)

    if heatmap:
        print("--- Fixation intensity heatmaps ---")
        plot_heatmap_per_run(_fix(), cfg, out_dir)
        plot_heatmap_averaged(_fix(), cfg, out_dir)
        plot_heatmap_by_expertise(_fix(), cfg, out_dir)

    if bubble:
        print("--- Bubble maps (AOI attention) ---")
        plot_bubble_map(_fix(), cfg, out_dir)
        plot_bubble_map_expertise(_fix(), cfg, out_dir)

    if pertype:
        print("--- Per-type heatmap subplots ---")
        plot_pertype_heatmaps(_fix(), cfg, out_dir)
        plot_pertype_heatmaps_expertise(_fix(), cfg, out_dir)

    if circles:
        print("--- AOI circle maps ---")
        plot_aoi_circles_full(_fix(), cfg, out_dir)
        plot_aoi_circles_expertise(_fix(), cfg, out_dir)

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
    parser.add_argument("--scatter",      action="store_true", help="Scatter plots only")
    parser.add_argument("--heatmap",      action="store_true", help="Overall heatmaps only")
    parser.add_argument("--bubble",       action="store_true", help="Bubble maps only")
    parser.add_argument("--pertype",      action="store_true", help="Per-type heatmap subplots only")
    parser.add_argument("--circles",      action="store_true", help="AOI circle maps only")
    parser.add_argument("--summary",      action="store_true", help="Summary charts only")
    parser.add_argument("--no-normalize", action="store_true", help="Raw counts in transition matrix")
    args = parser.parse_args()

    run_all = not any([args.scatter, args.heatmap, args.bubble, args.pertype, args.circles, args.summary])
    run_visualizations(
        scatter=run_all or args.scatter,
        heatmap=run_all or args.heatmap,
        bubble=run_all or args.bubble,
        pertype=run_all or args.pertype,
        circles=run_all or args.circles,
        summary=run_all or args.summary,
        normalize=not args.no_normalize,
    )