from __future__ import annotations

import json

import pandas as pd

# Fallback viewport half-size when cam_ fields are absent (mirrors observations.py).
_CAM_FALLBACK_HALF = 6


def cam_bounds(game_row: pd.Series) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of visible viewport in grid coordinates."""
    if "cam_top_x" in game_row.index and pd.notna(game_row.get("cam_top_x")):
        x0, y0 = int(game_row["cam_top_x"]), int(game_row["cam_top_y"])
        return x0, y0, x0 + int(game_row["cam_view_w"]), y0 + int(game_row["cam_view_h"])
    ax, ay = int(game_row.get("agent_x", 0)), int(game_row.get("agent_y", 0))
    h = _CAM_FALLBACK_HALF
    return ax - h, ay - h, ax + h + 1, ay + h + 1


def extract_run_grid(game_stream: dict, game_df: pd.DataFrame) -> dict | None:
    """Return the most recent grid emitted at or before this run's first game frame.

    The game emits a grid frame at episode reset, which happens between the
    previous run's end and this run's first timestamped game step. Searching
    strictly within [t0, t1] misses that frame for any run after the first.
    """
    t0 = float(game_df["timestamp"].iloc[0])
    last_grid: dict | None = None
    for ts, v in zip(game_stream["time_stamps"], game_stream["time_series"]):
        if ts > t0:
            break
        try:
            d = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        if "grid" in d:
            last_grid = {"grid": d["grid"], "victim_health": d.get("victim_health", {})}
    return last_grid


def best_runs(
    store: pd.HDFStore, trials_cfg: list[str], cfg: dict
) -> set[tuple[str, str, int]]:
    """Return {(sid, trial, run_num)} keeping only the highest-metric run per subject/trial.

    Metric is cfg["glmm2"]["best_run_metric"] (default: "saved_victims").
    """
    metric = cfg.get("glmm2", {}).get("best_run_metric", "saved_victims")
    return set(
        pd.DataFrame([
            {"sid": p[0],
             "trial": next((t for t in trials_cfg if t in p[1]), None),
             "run": int(p[2].replace("run_", "")),
             metric: float(store[k][metric].max()) if metric in store[k].columns else 0.0}
            for k in store.keys() if k.endswith("/game")
            for p in [k.strip("/").split("/")]
        ]).dropna(subset=["trial"])
        .sort_values(metric, ascending=False)
        .groupby(["sid", "trial"], as_index=False).first()
        [["sid", "trial", "run"]].itertuples(index=False, name=None)
    )
