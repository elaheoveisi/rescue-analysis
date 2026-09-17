from __future__ import annotations

import json

import numpy as np
import pandas as pd

def cam_bounds(game_row: pd.Series) -> tuple[int, int, int, int]:
    """Recorded viewport, or AgentFOVCamera's current-room crop.

    Both supplied MOSAIC branches render room_from_pos(agent_pos). RoomGrid
    rooms share one wall, so their origins advance by room_size - 1. The
    13-tile fallback in observations.cam_bounds is not the rendered camera.
    """
    if "cam_top_x" in game_row.index and pd.notna(game_row.get("cam_top_x")):
        x0, y0 = int(game_row["cam_top_x"]), int(game_row["cam_top_y"])
        return (
            x0,
            y0,
            x0 + int(game_row["cam_view_w"]),
            y0 + int(game_row["cam_view_h"]),
        )
    size = int(game_row["room_size"])
    ax, ay = int(game_row["agent_x"]), int(game_row["agent_y"])
    if size <= 1 or ax < 0 or ay < 0:
        raise ValueError("Invalid room geometry in game recording")
    x0, y0 = (ax // (size - 1)) * (size - 1), (ay // (size - 1)) * (size - 1)
    return x0, y0, x0 + size, y0 + size


def extract_run_states(game_stream: dict, game_df: pd.DataFrame, trial_field="trial_id") -> pd.DataFrame:
    """Read each run's changing grid with the camera state from the SAME frame."""
    columns = ["timestamp", "grid", "agent_x", "agent_y", "room_size",
               "cam_top_x", "cam_top_y", "cam_view_w", "cam_view_h"]
    if game_df.empty:
        return pd.DataFrame(columns=columns)
    times = np.asarray(game_stream["time_stamps"], dtype=float)
    selected = np.flatnonzero((times >= game_df["timestamp"].min()) & (times <= game_df["timestamp"].max()))
    trials = set(game_df[trial_field].dropna()) if trial_field in game_df else None
    rows, previous_grid = [], None
    for i in selected:
        value = game_stream["time_series"][i]
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[0]
        try:
            state = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(state, dict) or "grid" not in state:
            continue
        if trials is not None and state.get(trial_field) not in trials:
            continue
        grid = np.asarray(state["grid"])
        if grid.ndim != 2 or not grid.size:
            raise ValueError(f"Invalid grid at timestamp {times[i]}")
        if previous_grid is not None and np.array_equal(grid, previous_grid):
            grid = previous_grid
        else:
            grid.setflags(write=False)
            previous_grid = grid
        rows.append({**{k: state.get(k) for k in columns}, "timestamp": times[i], "grid": grid})
    return pd.DataFrame(rows, columns=columns).sort_values("timestamp", kind="stable").reset_index(drop=True)


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
