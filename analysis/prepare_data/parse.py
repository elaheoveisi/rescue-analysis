"""Stream parsing and trial-splitting utilities."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def get_stream(streams: list, name: str) -> dict | None:
    return next((s for s in streams if s["info"].get("name", [""])[0] == name), None)


def xdf_path(subject_id: str, cfg: dict) -> Path:
    """Return the XDF file path for a subject."""
    raw = cfg["paths"]["raw"]
    return (
        Path(raw)
        / f"sub-{subject_id}/ses-S001/sarmissiong"
        / f"sub-{subject_id}_ses-S001_task-Default_run-001_sarmissiong.xdf"
    )


def extract_grids(
    game_stream: dict, trial_field: str, include_victim_health: bool = False
) -> dict:
    """Return grid data from the first frame per trial that contains a 'grid' key.

    Returns:
        If include_victim_health is False: {trial_id: grid}
        If include_victim_health is True:  {trial_id: {"grid": ..., "victim_health": ...}}
    """
    result: dict = {}
    for v in game_stream["time_series"]:
        try:
            d = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        tid = d.get(trial_field)
        if not tid or tid in result or "grid" not in d:
            continue
        if include_victim_health:
            result[tid] = {"grid": d["grid"], "victim_health": d.get("victim_health", {})}
        else:
            result[tid] = d["grid"]
    return result


def parse_game(stream: dict, tfield: str) -> dict[str, list[dict]]:
    """Parse game stream and group rows by trial_id.

    Returns:
        {trial_id: list[dict]} with one row list per trial.
    """
    trial_rows: dict[str, list[dict]] = {}
    for ts, v in zip(stream["time_stamps"], stream["time_series"]):
        try:
            d = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        tid = d.get(tfield)
        if tid is not None:
            trial_rows.setdefault(tid, []).append(
                {
                    "timestamp": ts,
                    **{k: w for k, w in d.items() if not isinstance(w, (dict, list))},
                }
            )
    return trial_rows


def parse_eye(
    stream: dict, channels: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract raw arrays from an eye-tracker stream.

    Returns:
        (timestamps, data, cols) — all as numpy arrays / list of column names.
    """
    ts = np.array(stream["time_stamps"])
    data = np.array(stream["time_series"], dtype=float)
    n = data.shape[1]
    if channels:
        cols = list(channels[:n]) + [f"ch{i}" for i in range(len(channels), n)]
    else:
        cols = [f"ch{i}" for i in range(n)]
    return ts, data, cols


def split_streams_by_trial(
    game_stream: dict,
    eye_stream: dict,
    cfg: dict,
) -> dict[str, dict[int, dict[str, pd.DataFrame]]]:
    """Split game and eye-tracking data by trial and episode.

    Episodes within a trial end on rows where any field in
    ``episode_end_fields`` (default: ``terminated``, ``truncated``) is True.

    Returns:
        {trial_id: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}
    """
    xcfg = cfg["xdf"]
    tfield = xcfg["trial_field"]
    end_fields = xcfg.get("episode_end_fields", ["terminated", "truncated"])
    trials = parse_game(game_stream, tfield)
    channels = cfg.get("eyetracker", {}).get("channels")
    eye_ts, eye_data, eye_cols = parse_eye(eye_stream, channels)

    result: dict[str, dict[int, dict[str, pd.DataFrame]]] = {}
    for tid, rows in trials.items():
        end_positions = sorted(set(
            [i for i, row in enumerate(rows) if any(row.get(col) for col in end_fields)]
            + [i for i in range(1, len(rows)) if (rows[i].get("step_count", 1) or 1) < (rows[i - 1].get("step_count", 0) or 0)]
        ))
        starts = [0] + [p + 1 for p in end_positions]
        ends = [p + 1 for p in end_positions] + [len(rows)]

        runs: dict[int, dict[str, pd.DataFrame]] = {}
        run_num = 1
        for s, e in zip(starts, ends):
            if s >= e:
                continue
            slc = pd.DataFrame(rows[s:e])
            t0, t1 = slc["timestamp"].iloc[0], slc["timestamp"].iloc[-1]
            mask = (eye_ts >= t0) & (eye_ts <= t1)
            eye_df = pd.DataFrame(eye_data[mask], columns=eye_cols)
            eye_df.insert(0, "timestamp", eye_ts[mask])
            runs[run_num] = {"game": slc, "eye_tracking": eye_df}
            run_num += 1

        result[tid] = runs
    return result
