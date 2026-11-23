"""Loading runs and walking event windows -- shared by help_vs_auto_features.py
(stepwin/meanwin) and general_gaze_features.py, so this logic exists once."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.game_steps import action_timestamps, trial_event_timeline, validate_eye_origin, window_slice


def valid_window(lo: float, hi: float) -> bool:
    """Window start is known, after eye tracking began, and before the event."""
    return bool(np.isfinite(lo)) and 0 <= lo < hi


def key_row(sid: str, trial: str, run: int, step, label, w, ts) -> dict:
    """The columns that identify one event x window row (same as EVENT_KEYS)."""
    return {
        "subject": sid, "trial": trial, "run": run,
        "step": int(step), "label": int(label), "window": w, "event_timestamp": float(ts),
    }


def time_window(step_ts, eye_min: float, step, w: float, event_ts: float) -> tuple[float, float, float]:
    """Window of the last `w` seconds before the event, in ms since eye start."""
    event_rel_ms = (event_ts - eye_min) * 1000.0
    return event_rel_ms - w * 1000.0, event_rel_ms, w


def llm_runs(store: pd.HDFStore, cfg: dict):
    """Yield (subject, trial, run number, game key) for every LLM-trial run with
    eye data. Skips help_vs_auto.exclude_participants and, when
    help_vs_auto.expertise_filter is set, subjects of the other group."""
    hva_cfg = cfg["help_vs_auto"]
    exclude = set(hva_cfg.get("exclude_participants", []))
    llm_trials = hva_cfg.get("llm_trials", ["gemini", "openai"])  # pooled, not split
    group = hva_cfg.get("expertise_filter")
    expertise = cfg.get("expertise", {})
    keys = store.keys()
    for gk in [k for k in keys if k.endswith("/game")]:
        _, sid, trial, run_dir, _ = gk.split("/")
        if sid in exclude or trial not in llm_trials:
            continue
        if group and expertise.get(sid, "unknown") != group:
            continue
        if f"/{sid}/{trial}/{run_dir}/eye_tracking" not in keys:
            continue
        yield sid, trial, int(run_dir.replace("run_", "")), gk


def run_events(store: pd.HDFStore, gk: str, game: pd.DataFrame, interval: int, timelines: dict) -> pd.DataFrame:
    """Help events (manual + reconstructed automatic) that fall inside this run.
    The timeline is built once per trial (all runs joined) and cached in `timelines`."""
    _, sid, trial, _, _ = gk.split("/")
    if (sid, trial) not in timelines:
        timelines[(sid, trial)] = trial_event_timeline(store, gk, interval)
    timeline = timelines[(sid, trial)]
    return timeline[timeline["timestamp"].between(game["timestamp"].min(), game["timestamp"].max())]


def load_run(store: pd.HDFStore, sid: str, trial: str, run: int):
    """(game, eye, eye_min) for one run, timestamps numeric and eye["rel_ms"] in ms
    since eye start. None if the run's game or eye data is missing or empty."""
    gk, ek = f"/{sid}/{trial}/run_{run}/game", f"/{sid}/{trial}/run_{run}/eye_tracking"
    if gk not in store or ek not in store:
        return None
    game, eye = store[gk].copy(), store[ek].copy()
    if game.empty or eye.empty:
        return None
    game["timestamp"] = pd.to_numeric(game["timestamp"], errors="coerce")
    game["step_count"] = pd.to_numeric(game["step_count"], errors="coerce")
    eye["timestamp"] = pd.to_numeric(eye["timestamp"], errors="coerce")
    eye_min = eye["timestamp"].min()
    eye["rel_ms"] = (eye["timestamp"] - eye_min) * 1000.0
    return game, eye, eye_min


def event_windows(cfg: dict, events_df: pd.DataFrame, compute_window, fixations_for_run):
    """Yield (key_row, fixations, saccades, eye samples, lo, w_arg) for every
    event row whose window is valid; the three tables are cut to [lo, event).

    compute_window(step_ts, eye_min, step, w, event_ts) -> (lo, event_ms, w_arg) or None
    fixations_for_run(sid, trial, run) -> that run's fixations, or None to skip the run
    """
    processed = Path(cfg["paths"]["processed"])
    with pd.HDFStore(str(processed / "data.h5"), mode="r") as store:
        for (sid, trial, run), sub_events in events_df.groupby(["subject", "trial", "run"]):
            loaded = load_run(store, sid, trial, run)
            if loaded is None:
                continue
            game, eye, eye_min = loaded
            fix = fixations_for_run(sid, trial, run)
            if fix is None:
                continue
            sacc_path = processed / sid / trial / f"run_{run}_saccades.csv"
            if not sacc_path.exists():
                raise FileNotFoundError(f"Missing {sacc_path}; rerun extract_features.")
            sacc = pd.read_csv(sacc_path)
            validate_eye_origin(fix, eye_min)
            validate_eye_origin(sacc, eye_min)
            step_ts = action_timestamps(game)

            for _, ev in sub_events.iterrows():
                window = compute_window(step_ts, eye_min, ev["step"], ev["window"], ev["event_timestamp"])
                if window is None or not valid_window(window[0], window[1]):
                    continue
                lo, hi, w_arg = window
                yield (
                    key_row(sid, trial, run, ev["step"], ev["label"], ev["window"], ev["event_timestamp"]),
                    window_slice(fix, "start_ms", lo, hi),
                    window_slice(sacc, "start_ms", lo, hi),
                    window_slice(eye, "rel_ms", lo, hi),
                    lo, w_arg,
                )
