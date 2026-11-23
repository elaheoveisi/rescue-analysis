
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.eye_tracking_features import build_eye_features
from features.gaze_window_stats import (
    blink_rate, event_rate, fixation_dwell_fraction, last_saccade_features,
    pupil_trend, saccade_amplitude_sd, saccade_peak_velocity, saccade_velocity,
    scanpath_length_rate,
)
from features.game_steps import EVENT_KEYS
from features.event_windows import event_windows, key_row, llm_runs, run_events, time_window


def build_events(cfg: dict) -> pd.DataFrame:
    """One row per help event x time window. Window validity is checked later, in build_features."""
    hva_cfg = cfg["help_vs_auto"]
    windows = hva_cfg["windows_s"]
    interval = hva_cfg["auto_interval_steps"]

    rows, timelines = [], {}
    with pd.HDFStore(str(Path(cfg["paths"]["processed"]) / "data.h5"), mode="r") as store:
        for sid, trial, run, gk in llm_runs(store, cfg):
            game = store[gk]
            for _, ev in run_events(store, gk, game, interval, timelines).iterrows():
                for w in windows:
                    rows.append({
                        **key_row(sid, trial, run, ev["step"], ev["label"], w, ev["timestamp"]),
                        "event_basis": ev["event_basis"],
                    })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=EVENT_KEYS)


# ---------------------------------------------------------------------------
# 2. Feature building blocks (AOI-free only -- raw kinematics)
# ---------------------------------------------------------------------------

def gaze_dispersion(wf: pd.DataFrame) -> float:
    """Spatial gaze dispersion is a measure of the general tendency for the eyes to move around. 
    It was calculated as the SD in gaze across time points, averaged across x and y coordinates, and transformed to logarithmic values. Smaller values indicate less gaze dispersion.
    https://www.jneurosci.org/content/jneuro/43/32/5856.full.pdf"""
    if len(wf) < 2:
        return np.nan
    avg_sd = (wf["x"].std(ddof=0) + wf["y"].std(ddof=0)) / 2.0
    return float(np.log(avg_sd)) if avg_sd > 0 else np.nan


# ---------------------------------------------------------------------------
# 3. Per-window feature assembly
# ---------------------------------------------------------------------------

def window_features(
    cfg: dict, wf: pd.DataFrame, ws: pd.DataFrame, we: pd.DataFrame, lo: float, w: int,
) -> dict:

    eye_feats = build_eye_features(wf, ws, we, cfg["eyetracker"])
    n_fix, n_sacc = eye_feats["n_fixations"], eye_feats["n_saccades"]

    return {
        "fixation_rate_hz": event_rate(n_fix, w),
        "mean_fixation_dur_ms": eye_feats["mean_fixation_dur_ms"],
        "max_fixation_dur_ms": eye_feats["max_fixation_dur_ms"],
        "fixation_dwell_fraction": fixation_dwell_fraction(eye_feats["total_fixation_dur_ms"], n_fix, w),
        "dwell_time_ms": eye_feats["total_fixation_dur_ms"],
        "n_saccades": n_sacc,
        "saccade_rate_hz": event_rate(n_sacc, w),
        "mean_saccade_dur_ms": eye_feats["mean_saccade_dur_ms"],
        "mean_saccade_amp_px": eye_feats["mean_saccade_amp_px"],
        "saccade_velocity_px_s": saccade_velocity(ws),
        "peak_saccade_velocity_px_s": saccade_peak_velocity(ws, we, cfg["eyetracker"]),
        "saccade_amp_sd_px": saccade_amplitude_sd(ws),
        "std_pupil_diam": eye_feats["std_pupil_diam"],
        "gaze_dispersion_px": gaze_dispersion(wf),
        "scanpath_length_rate_px_s": scanpath_length_rate(wf, w),
        "blink_rate_hz": blink_rate(we, w),
        **last_saccade_features(ws, we, cfg["eyetracker"]),
        **pupil_trend(we, lo, missing=cfg["eyetracker"].get("missing", 0.0)),
    }


def build_features(cfg: dict, events_df: pd.DataFrame) -> pd.DataFrame:
    """Time-window features from each run's own fixation CSV (no AOI labels needed)."""
    processed = Path(cfg["paths"]["processed"])

    def run_fixations(sid, trial, run):
        path = processed / sid / trial / f"run_{run}_fixations.csv"
        return pd.read_csv(path).sort_values("start_ms") if path.exists() else None

    rows = [
        {**keys, **window_features(cfg, wf, ws, we, lo, w)}
        for keys, wf, ws, we, lo, w in event_windows(cfg, events_df, time_window, run_fixations)
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=EVENT_KEYS)


def build_features_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
   
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]

    events_df = build_events(cfg)
    events_out = processed / hva_cfg.get("general_events_file", "general_gaze_events.csv")
    events_df.to_csv(events_out, index=False)
    print(f"Saved {len(events_df)} rows -> {events_out}")

    features_df = events_df.merge(
        build_features(cfg, events_df), on=["subject", "trial", "run", "step", "label", "window", "event_timestamp"], how="left"
    )
    features_out = processed / hva_cfg.get("general_features_file", "general_gaze_features.csv")
    features_df.to_csv(features_out, index=False)
    print(f"Saved {len(features_df)} rows -> {features_out}")

    return events_df, features_df
