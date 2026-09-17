
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


from features.game_steps import (
    EVENT_KEYS, action_timestamps, trial_event_timeline, validate_eye_origin, window_slice,
)


def build_events(cfg: dict) -> pd.DataFrame:
    hva_cfg = cfg["help_vs_auto"]
    exclude = set(hva_cfg.get("exclude_participants", []))
    llm_trials = hva_cfg.get("llm_trials", ["gemini", "openai"])  # pooled, not split
    windows = hva_cfg.get("windows_s", [5, 10, 15, 20])
    auto_interval_steps = hva_cfg["auto_interval_steps"]

    processed = Path(cfg["paths"]["processed"])

    rows = []
    timelines = {}
    with pd.HDFStore(str(processed / "data.h5"), mode="r") as store:
        keys = store.keys()
        for gk in [k for k in keys if k.endswith("/game")]:
            _, sid, trial, run_dir, _ = gk.split("/")
            if sid in exclude or trial not in llm_trials:
                continue
            ek = f"/{sid}/{trial}/{run_dir}/eye_tracking"
            if ek not in keys:
                continue
            run_num = int(run_dir.replace("run_", ""))

            game = store[gk]
            trial_key = (sid, trial)
            if trial_key not in timelines:
                timelines[trial_key] = trial_event_timeline(store, gk, auto_interval_steps)
            timeline = timelines[trial_key]
            events = timeline[timeline["timestamp"].between(game["timestamp"].min(), game["timestamp"].max())]
            for _, ev in events.iterrows():
                for w in windows:
                    rows.append({
                        "subject": sid, "trial": trial, "run": run_num,
                        "step": int(ev["step"]), "label": int(ev["label"]), "window": w,
                        "event_timestamp": float(ev["timestamp"]), "event_basis": ev["event_basis"],
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
    processed = Path(cfg["paths"]["processed"])

    rows = []
    with pd.HDFStore(str(processed / "data.h5"), mode="r") as store:
        keys = store.keys()
        for (sid, trial, run_num), sub_events in events_df.groupby(["subject", "trial", "run"]):
            run_dir = f"run_{run_num}"
            gk, ek = f"/{sid}/{trial}/{run_dir}/game", f"/{sid}/{trial}/{run_dir}/eye_tracking"
            if gk not in keys or ek not in keys:
                continue

            game = store[gk].copy()
            game["timestamp"] = pd.to_numeric(game["timestamp"], errors="coerce")
            game["step_count"] = pd.to_numeric(game["step_count"], errors="coerce")
            eye = store[ek].copy()
            if game.empty or eye.empty:
                continue
            eye["timestamp"] = pd.to_numeric(eye["timestamp"], errors="coerce")
            eye_min = eye["timestamp"].min()
            eye["rel_ms"] = (eye["timestamp"] - eye_min) * 1000.0

            fix_path = processed / sid / trial / f"run_{run_num}_fixations.csv"
            sacc_path = processed / sid / trial / f"run_{run_num}_saccades.csv"
            if not fix_path.exists() or not sacc_path.exists():
                continue
            sub_fix = pd.read_csv(fix_path).sort_values("start_ms")
            sacc = pd.read_csv(sacc_path)

            validate_eye_origin(sub_fix, eye_min)
            validate_eye_origin(sacc, eye_min)
            action_timestamps(game)

            for _, ev in sub_events.iterrows():
                step, label, w = ev["step"], ev["label"], ev["window"]
                ts = ev["event_timestamp"]
                event_rel_ms = (ts - eye_min) * 1000.0
                lo = event_rel_ms - w * 1000.0
                if not np.isfinite(lo) or lo < 0 or lo >= event_rel_ms:
                    continue

                wf = window_slice(sub_fix, "start_ms", lo, event_rel_ms)
                ws = window_slice(sacc, "start_ms", lo, event_rel_ms)
                we = window_slice(eye, "rel_ms", lo, event_rel_ms)

                rows.append({
                    "subject": sid, "trial": trial, "run": run_num,
                    "step": int(step), "label": int(label), "window": w,
                    "event_timestamp": float(ev["event_timestamp"]),
                    **window_features(cfg, wf, ws, we, lo, w),
                })

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
