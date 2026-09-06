
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.eye_tracking_features import build_eye_features


def window_slice(df: pd.DataFrame, ts_col: str, lo: float, hi: float) -> pd.DataFrame:
    """Rows of df whose ts_col falls in [lo, hi)."""
    return df[(df[ts_col] >= lo) & (df[ts_col] < hi)]


# ---------------------------------------------------------------------------
# 1. Event timeline
# ---------------------------------------------------------------------------

def first_timestamp_per_step(game: pd.DataFrame, mask: pd.Series) -> pd.Series:
    return game.loc[mask].dropna(subset=["step_count"]).groupby("step_count")["timestamp"].min()


def event_timeline(game: pd.DataFrame, auto_interval_steps: int) -> pd.DataFrame:
    """One row per event (step, label, timestamp), chronological.

    label: 1 = manual alt-press, 0 = automatic recommendation every
    `auto_interval_steps` game steps.
    """
    game = game.copy()
    game["timestamp"] = pd.to_numeric(game["timestamp"], errors="coerce")
    game["step_count"] = pd.to_numeric(game["step_count"], errors="coerce")

    manual_steps = first_timestamp_per_step(game, game["alt_pressed"] == True)

    max_step = game["step_count"].max()
    auto_candidates = np.arange(
        auto_interval_steps, (max_step // auto_interval_steps + 1) * auto_interval_steps, auto_interval_steps
    )
    auto_mask = game["step_count"].isin(auto_candidates) & ~game["step_count"].isin(manual_steps.index)
    auto_steps = first_timestamp_per_step(game, auto_mask)

    events = pd.DataFrame(
        [{"step": int(s), "timestamp": t, "label": 1} for s, t in manual_steps.items()]
        + [{"step": int(s), "timestamp": t, "label": 0} for s, t in auto_steps.items()]
    )
    if events.empty:
        return events
    return events.sort_values("timestamp").reset_index(drop=True)


def build_events(cfg: dict) -> pd.DataFrame:
    hva_cfg = cfg["help_vs_auto"]
    exclude = set(hva_cfg.get("exclude_participants", []))
    llm_trials = hva_cfg.get("llm_trials", ["gemini", "openai"])  # pooled, not split
    windows = hva_cfg.get("windows_s", [5, 10, 15, 20])
    auto_interval_steps = hva_cfg["auto_interval_steps"]

    processed = Path(cfg["paths"]["processed"])

    rows = []
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
            events = event_timeline(game, auto_interval_steps)
            for _, ev in events.iterrows():
                for w in windows:
                    rows.append({
                        "subject": sid, "trial": trial, "run": run_num,
                        "step": int(ev["step"]), "label": int(ev["label"]), "window": w,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Feature building blocks (AOI-free only -- raw kinematics)
# ---------------------------------------------------------------------------

def fixation_dwell_fraction(total_fixation_dur_ms: float | None, n_fix: int, w: int) -> float:
    """Fraction of the whole window spent fixating on anything at all (vs. eyes
    moving between fixations)."""
    if not n_fix:
        return np.nan
    return (total_fixation_dur_ms or 0) / (w * 1000.0)


def event_rate(count: int, w: int) -> float:
    """Events per second within the window (used for fixation and saccade rate)."""
    return count / w


def gaze_dispersion(wf: pd.DataFrame) -> float:
    """Spread (std) of fixation x/y positions in the window -- how far the eyes
    roamed, independent of what they were looking at."""
    if len(wf) < 2:
        return np.nan
    return float(np.sqrt(wf["x"].var(ddof=0) + wf["y"].var(ddof=0)))


def scanpath_length_rate(wf: pd.DataFrame, w: int) -> float:
    """Total distance traveled fixation-to-fixation in the window (scanpath
    length: the path connecting consecutive fixation centers), normalized by
    window duration.

    https://www.sciencedirect.com/science/article/pii/S0165027003001511
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8460493/
    """
    if len(wf) < 2:
        return np.nan
    x, y = wf["x"].to_numpy(), wf["y"].to_numpy()
    length = np.hypot(np.diff(x), np.diff(y)).sum()
    return float(length / w)


def saccade_velocity(ws: pd.DataFrame) -> float:
    if ws.empty:
        return np.nan
    dur_s = ws["duration_ms"] / 1000.0
    valid = dur_s > 0
    if not valid.any():
        return np.nan
    return float((ws.loc[valid, "amplitude"] / dur_s[valid]).mean())


def saccade_amplitude_sd(ws: pd.DataFrame) -> float:
    """Std of saccade amplitude in the window -- erratic, variable-sized search
    vs. a run of similar, purposeful saccades."""
    if len(ws) < 2:
        return np.nan
    return float(ws["amplitude"].std())


def saccade_peak_velocity(ws: pd.DataFrame, we: pd.DataFrame, eye_cfg: dict) -> float:
    """
    https://link.springer.com/article/10.1140/epjs/s11734-026-02324-9
    https://www.nature.com/articles/s41467-018-05319-w
    """
    if ws.empty or we.empty:
        return np.nan
    x_col, y_col = eye_cfg["x_col"], eye_cfg["y_col"]
    we = we.dropna(subset=[x_col, y_col]).sort_values("rel_ms")
    if len(we) < 2:
        return np.nan
    t = we["rel_ms"].to_numpy()
    x = we[x_col].to_numpy() * eye_cfg["screen_w"]
    y = we[y_col].to_numpy() * eye_cfg["screen_h"]

    peaks = []
    for _, sacc in ws.iterrows():
        mask = (t >= sacc["start_ms"]) & (t <= sacc["end_ms"])
        if mask.sum() < 2:
            continue
        tt, xx, yy = t[mask], x[mask], y[mask]
        dt_s = np.diff(tt) / 1000.0
        valid = dt_s > 0
        if not valid.any():
            continue
        dist = np.hypot(np.diff(xx)[valid], np.diff(yy)[valid])
        peaks.append(float((dist / dt_s[valid]).max()))
    return max(peaks) if peaks else np.nan


def last_saccade_features(ws: pd.DataFrame, we: pd.DataFrame, eye_cfg: dict) -> dict:
    """

    https://www.nature.com/articles/s41467-018-05319-w.pdf
    """
    keys = [
        "last_saccade_amplitude_px", "last_saccade_duration_ms",
        "last_saccade_velocity_px_s", "last_saccade_peak_velocity_px_s",
    ]
    if ws.empty:
        return {k: np.nan for k in keys}
    last = ws.sort_values("start_ms").iloc[-1]
    amplitude = float(last["amplitude"])
    duration_ms = float(last["duration_ms"])
    dur_s = duration_ms / 1000.0
    velocity = amplitude / dur_s if dur_s > 0 else np.nan
    peak_velocity = saccade_peak_velocity(pd.DataFrame([last]), we, eye_cfg)
    return {
        "last_saccade_amplitude_px": amplitude,
        "last_saccade_duration_ms": duration_ms,
        "last_saccade_velocity_px_s": velocity,
        "last_saccade_peak_velocity_px_s": peak_velocity,
    }


def pupil_trend(we: pd.DataFrame, lo: float, missing: float = 0.0) -> dict:
    """Mean, peak, and linear trend (slope, per second) of pupil diameter within
    the window. build_eye_features only reports the pupil SD, not these.

    pupil_slope_per_s: linear regression of pupil diameter on time-in-window
    (OLS slope, beta_1). Kontogiorgos et al. (2021) used linear regression the
    same way to get pupil-diameter slope per segment:
    https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.623657/full
    """
    pupil = we["avg_pupil_diam"].replace(missing, np.nan).dropna()
    if pupil.empty:
        return {"mean_pupil_diameter": np.nan, "peak_pupil_diameter": np.nan, "pupil_slope_per_s": np.nan}
    mean_pupil_diameter = pupil.mean()
    peak_pupil_diameter = pupil.max()
    if len(pupil) > 1:
        t_s = (we.loc[pupil.index, "rel_ms"] - lo) / 1000.0
        pupil_slope_per_s = float(np.polyfit(t_s, pupil, 1)[0])
    else:
        pupil_slope_per_s = np.nan
    return {
        "mean_pupil_diameter": mean_pupil_diameter,
        "peak_pupil_diameter": peak_pupil_diameter,
        "pupil_slope_per_s": pupil_slope_per_s,
    }


def blink_rate(we: pd.DataFrame, w: int) -> float:
    """Blink onsets per second: a sample where both eyes are valid
    (eye_validities == 3) followed by one where they aren't. Counts onsets only
    (not recoveries), so a blink that's still ongoing at the end of the window
    still counts once."""
    if we.empty:
        return np.nan
    valid = (we["eye_validities"] == 3).to_numpy()
    prev_valid = np.concatenate(([True], valid[:-1]))  # window boundary assumed valid
    n_onsets = int((prev_valid & ~valid).sum())
    return n_onsets / w


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
            eye["timestamp"] = pd.to_numeric(eye["timestamp"], errors="coerce")
            eye_min = eye["timestamp"].min()
            eye["rel_ms"] = (eye["timestamp"] - eye_min) * 1000.0

            fix_path = processed / sid / trial / f"run_{run_num}_fixations.csv"
            sacc_path = processed / sid / trial / f"run_{run_num}_saccades.csv"
            if not fix_path.exists() or not sacc_path.exists():
                continue
            sub_fix = pd.read_csv(fix_path).sort_values("start_ms")
            sacc = pd.read_csv(sacc_path)

            step_ts = game.groupby("step_count")["timestamp"].min()

            for _, ev in sub_events.iterrows():
                step, label, w = ev["step"], ev["label"], ev["window"]
                ts = step_ts.get(step)
                event_rel_ms = (ts - eye_min) * 1000.0
                lo = event_rel_ms - w * 1000.0

                wf = window_slice(sub_fix, "start_ms", lo, event_rel_ms)
                ws = window_slice(sacc, "start_ms", lo, event_rel_ms)
                we = window_slice(eye, "rel_ms", lo, event_rel_ms)

                rows.append({
                    "subject": sid, "trial": trial, "run": run_num,
                    "step": int(step), "label": int(label), "window": w,
                    **window_features(cfg, wf, ws, we, lo, w),
                })

    return pd.DataFrame(rows)


def build_features_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
   
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]

    events_df = build_events(cfg)
    events_out = processed / hva_cfg.get("general_events_file", "general_gaze_events.csv")
    events_df.to_csv(events_out, index=False)
    print(f"Saved {len(events_df)} rows -> {events_out}")

    features_df = events_df.merge(
        build_features(cfg, events_df), on=["subject", "trial", "run", "step", "label", "window"], how="left"
    )
    features_out = processed / hva_cfg.get("general_features_file", "general_gaze_features.csv")
    features_df.to_csv(features_out, index=False)
    print(f"Saved {len(features_df)} rows -> {features_out}")

    return events_df, features_df
