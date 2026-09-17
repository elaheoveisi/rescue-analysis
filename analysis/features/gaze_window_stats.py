"""Per-window gaze/saccade/pupil statistics shared by help_vs_auto_features.py
and general_gaze_features.py. Extracted here because both modules had grown
identical copies of these functions -- keep the single implementation here
rather than re-duplicating a fix or a docstring in only one file next time.

Note: gaze_dispersion is deliberately NOT here -- the two modules compute
genuinely different statistics under that name (Euclidean spread vs. a
log-transformed average SD), so each keeps its own version.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def event_rate(count: int, w: int) -> float:
    """Events per second within the window (used for fixation and saccade rate)."""
    return count / w


def fixation_dwell_fraction(total_fixation_dur_ms: float | None, n_fix: int, w: int) -> float:
    """Fraction of the whole window spent fixating on anything at all (vs. eyes
    moving between fixations) -- not split by category, unlike dwell_fractions."""
    if not n_fix:
        return np.nan
    return (total_fixation_dur_ms or 0) / (w * 1000.0)


def scanpath_length_rate(wf: pd.DataFrame, w: int) -> float:
    """Total distance traveled fixation-to-fixation in the window (scanpath
    length: the path connecting consecutive fixation centers), normalized by
    window duration. Unlike gaze_dispersion_px (spread around a center), this
    measures the actual path length traveled -- how much visual searching per
    second, not just how far apart the fixations ended up.

    https://www.sciencedirect.com/science/article/pii/S0165027003001511
    https://pmc.ncbi.nlm.nih.gov/articles/PMC8460493/
    """
    if len(wf) < 2:
        return np.nan
    x, y = wf["x"].to_numpy(), wf["y"].to_numpy()
    length = np.hypot(np.diff(x), np.diff(y)).sum()
    return float(length / w)


def saccade_velocity(ws: pd.DataFrame) -> float:
    """Mean saccade velocity in the window (amplitude / duration), px per second.
    igaze's saccade detector reports amplitude and duration but not velocity
    directly, so it's derived here rather than read off a column."""
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
    """Fastest instantaneous point-to-point gaze speed during any saccade in
    the window, px/s. saccade_velocity() only has each saccade's amplitude and
    total duration, which gives an average speed and misses the true
    within-saccade peak -- this instead walks the raw eye samples inside each
    saccade's [start_ms, end_ms] span and takes the largest consecutive-sample
    speed, then reports the max across all saccades in the window.

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
    """Amplitude, duration, mean velocity, and peak velocity of the single
    saccade closest to the event -- the last one to occur within the window,
    as opposed to the window-wide saccade features above which summarize
    every saccade in the window.

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

    Invalid samples are recorded as `missing` (cfg["eyetracker"]["missing"]), not
    NaN -- same replace-then-dropna build_eye_features uses, so the missing
    sentinel doesn't get averaged in as a real (near-zero) pupil reading.

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
    if len(pupil) > 1 and we.loc[pupil.index, "rel_ms"].nunique() > 1:
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
    prev_valid = np.concatenate(([valid[0]], valid[:-1]))  # first sample has no observed predecessor
    n_onsets = int((prev_valid & ~valid).sum())
    return n_onsets / w
