import numpy as np
import pandas as pd
from igaze.detectors import fixation_detection, saccade_detection

"The reason we used this: https://link.springer.com/article/10.3758/s13428-013-0422-2"


def _preprocess_eye(eye_df: pd.DataFrame, eye_cfg: dict):
    """Preprocess eye-tracking DataFrame and return (df, x, y, time) arrays."""
    x_col = eye_cfg["x_col"]
    y_col = eye_cfg["y_col"]
    time_col = eye_cfg["time_col"]

    eye_df = eye_df.copy()
    eye_df[time_col] = (eye_df[time_col] - eye_df[time_col].iloc[0]) * 1000.0
    eye_df = eye_df.dropna(subset=[x_col, y_col])
    eye_df[x_col] = eye_df[x_col] * eye_cfg["screen_w"]
    eye_df[y_col] = eye_df[y_col] * eye_cfg["screen_h"]

    x = eye_df[x_col].to_numpy(dtype=float)
    y = eye_df[y_col].to_numpy(dtype=float)
    time = eye_df[time_col].to_numpy(dtype=float)
    return eye_df, x, y, time


def _detect_fixations(x, y, time, eye_cfg: dict, fix_cfg: dict) -> pd.DataFrame:
    _, Efix = fixation_detection(
        x=x,
        y=y,
        time=time,
        missing=eye_cfg["missing"],
        maxdist=fix_cfg["maxdist"],
        mindur=fix_cfg["mindur"],
    )
    return pd.DataFrame(Efix or [], columns=fix_cfg["columns"])


def _detect_saccades(x, y, time, eye_cfg: dict, sac_cfg: dict) -> pd.DataFrame:
    _, end_saccades = saccade_detection(
        x,
        y,
        time,
        missing=eye_cfg["missing"],
        minlen=sac_cfg["minlen"],
        maxvel=sac_cfg["maxvel"],
        maxacc=sac_cfg["maxacc"],
    )
    raw_cols = [
        "start_ms",
        "end_ms",
        "duration_ms",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
    ]
    sac_df = pd.DataFrame(end_saccades or [], columns=raw_cols)

    if not sac_df.empty:
        sac_df.insert(0, "saccade_id", sac_df.index + 1)
        sac_df["amplitude"] = np.hypot(
            sac_df["x_end"] - sac_df["x_start"],
            sac_df["y_end"] - sac_df["y_start"],
        )
    else:
        sac_df = pd.DataFrame(columns=sac_cfg["columns"])
    return sac_df


def run_eyetracking(eye_df: pd.DataFrame, cfg: dict) -> dict:
    """Run fixation and saccade detection for a single trial.

    Returns:
        {"fixations": DataFrame, "saccades": DataFrame}
    """
    fix_cfg = cfg.get("fixation", {})
    sac_cfg = cfg.get("saccade", {})

    if eye_df is None or eye_df.empty:
        return {
            "fixations": pd.DataFrame(columns=fix_cfg["columns"]),
            "saccades": pd.DataFrame(columns=sac_cfg["columns"]),
        }

    eye_cfg = cfg.get("eyetracker", {})
    _, x, y, time = _preprocess_eye(eye_df, eye_cfg)

    return {
        "fixations": _detect_fixations(x, y, time, eye_cfg, fix_cfg),
        "saccades": _detect_saccades(x, y, time, eye_cfg, sac_cfg),
    }


def build_eye_features(
    fix_df: pd.DataFrame,
    sac_df: pd.DataFrame,
    eye_df: pd.DataFrame,
    eye_cfg: dict,
) -> dict:
    """Build eye-tracking feature dict from fixations, saccades, and raw eye data."""
    pupil_col = eye_cfg.get("pupil_col")
    if not eye_df.empty and pupil_col and pupil_col in eye_df.columns:
        pupil = pd.to_numeric(eye_df[pupil_col], errors="coerce")
        pupil = pupil.replace(eye_cfg.get("missing", 0.0), pd.NA).dropna()
        std_pupil = float(pupil.std()) if not pupil.empty else None
    else:
        std_pupil = None

    return {
        "n_fixations": len(fix_df),
        "mean_fixation_dur_ms": float(fix_df["duration_ms"].mean())
        if not fix_df.empty
        else None,
        "max_fixation_dur_ms": float(fix_df["duration_ms"].max())
        if not fix_df.empty
        else None,
        "total_fixation_dur_ms": float(fix_df["duration_ms"].sum())
        if not fix_df.empty
        else None,
        "n_saccades": len(sac_df),
        "mean_saccade_dur_ms": float(sac_df["duration_ms"].mean())
        if not sac_df.empty
        else None,
        "mean_saccade_amp_px": float(sac_df["amplitude"].mean())
        if not sac_df.empty
        else None,
        "saccades_total_duration_ms": float(sac_df["duration_ms"].sum())
        if not sac_df.empty
        else None,
        "std_pupil_diam": std_pupil,
    }
