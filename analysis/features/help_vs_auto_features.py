
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.eye_tracking_features import build_eye_features
from features.gaze_entropy import build_transition_matrix, gte as compute_gte, regroup_obj_type, sge as compute_sge


def window_slice(df: pd.DataFrame, ts_col: str, lo: float, hi: float) -> pd.DataFrame:
    """Rows of df whose ts_col falls in [lo, hi)."""
    return df[(df[ts_col] >= lo) & (df[ts_col] < hi)]


# ---------------------------------------------------------------------------
# 1. Event timeline
# ---------------------------------------------------------------------------

def first_timestamp_per_step(game: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """One timestamp per step_count among the rows where `mask` is True.

    The game log has one row per eye-tracker sample, not one row per game step,
    so a single step (or a single held-down key press) spans many rows -- this
    collapses that back down to one event per step.
    """
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
    min_fix = hva_cfg.get("min_fixations", 2)
    auto_interval_steps = hva_cfg["auto_interval_steps"]

    groups = cfg["entropy_groups"]
    gte_types = list(groups.keys())

    processed = Path(cfg["paths"]["processed"])
    fix_all = pd.read_csv(processed / "object_aoi_fixations.csv")
    fix_all["obj_type_grouped"] = fix_all["obj_type"].apply(lambda t: regroup_obj_type(t, groups))

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

            sub_fix = fix_all[
                (fix_all["subject"] == sid) & (fix_all["trial"] == trial) & (fix_all["run"] == run_num)
            ]
            if sub_fix.empty:
                continue

            game = store[gk]
            eye = store[ek]
            eye_min = pd.to_numeric(eye["timestamp"], errors="coerce").min()

            events = event_timeline(game, auto_interval_steps)
            for _, ev in events.iterrows():
                event_rel_ms = (ev["timestamp"] - eye_min) * 1000.0
                for w in windows:
                    lo = event_rel_ms - w * 1000.0
                    sub_fix_win = window_slice(sub_fix, "start_ms", lo, event_rel_ms)
                    win_fix = sub_fix_win[sub_fix_win["obj_type_grouped"].isin(gte_types)]
                    n_fix = len(win_fix)
                    if n_fix >= min_fix:
                        win_grouped = win_fix[["obj_type_grouped", "duration_ms"]].rename(
                            columns={"obj_type_grouped": "obj_type"}
                        )
                        sge = compute_sge(win_grouped)
                        gte = compute_gte(build_transition_matrix(win_grouped, gte_types))
                    else:
                        sge, gte = None, None
                    rows.append({
                        "subject": sid, "trial": trial, "run": run_num,
                        "step": int(ev["step"]), "label": int(ev["label"]), "window": w,
                        "sge": sge, "gte": gte,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. New-feature building blocks (no existing function covers these)
# ---------------------------------------------------------------------------

def dwell_fractions(wf: pd.DataFrame, group_names: list[str]) -> dict:
    """Of the time spent fixating on anything, what fraction went to each
    entropy_groups category (Victim/Hazard/Access/Interface).

    `wf["obj_type"]` must already be regrouped via gaze_entropy.regroup_obj_type,
    same categories SGE/GTE use -- no separate category list here. Same
    computation gaze_entropy.dte does internally
    (groupby("obj_type")["duration_ms"].sum(), normalized) -- just returned as
    per-category fractions instead of collapsed into an entropy scalar.
    """
    grouped = wf[wf["obj_type"].isin(group_names)]
    total = grouped["duration_ms"].sum()
    if total <= 0 or grouped.empty:
        return {f"{g.lower()}_dwell_fraction": np.nan for g in group_names}
    by_type = grouped.groupby("obj_type")["duration_ms"].sum()
    return {f"{g.lower()}_dwell_fraction": by_type.get(g, 0.0) / total for g in group_names}


def fixation_dwell_fraction(total_fixation_dur_ms: float | None, n_fix: int, w: int) -> float:
    """Fraction of the whole window spent fixating on anything at all (vs. eyes
    moving between fixations) -- not split by category, unlike dwell_fractions."""
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


def aoi_transition_rate(wf: pd.DataFrame, w: int) -> float:
    """How often consecutive fixations land on a different obj_type, per second."""
    if len(wf) < 2:
        return np.nan
    seq = wf["obj_type"].tolist()
    n_trans = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
    return n_trans / w


def last_aoi_fixated(wf: pd.DataFrame, group_names: list[str]) -> dict:
    """One-hot of which entropy_groups category the last fixation in the window
    landed on -- e.g. last_aoi_interface=1 means they were looking at the
    chat/info panel right before the event."""
    if wf.empty:
        return {f"last_aoi_{g.lower()}": np.nan for g in group_names}
    last = wf.iloc[-1]["obj_type"]
    return {f"last_aoi_{g.lower()}": float(g == last) for g in group_names}


def dwell_diff(dwell: dict, group_a: str, group_b: str) -> float:
    """Dwell-fraction difference between two competing AOI groups, e.g. how much
    more time went to Interface (chat/info) than Victim right before the event."""
    a, b = dwell.get(f"{group_a.lower()}_dwell_fraction"), dwell.get(f"{group_b.lower()}_dwell_fraction")
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return np.nan
    return a - b


def last_transition(wf: pd.DataFrame, group: str) -> dict:
    """Direction of the most recent AOI switch relative to one group (default
    Interface): did the last fixation-to-fixation jump move into that group, or
    out of it? Both are 0 if the last two fixations stayed in the same AOI."""
    key_to, key_from = f"last_transition_to_{group.lower()}", f"last_transition_from_{group.lower()}"
    if len(wf) < 2:
        return {key_to: np.nan, key_from: np.nan}
    prev, last = wf.iloc[-2]["obj_type"], wf.iloc[-1]["obj_type"]
    if prev == last:
        return {key_to: 0.0, key_from: 0.0}
    return {key_to: float(last == group), key_from: float(prev == group)}


def revisit_count(wf: pd.DataFrame) -> float:
    """How many times gaze re-enters an AOI category it had already left within
    the window (e.g. Victim -> Interface -> Victim counts as 1 revisit)."""
    if wf.empty:
        return np.nan
    seq = wf["obj_type"].tolist()
    collapsed = [seq[0]] + [t for prev, t in zip(seq[:-1], seq[1:]) if t != prev]
    counts = pd.Series(collapsed).value_counts()
    return float((counts - 1).clip(lower=0).sum())


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


def pupil_trend(we: pd.DataFrame, lo: float, missing: float = 0.0) -> dict:
    """Mean, peak, and linear trend (slope, per second) of pupil diameter within
    the window. build_eye_features only reports the pupil SD, not these.

    Invalid samples are recorded as `missing` (cfg["eyetracker"]["missing"]), not
    NaN -- same replace-then-dropna build_eye_features uses, so the missing
    sentinel doesn't get averaged in as a real (near-zero) pupil reading.
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


def recent_dwell_fractions(wf: pd.DataFrame, group_names: list[str], lo: float, w: int, recent_window_s: float) -> dict:
    """Same as dwell_fractions, but restricted to the last `recent_window_s`
    seconds of the window -- a recency-weighted version of attention share,
    since gaze in the last couple seconds before an event is more diagnostic
    than gaze from the start of a long look-back window."""
    recent_span = min(recent_window_s, w)
    recent_lo = lo + (w - recent_span) * 1000.0
    recent_wf = wf[wf["start_ms"] >= recent_lo]
    return {f"recent_{k}": v for k, v in dwell_fractions(recent_wf, group_names).items()}


# ---------------------------------------------------------------------------
# 3. Per-window feature assembly
# ---------------------------------------------------------------------------

def window_features(
    cfg: dict, wf: pd.DataFrame, ws: pd.DataFrame, we: pd.DataFrame, lo: float, w: int, group_names: list[str]
) -> dict:
    
    eye_feats = build_eye_features(wf, ws, we, cfg["eyetracker"])
    n_fix, n_sacc = eye_feats["n_fixations"], eye_feats["n_saccades"]

    # AOI-category features (dwell, transitions, revisits, last-AOI) all reason
    # about *which meaningful thing* gaze was on -- same principle SGE/GTE
    # already use (entropy_groups excludes "other"/wall/offscreen). wf_aoi
    # restricts to those categories so a glance at background terrain doesn't
    # get treated as a real AOI, dilute a dwell fraction, or count as a
    # transition/revisit. Raw kinematics below (fixation rate/duration, overall
    # dwell fraction, dispersion) intentionally keep using the full `wf`,
    # since those measure general visual engagement, not AOI identity.
    wf_aoi = wf[wf["obj_type"].isin(group_names)]
    dwell = dwell_fractions(wf_aoi, group_names)

    return {
        "fixation_rate_hz": event_rate(n_fix, w),
        "mean_fixation_dur_ms": eye_feats["mean_fixation_dur_ms"],
        "fixation_dwell_fraction": fixation_dwell_fraction(eye_feats["total_fixation_dur_ms"], n_fix, w),
        "saccade_rate_hz": event_rate(n_sacc, w),
        "mean_saccade_dur_ms": eye_feats["mean_saccade_dur_ms"],
        "mean_saccade_amp_px": eye_feats["mean_saccade_amp_px"],
        "saccade_velocity_px_s": saccade_velocity(ws),
        "saccade_amp_sd_px": saccade_amplitude_sd(ws),
        "std_pupil_diam": eye_feats["std_pupil_diam"],
        "gaze_dispersion_px": gaze_dispersion(wf),
        "aoi_transition_rate_hz": aoi_transition_rate(wf_aoi, w),
        "revisit_count": revisit_count(wf_aoi),
        "blink_rate_hz": blink_rate(we, w),
        "dwell_diff_interface_victim": dwell_diff(dwell, "Interface", "Victim"),
        **pupil_trend(we, lo, missing=cfg["eyetracker"].get("missing", 0.0)),
        **dwell,
        **last_aoi_fixated(wf_aoi, group_names),
        **last_transition(wf_aoi, "Interface"),
        **recent_dwell_fractions(wf, group_names, lo, w, cfg["help_vs_auto"].get("recent_window_s", 2)),
    }


def build_features(cfg: dict, events_df: pd.DataFrame) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed"])
    groups = cfg["entropy_groups"]
    group_names = list(groups.keys())
    fix_all = pd.read_csv(processed / "object_aoi_fixations.csv")
    fix_all["obj_type"] = fix_all["obj_type"].apply(lambda t: regroup_obj_type(t, groups))

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

            sub_fix = fix_all[
                (fix_all["subject"] == sid) & (fix_all["trial"] == trial) & (fix_all["run"] == run_num)
            ].sort_values("start_ms")

            sacc_path = processed / sid / trial / f"run_{run_num}_saccades.csv"
            sacc = (
                pd.read_csv(sacc_path)
                if sacc_path.exists()
                else pd.DataFrame(columns=["start_ms", "end_ms", "duration_ms", "amplitude"])
            )

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
                    **window_features(cfg, wf, ws, we, lo, w, group_names),
                })

    return pd.DataFrame(rows)
