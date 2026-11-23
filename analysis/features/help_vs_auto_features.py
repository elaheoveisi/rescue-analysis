
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.eye_tracking_features import build_eye_features
from features.gaze_entropy import build_transition_matrix, gte as compute_gte, regroup_obj_type, sge as compute_sge
from features.gaze_window_stats import (
    blink_rate, event_rate, fixation_dwell_fraction, last_saccade_features,
    pupil_trend, saccade_amplitude_sd, saccade_peak_velocity, saccade_velocity,
    scanpath_length_rate,
)


from features.game_steps import EVENT_KEYS, action_timestamps, validate_eye_origin, window_slice
from features.event_windows import (
    event_windows, key_row, llm_runs, load_run, run_events, time_window, valid_window,
)


def build_events_core(cfg: dict, windows: list, get_lo) -> pd.DataFrame:
    """Shared event/window-label loop behind build_events() and
    model.help_vs_auto_stepwin.build_events_steps() -- the two only differ in
    how a window's start (`lo`, in ms) is derived from `w`, which is up to
    `get_lo(step_ts, eye_min, event_rel_ms, event_step, w) -> lo`.
    """
    hva_cfg = cfg["help_vs_auto"]
    min_fix = hva_cfg.get("min_fixations", 2)
    interval = hva_cfg["auto_interval_steps"]
    groups = cfg["entropy_groups"]
    gte_types = list(groups.keys())

    processed = Path(cfg["paths"]["processed"])
    fix_all = pd.read_csv(processed / "object_aoi_fixations.csv")
    fix_all["obj_type_grouped"] = fix_all["obj_type"].apply(lambda t: regroup_obj_type(t, groups))

    rows, timelines = [], {}
    with pd.HDFStore(str(processed / "data.h5"), mode="r") as store:
        for sid, trial, run, gk in llm_runs(store, cfg):
            sub_fix = fix_all[(fix_all["subject"] == sid) & (fix_all["trial"] == trial) & (fix_all["run"] == run)]
            if sub_fix.empty:
                continue
            loaded = load_run(store, sid, trial, run)
            if loaded is None:
                continue
            game, _, eye_min = loaded
            validate_eye_origin(sub_fix, eye_min)
            step_ts = action_timestamps(game)

            for _, ev in run_events(store, gk, game, interval, timelines).iterrows():
                event_rel_ms = (ev["timestamp"] - eye_min) * 1000.0
                for w in windows:
                    lo = get_lo(step_ts, eye_min, event_rel_ms, ev["step"], w)
                    if not valid_window(lo, event_rel_ms):
                        continue
                    win_fix = window_slice(sub_fix, "start_ms", lo, event_rel_ms)
                    win_fix = win_fix[win_fix["obj_type_grouped"].isin(gte_types)]
                    sge, gte = window_entropies(win_fix, gte_types) if len(win_fix) >= min_fix else (None, None)
                    rows.append({
                        **key_row(sid, trial, run, ev["step"], ev["label"], w, ev["timestamp"]),
                        "event_basis": ev["event_basis"], "sge": sge, "gte": gte,
                    })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=EVENT_KEYS)


def window_entropies(win_fix: pd.DataFrame, gte_types: list[str]) -> tuple[float, float]:
    """(SGE, GTE) over the AOI-group fixations of one window."""
    grouped = win_fix[["obj_type_grouped", "duration_ms"]].rename(columns={"obj_type_grouped": "obj_type"})
    return compute_sge(grouped), compute_gte(build_transition_matrix(grouped, gte_types))


def build_events(cfg: dict) -> pd.DataFrame:
    """Time windows: each starts `w` seconds before the event."""
    return build_events_core(
        cfg, cfg["help_vs_auto"]["windows_s"],
        get_lo=lambda step_ts, eye_min, event_rel_ms, event_step, w: event_rel_ms - w * 1000.0,
    )


# ---------------------------------------------------------------------------
# 2. New-feature building blocks (no existing function covers these)
# ---------------------------------------------------------------------------

def dwell_fractions(wf: pd.DataFrame, group_names: list[str]) -> dict:

    grouped = wf[wf["obj_type"].isin(group_names)]
    total = grouped["duration_ms"].sum()
    if total <= 0 or grouped.empty:
        return {f"{g.lower()}_dwell_fraction": np.nan for g in group_names}
    by_type = grouped.groupby("obj_type")["duration_ms"].sum()
    return {f"{g.lower()}_dwell_fraction": by_type.get(g, 0.0) / total for g in group_names}


def pct_duration_by_group(wf: pd.DataFrame, w: int, group_names: list[str]) -> dict:
    """Fraction of the whole window's duration spent looking at each
    entropy_groups category (victim+fake_victim as one "Victim" group, etc.
    -- same grouping dwell_fractions uses). Unlike dwell_fractions, these
    don't renormalize to sum to 1 -- idle/untracked gaze time isn't
    redistributed -- and an absent group is 0.0 (it simply wasn't looked
    at), not NaN."""
    if wf.empty:
        return {f"{g.lower()}_pct_dur": 0.0 for g in group_names}
    by_group = wf.groupby("obj_type")["duration_ms"].sum()
    return {f"{g.lower()}_pct_dur": by_group.get(g, 0.0) / (w * 1000.0) for g in group_names}


def fixation_stats_by_group(wf: pd.DataFrame, group_names: list[str]) -> dict:
    """Per entropy_groups category, fixation count and mean duration."""
    stats = {}
    for g in group_names:
        on_group = wf[wf["obj_type"] == g]
        stats[f"{g.lower()}_n_fixations"] = len(on_group)
        stats[f"{g.lower()}_mean_fixation_dur_ms"] = float(on_group["duration_ms"].mean()) if not on_group.empty else np.nan
    return stats


def saccade_stats_by_group(ws: pd.DataFrame, wf: pd.DataFrame, group_names: list[str]) -> dict:
    """Per entropy_groups category, saccade count, mean amplitude, and mean
    velocity. Saccades carry no AOI label of their own, so each is
    attributed to the group of the fixation it lands on -- the first
    fixation starting at or after the saccade ends."""
    keys = [f"{g.lower()}_{suffix}" for g in group_names for suffix in ("n_saccades", "mean_saccade_amp_px", "mean_saccade_velocity_px_s")]
    if ws.empty or wf.empty:
        return {k: (0 if k.endswith("n_saccades") else np.nan) for k in keys}

    saccades_by_end = ws.sort_values("end_ms")
    fixations_by_start = (
        wf[["start_ms", "obj_type"]]
        .rename(columns={"start_ms": "landing_fixation_start_ms"})
        .sort_values("landing_fixation_start_ms")
    )
    landed_on = pd.merge_asof(
        saccades_by_end, fixations_by_start,
        left_on="end_ms", right_on="landing_fixation_start_ms", direction="forward",
    )

    stats = {}
    for g in group_names:
        on_group = landed_on[landed_on["obj_type"] == g]
        stats[f"{g.lower()}_n_saccades"] = len(on_group)
        stats[f"{g.lower()}_mean_saccade_amp_px"] = float(on_group["amplitude"].mean()) if not on_group.empty else np.nan
        stats[f"{g.lower()}_mean_saccade_velocity_px_s"] = saccade_velocity(on_group) if not on_group.empty else np.nan
    return stats


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
    cfg: dict, wf: pd.DataFrame, ws: pd.DataFrame, we: pd.DataFrame, lo: float, w: int,
    group_names: list[str],
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
    recent = recent_dwell_fractions(wf, group_names, lo, w, cfg["help_vs_auto"].get("recent_window_s", 2))

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
        "aoi_transition_rate_hz": aoi_transition_rate(wf_aoi, w),
        "revisit_count": revisit_count(wf_aoi),
        "blink_rate_hz": blink_rate(we, w),
        "dwell_diff_interface_victim": dwell_diff(dwell, "Interface", "Victim"),
        **last_saccade_features(ws, we, cfg["eyetracker"]),
        **pupil_trend(we, lo, missing=cfg["eyetracker"].get("missing", 0.0)),
        **dwell,
        **last_aoi_fixated(wf_aoi, group_names),
        **last_transition(wf_aoi, "Interface"),
        **recent,

        # Per-group breakdown over the full `wf` (pct_dur is a fraction of
        # window time, not of AOI-only time).
        **pct_duration_by_group(wf, w, group_names),
        **fixation_stats_by_group(wf, group_names),
        **saccade_stats_by_group(ws, wf, group_names),
        # Same values as `recent` above, under the "<group>_recent_dwell_fraction" names.
        **{k.removeprefix("recent_").replace("_dwell_fraction", "_recent_dwell_fraction"): v
           for k, v in recent.items()},
    }


def build_features_core(cfg: dict, events_df: pd.DataFrame, compute_window) -> pd.DataFrame:
    """Shared per-event feature-window loop behind build_features() and
    model.help_vs_auto_stepwin.build_features_steps() -- the two only differ
    in how a row's (lo, event_rel_ms, window-features-w-arg) triple is
    derived, which is up to `compute_window(step_ts, eye_min, step, w, event_ts) ->
    (lo, event_rel_ms, w_arg) | None` (None skips the row).
    """
    groups = cfg["entropy_groups"]
    group_names = list(groups.keys())
    fix_all = pd.read_csv(Path(cfg["paths"]["processed"]) / "object_aoi_fixations.csv")
    fix_all["obj_type"] = fix_all["obj_type"].apply(lambda t: regroup_obj_type(t, groups))

    def run_fixations(sid, trial, run):
        return fix_all[
            (fix_all["subject"] == sid) & (fix_all["trial"] == trial) & (fix_all["run"] == run)
        ].sort_values("start_ms")

    rows = [
        {**keys, **window_features(cfg, wf, ws, we, lo, w, group_names)}
        for keys, wf, ws, we, lo, w in event_windows(cfg, events_df, compute_window, run_fixations)
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=EVENT_KEYS)


def build_features(cfg: dict, events_df: pd.DataFrame) -> pd.DataFrame:
    """Time windows: each starts `w` seconds before the event."""
    return build_features_core(cfg, events_df, time_window)
