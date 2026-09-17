"""MOSAIC action clocks and help requests (main and experiment-fov).

Custom victim pickups bypass env.step_count. User.total_steps counts all
actions and, like the help countdown, survives episode resets.
"""

import numpy as np
import pandas as pd

EVENT_KEYS = ["subject", "trial", "run", "step", "label", "window", "event_timestamp"]


def action_timestamps(game: pd.DataFrame) -> pd.Series:
    frame = game[["total_steps", "timestamp"]].apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Missing or invalid action counter/timestamp")
    steps = frame["total_steps"]
    if ((steps < 0) | (steps % 1 != 0) | (steps.diff() < 0)).any():
        raise ValueError("Expected nondecreasing integer total_steps within a trial")
    if "step_count" in game and pd.to_numeric(game["step_count"], errors="coerce").dropna().diff().lt(0).any():
        raise ValueError("Old run boundaries detected: rerun split_data_by_trial and feature extraction.")
    return frame.groupby("total_steps")["timestamp"].min()


def event_timeline(game: pd.DataFrame, auto_interval_steps: int) -> pd.DataFrame:
    """Reconstruct request onsets across a COMPLETE trial.

    Study assumption: recorded Alt presses are accepted and no actions occur
    during an AI call. Thus request and completion have the same action count.
    Automatic labels remain reconstructed, not observed response timestamps.
    """
    if not isinstance(auto_interval_steps, (int, np.integer)) or auto_interval_steps <= 0:
        raise ValueError("auto_interval_steps must be a positive integer")
    frame = game.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    # Episode step resets are valid here; total_steps must not reset in a trial.
    action_timestamps(frame.drop(columns="step_count", errors="ignore"))
    columns = ["step", "timestamp", "label", "event_basis"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["total_steps"] = pd.to_numeric(frame["total_steps"])
    frame["timestamp"] = pd.to_numeric(frame["timestamp"])
    if "llm_request_source" in frame:
        source = frame["llm_request_source"]
        mask = source.isin(["manual", "auto"])
        if "llm_request_id" in frame:
            mask &= frame["llm_request_id"].notna() & ~frame["llm_request_id"].duplicated()
        else:
            mask &= source.ne(source.shift())
        result = frame.loc[mask, ["total_steps", "timestamp"]].rename(columns={"total_steps": "step"})
        result["label"] = source[mask].map({"manual": 1, "auto": 0})
        result["event_basis"] = "recorded_request"
        return result.reset_index(drop=True)
    alt = frame["alt_pressed"].isin([True, 1, "true", "True"])
    manual = alt & ~alt.shift(fill_value=False)
    changed = frame["total_steps"].ne(frame["total_steps"].shift())
    due = auto_interval_steps if frame["total_steps"].iloc[0] <= 1 else None
    selected = frame.loc[changed | manual, ["total_steps", "timestamp"]].copy()
    selected["manual"] = manual.loc[selected.index]
    rows = []
    for step, timestamp, is_manual in selected.itertuples(index=False, name=None):
        step = int(step)
        if due is not None and step >= due:
            due += ((step - due) // auto_interval_steps) * auto_interval_steps
            if step == due and not is_manual:
                rows.append((step, timestamp, 0, "reconstructed_countdown"))
            # Never assign the timestamp of a later action to a missed threshold.
            # A simultaneous Alt/threshold frame has unresolved ordering; do
            # not fabricate an additional automatic call on that frame.
            due += auto_interval_steps
        if is_manual:
            rows.append((step, timestamp, 1, "recorded_alt"))
            due = step + auto_interval_steps
    return pd.DataFrame(rows, columns=columns)


def trial_event_timeline(store, game_key: str, interval: int) -> pd.DataFrame:
    prefix = game_key.rsplit("/", 2)[0] + "/"
    games = [store[k] for k in store.keys() if k.startswith(prefix) and k.endswith("/game")]
    # All-null optional fields differ across reset-only and active episodes.
    game = pd.concat([g.dropna(axis=1, how="all") for g in games], ignore_index=True).sort_values("timestamp", kind="stable")
    return event_timeline(game.drop_duplicates("timestamp"), interval)


def validate_eye_origin(frame: pd.DataFrame, origin: float) -> None:
    """Prevent old CSV clocks from being combined with newly split H5 runs."""
    if "eye_origin_timestamp" not in frame:
        raise ValueError("Stale eye/AOI CSV: rerun extract_features and victim_aoi after splitting data.")
    values = pd.to_numeric(frame["eye_origin_timestamp"], errors="coerce")
    if not np.isclose(values, origin, rtol=0, atol=1e-6).all():
        raise ValueError("Eye/AOI CSV origin differs from H5; rebuild features for these runs.")


def window_slice(df: pd.DataFrame, ts_col: str, lo: float, hi: float) -> pd.DataFrame:
    """Completed events with onsets in [lo, hi); never include future gaze."""
    mask = df[ts_col].ge(lo) & df[ts_col].lt(hi)
    if "end_ms" in df:
        mask &= df["end_ms"].le(hi)
    elif ts_col == "start_ms" and "duration_ms" in df:
        mask &= (df[ts_col] + df["duration_ms"]).le(hi)
    return df.loc[mask]
