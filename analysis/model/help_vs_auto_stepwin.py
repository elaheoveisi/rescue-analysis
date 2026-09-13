
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from features.help_vs_auto_features import (
    build_events_core,
    build_features_core,
)
from model.help_vs_auto import run_classifiers


def timestamp_at_step(step_timestamps: pd.Series, target_step: float) -> float:
    """Converts a game step into the real timestamp when that step occurred."""
    idx = step_timestamps.index.to_numpy()
    vals = step_timestamps.to_numpy()
    if target_step <= idx.min():
        return vals[0]
    if target_step >= idx.max():
        return vals[-1]
    pos = np.searchsorted(idx, target_step, side="left")
    return vals[pos]


def build_events_steps(cfg: dict) -> pd.DataFrame:
    step_windows = cfg["help_vs_auto"]["step_windows"]

    def compute_window_start_ms(step_timestamps, eye_start_ts, _event_ts_ms, event_step, window_steps):
        window_start_ts = timestamp_at_step(step_timestamps, event_step - window_steps)
        return (window_start_ts - eye_start_ts) * 1000.0

    return build_events_core(cfg, step_windows, compute_window_start_ms)


def build_features_steps(cfg: dict, events_df: pd.DataFrame) -> pd.DataFrame:
    def compute_window_bounds(step_timestamps, eye_start_ts, step, window_steps):
        event_ts = step_timestamps.get(step)
        if event_ts is None or pd.isna(event_ts):
            return None
        event_ts_ms = (event_ts - eye_start_ts) * 1000.0
        window_start_ts = timestamp_at_step(step_timestamps, step - window_steps)
        window_start_ms = (window_start_ts - eye_start_ts) * 1000.0
        window_duration_s = max((event_ts_ms - window_start_ms) / 1000.0, 1e-6)
        return window_start_ms, event_ts_ms, window_duration_s

    return build_features_core(cfg, events_df, compute_window_bounds)


def run(cfg: dict):
    processed = Path(cfg["paths"]["processed"])

    #print(f"Building events for step-windows: {cfg['help_vs_auto']['step_windows']}")
    events_df = build_events_steps(cfg)
    print(f"{len(events_df)} event x window rows")

    #print("Building features...")
    features_df = events_df.merge(
        build_features_steps(cfg, events_df),
        on=["subject", "trial", "run", "step", "label", "window"], how="left",
    )

    events_df.to_csv(processed / "help_vs_auto_events_stepwin.csv", index=False)
    features_df.to_csv(processed / "help_vs_auto_features_stepwin.csv", index=False)

    print("\nRunning RF classifiers")
    results, shap_df, cm = run_classifiers(cfg, features_df)
    results = results.rename(columns={"window_s": "window_steps"})

    results["n_per_class_after_balancing"] = results["n_test_after_balancing"] / 2
    cm = cm.rename(columns={"window_s": "window_steps"})
    shap_df = shap_df.rename(columns={"window_s": "window_steps"}) if not shap_df.empty else shap_df

    results.to_csv(processed / "help_vs_auto_classifier_results_stepwin.csv", index=False)
    cm.to_csv(processed / "help_vs_auto_confusion_matrix_stepwin.csv", index=False)
    shap_df.to_csv(processed / "help_vs_auto_shap_values_stepwin.csv", index=False)

    print(f"\nSaved outputs -> {processed} (*_stepwin.csv)")
    print("\n=== Results (step window) ===")
    print(results.to_string(index=False))
