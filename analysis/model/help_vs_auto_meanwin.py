
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from features.help_vs_auto_features import build_events, build_features
from model.help_vs_auto import run_classifiers
from model.help_vs_auto_stepwin import event_counts, print_counts


def meanwin_windows(hva_cfg: dict) -> tuple[list[int], list[float]]:
    """Time windows equivalent to k steps (k from meanwin_steps): k * mean seconds per step."""
    step_s = hva_cfg.get("meanwin_step_s", 1.07)
    ks = hva_cfg.get("meanwin_steps", list(range(1, 11)))
    return ks, [round(k * step_s, 2) for k in ks]


def meanwin_tag(hva_cfg: dict) -> str:
    """File tag for the k range, e.g. "meanwin_k8-10", so runs over different k don't overwrite each other."""
    ks, _ = meanwin_windows(hva_cfg)
    return f"meanwin_k{min(ks)}-{max(ks)}"


def run_one_interval(cfg: dict, interval: int) -> pd.DataFrame:
    """Time-window pipeline (windows = k * mean step duration) with automatic
    help reconstructed every `interval` actions."""
    processed = Path(cfg["paths"]["processed"])
    cfg = copy.deepcopy(cfg)
    hva_cfg = cfg["help_vs_auto"]
    hva_cfg["auto_interval_steps"] = interval
    ks, windows = meanwin_windows(hva_cfg)
    hva_cfg["windows_s"] = windows
    steps_equiv = dict(zip(windows, ks))
    suffix = f"{meanwin_tag(hva_cfg)}_auto{interval}"

    print(f"\n===== auto help every {interval} actions, windows (s) = {windows} =====")
    events_df = build_events(cfg)
    print(f"{len(events_df)} event x window rows")

    features_df = events_df.merge(
        build_features(cfg, events_df),
        on=["subject", "trial", "run", "step", "label", "window", "event_timestamp"], how="left",
    )

    events_df.to_csv(processed / f"help_vs_auto_events_{suffix}.csv", index=False)
    features_df.to_csv(processed / f"help_vs_auto_features_{suffix}.csv", index=False)

    counts = event_counts(cfg, features_df)
    print_counts(counts, interval)
    counts = counts.rename(columns={"window_steps": "window_s"})
    counts.insert(0, "auto_interval", interval)
    counts.insert(2, "window_steps_equiv", counts["window_s"].map(steps_equiv))
    counts.to_csv(processed / f"help_vs_auto_event_counts_{suffix}.csv", index=False)

    print("\nRunning RF classifiers")
    results, shap_df, cm = run_classifiers(cfg, features_df)
    results.insert(0, "auto_interval", interval)
    results.insert(2, "window_steps_equiv", results["window_s"].map(steps_equiv))
    results["n_per_class_after_balancing"] = results["n_test_after_balancing"] / 2

    results.to_csv(processed / f"help_vs_auto_classifier_results_{suffix}.csv", index=False)
    cm.to_csv(processed / f"help_vs_auto_confusion_matrix_{suffix}.csv", index=False)
    shap_df.to_csv(processed / f"help_vs_auto_shap_values_{suffix}.csv", index=False)

    print(f"\nSaved outputs -> {processed} (*_{suffix}.csv)")
    print(f"\n=== Results (mean-step time window, auto every {interval} actions) ===")
    print(results.to_string(index=False))
    return results


def run(cfg: dict):
    hva_cfg = cfg["help_vs_auto"]
    intervals = hva_cfg.get("stepwin_auto_intervals", [hva_cfg["auto_interval_steps"]])
    all_results = pd.concat([run_one_interval(cfg, iv) for iv in intervals], ignore_index=True)

    processed = Path(cfg["paths"]["processed"])
    all_results.to_csv(processed / f"help_vs_auto_classifier_results_{meanwin_tag(hva_cfg)}_all_intervals.csv", index=False)
