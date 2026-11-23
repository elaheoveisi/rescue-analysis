
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from classification.features import build_events, build_features
from classification.stepwin import build_interval, classify_interval


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
    """Time-window pipeline (windows = k * mean step duration), auto help every `interval` actions."""
    cfg = copy.deepcopy(cfg)
    ks, windows = meanwin_windows(cfg["info_request"])
    cfg["info_request"].update(auto_interval_steps=interval, windows_s=windows)
    suffix = f"{meanwin_tag(cfg['info_request'])}_auto{interval}"
    features_df = build_interval(cfg, suffix, build_events, build_features)
    return classify_interval(cfg, interval, suffix, features_df, win_col="window_s", equiv=dict(zip(windows, ks)))


def run(cfg: dict):
    hva_cfg = cfg["info_request"]
    intervals = hva_cfg.get("stepwin_auto_intervals", [hva_cfg["auto_interval_steps"]])
    all_results = pd.concat([run_one_interval(cfg, iv) for iv in intervals], ignore_index=True)

    processed = Path(cfg["paths"]["processed"])
    all_results.to_csv(processed / f"info_request_classifier_results_{meanwin_tag(hva_cfg)}_all_intervals.csv", index=False)
