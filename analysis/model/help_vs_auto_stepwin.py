
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from features.help_vs_auto_features import (
    build_events_core,
    build_features_core,
)
from model.help_vs_auto import balanced_sample, make_rf, run_classifiers, window_rows

# Modes for run(): within-group LOSO, or train on one group and test on the other.
WITHIN_MODES = {"all": None, "expert": "expert", "novice": "novice"}
CROSS_MODES = {"expert_to_novice": ("expert", "novice"), "novice_to_expert": ("novice", "expert")}


def timestamp_at_step(step_timestamps: pd.Series, target_step: float) -> float:
    """An unobserved action has no timestamp; do not clamp/interpolate it."""
    return step_timestamps.get(target_step, np.nan)


def build_events_steps(cfg: dict) -> pd.DataFrame:
    step_windows = cfg["help_vs_auto"]["step_windows"]

    def compute_window_start_ms(step_timestamps, eye_start_ts, _event_ts_ms, event_step, window_steps):
        window_start_ts = timestamp_at_step(step_timestamps, event_step - window_steps)
        return (window_start_ts - eye_start_ts) * 1000.0

    return build_events_core(cfg, step_windows, compute_window_start_ms)


def build_features_steps(cfg: dict, events_df: pd.DataFrame) -> pd.DataFrame:
    def compute_window_bounds(step_timestamps, eye_start_ts, step, window_steps, event_ts):
        if event_ts is None or pd.isna(event_ts):
            return None
        event_ts_ms = (event_ts - eye_start_ts) * 1000.0
        window_start_ts = timestamp_at_step(step_timestamps, step - window_steps)
        window_start_ms = (window_start_ts - eye_start_ts) * 1000.0
        window_duration_s = (event_ts_ms - window_start_ms) / 1000.0
        if not np.isfinite(window_duration_s) or window_duration_s <= 0:
            return None
        return window_start_ms, event_ts_ms, window_duration_s

    return build_features_core(cfg, events_df, compute_window_bounds)


def event_counts(cfg: dict, features_df: pd.DataFrame) -> pd.DataFrame:
    """Manual/auto counts per subject and window: kept by the model (after the
    NaN rule in window_rows) and after LOSO test balancing (each held-out
    subject downsampled to their own minority class; subjects missing a class
    are not tested, so they get 0). Includes a "Total" row per window."""
    hva_cfg = cfg["help_vs_auto"]
    features = hva_cfg["features"]
    zero_fill = hva_cfg.get("zero_fill_features", [])
    rows = []
    for w in sorted(features_df["window"].unique()):
        base = window_rows(features_df, w, features, zero_fill)
        c = base.groupby(["subject", "label"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
        bal = np.where((c[0] > 0) & (c[1] > 0), c.min(axis=1), 0)
        per_subj = pd.DataFrame({
            "window_steps": w, "subject": c.index,
            "manual": c[1].to_numpy(), "auto": c[0].to_numpy(),
            "manual_balanced": bal, "auto_balanced": bal,
        })
        total = per_subj.drop(columns=["window_steps", "subject"]).sum().to_dict()
        rows.append(pd.concat([per_subj, pd.DataFrame([{"window_steps": w, "subject": "Total", **total}])]))
    return pd.concat(rows, ignore_index=True)


def print_counts(counts: pd.DataFrame, interval: int):
    """Subjects x windows, each cell "manual/auto"."""
    for label, m, a in [("kept", "manual", "auto"), ("after balancing", "manual_balanced", "auto_balanced")]:
        cell = counts.assign(cell=counts[m].astype(int).astype(str) + "/" + counts[a].astype(int).astype(str))
        table = cell.pivot(index="subject", columns="window_steps", values="cell")
        order = [s for s in table.index if s != "Total"] + ["Total"]
        print(f"\n=== manual/auto per subject, {label} (auto every {interval} actions) ===")
        print(table.loc[order].to_string())


def stepwin_name(cfg: dict) -> str:
    """File-name prefix: "stepwin", or "stepwin_<stepwin_tag>" when the config sets a tag,
    so a rerun (e.g. a different feature set) doesn't overwrite earlier outputs."""
    tag = cfg["help_vs_auto"].get("stepwin_tag")
    return f"stepwin_{tag}" if tag else "stepwin"


def run_one_interval(cfg: dict, interval: int, group: str | None = None) -> pd.DataFrame:
    """Full stepwin pipeline with automatic help reconstructed every `interval` actions.
    `group` ("expert"/"novice") keeps only that expertise group; outputs then get
    a "_<group>" tag so they don't overwrite the all-participant files."""
    processed = Path(cfg["paths"]["processed"])
    cfg = copy.deepcopy(cfg)
    cfg["help_vs_auto"]["auto_interval_steps"] = interval
    cfg["help_vs_auto"]["expertise_filter"] = group
    name = stepwin_name(cfg)
    suffix = f"{name}_{group}_auto{interval}" if group else f"{name}_auto{interval}"

    print(f"\n===== {group or 'all participants'}: auto help every {interval} actions =====")
    events_df = build_events_steps(cfg)
    subjects = sorted(events_df["subject"].unique())
    print(f"{len(subjects)} subjects ({group or 'all'}): {', '.join(subjects)}")
    print(f"{len(events_df)} event x window rows")

    features_df = events_df.merge(
        build_features_steps(cfg, events_df),
        on=["subject", "trial", "run", "step", "label", "window", "event_timestamp"], how="left",
    )

    events_df.to_csv(processed / f"help_vs_auto_events_{suffix}.csv", index=False)
    features_df.to_csv(processed / f"help_vs_auto_features_{suffix}.csv", index=False)

    counts = event_counts(cfg, features_df)
    counts.insert(0, "auto_interval", interval)
    counts.to_csv(processed / f"help_vs_auto_event_counts_{suffix}.csv", index=False)
    print_counts(counts, interval)

    print("\nRunning RF classifiers")
    results, shap_df, cm = run_classifiers(cfg, features_df)
    results = results.rename(columns={"window_s": "window_steps"})
    results.insert(0, "auto_interval", interval)

    results["n_per_class_after_balancing"] = results["n_test_after_balancing"] / 2
    cm = cm.rename(columns={"window_s": "window_steps"})
    shap_df = shap_df.rename(columns={"window_s": "window_steps"}) if not shap_df.empty else shap_df

    results.to_csv(processed / f"help_vs_auto_classifier_results_{suffix}.csv", index=False)
    cm.to_csv(processed / f"help_vs_auto_confusion_matrix_{suffix}.csv", index=False)
    shap_df.to_csv(processed / f"help_vs_auto_shap_values_{suffix}.csv", index=False)

    print(f"\nSaved outputs -> {processed} (*_{suffix}.csv)")
    print(f"\n=== Results (step window, auto every {interval} actions) ===")
    print(results.to_string(index=False))
    return results


def cross_group_predictor(expertise: dict, train_group: str, test_group: str):
    """Drop-in replacement for loso_predictions: one RF fit on every subject of
    `train_group` (balanced), then each `test_group` subject is tested on their
    own balanced events. Groups don't overlap, so no subject is held out."""
    def predict(base, features, hva_cfg, seed):
        group_of = base["subject"].map(lambda s: expertise.get(s, "unknown"))
        train_raw, test_df = base[group_of == train_group], base[group_of == test_group]
        y_true, y_pred, shap_rows, x_rows, per_subject = [], [], [], [], []
        if train_raw["label"].nunique() == 2:
            train_bal = balanced_sample(train_raw, seed)
            rf = make_rf(hva_cfg)
            rf.fit(train_bal[features], train_bal["label"])
            explainer = shap.TreeExplainer(rf)
            for s, test_raw in test_df.groupby("subject"):
                if test_raw["label"].nunique() < 2:
                    continue
                test_bal = balanced_sample(test_raw, seed)
                p = rf.predict_proba(test_bal[features])[:, 1]
                y_true.extend(test_bal["label"].tolist())
                y_pred.extend(p.tolist())
                per_subject.append((s, test_bal["label"].tolist(), p.tolist()))
                sv = explainer.shap_values(test_bal[features])
                if isinstance(sv, list):
                    sv = sv[1]  # positive-class (label=1) SHAP values
                elif sv.ndim == 3:
                    sv = sv[:, :, 1]
                shap_rows.append(sv)
                x_rows.append(test_bal[features].to_numpy(dtype=float))
        shap_matrix = np.vstack(shap_rows) if shap_rows else np.empty((0, len(features)))
        x_matrix = np.vstack(x_rows) if x_rows else np.empty((0, len(features)))
        return {"rf": (y_true, y_pred, shap_matrix, x_matrix, per_subject)}
    return predict


def run_cross_interval(cfg: dict, interval: int, train_group: str, test_group: str) -> pd.DataFrame:
    """Train on one expertise group, test on the other. Reuses the all-participant
    stepwin features for this interval (run mode "all" first), split by `expertise:`."""
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]
    name = stepwin_name(cfg)
    src = processed / f"help_vs_auto_features_{name}_auto{interval}.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}; run stepwin with mode 'all' first.")
    features_df = pd.read_csv(src)
    features_df = features_df[~features_df["subject"].isin(hva_cfg.get("exclude_participants", []))]

    expertise = cfg.get("expertise", {})
    group_of = features_df["subject"].map(lambda s: expertise.get(s, "unknown"))
    train_subj = sorted(features_df.loc[group_of == train_group, "subject"].unique())
    test_subj = sorted(features_df.loc[group_of == test_group, "subject"].unique())
    suffix = f"{name}_{train_group}_to_{test_group}_auto{interval}"

    print(f"\n===== train {train_group} -> test {test_group}: auto help every {interval} actions =====")
    print(f"train {len(train_subj)} subjects: {', '.join(train_subj)}")
    print(f"test  {len(test_subj)} subjects: {', '.join(test_subj)}")

    counts = event_counts(cfg, features_df[features_df["subject"].isin(test_subj)])
    counts.insert(0, "auto_interval", interval)
    counts.to_csv(processed / f"help_vs_auto_event_counts_{suffix}.csv", index=False)

    results, shap_df, cm = run_classifiers(
        cfg, features_df, predict=cross_group_predictor(expertise, train_group, test_group),
    )
    results = results.rename(columns={"window_s": "window_steps"})
    results.insert(0, "auto_interval", interval)
    results.insert(1, "train_group", train_group)
    results.insert(2, "test_group", test_group)
    results["n_per_class_after_balancing"] = results["n_test_after_balancing"] / 2
    cm = cm.rename(columns={"window_s": "window_steps"})
    shap_df = shap_df.rename(columns={"window_s": "window_steps"}) if not shap_df.empty else shap_df

    results.to_csv(processed / f"help_vs_auto_classifier_results_{suffix}.csv", index=False)
    cm.to_csv(processed / f"help_vs_auto_confusion_matrix_{suffix}.csv", index=False)
    shap_df.to_csv(processed / f"help_vs_auto_shap_values_{suffix}.csv", index=False)
    print(f"\nSaved outputs -> {processed} (*_{suffix}.csv)")
    return results


def run(cfg: dict, modes=("all",)):
    """modes: any of "all", "expert", "novice" (LOSO within that group) and
    "expert_to_novice", "novice_to_expert" (train on one group, test on the other).
    Each mode saves under its own file tag, e.g. *_stepwin_expert_auto20.csv."""
    hva_cfg = cfg["help_vs_auto"]
    intervals = hva_cfg.get("stepwin_auto_intervals", [hva_cfg["auto_interval_steps"]])
    processed = Path(cfg["paths"]["processed"])
    cols = ["auto_interval", "window_steps", "n_per_class_after_balancing",
            "accuracy_rf_mean", "accuracy_rf_subj_mean", "auc_rf_mean", "auc_rf_subj_mean"]

    for mode in modes:
        if mode in WITHIN_MODES:
            group = WITHIN_MODES[mode]
            all_results = pd.concat([run_one_interval(cfg, iv, group) for iv in intervals], ignore_index=True)
            tag = f"{stepwin_name(cfg)}_{group}" if group else stepwin_name(cfg)
        elif mode in CROSS_MODES:
            train_group, test_group = CROSS_MODES[mode]
            all_results = pd.concat(
                [run_cross_interval(cfg, iv, train_group, test_group) for iv in intervals], ignore_index=True,
            )
            tag = f"{stepwin_name(cfg)}_{mode}"
        else:
            raise ValueError(f"Unknown stepwin mode {mode!r}; use one of {[*WITHIN_MODES, *CROSS_MODES]}")
        all_results.to_csv(processed / f"help_vs_auto_classifier_results_{tag}_all_intervals.csv", index=False)

        print(f"\n=== Accuracy by auto interval ({mode}) ===")
        print(all_results[cols].round(3).to_string(index=False))
