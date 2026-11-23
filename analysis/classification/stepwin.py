from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from classification.classifier import (
    balanced_sample,
    make_rf,
    run_classifiers,
    window_rows,
)
from classification.features import (
    build_events_core,
    build_features_core,
)
from features.game_steps import EVENT_KEYS

# Modes for run(): within-group LOSO, or train on one group and test on the other.
WITHIN_MODES = {"all": None, "expert": "expert", "novice": "novice"}
CROSS_MODES = {
    "expert_to_novice": ("expert", "novice"),
    "novice_to_expert": ("novice", "expert"),
}


def timestamp_at_step(step_timestamps: pd.Series, target_step: float) -> float:
    """An unobserved action has no timestamp; do not clamp/interpolate it."""
    return step_timestamps.get(target_step, np.nan)


def build_events_steps(cfg: dict) -> pd.DataFrame:
    step_windows = cfg["info_request"]["step_windows"]
    lead = cfg["info_request"].get("stepwin_lead_steps", 0)

    def compute_window_start_ms(
        step_timestamps,
        eye_start_ts,
        _event_ts_ms,
        event_step,
        window_steps,
        label=None,
    ):
        shift = (
            lead if label == 1 else 0
        )  # only Alt presses are shifted; auto fires on an action
        window_start_ts = timestamp_at_step(
            step_timestamps, event_step - window_steps - shift
        )
        return (window_start_ts - eye_start_ts) * 1000.0

    return build_events_core(cfg, step_windows, compute_window_start_ms)


def build_features_steps(cfg: dict, events_df: pd.DataFrame) -> pd.DataFrame:
    lead = cfg["info_request"].get("stepwin_lead_steps", 0)
    at_action = (
        cfg["info_request"].get("stepwin_end") == "action"
    )  # end Alt windows at the last action, not the press

    def compute_window_bounds(
        step_timestamps, eye_start_ts, step, window_steps, event_ts, label=None
    ):
        """Alt (label 1) with lead=L: window = [ts(step-k-L), ts(step-L)], i.e. k actions of gaze
        ending L actions BEFORE the press, so nothing from the press moment is used.
        Auto (label 0) is unchanged: [ts(step-k), event], since it fires on an action
        timestamp (no gap), a fixed N actions after the previous help."""
        shift = lead if label == 1 else 0
        if event_ts is None or pd.isna(event_ts):
            return None
        end_ts = (
            timestamp_at_step(step_timestamps, step - shift)
            if shift or at_action
            else event_ts
        )
        end_ms = (end_ts - eye_start_ts) * 1000.0
        window_start_ts = timestamp_at_step(
            step_timestamps, step - window_steps - shift
        )
        window_start_ms = (window_start_ts - eye_start_ts) * 1000.0
        window_duration_s = (end_ms - window_start_ms) / 1000.0
        if not np.isfinite(window_duration_s) or window_duration_s <= 0:
            return None
        return window_start_ms, end_ms, window_duration_s

    return build_features_core(cfg, events_df, compute_window_bounds)


def event_counts(cfg: dict, features_df: pd.DataFrame) -> pd.DataFrame:
    """Request/non-request counts per subject and window: kept by the model (after the
    NaN rule in window_rows) and after LOSO test balancing (each held-out
    subject downsampled to their own minority class; subjects missing a class
    are not tested, so they get 0). Includes a "Total" row per window."""
    hva_cfg = cfg["info_request"]
    features = hva_cfg["features"]
    zero_fill = hva_cfg.get("zero_fill_features", [])
    rows = []
    for w in sorted(features_df["window"].unique()):
        base = window_rows(features_df, w, features, zero_fill)
        c = (
            base.groupby(["subject", "label"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
        )
        bal = np.where((c[0] > 0) & (c[1] > 0), c.min(axis=1), 0)
        per_subj = pd.DataFrame(
            {
                "window_steps": w,
                "subject": c.index,
                "request": c[1].to_numpy(),
                "non_request": c[0].to_numpy(),
                "request_balanced": bal,
                "non_request_balanced": bal,
            }
        )
        total = per_subj.drop(columns=["window_steps", "subject"]).sum().to_dict()
        rows.append(
            pd.concat(
                [
                    per_subj,
                    pd.DataFrame([{"window_steps": w, "subject": "Total", **total}]),
                ]
            )
        )
    return pd.concat(rows, ignore_index=True)


def print_counts(counts: pd.DataFrame, interval: int):
    """Subjects x windows, each cell "request/non_request"."""
    for label, m, a in [
        ("kept", "request", "non_request"),
        ("after balancing", "request_balanced", "non_request_balanced"),
    ]:
        cell = counts.assign(
            cell=counts[m].astype(int).astype(str)
            + "/"
            + counts[a].astype(int).astype(str)
        )
        table = cell.pivot(index="subject", columns="window_steps", values="cell")
        order = [s for s in table.index if s != "Total"] + ["Total"]
        print(
            f"\n=== request/non_request per subject, {label} (auto every {interval} actions) ==="
        )
        print(table.loc[order].to_string())


def stepwin_name(cfg: dict) -> str:
    """File-name prefix: "stepwin", or "stepwin_<stepwin_tag>" when the config sets a tag,
    so a rerun (e.g. a different feature set) doesn't overwrite earlier outputs."""
    tag = cfg["info_request"].get("stepwin_tag")
    return f"stepwin_{tag}" if tag else "stepwin"


def build_interval(cfg: dict, suffix: str, events_fn, features_fn) -> pd.DataFrame:
    """Build and save events + features for one auto interval."""
    events_df = events_fn(cfg)
    features_df = events_df.merge(
        features_fn(cfg, events_df), on=EVENT_KEYS, how="left"
    )
    for name, df in (("events", events_df), ("features", features_df)):
        df.to_csv(
            Path(cfg["paths"]["processed"]) / f"info_request_{name}_{suffix}.csv",
            index=False,
        )
    return features_df


def classify_interval(
    cfg,
    interval,
    suffix,
    features_df,
    win_col="window_steps",
    front=None,
    equiv=None,
    predict=None,
    counts_df=None,
):
    """Event counts, classifiers and CSV outputs shared by the stepwin, cross-group and meanwin runs.
    `front` adds leading columns, `equiv` ({window: k}) adds the step-equivalent column after the window."""
    processed, front = (
        Path(cfg["paths"]["processed"]),
        {"auto_interval": interval, **(front or {})},
    )

    def tag(df):
        df = df.rename(columns={"window_steps": win_col, "window_s": win_col})
        for i, (k, v) in enumerate(front.items()):
            df.insert(i, k, v)
        if equiv:
            df.insert(len(front) + 1, "window_steps_equiv", df[win_col].map(equiv))
        return df

    counts = event_counts(cfg, features_df if counts_df is None else counts_df)
    print_counts(counts, interval)
    tag(counts).to_csv(
        processed / f"info_request_event_counts_{suffix}.csv", index=False
    )
    results, shap_df, cm = run_classifiers(cfg, features_df, predict=predict)
    results = tag(results)
    results["n_per_class_after_balancing"] = results["n_test_after_balancing"] / 2
    for name, df in (
        ("classifier_results", results),
        ("confusion_matrix", cm),
        ("shap_values", shap_df),
    ):
        df.rename(columns={"window_s": win_col}).to_csv(
            processed / f"info_request_{name}_{suffix}.csv", index=False
        )
    print(f"\n=== Results ({suffix}) ===\n{results.to_string(index=False)}")
    return results


def run_one_interval(
    cfg: dict, interval: int, group: str | None = None
) -> pd.DataFrame:
    """Full stepwin pipeline with automatic help every `interval` actions; `group`
    ("expert"/"novice") keeps only that expertise group and tags the outputs."""
    cfg = copy.deepcopy(cfg)
    cfg["info_request"].update(auto_interval_steps=interval, expertise_filter=group)
    suffix = "_".join(filter(None, [stepwin_name(cfg), group, f"auto{interval}"]))
    features_df = build_interval(cfg, suffix, build_events_steps, build_features_steps)
    return classify_interval(cfg, interval, suffix, features_df)


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
        shap_matrix = (
            np.vstack(shap_rows) if shap_rows else np.empty((0, len(features)))
        )
        x_matrix = np.vstack(x_rows) if x_rows else np.empty((0, len(features)))
        return {"rf": (y_true, y_pred, shap_matrix, x_matrix, per_subject)}

    return predict


def run_cross_interval(
    cfg: dict, interval: int, train_group: str, test_group: str
) -> pd.DataFrame:
    """Train on one expertise group, test on the other. Reuses the all-participant
    stepwin features for this interval (run mode "all" first), split by `expertise:`."""
    hva_cfg, name = cfg["info_request"], stepwin_name(cfg)
    src = (
        Path(cfg["paths"]["processed"])
        / f"info_request_features_{name}_auto{interval}.csv"
    )
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}; run stepwin with mode 'all' first.")
    features_df = pd.read_csv(src)
    features_df = features_df[
        ~features_df["subject"].isin(hva_cfg.get("exclude_participants", []))
    ]
    expertise = cfg.get("expertise", {})
    group_of = features_df["subject"].map(lambda s: expertise.get(s, "unknown"))
    test_df = features_df[group_of == test_group]
    print(
        f"\n===== train {train_group} -> test {test_group}: auto help every {interval} actions ====="
    )
    return classify_interval(
        cfg,
        interval,
        f"{name}_{train_group}_to_{test_group}_auto{interval}",
        features_df,
        front={"train_group": train_group, "test_group": test_group},
        predict=cross_group_predictor(expertise, train_group, test_group),
        counts_df=test_df,
    )


def run(cfg: dict, modes=("all",)):
    """modes: any of "all", "expert", "novice" (LOSO within that group) and
    "expert_to_novice", "novice_to_expert" (train on one group, test on the other).
    Each mode saves under its own file tag, e.g. *_stepwin_expert_auto20.csv."""
    hva_cfg = cfg["info_request"]
    intervals = hva_cfg.get("stepwin_auto_intervals", [hva_cfg["auto_interval_steps"]])
    processed = Path(cfg["paths"]["processed"])
    cols = [
        "auto_interval",
        "window_steps",
        "n_per_class_after_balancing",
        "accuracy_rf_mean",
        "accuracy_rf_subj_mean",
        "auc_rf_mean",
        "auc_rf_subj_mean",
    ]

    for mode in modes:
        if mode in WITHIN_MODES:
            group = WITHIN_MODES[mode]
            all_results = pd.concat(
                [run_one_interval(cfg, iv, group) for iv in intervals],
                ignore_index=True,
            )
            tag = f"{stepwin_name(cfg)}_{group}" if group else stepwin_name(cfg)
        elif mode in CROSS_MODES:
            train_group, test_group = CROSS_MODES[mode]
            all_results = pd.concat(
                [
                    run_cross_interval(cfg, iv, train_group, test_group)
                    for iv in intervals
                ],
                ignore_index=True,
            )
            tag = f"{stepwin_name(cfg)}_{mode}"
        else:
            raise ValueError(
                f"Unknown stepwin mode {mode!r}; use one of {[*WITHIN_MODES, *CROSS_MODES]}"
            )
        all_results.to_csv(
            processed / f"info_request_classifier_results_{tag}_all_intervals.csv",
            index=False,
        )

        print(f"\n=== Accuracy by auto interval ({mode}) ===")
        print(all_results[cols].round(3).to_string(index=False))
