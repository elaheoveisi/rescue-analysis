

from __future__ import annotations

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from features.help_vs_auto_features import build_events, build_features

warnings.filterwarnings("ignore", category=RuntimeWarning)


def balanced_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    n_min = df["label"].value_counts().min()
    rng = np.random.RandomState(seed)
    return pd.concat(
        [g.sample(n=n_min, random_state=rng) for _, g in df.groupby("label")],
        ignore_index=True,
    )


def window_rows(df: pd.DataFrame, w, features: list[str]) -> pd.DataFrame:
    return df[df["window"] == w].dropna(subset=features).copy()


def make_rf(hva_cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=hva_cfg.get("rf_n_estimators", 300),
        max_depth=hva_cfg.get("rf_max_depth", 4),
        min_samples_leaf=hva_cfg.get("rf_min_samples_leaf", 5),
        random_state=hva_cfg.get("rf_random_state", 0),
        n_jobs=-1,
    )


def loso_predictions(
    base: pd.DataFrame, features: list[str], hva_cfg: dict, seed: int
) -> dict[str, tuple[list, list, np.ndarray]]:
    """One leave-one-subject-out pass with the Random Forest.

    For each held-out subject: balance using only the remaining (training)
    subjects, then test on the held-out subject's full, untouched data --
    their rows are never balanced away. SHAP values for the held-out rows are
    computed from that same fitted model, so they explain exactly the
    predictions returned alongside them.
    """
    y_true = {"rf": []}
    y_pred = {"rf": []}
    shap_rows = {"rf": []}
    for s in base["subject"].unique():
        train_raw, test_raw = base[base["subject"] != s], base[base["subject"] == s]
        if test_raw.empty:
            continue
        train_bal = balanced_sample(train_raw, seed)
        if train_bal["label"].nunique() < 2:
            continue

        rf = make_rf(hva_cfg)
        rf.fit(train_bal[features], train_bal["label"])
        p = rf.predict_proba(test_raw[features])[:, 1]
        y_true["rf"].extend(test_raw["label"].tolist())
        y_pred["rf"].extend(p.tolist())

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(test_raw[features])
        if isinstance(sv, list):
            sv = sv[1]  # positive-class (label=1) SHAP values
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_rows["rf"].append(np.abs(sv))

    shap_matrix = np.vstack(shap_rows["rf"]) if shap_rows["rf"] else np.empty((0, len(features)))
    return {"rf": (y_true["rf"], y_pred["rf"], shap_matrix)}


def score(y_true: list, y_pred: list, threshold: float) -> dict[str, float | None]:
    if len(set(y_true)) < 2:
        return {"auc": None, "accuracy": None, "precision": None, "recall": None, "f1": None}
    pred = (np.array(y_pred) >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def run_classifiers(cfg: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pooled scores/SHAP values are per held-out decision, not per subject --
    subjects who contribute more decisions get proportionally more weight in
    each repeat's numbers. That matches "decision" as the modeling unit, but
    should be stated in the methods write-up.
    """
    hva_cfg = cfg["help_vs_auto"]
    features = hva_cfg["features"]
    n_repeats = hva_cfg.get("n_balance_repeats", 10)
    seed0 = hva_cfg.get("random_state", 42)
    threshold = hva_cfg.get("classification_threshold", 0.5)

    results = []
    cm_rows = []
    shap_rows = []
    for w in sorted(df["window"].unique()):
        base = window_rows(df, w, features)
        if base.empty:
            continue

        scores = {"rf": []}
        cms = {"rf": np.zeros((2, 2), dtype=int)}
        shap_pool = {"rf": []}
        for r in range(n_repeats):
            preds = loso_predictions(base, features, hva_cfg, seed0 + r)
            for model_name, (yt, yp, sv) in preds.items():
                scores[model_name].append(score(yt, yp, threshold))
                if len(set(yt)) >= 2:
                    pred = (np.array(yp) >= threshold).astype(int)
                    cms[model_name] += confusion_matrix(yt, pred, labels=[0, 1])
                if sv.size:
                    # One mean-abs-SHAP vector per repeat, not per row -- so the SD below
                    # reflects repeat-to-repeat stability, matching how accuracy_*_sd is computed.
                    shap_pool[model_name].append(sv.mean(axis=0))

        if shap_pool["rf"]:
            repeat_means = np.vstack(shap_pool["rf"])
            shap_rows.append(pd.DataFrame({
                "window_s": w, "feature": features,
                "mean_abs_shap": repeat_means.mean(axis=0),
                "mean_abs_shap_sd": repeat_means.std(axis=0, ddof=1) if len(repeat_means) > 1 else 0.0,
            }))

        # Confusion-matrix counts (below) are summed across all n_repeats repeats, so
        # tn/fp/fn/tp are ~n_repeats x the number of held-out decisions, not per-repeat counts.
        # recall_alt/precision_alt/specificity_auto are ratios, so they're unaffected by the scaling.
        row = {"window_s": w, "minority_class_n_full_data": base["label"].value_counts().min()}
        for model_name, sc in scores.items():
            for metric in ("auc", "accuracy", "precision", "recall", "f1"):
                vals = [d[metric] for d in sc if d[metric] is not None]
                row[f"{metric}_{model_name}_mean"] = np.mean(vals) if vals else None
                row[f"{metric}_{model_name}_sd"] = np.std(vals, ddof=1) if len(vals) > 1 else (0.0 if vals else None)
        results.append(row)
        rf_accuracy = row["accuracy_rf_mean"]
        rf_str = f"{rf_accuracy:.3f}" if rf_accuracy is not None else "n/a"
        print(f"window={w}s  RF accuracy={rf_str}")

        for model_name, cm in cms.items():
            tn, fp, fn, tp = (int(v) for v in cm.ravel())
            cm_rows.append({
                "window_s": w, "model": model_name, "n_repeats": n_repeats,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "recall_alt": tp / (tp + fn) if (tp + fn) else None,
                "precision_alt": tp / (tp + fp) if (tp + fp) else None,
                "specificity_auto": tn / (tn + fp) if (tn + fp) else None,
            })
        print(f"window={w}s  confusion matrix computed (RF)")

    shap_df = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    return pd.DataFrame(results), shap_df, pd.DataFrame(cm_rows)


def build_features_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build events + eye features and save them to CSV. Independent of the
    models below -- rerun this only when the raw data, AOI labels, or feature
    computation itself changes, not every time you want to retrain."""
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]

    events_df = build_events(cfg)
    events_out = processed / hva_cfg.get("events_file", "help_vs_auto_events.csv")
    events_df.to_csv(events_out, index=False)
    print(f"Saved {len(events_df)} rows -> {events_out}")

    features_df = events_df.merge(
        build_features(cfg, events_df), on=["subject", "trial", "run", "step", "label", "window"], how="left"
    )
    features_out = processed / hva_cfg.get("features_file", "help_vs_auto_features.csv")
    features_df.to_csv(features_out, index=False)
    print(f"Saved {len(features_df)} rows -> {features_out}")

    return events_df, features_df


def run_models(cfg: dict, features_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random Forest classification, SHAP values, and confusion matrices.
    Reads the saved features CSV if features_df isn't passed in-memory --
    lets this run on its own (e.g. to try a different feature list) without
    recomputing eye features via build_features_dataset() first."""
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]

    if features_df is None:
        features_out = processed / hva_cfg.get("features_file", "help_vs_auto_features.csv")
        features_df = pd.read_csv(features_out)

    results, shap_df, cm = run_classifiers(cfg, features_df)
    results_out = processed / hva_cfg.get("results_file", "help_vs_auto_classifier_results.csv")
    results.to_csv(results_out, index=False)
    print(f"Saved -> {results_out}")

    cm_out = processed / hva_cfg.get("confusion_matrix_file", "help_vs_auto_confusion_matrix.csv")
    cm.to_csv(cm_out, index=False)
    print(f"Saved -> {cm_out}")

    shap_out = processed / hva_cfg.get("shap_file", "help_vs_auto_shap_values.csv")
    shap_df.to_csv(shap_out, index=False)
    print(f"Saved -> {shap_out}")

    return results, shap_df, cm


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: build_features_dataset() + run_models(), in one call --
    kept for backward compatibility. Prefer calling the two separately (e.g.
    from main.py) when you only need to rerun one half."""
    events_df, features_df = build_features_dataset(cfg)
    results, shap_df, cm = run_models(cfg, features_df)
    return events_df, features_df, results, shap_df, cm
