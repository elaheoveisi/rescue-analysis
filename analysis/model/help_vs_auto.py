"""Classify manual Alt-press-for-help (label=1) vs automatic step-interval LLM
recommendation (label=0) events from pre-event eye-tracking/gaze/pupil features.

Event timeline and feature extraction live in
features/help_vs_auto_features.py -- this module only fits and evaluates
GLMM + Random Forest classifiers on the resulting per-event, per-window features.

Each model is evaluated by leave-one-participant-out cross-validation (train on
all but one participant, test on the held-out one) so reported performance
reflects generalization to a new person, not memorization. Classes are balanced
by downsampling the majority class before each fit -- averaged over
`n_balance_repeats` random draws, since a single draw is noisy at this sample size.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))

from features.help_vs_auto_features import build_events, build_features

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _balanced_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    n_min = df["label"].value_counts().min()
    rng = np.random.RandomState(seed)
    return pd.concat(
        [g.sample(n=n_min, random_state=rng) for _, g in df.groupby("label")],
        ignore_index=True,
    )


def _loso_glmm(bal: pd.DataFrame, zcols: list[str]) -> tuple[list, list]:
    formula = "label ~ " + " + ".join(zcols)
    y_true, y_pred = [], []
    for s in bal["subject_c"].unique():
        train, test = bal[bal["subject_c"] != s], bal[bal["subject_c"] == s]
        if train["label"].nunique() < 2 or test.empty:
            continue
        try:
            model = BinomialBayesMixedGLM.from_formula(formula, {"subject": "0 + C(subject_c)"}, train)
            fit = model.fit_vb()
            test_exog = np.column_stack([np.ones(len(test))] + [test[c].to_numpy() for c in zcols])
            p = fit.predict(exog=test_exog)
            y_true.extend(test["label"].tolist())
            y_pred.extend(np.atleast_1d(p).tolist())
        except Exception:
            continue
    return y_true, y_pred


def _loso_sklearn(bal: pd.DataFrame, zcols: list[str], make_model) -> tuple[list, list]:
    y_true, y_pred = [], []
    for s in bal["subject_c"].unique():
        train, test = bal[bal["subject_c"] != s], bal[bal["subject_c"] == s]
        if train["label"].nunique() < 2 or test.empty:
            continue
        model = make_model()
        model.fit(train[zcols], train["label"])
        p = model.predict_proba(test[zcols])[:, 1]
        y_true.extend(test["label"].tolist())
        y_pred.extend(p.tolist())
    return y_true, y_pred


def _score(y_true: list, y_pred: list, threshold: float) -> tuple[float | None, float | None]:
    if len(set(y_true)) < 2:
        return None, None
    pred = (np.array(y_pred) > threshold).astype(int)
    return roc_auc_score(y_true, y_pred), accuracy_score(y_true, pred)


def _make_rf(hva_cfg: dict, n_estimators_key: str) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=hva_cfg.get(n_estimators_key, 300),
        max_depth=hva_cfg.get("rf_max_depth", 4),
        min_samples_leaf=hva_cfg.get("rf_min_samples_leaf", 5),
        random_state=0,
    )


def _prep_window(df: pd.DataFrame, w, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One window's rows, globally z-scored per feature, with a categorical
    subject column for grouping -- shared prep step for run_classifiers,
    feature_importance, and confusion_matrices."""
    base = df[df["window"] == w].dropna(subset=features).copy()
    if base.empty:
        return base, []
    base["subject_c"] = base["subject"].astype("category")
    zcols = [f + "_z" for f in features]
    for f, z in zip(features, zcols):
        sd = base[f].std()
        base[z] = (base[f] - base[f].mean()) / sd if sd > 0 else 0.0
    return base, zcols


def run_classifiers(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    hva_cfg = cfg["help_vs_auto"]
    features = hva_cfg["features"]
    n_repeats = hva_cfg.get("n_balance_repeats", 10)
    seed0 = hva_cfg.get("random_state", 42)
    threshold = hva_cfg.get("classification_threshold", 0.5)

    results = []
    for w in sorted(df["window"].unique()):
        base, zcols = _prep_window(df, w, features)
        if base.empty:
            continue

        scores = {"glmm": [], "rf": []}
        for r in range(n_repeats):
            bal = _balanced_sample(base, seed0 + r)

            scores["glmm"].append(_score(*_loso_glmm(bal, zcols), threshold))
            scores["rf"].append(_score(*_loso_sklearn(
                bal, zcols, lambda: _make_rf(hva_cfg, "rf_n_estimators"),
            ), threshold))

        row = {"window_s": w, "n_per_class": base["label"].value_counts().min()}
        for model_name, sc in scores.items():
            aucs = [a for a, _ in sc if a is not None]
            accs = [c for _, c in sc if c is not None]
            row[f"auc_{model_name}_mean"] = np.mean(aucs) if aucs else None
            row[f"auc_{model_name}_sd"] = np.std(aucs) if aucs else None
            row[f"acc_{model_name}_mean"] = np.mean(accs) if accs else None
        results.append(row)
        print(f"window={w}s  GLMM AUC={row['auc_glmm_mean']:.3f}  RF AUC={row['auc_rf_mean']:.3f}")

    return pd.DataFrame(results)


def confusion_matrices(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Confusion matrix (summed over the n_balance_repeats draws) for GLMM and
    RF at every window: tn/fp/fn/tp plus recall/precision on the Alt class and
    specificity on the Automatic class."""
    hva_cfg = cfg["help_vs_auto"]
    features = hva_cfg["features"]
    n_repeats = hva_cfg.get("n_balance_repeats", 10)
    seed0 = hva_cfg.get("random_state", 42)
    threshold = hva_cfg.get("classification_threshold", 0.5)

    rows = []
    for w in sorted(df["window"].unique()):
        base, zcols = _prep_window(df, w, features)
        if base.empty:
            continue

        cms = {"glmm": np.zeros((2, 2), dtype=int), "rf": np.zeros((2, 2), dtype=int)}
        for r in range(n_repeats):
            bal = _balanced_sample(base, seed0 + r)

            yt, yp = _loso_glmm(bal, zcols)
            pred = (np.array(yp) > threshold).astype(int)
            cms["glmm"] += confusion_matrix(yt, pred, labels=[0, 1])

            yt, yp = _loso_sklearn(bal, zcols, lambda: _make_rf(hva_cfg, "rf_n_estimators"))
            pred = (np.array(yp) > threshold).astype(int)
            cms["rf"] += confusion_matrix(yt, pred, labels=[0, 1])

        for model_name, cm in cms.items():
            tn, fp, fn, tp = (int(v) for v in cm.ravel())
            rows.append({
                "window_s": w, "model": model_name,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "recall_alt": tp / (tp + fn) if (tp + fn) else None,
                "precision_alt": tp / (tp + fp) if (tp + fp) else None,
                "specificity_auto": tn / (tn + fp) if (tn + fp) else None,
            })
        print(f"window={w}s  confusion matrices computed (GLMM, RF)")

    return pd.DataFrame(rows)


def feature_importance(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Permutation importance (drop in AUC when a feature is shuffled), from a
    Random Forest fit on the balanced data per window."""
    hva_cfg = cfg["help_vs_auto"]
    features = hva_cfg["features"]
    seed0 = hva_cfg.get("random_state", 42)

    rows = []
    for w in sorted(df["window"].unique()):
        sub = df[df["window"] == w].dropna(subset=features).copy()
        if sub.empty:
            continue
        bal = _balanced_sample(sub, seed0)
        X, y = bal[features], bal["label"]
        rf = _make_rf(hva_cfg, "rf_importance_n_estimators")
        rf.fit(X, y)
        perm = permutation_importance(rf, X, y, n_repeats=20, random_state=0, scoring="roc_auc")
        rows.append(pd.DataFrame({
            "window_s": w, "feature": features,
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_sd": perm.importances_std,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed = ROOT / cfg["paths"]["processed"]
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

    results = run_classifiers(cfg, features_df)
    results_out = processed / hva_cfg.get("results_file", "help_vs_auto_classifier_results.csv")
    results.to_csv(results_out, index=False)
    print(f"Saved -> {results_out}")

    importance = feature_importance(cfg, features_df)
    importance_out = processed / hva_cfg.get("importance_file", "help_vs_auto_feature_importance.csv")
    importance.to_csv(importance_out, index=False)
    print(f"Saved -> {importance_out}")

    cm = confusion_matrices(cfg, features_df)
    cm_out = processed / hva_cfg.get("confusion_matrix_file", "help_vs_auto_confusion_matrix.csv")
    cm.to_csv(cm_out, index=False)
    print(f"Saved -> {cm_out}")

    return events_df, features_df, results, importance, cm


if __name__ == "__main__":
    with open(ROOT / "configs" / "analysis.yml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
