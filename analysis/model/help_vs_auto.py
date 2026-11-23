from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
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
warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning
)

METRICS = ("auc", "accuracy", "precision", "recall", "f1")


# ---------- data ----------

def balanced_sample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Downsample every label to the size of the smallest one."""
    n_min = df["label"].value_counts().min()
    rng = np.random.RandomState(seed)
    return pd.concat(
        [g.sample(n=n_min, random_state=rng) for _, g in df.groupby("label")],
        ignore_index=True,
    )


def window_rows(df: pd.DataFrame, w, features: list[str], zero_fill: list[str]) -> pd.DataFrame:
    """Rows of one window, with zero_fill features NaN -> 0 and any other NaN row dropped."""
    rows = df[df["window"] == w].copy()
    fill_cols = [f for f in zero_fill if f in features]
    rows[fill_cols] = rows[fill_cols].fillna(0)
    return rows.dropna(subset=features)


# ---------- model ----------

def make_rf(hva_cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=hva_cfg.get("rf_n_estimators", 300),
        max_depth=hva_cfg.get("rf_max_depth", 4),
        min_samples_leaf=hva_cfg.get("rf_min_samples_leaf", 5),
        random_state=hva_cfg.get("rf_random_state", 0),
        n_jobs=-1,
    )


def fit_rf(train_raw: pd.DataFrame, features: list[str], hva_cfg: dict, seed: int) -> RandomForestClassifier:
    """Balance the training rows, then fit one Random Forest."""
    train = balanced_sample(train_raw, seed)
    rf = make_rf(hva_cfg)
    rf.fit(train[features], train["label"])
    return rf


def positive_class_shap(rf: RandomForestClassifier, x: pd.DataFrame) -> np.ndarray:
    """SHAP values toward label 1 (Alt), shape (rows, features).
    Older shap returns a list per class, newer a 3-D array."""
    sv = shap.TreeExplainer(rf).shap_values(x)
    if isinstance(sv, list):
        return sv[1]
    return sv[:, :, 1] if sv.ndim == 3 else sv


def test_subject(rf: RandomForestClassifier, subject: str, test_raw: pd.DataFrame,
                 features: list[str], seed: int) -> dict:
    """Test one subject on their own balanced rows. Returns one "fold":
    true labels, predicted P(Alt), SHAP values and feature values of those rows."""
    test = balanced_sample(test_raw, seed)
    x = test[features]
    return {
        "subject": subject,
        "y_true": test["label"].tolist(),
        "y_pred": rf.predict_proba(x)[:, 1].tolist(),
        "shap": positive_class_shap(rf, x),
        "x": x.to_numpy(dtype=float),
    }


def loso_predictions(base: pd.DataFrame, features: list[str], hva_cfg: dict, seed: int) -> list[dict]:
    """Leave-one-subject-out: for each subject, train on everyone else and
    test on that subject. Subjects missing a label are skipped."""
    folds = []
    for s in base["subject"].unique():
        train_raw, test_raw = base[base["subject"] != s], base[base["subject"] == s]
        if test_raw["label"].nunique() < 2 or train_raw["label"].nunique() < 2:
            continue
        rf = fit_rf(train_raw, features, hva_cfg, seed)
        folds.append(test_subject(rf, s, test_raw, features, seed))
    return folds


# ---------- scoring ----------

def score(y_true: list, y_pred: list, threshold: float) -> dict[str, float | None]:
    if len(set(y_true)) < 2:
        return dict.fromkeys(METRICS)
    pred = (np.array(y_pred) >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def mean_sd(values) -> tuple[float | None, float | None]:
    """Mean and SD, ignoring None. SD is 0 for one value, None for no values."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)


def pooled(folds: list[dict], key: str) -> list:
    """One repeat's y_true or y_pred, all held-out subjects joined together."""
    return [v for f in folds for v in f[key]]


def subject_scores(repeats: list[list[dict]], threshold: float) -> dict[str, list[dict]]:
    """subject -> one score dict per repeat that tested them."""
    out = {}
    for folds in repeats:
        for f in folds:
            out.setdefault(f["subject"], []).append(score(f["y_true"], f["y_pred"], threshold))
    return out


def score_row(w, base: pd.DataFrame, repeats: list[list[dict]], threshold: float) -> dict:
    """One results row for a window.
    *_rf_mean/_sd: pooled over all held-out decisions, mean/SD across repeats.
    *_rf_subj_mean/_sd: each subject averaged over repeats, then mean/SD across
    subjects (each subject weighted equally)."""
    pooled_scores = [score(pooled(f, "y_true"), pooled(f, "y_pred"), threshold) for f in repeats]
    per_subject = subject_scores(repeats, threshold)
    row = {
        "window_s": w,
        "minority_class_n_full_data": base["label"].value_counts().min(),
        # Same in every repeat (each subject's minority count x2); only which rows are sampled changes.
        "n_test_after_balancing": len(pooled(repeats[0], "y_true")),
    }
    for m in METRICS:
        row[f"{m}_rf_mean"], row[f"{m}_rf_sd"] = mean_sd(s[m] for s in pooled_scores)
    row["n_subjects_tested_rf"] = len(per_subject)
    for m in METRICS:
        subject_means = [mean_sd(s[m] for s in scores)[0] for scores in per_subject.values()]
        row[f"{m}_rf_subj_mean"], row[f"{m}_rf_subj_sd"] = mean_sd(subject_means)
    return row


def shap_table(w, features: list[str], repeats: list[list[dict]]) -> pd.DataFrame | None:
    """Per feature: mean |SHAP|, its % of the total, and direction, each as
    mean/SD across repeats. Direction = Spearman r between a feature's value and
    its SHAP value over held-out rows (> 0: higher values push toward Alt)."""
    abs_shap, direction = [], []
    for folds in repeats:
        if not folds:
            continue
        sv = np.vstack([f["shap"] for f in folds])
        x = np.vstack([f["x"] for f in folds])
        abs_shap.append(np.abs(sv).mean(axis=0))
        direction.append([spearmanr(x[:, i], sv[:, i]).correlation for i in range(len(features))])
    if not abs_shap:
        return None

    abs_shap = np.vstack(abs_shap)
    pct = 100 * abs_shap / abs_shap.sum(axis=1, keepdims=True)
    direction = np.array(direction, dtype=float)
    return pd.DataFrame({
        "window_s": w, "feature": features,
        "mean_abs_shap": abs_shap.mean(axis=0), "mean_abs_shap_sd": sd_across_repeats(abs_shap),
        "shap_pct": pct.mean(axis=0), "shap_pct_sd": sd_across_repeats(pct),
        "direction_r": np.nanmean(direction, axis=0), "direction_r_sd": sd_across_repeats(direction, np.nanstd),
    })


def sd_across_repeats(a: np.ndarray, std=np.std):
    """Column SD over repeats (rows); 0 when there is only one repeat."""
    return std(a, axis=0, ddof=1) if len(a) > 1 else 0.0


def confusion_row(w, repeats: list[list[dict]], threshold: float) -> dict:
    """Counts summed over all repeats (so ~n_repeats x the held-out decisions);
    the ratios are unaffected by that scaling."""
    cm = np.zeros((2, 2), dtype=int)
    for folds in repeats:
        yt = pooled(folds, "y_true")
        if len(set(yt)) == 2:
            pred = (np.array(pooled(folds, "y_pred")) >= threshold).astype(int)
            cm += confusion_matrix(yt, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "window_s": w, "model": "rf", "n_repeats": len(repeats),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall_alt": tp / (tp + fn) if (tp + fn) else None,
        "precision_alt": tp / (tp + fp) if (tp + fp) else None,
        "specificity_auto": tn / (tn + fp) if (tn + fp) else None,
    }


def fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "n/a"


def print_window(row: dict, repeats: list[list[dict]], threshold: float):
    print(f"window={row['window_s']}  RF accuracy={fmt(row['accuracy_rf_mean'])}")
    print("  per-subject accuracy (mean over repeats):")
    for s, scores in subject_scores(repeats, threshold).items():
        print(f"    subject={s}  accuracy={fmt(mean_sd(d['accuracy'] for d in scores)[0])}")
    if row["accuracy_rf_subj_mean"] is not None:
        print(f"  average across {row['n_subjects_tested_rf']} subjects: "
              f"{row['accuracy_rf_subj_mean']:.3f} (SD {row['accuracy_rf_subj_sd']:.3f})")


def run_classifiers(cfg: dict, df: pd.DataFrame, predict=loso_predictions) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """For each window: repeat `predict` n_balance_repeats times (new balancing
    seed each time), then return (results, shap, confusion matrix) tables.

    Pooled scores/SHAP are per held-out decision, so subjects with more
    decisions weigh more (worth stating in the methods write-up).
    `predict(base, features, hva_cfg, seed) -> list of folds` (see test_subject);
    e.g. help_vs_auto_stepwin.cross_group_predictor trains on one expertise group
    and tests on the other.
    """
    if "event_timestamp" not in df or df["event_timestamp"].isna().any():
        raise ValueError("Stale event features: rebuild using corrected request timestamps before training.")
    hva_cfg = cfg["help_vs_auto"]
    features = hva_cfg["features"]
    n_repeats = hva_cfg.get("n_balance_repeats", 10)
    seed0 = hva_cfg.get("random_state", 42)
    threshold = hva_cfg.get("classification_threshold", 0.5)

    results, shap_tables, cm_rows = [], [], []
    for w in sorted(df["window"].unique()):
        base = window_rows(df, w, features, hva_cfg.get("zero_fill_features", []))
        if base.empty:
            continue
        repeats = [predict(base, features, hva_cfg, seed0 + r) for r in range(n_repeats)]
        row = score_row(w, base, repeats, threshold)
        results.append(row)
        shap_tables.append(shap_table(w, features, repeats))
        cm_rows.append(confusion_row(w, repeats, threshold))
        print_window(row, repeats, threshold)

    shap_tables = [t for t in shap_tables if t is not None]
    shap_df = pd.concat(shap_tables, ignore_index=True) if shap_tables else pd.DataFrame()
    return pd.DataFrame(results), shap_df, pd.DataFrame(cm_rows)


# ---------- pipeline ----------

def build_features_dataset(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build events + eye features and save them to CSV. Rerun only when the raw
    data, AOI labels, or feature code change -- not to retrain."""
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]

    events_df = build_events(cfg)
    events_out = processed / hva_cfg.get("events_file", "help_vs_auto_events.csv")
    events_df.to_csv(events_out, index=False)
    print(f"Saved {len(events_df)} rows -> {events_out}")

    features_df = events_df.merge(
        build_features(cfg, events_df), on=["subject", "trial", "run", "step", "label", "window", "event_timestamp"], how="left"
    )
    features_out = processed / hva_cfg.get("features_file", "help_vs_auto_features.csv")
    features_df.to_csv(features_out, index=False)
    print(f"Saved {len(features_df)} rows -> {features_out}")

    return events_df, features_df


def run_models(cfg: dict, features_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random Forest, SHAP and confusion matrices. Reads the saved features CSV
    when features_df isn't given, so a new feature list doesn't need a rebuild."""
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["help_vs_auto"]

    if features_df is None:
        features_df = pd.read_csv(processed / hva_cfg.get("features_file", "help_vs_auto_features.csv"))

    results, shap_df, cm = run_classifiers(cfg, features_df)
    for table, key, default in [
        (results, "results_file", "help_vs_auto_classifier_results.csv"),
        (cm, "confusion_matrix_file", "help_vs_auto_confusion_matrix.csv"),
        (shap_df, "shap_file", "help_vs_auto_shap_values.csv"),
    ]:
        out = processed / hva_cfg.get(key, default)
        table.to_csv(out, index=False)
        print(f"Saved -> {out}")

    return results, shap_df, cm


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """build_features_dataset() + run_models() in one call."""
    events_df, features_df = build_features_dataset(cfg)
    results, shap_df, cm = run_models(cfg, features_df)
    return events_df, features_df, results, shap_df, cm
