from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from features.game_steps import EVENT_KEYS
from classification.features import build_events, build_features
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


<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
def window_rows(df: pd.DataFrame, w, features: list[str], zero_fill: list[str]) -> pd.DataFrame:
    """Rows of one window, with zero_fill features NaN -> 0 and any other NaN row dropped."""
=======
def window_rows(
    df: pd.DataFrame, w, features: list[str], zero_fill: list[str]
) -> pd.DataFrame:
>>>>>>> Stashed changes:analysis/classification/classifier.py
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
<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
        if test_raw["label"].nunique() < 2 or train_raw["label"].nunique() < 2:
=======
        if (
            test_raw.empty
            or test_raw["label"].nunique() < 2
            or train_raw["label"].nunique() < 2
        ):
>>>>>>> Stashed changes:analysis/classification/classifier.py
            continue
        rf = fit_rf(train_raw, features, hva_cfg, seed)
        folds.append(test_subject(rf, s, test_raw, features, seed))
    return folds

<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
=======
        rf = make_rf(hva_cfg)
        rf.fit(train_bal[features], train_bal["label"])
        p = rf.predict_proba(test_bal[features])[:, 1]
        y_true["rf"].extend(test_bal["label"].tolist())
        y_pred["rf"].extend(p.tolist())
        per_subject["rf"].append((s, test_bal["label"].tolist(), p.tolist()))

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(test_bal[features])
        if isinstance(sv, list):
            sv = sv[1]  # positive-class (label=1) SHAP values
        elif sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_rows["rf"].append(sv)
        x_rows["rf"].append(test_bal[features].to_numpy(dtype=float))

    shap_matrix = (
        np.vstack(shap_rows["rf"]) if shap_rows["rf"] else np.empty((0, len(features)))
    )
    x_matrix = np.vstack(x_rows["rf"]) if x_rows["rf"] else np.empty((0, len(features)))
    return {
        "rf": (y_true["rf"], y_pred["rf"], shap_matrix, x_matrix, per_subject["rf"])
    }


def per_subject_balanced_loso(
    base: pd.DataFrame, features: list[str], hva_cfg: dict, seed: int
) -> dict[str, tuple[list, list, np.ndarray]]:
    """Balance each subject ONCE (downsample to their own minority class), then plain
    leave-one-subject-out on those fixed balanced sets: train on the other subjects'
    balanced data, test on the held-out subject's balanced data. Subjects missing a
    class contribute nothing. Set info_request.loso_balance: per_subject to use it;
    compute_shap: false skips SHAP (the slow part)."""
    balanced = {
        s: balanced_sample(g, seed)
        for s, g in base.groupby("subject")
        if g["label"].nunique() == 2
    }
    want_shap = hva_cfg.get("compute_shap", True)
    y_true, y_pred, shap_rows, x_rows, per_subject = [], [], [], [], []
    for s, test_bal in balanced.items():
        others = [g for k, g in balanced.items() if k != s]
        if not others:
            continue
        train_bal = pd.concat(others, ignore_index=True)
        rf = make_rf(hva_cfg)
        rf.fit(train_bal[features], train_bal["label"])
        p = rf.predict_proba(test_bal[features])[:, 1]
        y_true.extend(test_bal["label"].tolist())
        y_pred.extend(p.tolist())
        per_subject.append((s, test_bal["label"].tolist(), p.tolist()))
        if want_shap:
            sv = shap.TreeExplainer(rf).shap_values(test_bal[features])
            if isinstance(sv, list):
                sv = sv[1]
            elif sv.ndim == 3:
                sv = sv[:, :, 1]
            shap_rows.append(sv)
            x_rows.append(test_bal[features].to_numpy(dtype=float))
    shap_matrix = np.vstack(shap_rows) if shap_rows else np.empty((0, len(features)))
    x_matrix = np.vstack(x_rows) if x_rows else np.empty((0, len(features)))
    return {"rf": (y_true, y_pred, shap_matrix, x_matrix, per_subject)}
>>>>>>> Stashed changes:analysis/classification/classifier.py

# ---------- scoring ----------

def score(y_true: list, y_pred: list, threshold: float) -> dict[str, float | None]:
    if len(set(y_true)) < 2:
<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
        return dict.fromkeys(METRICS)
=======
        return {
            "auc": None,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
        }
>>>>>>> Stashed changes:analysis/classification/classifier.py
    pred = (np.array(y_pred) >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
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
=======
def run_classifiers(
    cfg: dict, df: pd.DataFrame, predict=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pooled scores/SHAP values are per held-out decision, not per subject --
    subjects who contribute more decisions get proportionally more weight in
    each repeat's numbers. That matches "decision" as the modeling unit, but
    should be stated in the methods write-up.

    `predict` defaults to loso_predictions; any function with the same
    signature and return shape can replace it (e.g. train on one expertise
    group, test on the other -- see stepwin.cross_group_predictor).
>>>>>>> Stashed changes:analysis/classification/classifier.py
    """
    if "event_timestamp" not in df or df["event_timestamp"].isna().any():
        raise ValueError(
            "Stale event features: rebuild using corrected request timestamps before training."
        )
    hva_cfg = cfg["info_request"]
    if predict is None:
        predict = (
            per_subject_balanced_loso
            if hva_cfg.get("loso_balance") == "per_subject"
            else loso_predictions
        )
    features = hva_cfg["features"]
    n_repeats = hva_cfg.get("n_balance_repeats", 10)
    seed0 = hva_cfg.get("random_state", 42)
    threshold = hva_cfg.get("classification_threshold", 0.5)

    results, shap_tables, cm_rows = [], [], []
    for w in sorted(df["window"].unique()):
        base = window_rows(df, w, features, hva_cfg.get("zero_fill_features", []))
        if base.empty:
            continue
<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
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

=======

        scores = {"rf": []}
        cms = {"rf": np.zeros((2, 2), dtype=int)}
        shap_pool = {"rf": []}
        direction_pool = {"rf": []}
        subj_scores = {"rf": {}}  # subject -> list of score dicts, one per repeat
        n_test_after_balancing = None
        for r in range(n_repeats):
            preds = predict(base, features, hva_cfg, seed0 + r)
            for model_name, (yt, yp, sv, xv, per_subj) in preds.items():
                for s, s_yt, s_yp in per_subj:
                    subj_scores[model_name].setdefault(s, []).append(
                        score(s_yt, s_yp, threshold)
                    )
                if n_test_after_balancing is None:
                    # Balanced test size is fixed per fold (each held-out subject's own
                    # minority class count x2), so it doesn't vary across repeats --
                    # only which rows get sampled does. Pooled across all LOSO folds.
                    n_test_after_balancing = len(yt)
                scores[model_name].append(score(yt, yp, threshold))
                if len(set(yt)) >= 2:
                    pred = (np.array(yp) >= threshold).astype(int)
                    cms[model_name] += confusion_matrix(yt, pred, labels=[0, 1])
                if sv.size:
                    # One mean-abs-SHAP vector per repeat, not per row -- so the SD below
                    # reflects repeat-to-repeat stability, matching how accuracy_*_sd is computed.
                    shap_pool[model_name].append(np.abs(sv).mean(axis=0))
                    # Direction: Spearman r between a feature's value and its SHAP value
                    # over held-out rows. > 0: higher values push toward Alt (label 1).
                    direction_pool[model_name].append(
                        [
                            spearmanr(xv[:, i], sv[:, i]).correlation
                            for i in range(len(features))
                        ]
                    )

        if shap_pool["rf"]:
            repeat_means = np.vstack(shap_pool["rf"])
            # Relative importance: each repeat's mean |SHAP| as % of that repeat's total
            # across features, so the SD below is the repeat-to-repeat SD of the percentage.
            repeat_pct = 100 * repeat_means / repeat_means.sum(axis=1, keepdims=True)
            repeat_dir = np.array(direction_pool["rf"], dtype=float)
            shap_rows.append(
                pd.DataFrame(
                    {
                        "window_s": w,
                        "feature": features,
                        "mean_abs_shap": repeat_means.mean(axis=0),
                        "mean_abs_shap_sd": repeat_means.std(axis=0, ddof=1)
                        if len(repeat_means) > 1
                        else 0.0,
                        "shap_pct": repeat_pct.mean(axis=0),
                        "shap_pct_sd": repeat_pct.std(axis=0, ddof=1)
                        if len(repeat_pct) > 1
                        else 0.0,
                        "direction_r": np.nanmean(repeat_dir, axis=0),
                        "direction_r_sd": np.nanstd(repeat_dir, axis=0, ddof=1)
                        if len(repeat_dir) > 1
                        else 0.0,
                    }
                )
            )

        # Confusion-matrix counts (below) are summed across all n_repeats repeats, so
        # tn/fp/fn/tp are ~n_repeats x the number of held-out decisions, not per-repeat counts.
        # recall_request/precision_request/specificity_non_request are ratios, so they're unaffected by the scaling.
        row = {
            "window_s": w,
            "minority_class_n_full_data": base["label"].value_counts().min(),
            "n_test_after_balancing": n_test_after_balancing,
        }
        for model_name, sc in scores.items():
            for metric in ("auc", "accuracy", "precision", "recall", "f1"):
                vals = [d[metric] for d in sc if d[metric] is not None]
                row[f"{metric}_{model_name}_mean"] = np.mean(vals) if vals else None
                row[f"{metric}_{model_name}_sd"] = (
                    np.std(vals, ddof=1) if len(vals) > 1 else (0.0 if vals else None)
                )

            # Per-subject: average each held-out subject's score over repeats, then
            # mean +/- SD across subjects (each subject weighted equally).
            per_subj = subj_scores[model_name]
            row[f"n_subjects_tested_{model_name}"] = len(per_subj)
            for metric in ("auc", "accuracy", "precision", "recall", "f1"):
                subj_means = []
                for s_list in per_subj.values():
                    vals = [d[metric] for d in s_list if d[metric] is not None]
                    if vals:
                        subj_means.append(np.mean(vals))
                row[f"{metric}_{model_name}_subj_mean"] = (
                    np.mean(subj_means) if subj_means else None
                )
                row[f"{metric}_{model_name}_subj_sd"] = (
                    np.std(subj_means, ddof=1)
                    if len(subj_means) > 1
                    else (0.0 if subj_means else None)
                )
        results.append(row)
        rf_accuracy = row["accuracy_rf_mean"]
        rf_str = f"{rf_accuracy:.3f}" if rf_accuracy is not None else "n/a"
        print(f"window={w}s  RF accuracy={rf_str}")
        print("  per-subject accuracy (mean over repeats):")
        for s, s_list in subj_scores["rf"].items():
            vals = [d["accuracy"] for d in s_list if d["accuracy"] is not None]
            s_acc = f"{np.mean(vals):.3f}" if vals else "n/a"
            print(f"    subject={s}  accuracy={s_acc}")
        subj_mean = row["accuracy_rf_subj_mean"]
        subj_sd = row["accuracy_rf_subj_sd"]
        if subj_mean is not None:
            print(
                f"  average across {len(subj_scores['rf'])} subjects: {subj_mean:.3f} (SD {subj_sd:.3f})"
            )

        for model_name, cm in cms.items():
            tn, fp, fn, tp = (int(v) for v in cm.ravel())
            cm_rows.append(
                {
                    "window_s": w,
                    "model": model_name,
                    "n_repeats": n_repeats,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "recall_request": tp / (tp + fn) if (tp + fn) else None,
                    "precision_request": tp / (tp + fp) if (tp + fp) else None,
                    "specificity_non_request": tn / (tn + fp) if (tn + fp) else None,
                }
            )
        print(f"window={w}s  confusion matrix computed (RF)")

    shap_df = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    return pd.DataFrame(results), shap_df, pd.DataFrame(cm_rows)


def build_features_dataset(cfg: dict, general: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build events + eye features and save them to CSV. general=True: AOI-free features only
    (general_gaze_* files). Rerun only when the raw data, AOI labels or feature code change."""
>>>>>>> Stashed changes:analysis/classification/classifier.py
    events_df = build_events(cfg)
    features_df = events_df.merge(build_features(cfg, events_df, aoi=not general), on=EVENT_KEYS, how="left")
    for name, df in (("events", events_df), ("features", features_df)):
        key, default = (f"general_{name}_file", f"general_gaze_{name}.csv") if general else (f"{name}_file", f"info_request_{name}.csv")
        out = Path(cfg["paths"]["processed"]) / cfg["info_request"].get(key, default)
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} rows -> {out}")
    return events_df, features_df


<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
def run_models(cfg: dict, features_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random Forest, SHAP and confusion matrices. Reads the saved features CSV
    when features_df isn't given, so a new feature list doesn't need a rebuild."""
=======
def run_models(
    cfg: dict, features_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random Forest classification, SHAP values, and confusion matrices.
    Reads the saved features CSV if features_df isn't passed in-memory --
    lets this run on its own (e.g. to try a different feature list) without
    recomputing eye features via build_features_dataset() first."""
>>>>>>> Stashed changes:analysis/classification/classifier.py
    processed = Path(cfg["paths"]["processed"])
    hva_cfg = cfg["info_request"]

    if features_df is None:
<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
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
=======
        features_out = processed / hva_cfg.get(
            "features_file", "info_request_features.csv"
        )
        features_df = pd.read_csv(features_out)

    results, shap_df, cm = run_classifiers(cfg, features_df)
    results_out = processed / hva_cfg.get(
        "results_file", "info_request_classifier_results.csv"
    )
    results.to_csv(results_out, index=False)
    print(f"Saved -> {results_out}")

    cm_out = processed / hva_cfg.get(
        "confusion_matrix_file", "info_request_confusion_matrix.csv"
    )
    cm.to_csv(cm_out, index=False)
    print(f"Saved -> {cm_out}")

    shap_out = processed / hva_cfg.get("shap_file", "info_request_shap_values.csv")
    shap_df.to_csv(shap_out, index=False)
    print(f"Saved -> {shap_out}")
>>>>>>> Stashed changes:analysis/classification/classifier.py

    return results, shap_df, cm


<<<<<<< Updated upstream:analysis/model/help_vs_auto.py
def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """build_features_dataset() + run_models() in one call."""
=======
def run(
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: build_features_dataset() + run_models(), in one call --
    kept for backward compatibility. Prefer calling the two separately (e.g.
    from main.py) when you only need to rerun one half."""
>>>>>>> Stashed changes:analysis/classification/classifier.py
    events_df, features_df = build_features_dataset(cfg)
    results, shap_df, cm = run_models(cfg, features_df)
    return events_df, features_df, results, shap_df, cm
