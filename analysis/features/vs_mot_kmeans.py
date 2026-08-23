from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

from visualization.visualize_vs_mot import plot_vs_mot_scores


def load_subject_scores(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, skipped = [], []

    for vs_path in sorted(
        raw_dir.glob("sub-*/vs_mot/sub-*_visual_search_*.csv")
    ) + sorted(raw_dir.glob("sub-*/vs-mot/sub-*_visual_search_*.csv")):
        subject_id = vs_path.parts[-3]
        mot_paths = list(vs_path.parent.glob("sub-*_multi_object_tracking_*.csv"))

        if not mot_paths:
            skipped.append({"subject_id": subject_id, "reason": "missing MOT file"})
            continue

        vs = pd.read_csv(vs_path)
        mot = pd.read_csv(mot_paths[0])

        correct_mask = vs["correct"].astype(str).str.lower() == "true"
        correct_num = correct_mask.astype(float)
        timeout_rate = vs["response"].astype(str).str.lower().eq("timeout").mean()
        rt = pd.to_numeric(vs["rt"], errors="coerce")
        vs_score = float(correct_num.mean()) - float(timeout_rate)
        vs_rt_correct = float(rt[correct_mask].mean())
        mot_score = float(mot["accuracy"].mean())

        rows.append(
            {
                "subject_id": subject_id,
                "vs_score": vs_score,
                "vs_rt_correct": vs_rt_correct,
                "mot_score": mot_score,
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(skipped)


def run_vs_mot_kmeans(raw_dir: Path, output_dir: Path) -> dict[str, pd.DataFrame]:
    combined, skipped = load_subject_scores(raw_dir)

    combined["vs_rt_correct"] = combined["vs_rt_correct"].fillna(
        combined["vs_rt_correct"].median()
    )
    x = combined[["vs_score", "vs_rt_correct", "mot_score"]].to_numpy()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    combined["cluster_id"] = kmeans.fit_predict(x)
    combined["predicted_expertise"] = combined["cluster_id"].map(
        {0: "cluster_0", 1: "cluster_1"}
    )
    combined = combined.sort_values("subject_id").reset_index(drop=True)

    summary = (
        combined.groupby("predicted_expertise", as_index=False)
        .agg(subject_count=("subject_id", "nunique"))
        .sort_values("predicted_expertise")
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = output_dir / "vs_mot_subject_clusters.csv"
    summary_path = output_dir / "vs_mot_cluster_summary.csv"
    skipped_path = output_dir / "vs_mot_skipped_subjects.csv"
    plot_path = output_dir / "vs_mot_scores.png"

    plot_vs_mot_scores(combined, plot_path)
    combined.to_csv(subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    return {"subject_clusters": combined, "summary": summary, "skipped": skipped}
