from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


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


def plot_vs_mot_scores(subject_clusters: pd.DataFrame, output_path: Path) -> None:
    colors = {"expert": "#2196F3", "novice": "#FF9800"}
    fig, ax = plt.subplots(figsize=(7, 5))

    for expertise, group in subject_clusters.groupby("predicted_expertise"):
        ax.scatter(
            group["vs_score"],
            group["mot_score"],
            label=expertise,
            color=colors.get(expertise, "gray"),
            s=80,
            alpha=0.85,
        )
        for _, row in group.iterrows():
            ax.annotate(
                row["subject_id"],
                (row["vs_score"], row["mot_score"]),
                fontsize=7,
                textcoords="offset points",
                xytext=(5, 3),
            )

    ax.set_xlabel("VS Score (accuracy − timeout rate)")
    ax.set_ylabel("MOT Score (accuracy mean)")
    ax.set_title("VS vs. MOT Scores by Predicted Expertise")
    ax.legend(title="Expertise")
    fig.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    results = run_vs_mot_kmeans(Path(args.raw_dir), Path(args.output_dir))

    print("\nPredicted subject expertise:")
    print(
        results["subject_clusters"][["subject_id", "predicted_expertise"]].to_string(
            index=False
        )
    )

    if not results["skipped"].empty:
        print("\nSkipped:", results["skipped"].to_string(index=False))


if __name__ == "__main__":
    main()
