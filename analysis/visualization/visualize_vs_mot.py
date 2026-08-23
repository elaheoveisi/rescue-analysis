"""Visualize VS/MOT subject clusters (see features.vs_mot_kmeans)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
