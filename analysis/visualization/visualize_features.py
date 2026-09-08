"""Visualize AOI attention, performance, and gaze features computed from data.h5."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "legend.title_fontsize": 13,
        "figure.titlesize": 18,
    }
)

ROOT = Path(__file__).resolve().parent.parent.parent
H5_PATH = ROOT / "data" / "processed" / "data.h5"
CFG_PATH = ROOT / "configs" / "analysis.yml"
OUT = ROOT / "analysis" / "visualization" / "figures"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "analysis"))
from features.extract_features import extract_features  # noqa: E402


def load_features() -> pd.DataFrame:
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    for key, rel in cfg["paths"].items():
        cfg["paths"][key] = str(ROOT / rel)
    print(f"Extracting features from {H5_PATH} …")
    return extract_features(cfg)


TRIAL_ORDER = ["dummy", "gemini", "openai"]
TRIAL_LABELS = {"dummy": "Dummy", "gemini": "Gemini", "openai": "OpenAI"}
EXPERTISE_ORDER = ["novice", "expert"]

TRIAL_PALETTE = {"dummy": "#7f7f7f", "gemini": "#4C72B0", "openai": "#DD8452"}
EXPERTISE_PALETTE = {"novice": "#55A868", "expert": "#C44E52"}

NUMERIC_COLS = [
    "n_fixations",
    "mean_fixation_dur_ms",
    "total_fixation_dur_ms",
    "n_saccades",
    "mean_saccade_dur_ms",
    "mean_saccade_amp_px",
    "saccades_total_duration_ms",
    "std_pupil_diam",
    "game_area_pct_dur",
    "info_panel_pct_dur",
    "chat_panel_pct_dur",
    "offscreen_pct_dur",
    "n_fixations_game_area",
    "n_fixations_info_panel",
    "n_fixations_chat_panel",
    "transitions_game_area_game_area",
    "transitions_game_area_info_panel",
    "transitions_game_area_chat_panel",
    "transitions_info_panel_game_area",
    "transitions_info_panel_info_panel",
    "transitions_info_panel_chat_panel",
    "transitions_chat_panel_game_area",
    "transitions_chat_panel_info_panel",
    "transitions_chat_panel_chat_panel",
    "n_actions",
    "n_llm_calls",
    "saved_victims",
    "mean_reward",
    "total_reward",
    "victims_per_step",
]

CORR_LABELS = {
    "n_fixations": "n_fix",
    "mean_fixation_dur_ms": "fix_dur",
    "total_fixation_dur_ms": "fix_dur_tot",
    "n_saccades": "n_sacc",
    "mean_saccade_dur_ms": "sacc_dur",
    "mean_saccade_amp_px": "sacc_amp",
    "saccades_total_duration_ms": "sacc_dur_tot",
    "std_pupil_diam": "pupil_SD",
    "game_area_pct_dur": "game%",
    "info_panel_pct_dur": "info%",
    "chat_panel_pct_dur": "chat%",
    "offscreen_pct_dur": "offscreen%",
    "n_fixations_game_area": "nfix_G",
    "n_fixations_info_panel": "nfix_I",
    "n_fixations_chat_panel": "nfix_C",
    "transitions_game_area_game_area": "tr_G>G",
    "transitions_game_area_info_panel": "tr_G>I",
    "transitions_game_area_chat_panel": "tr_G>C",
    "transitions_info_panel_game_area": "tr_I>G",
    "transitions_info_panel_info_panel": "tr_I>I",
    "transitions_info_panel_chat_panel": "tr_I>C",
    "transitions_chat_panel_game_area": "tr_C>G",
    "transitions_chat_panel_info_panel": "tr_C>I",
    "transitions_chat_panel_chat_panel": "tr_C>C",
    "n_actions": "n_actions",
    "n_llm_calls": "n_llm",
    "saved_victims": "saved",
    "mean_reward": "mean_rew",
    "total_reward": "tot_rew",
    "victims_per_step": "vic/step",
}

_raw = load_features()
df = (
    _raw.sort_values("saved_victims", ascending=False)
    .groupby(["participant", "trial"], as_index=False)
    .first()
)
print(f"Sessions after best-run filter: {len(df)}  (from {len(_raw)} total runs)")

df["trial"] = pd.Categorical(df["trial"], categories=TRIAL_ORDER, ordered=True)
df["expertise"] = pd.Categorical(
    df["expertise"], categories=EXPERTISE_ORDER, ordered=True
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. AOI STACKED BAR CHART
# ──────────────────────────────────────────────────────────────────────────────


def plot_aoi_stacked():
    aoi_cols = [
        "game_area_pct_dur",
        "info_panel_pct_dur",
        "chat_panel_pct_dur",
        "offscreen_pct_dur",
    ]
    aoi_labels = ["Game Area", "Info Panel", "Chat Panel", "Off-screen"]
    aoi_colors = ["#4C72B0", "#DD8452", "#55A868", "#d3d3d3"]

    groups = df.groupby(["trial", "expertise"])[aoi_cols].mean().reset_index()
    groups["trial_label"] = groups["trial"].astype(str).map(TRIAL_LABELS)
    groups["group"] = (
        groups["trial_label"].astype(str)
        + "\n"
        + groups["expertise"].astype(str).str.capitalize()
    )
    groups = groups.sort_values(["expertise", "trial"])

    x = np.arange(len(groups))
    width = 0.65
    bottom = np.zeros(len(groups))

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for col, label, color in zip(aoi_cols, aoi_labels, aoi_colors):
        vals = groups[col].values
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color)
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0.04:
                ax.text(
                    xi,
                    b + v / 2,
                    f"{v:.0%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white",
                    fontweight="bold",
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(groups["group"])
    ax.set_ylabel("Mean proportion of dwell time")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("AOI Dwell-Time Distribution by AI Condition and Expertise", pad=10)
    ax.legend(loc="upper right", framealpha=0.9)

    mid = len(groups) // 2
    ax.axvline(mid - 0.5, color="black", linewidth=1.2, linestyle="--", alpha=0.5)

    novice_cx = (mid - 1) / 2
    expert_cx = mid + (len(groups) - mid - 1) / 2
    for cx, lbl in [(novice_cx, "Novice"), (expert_cx, "Expert")]:
        ax.annotate(
            lbl,
            xy=(cx, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -52),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            annotation_clip=False,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    path = OUT / "aoi_stacked_bar.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 2. PERFORMANCE VIOLIN PLOTS
# ──────────────────────────────────────────────────────────────────────────────


def plot_performance_violins():
    metrics = [
        ("saved_victims", "Saved Victims"),
        ("total_reward", "Total Reward"),
        ("victims_per_step", "Victims per Step"),
    ]

    def _draw(ax, col, label, legend=False):
        sns.violinplot(
            data=df,
            x="trial",
            y=col,
            hue="expertise",
            order=TRIAL_ORDER,
            hue_order=EXPERTISE_ORDER,
            palette=EXPERTISE_PALETTE,
            inner="box",
            split=True,
            ax=ax,
            linewidth=1.2,
            cut=0,
        )
        sns.stripplot(
            data=df,
            x="trial",
            y=col,
            hue="expertise",
            order=TRIAL_ORDER,
            hue_order=EXPERTISE_ORDER,
            palette=EXPERTISE_PALETTE,
            dodge=True,
            size=4,
            alpha=0.6,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("AI Condition")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(TRIAL_ORDER)))
        ax.set_xticklabels([TRIAL_LABELS[t] for t in TRIAL_ORDER])
        ax.set_title(label)
        if legend:
            handles, lbs = ax.get_legend_handles_labels()
            ax.legend(
                handles[:2],
                [l.capitalize() for l in EXPERTISE_ORDER],
                title="Expertise",
            )
        elif ax.get_legend():
            ax.get_legend().remove()

    # Individual plots
    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        _draw(ax, col, label, legend=True)
        plt.tight_layout()
        path = OUT / f"perf_{col}.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for i, (ax, (col, label)) in enumerate(zip(axes, metrics)):
        _draw(ax, col, label, legend=(i == 0))
    fig.suptitle("Task Performance by AI Condition and Expertise")
    plt.tight_layout()
    path = OUT / "performance_violins.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 3. GAZE TRANSITION HEATMAPS
# ──────────────────────────────────────────────────────────────────────────────


def plot_transition_heatmaps():
    aoi_names = ["Game", "Info", "Chat"]
    trans_cols = [
        [
            "transitions_game_area_game_area",
            "transitions_game_area_info_panel",
            "transitions_game_area_chat_panel",
        ],
        [
            "transitions_info_panel_game_area",
            "transitions_info_panel_info_panel",
            "transitions_info_panel_chat_panel",
        ],
        [
            "transitions_chat_panel_game_area",
            "transitions_chat_panel_info_panel",
            "transitions_chat_panel_chat_panel",
        ],
    ]

    matrices = {}
    for trial in TRIAL_ORDER:
        sub = df[df["trial"] == trial]
        mat = np.array([[sub[c].mean() for c in row] for row in trans_cols])
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrices[trial] = mat / row_sums

    def _draw(ax, trial, cbar=False):
        sns.heatmap(
            matrices[trial],
            ax=ax,
            xticklabels=aoi_names,
            yticklabels=aoi_names,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar=cbar,
            annot_kws={"size": 13},
        )
        ax.set_title(TRIAL_LABELS[trial])
        ax.set_xlabel("To AOI")
        ax.set_ylabel("From AOI" if trial == TRIAL_ORDER[0] else "")

    # Individual plots
    for trial in TRIAL_ORDER:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        _draw(ax, trial, cbar=True)
        fig.suptitle(
            f"Gaze Transitions — {TRIAL_LABELS[trial]}\n(row-normalised: P(to | from))"
        )
        plt.tight_layout()
        path = OUT / f"transition_heatmap_{trial}.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, trial in zip(axes, TRIAL_ORDER):
        _draw(ax, trial, cbar=(trial == TRIAL_ORDER[-1]))
    fig.suptitle(
        "Gaze Transition Probabilities by AI Condition\n(row-normalised: P(to | from))"
    )
    plt.tight_layout()
    path = OUT / "transition_heatmaps.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4. EYE-TRACKING FEATURE BOXPLOTS
# ──────────────────────────────────────────────────────────────────────────────


def plot_gaze_boxplots():
    gaze_metrics = [
        ("n_fixations", "Number of Fixations"),
        ("mean_fixation_dur_ms", "Mean Fixation Duration (ms)"),
        ("mean_saccade_amp_px", "Mean Saccade Amplitude (px)"),
        ("std_pupil_diam", "Pupil Diameter Std Dev"),
    ]

    def _draw(ax, col, label, legend=False):
        sns.boxplot(
            data=df,
            x="trial",
            y=col,
            hue="expertise",
            order=TRIAL_ORDER,
            hue_order=EXPERTISE_ORDER,
            palette=EXPERTISE_PALETTE,
            ax=ax,
            linewidth=1.2,
        )
        sns.stripplot(
            data=df,
            x="trial",
            y=col,
            hue="expertise",
            order=TRIAL_ORDER,
            hue_order=EXPERTISE_ORDER,
            palette=EXPERTISE_PALETTE,
            dodge=True,
            size=4,
            alpha=0.6,
            ax=ax,
            legend=False,
        )
        ax.set_xlabel("AI Condition")
        ax.set_ylabel(label)
        ax.set_xticks(range(len(TRIAL_ORDER)))
        ax.set_xticklabels([TRIAL_LABELS[t] for t in TRIAL_ORDER])
        ax.set_title(label)
        if legend:
            handles, lbs = ax.get_legend_handles_labels()
            ax.legend(
                handles[:2],
                [l.capitalize() for l in EXPERTISE_ORDER],
                title="Expertise",
            )
        elif ax.get_legend():
            ax.get_legend().remove()

    # Individual plots
    for col, label in gaze_metrics:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        _draw(ax, col, label, legend=True)
        plt.tight_layout()
        path = OUT / f"gaze_{col}.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ax, (col, label)) in enumerate(zip(axes.flatten(), gaze_metrics)):
        _draw(ax, col, label, legend=(i == 0))
    fig.suptitle("Eye-Tracking Features by AI Condition and Expertise")
    plt.tight_layout()
    path = OUT / "gaze_boxplots.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────────────────────


def plot_correlation_heatmap():
    corr = df[NUMERIC_COLS].corr()
    short_labels = [CORR_LABELS[c] for c in NUMERIC_COLS]

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=False,
        linewidths=0.3,
        linecolor="#eeeeee",
        cbar_kws={"shrink": 0.6, "label": "Pearson r", "pad": 0.02},
        xticklabels=short_labels,
        yticklabels=short_labels,
    )
    ax.set_title("Feature Correlation Matrix", pad=14)
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, rotation=0)

    plt.tight_layout()
    path = OUT / "correlation_heatmap.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 6. LEARNING CURVES ACROSS RUNS
# ──────────────────────────────────────────────────────────────────────────────


def plot_learning_curves():
    metrics = [
        ("saved_victims", "Saved Victims"),
        ("total_reward", "Total Reward"),
        ("victims_per_step", "Victims per Step"),
    ]

    def _draw(ax, col, label):
        lines, lbs = [], []
        for expertise, exp_df in df.groupby("expertise"):
            for trial, t_df in exp_df.groupby("trial"):
                run_means = t_df.groupby("run")[col].mean()
                run_sems = t_df.groupby("run")[col].sem()
                runs = run_means.index.values

                color = EXPERTISE_PALETTE[expertise]
                linestyle = {"dummy": ":", "gemini": "--", "openai": "-"}[trial]

                (line,) = ax.plot(
                    runs,
                    run_means.values,
                    marker="o",
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    label=f"{expertise.capitalize()} / {TRIAL_LABELS[trial]}",
                )
                ax.fill_between(
                    runs,
                    run_means.values - run_sems.values,
                    run_means.values + run_sems.values,
                    alpha=0.12,
                    color=color,
                )
                lines.append(line)
                lbs.append(f"{expertise.capitalize()} / {TRIAL_LABELS[trial]}")

        ax.set_xlabel("Run")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(sorted(df["run"].unique()))
        return lines, lbs

    # Individual plots
    for col, label in metrics:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        lines, lbs = _draw(ax, col, label)
        ax.legend(lines, lbs, title="Group", loc="best")
        plt.tight_layout()
        path = OUT / f"learning_{col}.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    all_lines, all_lbs = None, None
    for ax, (col, label) in zip(axes, metrics):
        lines, lbs = _draw(ax, col, label)
        if all_lines is None:
            all_lines, all_lbs = lines, lbs

    fig.legend(
        all_lines,
        all_lbs,
        title="Group",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        ncol=1,
        framealpha=0.9,
    )
    fig.suptitle("Learning Curves: Performance across Runs")
    plt.tight_layout(rect=[0, 0, 0.80, 0.95])
    path = OUT / "learning_curves.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 7. PCA SCATTER
# ──────────────────────────────────────────────────────────────────────────────


def plot_pca_scatter():
    X = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    pca_df = df[["expertise", "trial", "participant"]].copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]

    def _draw_expertise(ax):
        for exp in EXPERTISE_ORDER:
            sub = pca_df[pca_df["expertise"] == exp]
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                color=EXPERTISE_PALETTE[exp],
                label=exp.capitalize(),
                s=70,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
        ax.set_title("Coloured by Expertise")
        ax.legend(title="Expertise")

    def _draw_trial(ax):
        markers = {"dummy": "s", "gemini": "^", "openai": "o"}
        for trial in TRIAL_ORDER:
            sub = pca_df[pca_df["trial"] == trial]
            ax.scatter(
                sub["PC1"],
                sub["PC2"],
                color=TRIAL_PALETTE[trial],
                label=TRIAL_LABELS[trial],
                s=70,
                alpha=0.8,
                marker=markers[trial],
                edgecolors="white",
                linewidth=0.5,
            )
        ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
        ax.set_title("Coloured by AI Condition")
        ax.legend(title="AI Condition")

    # Individual plots
    for draw_fn, name in [
        (_draw_expertise, "pca_expertise"),
        (_draw_trial, "pca_trial"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        draw_fn(ax)
        fig.suptitle("PCA of All Features (2 Components)")
        plt.tight_layout()
        path = OUT / f"{name}.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _draw_expertise(axes[0])
    _draw_trial(axes[1])
    fig.suptitle("PCA of All Features (2 Components)")
    plt.tight_layout()
    path = OUT / "pca_scatter.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 8. PARALLEL COORDINATES
# ──────────────────────────────────────────────────────────────────────────────


def plot_parallel_coordinates():
    selected = [
        "mean_fixation_dur_ms",
        "mean_saccade_amp_px",
        "std_pupil_diam",
        "game_area_pct_dur",
        "info_panel_pct_dur",
        "chat_panel_pct_dur",
        "n_actions",
        "saved_victims",
        "victims_per_step",
    ]
    labels = [
        "Fix Dur",
        "Sacc Amp",
        "Pupil SD",
        "Game%",
        "Info%",
        "Chat%",
        "Actions",
        "Victims",
        "Vic/Step",
    ]

    plot_df = df[selected + ["expertise"]].copy()
    for col in selected:
        col_min, col_max = plot_df[col].min(), plot_df[col].max()
        denom = col_max - col_min if col_max != col_min else 1.0
        plot_df[col] = (plot_df[col] - col_min) / denom

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x_pos = np.arange(len(selected))

    for _, row in plot_df.iterrows():
        exp = row["expertise"]
        ax.plot(
            x_pos,
            row[selected].values,
            color=EXPERTISE_PALETTE[exp],
            alpha=0.35,
            linewidth=1.2,
        )

    for exp in EXPERTISE_ORDER:
        sub = plot_df[plot_df["expertise"] == exp][selected].mean()
        ax.plot(
            x_pos,
            sub.values,
            color=EXPERTISE_PALETTE[exp],
            linewidth=3,
            label=exp.capitalize(),
            zorder=5,
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["Min", "25%", "50%", "75%", "Max"])
    ax.set_ylabel("Normalised value")
    ax.set_title(
        "Parallel Coordinates — Selected Features by Expertise\n"
        "(bold lines = group mean; thin lines = individual trials)"
    )
    ax.legend(title="Expertise")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = OUT / "parallel_coordinates.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 9. EYE-TRACKING vs PERFORMANCE SCATTER
# ──────────────────────────────────────────────────────────────────────────────


def plot_gaze_vs_performance():
    pairs = [
        ("mean_fixation_dur_ms", "total_reward", "Fix Duration (ms)", "Total Reward"),
        ("n_fixations", "saved_victims", "N Fixations", "Saved Victims"),
        ("mean_saccade_amp_px", "victims_per_step", "Saccade Amp (px)", "Victims/Step"),
        ("std_pupil_diam", "total_reward", "Pupil Diam SD", "Total Reward"),
    ]

    def _draw(ax, xcol, ycol, xlabel, ylabel, legend=False):
        for exp in EXPERTISE_ORDER:
            sub = df[df["expertise"] == exp]
            x_vals = sub[xcol].values
            y_vals = sub[ycol].values
            ax.scatter(
                x_vals,
                y_vals,
                color=EXPERTISE_PALETTE[exp],
                marker={"novice": "o", "expert": "^"}[exp],
                label=exp.capitalize(),
                s=60,
                alpha=0.75,
                edgecolors="white",
                linewidth=0.5,
            )
            if len(x_vals) > 2 and np.std(x_vals) > 0:
                m, b = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                ax.plot(
                    x_line,
                    m * x_line + b,
                    color=EXPERTISE_PALETTE[exp],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.7,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel}  vs  {ylabel}")
        if legend:
            ax.legend(title="Expertise")

    # Individual plots
    for xcol, ycol, xlabel, ylabel in pairs:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        _draw(ax, xcol, ycol, xlabel, ylabel, legend=True)
        plt.tight_layout()
        path = OUT / f"gaze_vs_perf_{xcol}_vs_{ycol}.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
        plt.close(fig)

    # Combined figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for i, (ax, (xcol, ycol, xlabel, ylabel)) in enumerate(zip(axes, pairs)):
        _draw(ax, xcol, ycol, xlabel, ylabel, legend=(i == 1))
    fig.suptitle("Eye-Tracking Features vs Task Performance")
    plt.tight_layout()
    path = OUT / "gaze_vs_performance.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("1/9  AOI stacked bar chart …")
    plot_aoi_stacked()

    print("2/9  Performance violin plots …")
    plot_performance_violins()

    print("3/9  Gaze transition heatmaps …")
    plot_transition_heatmaps()

    print("4/9  Eye-tracking boxplots …")
    plot_gaze_boxplots()

    print("5/9  Correlation heatmap …")
    plot_correlation_heatmap()

    print("6/9  Learning curves …")
    plot_learning_curves()

    print("7/9  PCA scatter …")
    plot_pca_scatter()

    print("8/9  Parallel coordinates …")
    plot_parallel_coordinates()

    print("9/9  Gaze vs performance scatter …")
    plot_gaze_vs_performance()

    print(f"\nAll figures saved to: {OUT}")
