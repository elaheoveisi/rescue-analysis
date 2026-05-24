"""Visualize GLMM results from glmm2_results.csv."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed" / "glmm2_results.csv"
OUT  = ROOT / "analysis" / "visualization" / "figures"
OUT.mkdir(exist_ok=True)

# ── Label mappings ────────────────────────────────────────────────────────────

TERM_MAP = {
    "C(condition, Treatment('no_llm'))[T.llm]":
        "LLM effect",
    "C(expertise, Treatment('novice'))[T.expert]":
        "Expertise effect",
    "C(condition, Treatment('no_llm'))[T.llm]:C(expertise, Treatment('novice'))[T.expert]":
        "LLM × Expertise",
}
TERM_ORDER  = ["LLM effect", "Expertise effect", "LLM × Expertise"]
TERM_COLORS = {
    "LLM effect":       "#4C72B0",
    "Expertise effect": "#C44E52",
    "LLM × Expertise":  "#8172B2",
}

OUTCOME_MAP = {
    "victims_per_step":   "Victims / Step",
    "total_reward":       "Total Reward",
    "mean_fixation_dur_ms": "Fix Duration (ms)†",
    "std_pupil_diam":     "Pupil Diam SD",
    "game_area_pct_dur":  "Game Area %",
    "chat_panel_pct_dur": "Chat Panel %",
    "n_fixations":        "N Fixations†",
    "n_saccades":         "N Saccades†",
    "saved_victims":      "Saved Victims†",
}
OUTCOME_ORDER = list(OUTCOME_MAP.values())

SIG_ALPHA   = 0.05
MARG_ALPHA  = 0.10

# ── Load & clean ──────────────────────────────────────────────────────────────

raw = pd.read_csv(DATA)
raw = raw[raw["term"] != "Intercept"].copy()
raw["term_label"]    = raw["term"].map(TERM_MAP)
raw["outcome_label"] = raw["outcome"].map(OUTCOME_MAP)
raw["ci95"]          = 1.96 * raw["se"]
raw["sig"]           = raw["p_value_fdr"] < SIG_ALPHA
raw["marginal"]      = (raw["p_value_fdr"] >= SIG_ALPHA) & (raw["p_value_fdr"] < MARG_ALPHA)
raw["z_score"]       = raw["coef"] / raw["se"]  # comparable across outcomes

# ── Helpers ───────────────────────────────────────────────────────────────────

def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "†"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# 1. FOREST PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_forest():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    y_pos = {lbl: i for i, lbl in enumerate(reversed(OUTCOME_ORDER))}

    for ax, term in zip(axes, TERM_ORDER):
        sub = raw[raw["term_label"] == term].copy()
        color = TERM_COLORS[term]

        for _, row in sub.iterrows():
            y   = y_pos[row["outcome_label"]]
            col = color if row["sig"] else ("#aaaaaa" if not row["marginal"] else "#ddaa44")
            ms  = 9 if row["sig"] else 6
            ax.errorbar(
                row["coef"], y,
                xerr=row["ci95"],
                fmt="o", color=col, ecolor=col,
                markersize=ms, capsize=4, linewidth=1.5,
                zorder=3,
            )
            # significance label
            lbl = sig_label(row["p_value_fdr"])
            if lbl:
                ax.text(row["coef"] + row["ci95"] * 1.05, y, lbl,
                        va="center", ha="left", fontsize=10, color=col)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(term, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Coefficient (95% CI)", fontsize=10)
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(list(y_pos.values()))
    axes[0].set_yticklabels(list(y_pos.keys()), fontsize=10)

    # shared legend
    handles = [
        mpatches.Patch(color=TERM_COLORS["LLM effect"],      label="Significant (p_fdr < .05)"),
        mpatches.Patch(color="#ddaa44",                       label="Marginal  (p_fdr < .10)"),
        mpatches.Patch(color="#aaaaaa",                       label="n.s."),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Forest Plot of Mixed-Model Coefficients\n"
                 "(† outcome modelled on log scale)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = OUT / "glmm_forest.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 2. SIGNIFICANCE HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def plot_sig_heatmap():
    pivot = raw.pivot(index="outcome_label", columns="term_label", values="p_value_fdr")
    pivot = pivot.reindex(index=OUTCOME_ORDER, columns=TERM_ORDER)

    # Build colour matrix: 0=sig, 0.5=marginal, 1=ns
    color_vals = pivot.map(
        lambda p: 0.0 if p < SIG_ALPHA else (0.5 if p < MARG_ALPHA else 1.0)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdYlGn_r

    im = ax.imshow(color_vals.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Annotate each cell
    for i, outcome in enumerate(OUTCOME_ORDER):
        for j, term in enumerate(TERM_ORDER):
            p   = pivot.loc[outcome, term]
            lbl = f"p = {p:.3f}{sig_label(p)}"
            txt_color = "white" if p < SIG_ALPHA else "black"
            ax.text(j, i, lbl, ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold" if p < SIG_ALPHA else "normal")

    ax.set_xticks(range(len(TERM_ORDER)))
    ax.set_xticklabels(TERM_ORDER, fontsize=11)
    ax.set_yticks(range(len(OUTCOME_ORDER)))
    ax.set_yticklabels(OUTCOME_ORDER, fontsize=10)
    ax.set_title("FDR-corrected p-values  (* p<.05   † p<.10)\n"
                 "(† in outcome name = log-transformed in model)",
                 fontsize=12, pad=12)

    # Colour bar as a legend
    from matplotlib.colors import BoundaryNorm
    from matplotlib.cm import ScalarMappable
    bounds  = [0, SIG_ALPHA, MARG_ALPHA, 1.0]
    labels_ = ["sig\n(p<.05)", "marginal\n(p<.10)", "n.s."]
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Manual legend patches
    sig_patch  = mpatches.Patch(color=cmap(0.0),  label="Significant  (p_fdr < .05)")
    marg_patch = mpatches.Patch(color=cmap(0.5),  label="Marginal     (p_fdr < .10)")
    ns_patch   = mpatches.Patch(color=cmap(1.0),  label="n.s.")
    ax.legend(handles=[sig_patch, marg_patch, ns_patch],
              loc="upper left", bbox_to_anchor=(1.02, 1),
              fontsize=9, framealpha=0.9, title="Legend")

    plt.tight_layout()
    path = OUT / "glmm_sig_heatmap.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 3. VOLCANO PLOT
# ──────────────────────────────────────────────────────────────────────────────

def plot_volcano():
    fig, ax = plt.subplots(figsize=(10, 6))

    sig_threshold = -np.log10(SIG_ALPHA)

    for _, row in raw.iterrows():
        term    = row["term_label"]
        x       = row["z_score"]          # coef / SE — comparable across outcomes
        y       = -np.log10(row["p_value_fdr"] + 1e-30)
        color   = TERM_COLORS[term]
        marker  = {"LLM effect": "o", "Expertise effect": "^", "LLM × Expertise": "s"}[term]
        alpha   = 1.0 if row["sig"] else 0.45
        ms      = 90 if row["sig"] else 55

        ax.scatter(x, y, color=color, marker=marker, s=ms, alpha=alpha,
                   edgecolors="white", linewidth=0.5, zorder=3)

        # Label significant or marginal points
        if row["sig"] or row["marginal"]:
            ax.annotate(
                row["outcome_label"],
                xy=(x, y), xytext=(5, 4), textcoords="offset points",
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
            )

    ax.axhline(sig_threshold, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] < -3 else -4,
            sig_threshold + 0.08, "p_fdr = .05",
            fontsize=8.5, color="black", alpha=0.7)

    ax.set_xlabel("Z-score  (coef / SE) — comparable across outcomes", fontsize=11)
    ax.set_ylabel("−log₁₀(p_fdr)", fontsize=11)
    ax.set_title("Volcano Plot of GLMM Effects", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.35)

    # Legend: terms
    handles = [
        plt.scatter([], [], color=TERM_COLORS[t], marker=m, s=70, label=t, edgecolors="white")
        for t, m in zip(TERM_ORDER, ["o", "^", "s"])
    ]
    ax.legend(handles=handles, title="Predictor", fontsize=9, loc="upper left")

    plt.tight_layout()
    path = OUT / "glmm_volcano.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4. EFFECT DIRECTION GRID
# ──────────────────────────────────────────────────────────────────────────────

def plot_effect_grid():
    pivot_z = raw.pivot(index="outcome_label", columns="term_label", values="z_score")
    pivot_z = pivot_z.reindex(index=OUTCOME_ORDER, columns=TERM_ORDER)

    pivot_p = raw.pivot(index="outcome_label", columns="term_label", values="p_value_fdr")
    pivot_p = pivot_p.reindex(index=OUTCOME_ORDER, columns=TERM_ORDER)

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Clip z-scores for colour mapping
    z_vals  = pivot_z.values.astype(float)
    z_clipped = np.clip(z_vals, -5, 5)

    im = ax.imshow(z_clipped, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")

    for i, outcome in enumerate(OUTCOME_ORDER):
        for j, term in enumerate(TERM_ORDER):
            z = pivot_z.loc[outcome, term]
            p = pivot_p.loc[outcome, term]

            arrow = "▲" if z > 0 else "▼"
            star  = sig_label(p)
            txt   = f"{arrow}  {star}" if star else arrow

            # text colour: white on dark cells, black on light
            txt_color = "white" if abs(z) > 2.5 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=13 if star else 11,
                    color=txt_color, fontweight="bold" if p < SIG_ALPHA else "normal")

    ax.set_xticks(range(len(TERM_ORDER)))
    ax.set_xticklabels(TERM_ORDER, fontsize=11)
    ax.set_yticks(range(len(OUTCOME_ORDER)))
    ax.set_yticklabels(OUTCOME_ORDER, fontsize=10)
    ax.set_title("Effect Direction Grid\n"
                 "(▲ positive  ▼ negative  |  fill = z-score  |  * p<.05  † p<.10)\n"
                 "(† in outcome name = log-transformed in model)",
                 fontsize=11, pad=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Z-score (coef / SE)", fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(TERM_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(OUTCOME_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    plt.tight_layout()
    path = OUT / "glmm_effect_grid.pdf"
    plt.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("1/4  Forest plot …")
    plot_forest()

    print("2/4  Significance heatmap …")
    plot_sig_heatmap()

    print("3/4  Volcano plot …")
    plot_volcano()

    print("4/4  Effect direction grid …")
    plot_effect_grid()

    print(f"\nAll figures saved to: {OUT}")
