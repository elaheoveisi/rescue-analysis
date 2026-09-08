"""Power analysis for SGE/GTE, built on victim_aoi's existing best-run selection.

Participants and expertise come from configs/analysis.yml. SGE/GTE per participant x
trial are computed by features.victim_aoi.run_object_aoi() (unchanged) -- the same
best-run selection (cfg["glmm2"]["best_run_metric"], default saved_victims) already
used everywhere else in the pipeline. Condition (no_llm/llm) and expertise are
attached, then a paired-samples power analysis (Cohen's dz + two-sided TTestPower) is
run on llm vs no_llm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.power import TTestPower

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.victim_aoi import run_object_aoi

ALPHA = 0.05
TARGET_POWER = 0.8


def load_participants(cfg: dict) -> tuple[list[str], dict[str, str]]:
    """Subjects and expertise map, straight from config."""
    subjects = [str(s) for s in cfg.get("sub", [])]
    expertise_map = {str(k): v for k, v in cfg.get("expertise", {}).items()}
    return subjects, expertise_map


def build_entropy_dataset(cfg: dict) -> pd.DataFrame:
    """SGE/GTE per participant x trial, using victim_aoi's existing best-run selection."""
    feat_df, _ = run_object_aoi(cfg)
    _, expertise_map = load_participants(cfg)
    condition_map = cfg.get("analysis", {}).get("condition_by_category", {})

    df = feat_df[["subject", "trial", "run", "sge", "gte"]].copy()
    df = df.rename(columns={"subject": "participant"})
    df["condition"] = df["trial"].map(condition_map)
    df["expertise"] = df["participant"].map(expertise_map).fillna("unknown")
    return df


def paired_power_summary(
    df: pd.DataFrame, alpha: float = ALPHA, target_power: float = TARGET_POWER
) -> pd.DataFrame:
    """Cohen's dz + two-sided TTestPower for llm vs no_llm, per metric (sge, gte).

    Each participant has 1 no_llm row (dummy) but 2 llm rows (gemini, openai); the
    llm rows are averaged per participant first so the comparison is a proper 1-to-1
    paired difference.
    """
    analysis = TTestPower()
    rows = []
    for metric in ["sge", "gte"]:
        per_condition = (
            df.groupby(["participant", "condition"])[metric].mean().unstack("condition")
        )
        diff = (per_condition["llm"] - per_condition["no_llm"]).dropna()
        n = len(diff)
        mean_diff = diff.mean()
        sd_diff = diff.std(ddof=1)
        dz = mean_diff / sd_diff

        power_at_n = analysis.power(effect_size=dz, nobs=n, alpha=alpha, alternative="two-sided")
        n_required = analysis.solve_power(
            effect_size=dz, alpha=alpha, power=target_power, alternative="two-sided"
        )

        rows.append({
            "metric": metric,
            "n_participants": n,
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "cohens_dz": dz,
            "power_at_n": power_at_n,
            "n_required_80pct": int(np.ceil(n_required)),
        })
    return pd.DataFrame(rows)


def run(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = ROOT / cfg["paths"]["processed"]
    subjects, _ = load_participants(cfg)
    print(f"Participants from config: {len(subjects)}")

    entropy_df = build_entropy_dataset(cfg)
    entropy_out = processed / "entropy_features_by_condition.csv"
    entropy_df.to_csv(entropy_out, index=False)
    print(f"\nSaved {len(entropy_df)} rows -> {entropy_out}")
    print(entropy_df.round(3).to_string(index=False))

    power_df = paired_power_summary(entropy_df)
    power_out = processed / "power_analysis_summary.csv"
    power_df.to_csv(power_out, index=False)
    print(f"\nSaved -> {power_out}")
    print("\n=== Paired power summary (alpha=0.05, two-sided) ===")
    print(power_df.round(3).to_string(index=False))

    return entropy_df, power_df


if __name__ == "__main__":
    with open(ROOT / "configs" / "analysis.yml") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
