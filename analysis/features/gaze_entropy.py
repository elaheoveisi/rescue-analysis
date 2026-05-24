

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _sge(labeled: pd.DataFrame) -> float | None:
    """Stationary Gaze Entropy over obj_type, excluding offscreen fixations."""
    on = labeled[labeled["obj_type"] != "offscreen"]["obj_type"]
    if on.empty:
        return None
    p = on.value_counts(normalize=True).values
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _gte(matrix: pd.DataFrame) -> float | None:
    """Gaze Transition Entropy from a raw-count transition matrix."""
    counts = matrix.values.astype(float)
    row_totals = counts.sum(axis=1, keepdims=True)
    if row_totals.sum() == 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        p_cond = np.where(row_totals > 0, counts / row_totals, 0.0)
        h_cond = -np.sum(p_cond * np.log2(p_cond + 1e-12), axis=1)
    p_i = row_totals.flatten() / row_totals.sum()
    return float(np.sum(p_i * h_cond))


# ---------------------------------------------------------------------------
# Standalone entry point (reads from saved fixation CSV)
# ---------------------------------------------------------------------------

def build_transition_matrix(labeled: pd.DataFrame, types: list) -> pd.DataFrame:
    """Build a transition count matrix over obj_type for the given type labels."""
    matrix = pd.DataFrame(0, index=types, columns=types)
    seq = [t for t in labeled["obj_type"].tolist() if t in types]
    for src, dst in zip(seq[:-1], seq[1:]):
        matrix.loc[src, dst] += 1
    return matrix


def run_entropy(cfg: dict) -> pd.DataFrame:
    """Compute SGE and GTE from object_aoi_fixations.csv, writing entropy_features.csv.

    Requires victim_aoi.run_object_aoi to have been run first.
    """
    # Deferred import — victim_aoi imports _sge/_gte from this module at module level,
    # so importing victim_aoi here would be circular if done at module level.
    processed = ROOT / cfg["paths"]["processed"]
    fix_all = pd.read_csv(processed / "object_aoi_fixations.csv")

    panel_names = [a["name"] for a in cfg["aoi"] if a.get("type", "static") == "static"]
    gte_types = ["victim"] + panel_names

    rows = []
    for (sid, trial, run), group in fix_all.groupby(["subject", "trial", "run"]):
        group = group.sort_values("start_ms")
        gte_matrix = build_transition_matrix(group, gte_types)

        rows.append({
            "subject": sid,
            "trial": trial,
            "run": run,
            "sge": _sge(group),
            "gte": _gte(gte_matrix),
        })

    ent_df = pd.DataFrame(rows)
    ent_df.to_csv(processed / "entropy_features.csv", index=False)
    print(f"Saved {len(ent_df)} rows -> entropy_features.csv")
    return ent_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    with open(ROOT / "configs" / "analysis.yml") as f:
        cfg = yaml.safe_load(f)
    run_entropy(cfg)
