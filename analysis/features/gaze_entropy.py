

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


def _dte(labeled: pd.DataFrame) -> float | None:
    """Dwell-Time Entropy over obj_type: same formula as SGE, but weighted by
    summed fixation duration instead of fixation count. Excludes offscreen."""
    on = labeled[labeled["obj_type"] != "offscreen"]
    if on.empty:
        return None
    dur_by_type = on.groupby("obj_type")["duration_ms"].sum()
    total = dur_by_type.sum()
    if total <= 0:
        return None
    p = (dur_by_type / total).values
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


# ---------------------------------------------------------------------------
# Config-driven AOI grouping (same category set used for both SGE and GTE)
# ---------------------------------------------------------------------------

def regroup_obj_type(obj_type: str, groups: dict[str, list[str]]) -> str:
    """Map a raw obj_type to its configured entropy group, e.g. victim/fake_victim -> Victim.

    obj_types not listed in any group (e.g. "other", "offscreen") pass through unchanged.
    """
    for group_name, members in groups.items():
        if obj_type in members:
            return group_name
    return obj_type


def run_entropy_grouped(cfg: dict) -> pd.DataFrame:
    """Compute SGE and GTE using one consistent AOI grouping (cfg["entropy_groups"])
    for both metrics, from object_aoi_fixations.csv (already-labeled fixations).

    Fixations outside the configured groups (e.g. wall/empty "other", offscreen)
    are excluded from both metrics, so SGE and GTE regard the exact same AOIs.

    Requires victim_aoi.run_object_aoi to have been run first.
    """
    processed = ROOT / cfg["paths"]["processed"]
    fix_all = pd.read_csv(processed / "object_aoi_fixations.csv")
    groups = cfg.get("entropy_groups", {})
    gte_types = list(groups.keys())

    fix_all["obj_type"] = fix_all["obj_type"].apply(lambda t: regroup_obj_type(t, groups))
    fix_all = fix_all[fix_all["obj_type"].isin(gte_types)]

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
    out = processed / "entropy_features_grouped.csv"
    ent_df.to_csv(out, index=False)
    print(f"Saved {len(ent_df)} rows -> {out}")
    return ent_df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "analysis"))
    with open(ROOT / "configs" / "analysis.yml") as f:
        cfg = yaml.safe_load(f)
    run_entropy(cfg)
    run_entropy_grouped(cfg)
