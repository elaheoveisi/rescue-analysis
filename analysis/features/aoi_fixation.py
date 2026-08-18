import pandas as pd

DEFAULT_OFFSCREEN_LABEL: str = "offscreen"


def label_fixations(fix_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Vectorized AOI assignment based on bounding boxes."""
    df = fix_df.copy()
    df["aoi"] = DEFAULT_OFFSCREEN_LABEL
    for aoi in aois:
        mask = df["x"].between(aoi["x_min"], aoi["x_max"]) & df["y"].between(
            aoi["y_min"], aoi["y_max"]
        )
        df.loc[mask, "aoi"] = aoi["name"]
    return df


def aoi_fixation_stats(fix_aoi_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return dwell time (ms) and fixation counts per AOI label."""
    if fix_aoi_df.empty:
        empty = pd.Series(dtype=float)
        return empty, empty
    dur = fix_aoi_df.groupby("aoi")["duration_ms"].sum()
    counts = fix_aoi_df["aoi"].value_counts()
    return dur, counts


def run_aoi(fix_df: pd.DataFrame, aois: list[dict]) -> dict:
    """Label fixations with AOIs and compute transition matrix.

    Returns:
        {"fix_aoi": DataFrame, "transitions": DataFrame}
    """
    fix_aoi = label_fixations(fix_df, aois)
    return {
        "fix_aoi": fix_aoi,
        "transitions": aoi_transition_matrix(fix_aoi, aois),
    }


def aoi_transition_matrix(fix_aoi_df: pd.DataFrame, aois: list[dict]) -> pd.DataFrame:
    """Calculate transition counts between sequential AOI fixations."""
    labels = [a["name"] for a in aois]
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    sequence = fix_aoi_df[fix_aoi_df["aoi"] != DEFAULT_OFFSCREEN_LABEL]["aoi"].tolist()

    for src, dst in zip(sequence[:-1], sequence[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix


def aoi_labels(aois: list[dict], offscreen_label: str = DEFAULT_OFFSCREEN_LABEL) -> list[str]:
    """Return all AOI names plus the offscreen label."""
    return [a["name"] for a in aois] + [offscreen_label]


def build_aoi_features(
    fix_aoi: pd.DataFrame,
    trans: pd.DataFrame,
    aois: list[dict],
    offscreen_label: str = DEFAULT_OFFSCREEN_LABEL,
) -> dict:
    """Build AOI feature dict from labeled fixations and transition matrix."""
    dur, counts = aoi_fixation_stats(fix_aoi)
    total_dur = float(dur.sum())
    labels = aoi_labels(aois, offscreen_label)

    pct_dur = {
        f"{a}_pct_dur": float(dur.get(a, 0.0) / total_dur) if total_dur > 0 else None
        for a in labels
    }
    fix_counts = {f"n_fixations_{a}": int(counts.get(a, 0)) for a in labels}
    transitions = {
        f"transitions_{src}_{dst}": int(trans.loc[src, dst])
        for src in labels
        for dst in labels
        if src in trans.index and dst in trans.columns
    } if not trans.empty else {}

    return {**pct_dur, **fix_counts, **transitions}
