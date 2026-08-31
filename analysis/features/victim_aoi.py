from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.aoi_fixation import DEFAULT_OFFSCREEN_LABEL as _OFFSCREEN
from features.aoi_fixation import label_fixations
from features.eye_tracking_features import run_eyetracking
from features.gaze_entropy import _gte, _sge, build_transition_matrix, regroup_obj_type
from features.grid import best_runs, cam_bounds, extract_run_grid
from prepare_data.parse import get_stream, xdf_path

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Config-driven tile → AOI label
# ---------------------------------------------------------------------------


def _build_tile_labeler(tile_aois: list[dict]):

    exact: dict[int, tuple[str, bool]] = {}
    ranges: list[tuple[int, int, str, bool]] = []
    for a in tile_aois:
        positional = a.get("positional", True)
        if "tile_id" in a:
            exact[int(a["tile_id"])] = (a["name"], positional)
        elif "tile_id_min" in a:
            ranges.append(
                (int(a["tile_id_min"]), int(a["tile_id_max"]), a["name"], positional)
            )

    def _label(tile: int, gx: int, gy: int) -> str:
        if tile in exact:
            name, positional = exact[tile]
            return f"{name}_{gx}_{gy}" if positional else name
        for lo, hi, name, positional in ranges:
            if lo <= tile < hi:
                return f"{name}_{gx}_{gy}" if positional else name
        return "empty"

    return _label


def _object_types(tile_aois: list[dict]) -> tuple[str, ...]:
    """Ordered tuple of trackable object type names (positional tile AOIs)."""
    return tuple(a["name"] for a in tile_aois if a.get("positional", True))


def _make_aoi_to_type(panel_names: frozenset, object_types: tuple[str, ...]):
    """Return a function mapping an AOI label to its object-type string."""

    def _fn(aoi: str) -> str:
        for t in object_types:
            if aoi.startswith(t + "_"):
                return t
        if aoi in panel_names:
            return aoi
        return "offscreen" if aoi == "offscreen" else "other"

    return _fn


# ---------------------------------------------------------------------------
# Dynamic per-fixation labeling
# ---------------------------------------------------------------------------


def label_fixations_dynamic(
    fix_df: pd.DataFrame,
    game_df: pd.DataFrame,
    eye_df: pd.DataFrame,
    grid: list[list[int]],
    game_aoi: dict,
    panel_aois: list[dict],
    tile_aois: list[dict],
) -> pd.DataFrame:
    """Label each fixation dynamically based on the current camera viewport.

    Static panel AOIs (info_panel, chat_panel, …) are labeled first by
    delegating to ``label_fixations`` from ``aoi_fixation``.  Fixations that
    remain "offscreen" after that step are mapped to dynamic grid tiles using
    the game camera viewport at the time of each fixation.  Tile-type
    definitions (victim, door, key, …) are read from ``tile_aois`` so no
    game-design constants are hardcoded here.
    """
    # Step 1 — static panel labels (reuses shared label_fixations from aoi_fixation).
    df = label_fixations(fix_df, panel_aois)
    df["obj_type"] = pd.Series(dtype=str)
    df["grid_x"] = np.nan
    df["grid_y"] = np.nan
    df["tile_pixel_x_min"] = np.nan
    df["tile_pixel_x_max"] = np.nan
    df["tile_pixel_y_min"] = np.nan
    df["tile_pixel_y_max"] = np.nan

    panel_names = frozenset(p["name"] for p in panel_aois)
    tile_labeler = _build_tile_labeler(tile_aois)
    aoi_to_type = _make_aoi_to_type(panel_names, _object_types(tile_aois))

    # Pre-compute the game frame index that aligns with each fixation onset.
    t0_xdf = float(eye_df["timestamp"].iloc[0])
    game_ts = game_df["timestamp"].values.astype(float)
    fix_xdf = t0_xdf + df["start_ms"].values.astype(float) / 1000.0
    frame_idx = np.clip(np.searchsorted(game_ts, fix_xdf), 0, len(game_ts) - 1)

    sx0, sx1 = float(game_aoi["x_min"]), float(game_aoi["x_max"])
    sy0, sy1 = float(game_aoi["y_min"]), float(game_aoi["y_max"])
    grid_h, grid_w = len(grid), len(grid[0]) if grid else 0

    # Step 2 — dynamic grid mapping for fixations not claimed by a panel AOI.
    for i, (row_idx, row) in enumerate(df.iterrows()):
        if row["aoi"] != _OFFSCREEN:
            continue  # already labeled by a static panel

        px, py = float(row["x"]), float(row["y"])
        if sx0 <= px <= sx1 and sy0 <= py <= sy1:
            cx0, cy0, cx1, cy1 = cam_bounds(game_df.iloc[int(frame_idx[i])])
            vw, vh = cx1 - cx0, cy1 - cy0
            if vw > 0 and vh > 0:
                gx = cx0 + int((px - sx0) / ((sx1 - sx0) / vw))
                gy = cy0 + int((py - sy0) / ((sy1 - sy0) / vh))
                gx = max(0, min(gx, grid_w - 1))
                gy = max(0, min(gy, grid_h - 1))
                df.at[row_idx, "aoi"] = tile_labeler(int(grid[gy][gx]), gx, gy)
                df.at[row_idx, "grid_x"] = gx
                df.at[row_idx, "grid_y"] = gy
                tile_pw = (sx1 - sx0) / vw
                tile_ph = (sy1 - sy0) / vh
                df.at[row_idx, "tile_pixel_x_min"] = sx0 + (gx - cx0) * tile_pw
                df.at[row_idx, "tile_pixel_x_max"] = sx0 + (gx - cx0 + 1) * tile_pw
                df.at[row_idx, "tile_pixel_y_min"] = sy0 + (gy - cy0) * tile_ph
                df.at[row_idx, "tile_pixel_y_max"] = sy0 + (gy - cy0 + 1) * tile_ph

    df["obj_type"] = df["aoi"].apply(aoi_to_type)
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def _fixation_stats(on: pd.DataFrame, total_dur: float, prefix: str) -> dict:
    dur = float(on["duration_ms"].sum()) if not on.empty else 0.0
    return {
        f"n_fixations_{prefix}": len(on),
        f"total_dur_{prefix}_ms": dur,
        f"pct_dur_{prefix}": dur / total_dur if total_dur > 0 else None,
    }


def _panel_features(labeled: pd.DataFrame, total_dur: float, panel_name: str) -> dict:
    return _fixation_stats(labeled[labeled["aoi"] == panel_name], total_dur, panel_name)


def _type_features(labeled: pd.DataFrame, total_dur: float, t: str) -> dict:
    on = labeled[labeled["aoi"].str.startswith(t + "_")]
    return {
        **_fixation_stats(on, total_dur, f"on_{t}"),
        f"n_unique_{t}_fixated": on["aoi"].nunique() if not on.empty else 0,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_object_aoi(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run dynamic object AOI analysis for all subjects / trials / runs.

    Writes:
      {processed}/object_aoi_features.csv
      {processed}/object_aoi_transitions.csv
    """
    h5_path = ROOT / cfg["paths"]["processed"] / "data.h5"
    # Derive AOI roles from the config `type` field (dynamic = game viewport,
    # static = fixed panel bounding box).  Fallback: treat "game_area" as
    # dynamic for configs that pre-date the type field.
    game_aoi = next(
        (a for a in cfg["aoi"] if a.get("type") == "dynamic"),
        next(a for a in cfg["aoi"] if a["name"] == "game_area"),
    )
    panel_aois = [a for a in cfg["aoi"] if a.get("type", "static") == "static"]
    tile_aois = [a for a in cfg["aoi"] if a.get("type") == "tile"]
    obj_types = _object_types(tile_aois)  # e.g. (fake_victim, victim, lava, door, key)
    panel_names = [a["name"] for a in panel_aois]
    trans_types = list(obj_types) + panel_names + ["other"]
    entropy_groups = cfg.get("entropy_groups", {})
    entropy_types = list(entropy_groups.keys())
    trials_cfg = [str(t) for t in cfg.get("trials", [])]

    feat_rows, trans_rows, fix_rows = [], [], []

    with pd.HDFStore(str(h5_path), mode="r") as store:
        best_runs_set = best_runs(store, trials_cfg, cfg)
        for sid in [str(s) for s in cfg.get("sub", [])]:
            print(f"Processing {sid}")
            try:
                streams, _ = pyxdf.load_xdf(str(xdf_path(sid, cfg)))
            except FileNotFoundError:
                print(f"  XDF not found for {sid}, skipping")
                continue

            game_stream = get_stream(streams, cfg["xdf"]["game_stream"])
            if game_stream is None:
                print(f"  No game stream for {sid}, skipping")
                continue

            for eye_key in [
                k
                for k in store.keys()
                if f"/{sid}/" in k and k.endswith("/eye_tracking")
            ]:
                game_key = eye_key.replace("/eye_tracking", "/game")
                if game_key not in store:
                    continue

                parts = eye_key.strip("/").split("/")
                trial_h5, run_num = parts[1], int(parts[2].replace("run_", ""))

                trial_match = next((t for t in trials_cfg if t in trial_h5), None)
                if trial_match is None:
                    continue

                if (sid, trial_match, run_num) not in best_runs_set:
                    continue

                eye_df = store[eye_key]
                game_df = store[game_key]

                grid_info = extract_run_grid(game_stream, game_df)
                if grid_info is None:
                    print(f"  No grid for {sid}/{trial_match}, skipping")
                    continue
                fix_df = run_eyetracking(eye_df, cfg)["fixations"]

                meta = {"subject": sid, "trial": trial_match, "run": run_num}
                total_dur = (
                    float(fix_df["duration_ms"].sum()) if not fix_df.empty else 0.0
                )

                labeled = label_fixations_dynamic(
                    fix_df,
                    game_df,
                    eye_df,
                    grid_info["grid"],
                    game_aoi,
                    panel_aois,
                    tile_aois,
                )

                matrix = build_transition_matrix(labeled, trans_types)

                # SGE/GTE: regroup to the configured entropy categories (same AOIs
                # for both metrics) and drop anything outside those groups.
                entropy_labeled = labeled.copy()
                entropy_labeled["obj_type"] = entropy_labeled["obj_type"].apply(
                    lambda t: regroup_obj_type(t, entropy_groups)
                )
                entropy_labeled = entropy_labeled[entropy_labeled["obj_type"].isin(entropy_types)]
                gte_matrix = build_transition_matrix(entropy_labeled, entropy_types)

                feat_rows.append({
                    **meta,
                    "n_fixations_total": len(fix_df),
                    **{k: v for t in obj_types for k, v in _type_features(labeled, total_dur, t).items()},
                    **{k: v for p in panel_aois for k, v in _panel_features(labeled, total_dur, p["name"]).items()},
                    "sge": _sge(entropy_labeled),
                    "gte": _gte(gte_matrix),
                })

                grid_dir = ROOT / cfg["paths"]["processed"] / "grids"
                grid_dir.mkdir(exist_ok=True)
                grid_stem = f"grid_{sid}_{trial_match}"
                with open(grid_dir / f"{grid_stem}.json", "w") as _f:
                    json.dump(grid_info["grid"], _f)
                pd.DataFrame(grid_info["grid"]).to_csv(
                    grid_dir / f"{grid_stem}.csv", index=False, header=False
                )

                if not labeled.empty:
                    lf = labeled[
                        [
                            "start_ms",
                            "end_ms",
                            "duration_ms",
                            "x",
                            "y",
                            "grid_x",
                            "grid_y",
                            "tile_pixel_x_min",
                            "tile_pixel_x_max",
                            "tile_pixel_y_min",
                            "tile_pixel_y_max",
                            "aoi",
                            "obj_type",
                        ]
                    ].copy()
                    for k, v in meta.items():
                        lf[k] = v
                    fix_rows.append(lf)
                trans_rows.append(
                    {
                        **meta,
                        "n_fixations_total": len(fix_df),
                        **{
                            f"trans_{s}_to_{d}": int(matrix.loc[s, d])
                            for s in trans_types
                            for d in trans_types
                        },
                    }
                )

    processed = ROOT / cfg["paths"]["processed"]

    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(processed / "object_aoi_features.csv", index=False)
    print(f"Saved {len(feat_df)} rows -> object_aoi_features.csv")

    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(processed / "object_aoi_transitions.csv", index=False)
    print(f"Saved {len(trans_df)} rows -> object_aoi_transitions.csv")

    fix_df = pd.concat(fix_rows, ignore_index=True) if fix_rows else pd.DataFrame()
    fix_df.to_csv(processed / "object_aoi_fixations.csv", index=False)
    print(f"Saved {len(fix_df)} rows -> object_aoi_fixations.csv")

    return feat_df, trans_df
