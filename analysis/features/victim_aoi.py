"""Dynamic object AOI analysis using per-frame camera viewport.

For each fixation the nearest game frame is looked up by XDF timestamp.
The camera viewport (cam_top_x/y + cam_view_w/h) maps the visible grid
region onto the game area, converting screen pixels to grid tiles.
Falls back to an agent-centred 13×13 view when camera fields are absent.

Tile encoding (mirrors game observations.py):
  0=empty  1=wall  4=lava  5=victim  6=fake_victim
  10-27=door (DOOR_BASE + color*3 + state)  30-35=key (KEY_BASE + color)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.eye_tracking_features import run_eyetracking

ROOT = Path(__file__).resolve().parents[2]

_WALL, _LAVA, _VICTIM, _FAKE_VICTIM = 1, 4, 5, 6
_DOOR_BASE, _DOOR_END = 10, 28
_KEY_BASE, _KEY_END = 30, 36

# Defined once — used for feature extraction, iteration, and visualization import.
OBJECT_TYPES = ("victim", "fake_victim", "door", "key", "lava")

# Fallback viewport half-size when cam_ fields are absent (mirrors observations.py).
_CAM_FALLBACK_HALF = 6


# ---------------------------------------------------------------------------
# XDF helpers
# ---------------------------------------------------------------------------

def _load_streams(subject_id: str, cfg: dict) -> list:
    raw = cfg["paths"]["raw"]
    xdf = (
        Path(raw)
        / f"sub-{subject_id}/ses-S001/sarmissiong"
        / f"sub-{subject_id}_ses-S001_task-Default_run-001_sarmissiong.xdf"
    )
    streams, _ = pyxdf.load_xdf(str(xdf))
    return streams


def _get_stream(streams: list, name: str) -> dict | None:
    return next((s for s in streams if s["info"]["name"][0] == name), None)


# ---------------------------------------------------------------------------
# Grid extraction — first frame per trial (grid is dropped from HDF5)
# ---------------------------------------------------------------------------

def _extract_trial_grids(game_stream: dict, trial_field: str) -> dict:
    """Return {trial_id: {"grid": [[int]], "victim_health": {...}}}."""
    result = {}
    for v in game_stream["time_series"]:
        try:
            d = json.loads(v[0] if isinstance(v, (list, tuple)) else v)
        except (json.JSONDecodeError, TypeError):
            continue
        tid = d.get(trial_field)
        if tid is None or tid in result or "grid" not in d:
            continue
        result[tid] = {"grid": d["grid"], "victim_health": d.get("victim_health", {})}
    return result


# ---------------------------------------------------------------------------
# Tile → AOI label
# ---------------------------------------------------------------------------

def _tile_to_label(tile: int, gx: int, gy: int) -> str:
    """Position-encoded label, e.g. 'victim_3_5' or 'door_7_2'."""
    if tile == _VICTIM:
        return f"victim_{gx}_{gy}"
    if tile == _FAKE_VICTIM:
        return f"fake_victim_{gx}_{gy}"
    if tile == _LAVA:
        return f"lava_{gx}_{gy}"
    if _DOOR_BASE <= tile < _DOOR_END:
        return f"door_{gx}_{gy}"
    if _KEY_BASE <= tile < _KEY_END:
        return f"key_{gx}_{gy}"
    return "wall" if tile == _WALL else "empty"


def _make_aoi_to_type(panel_names: frozenset):
    """Return a function mapping an AOI label to its type string."""
    def _fn(aoi: str) -> str:
        # Check fake_victim before victim (both start with letters, but keep explicit order)
        for t in OBJECT_TYPES:
            if aoi.startswith(t + "_"):
                return t
        if aoi in panel_names:
            return aoi
        return "offscreen" if aoi == "offscreen" else "other"
    return _fn


# ---------------------------------------------------------------------------
# Camera viewport
# ---------------------------------------------------------------------------

def _cam_bounds(game_row: pd.Series) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of visible viewport in grid coordinates."""
    if "cam_top_x" in game_row.index and pd.notna(game_row.get("cam_top_x")):
        x0, y0 = int(game_row["cam_top_x"]), int(game_row["cam_top_y"])
        return x0, y0, x0 + int(game_row["cam_view_w"]), y0 + int(game_row["cam_view_h"])
    ax, ay = int(game_row.get("agent_x", 0)), int(game_row.get("agent_y", 0))
    h = _CAM_FALLBACK_HALF
    return ax - h, ay - h, ax + h + 1, ay + h + 1


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
) -> pd.DataFrame:
    """Label each fixation using the camera viewport at the time of the fixation.

    Columns added:
      aoi      — position-encoded ('victim_3_5', 'door_7_2', 'info_panel', …)
      obj_type — type prefix ('victim', 'door', 'other', 'offscreen', …)
    """
    df = fix_df.copy()
    df["aoi"] = "offscreen"
    df["obj_type"] = pd.Series(dtype=str)
    df["grid_x"] = np.nan
    df["grid_y"] = np.nan

    if df.empty or game_df.empty:
        return df

    panel_names = frozenset(p["name"] for p in panel_aois)
    aoi_to_type = _make_aoi_to_type(panel_names)

    # Fixation start_ms is ms from the first eye sample; convert to XDF seconds.
    t0_xdf = float(eye_df["timestamp"].iloc[0])
    game_ts = game_df["timestamp"].values.astype(float)
    fix_xdf = t0_xdf + df["start_ms"].values.astype(float) / 1000.0
    frame_idx = np.clip(np.searchsorted(game_ts, fix_xdf), 0, len(game_ts) - 1)

    sx0, sx1 = float(game_aoi["x_min"]), float(game_aoi["x_max"])
    sy0, sy1 = float(game_aoi["y_min"]), float(game_aoi["y_max"])
    grid_h, grid_w = len(grid), len(grid[0]) if grid else 0

    for i, (row_idx, row) in enumerate(df.iterrows()):
        px, py = float(row["x"]), float(row["y"])

        for panel in panel_aois:
            if panel["x_min"] <= px <= panel["x_max"] and panel["y_min"] <= py <= panel["y_max"]:
                df.at[row_idx, "aoi"] = panel["name"]
                break
        else:
            if sx0 <= px <= sx1 and sy0 <= py <= sy1:
                cx0, cy0, cx1, cy1 = _cam_bounds(game_df.iloc[int(frame_idx[i])])
                vw, vh = cx1 - cx0, cy1 - cy0
                if vw > 0 and vh > 0:
                    gx = cx0 + int((px - sx0) / ((sx1 - sx0) / vw))
                    gy = cy0 + int((py - sy0) / ((sy1 - sy0) / vh))
                    gx = max(0, min(gx, grid_w - 1))
                    gy = max(0, min(gy, grid_h - 1))
                    df.at[row_idx, "aoi"] = _tile_to_label(int(grid[gy][gx]), gx, gy)
                    df.at[row_idx, "grid_x"] = gx
                    df.at[row_idx, "grid_y"] = gy

    df["obj_type"] = df["aoi"].apply(aoi_to_type)
    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _panel_features(labeled: pd.DataFrame, total_dur: float, panel_name: str) -> dict:
    on = labeled[labeled["aoi"] == panel_name]
    dur = float(on["duration_ms"].sum()) if not on.empty else 0.0
    return {
        f"n_fixations_{panel_name}": len(on),
        f"total_dur_{panel_name}_ms": dur,
        f"pct_dur_{panel_name}": dur / total_dur if total_dur > 0 else None,
    }


def _type_features(labeled: pd.DataFrame, total_dur: float, t: str) -> dict:
    on = labeled[labeled["aoi"].str.startswith(t + "_")]
    dur = float(on["duration_ms"].sum()) if not on.empty else 0.0
    return {
        f"n_fixations_on_{t}": len(on),
        f"total_dur_on_{t}_ms": dur,
        f"pct_dur_on_{t}": dur / total_dur if total_dur > 0 else None,
        f"n_unique_{t}_fixated": on["aoi"].nunique() if not on.empty else 0,
    }


def _transition_matrix(labeled: pd.DataFrame, trans_types: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=trans_types, columns=trans_types)
    seq = labeled[labeled["obj_type"] != "offscreen"]["obj_type"].tolist()
    for src, dst in zip(seq[:-1], seq[1:]):
        if src in matrix.index and dst in matrix.columns:
            matrix.loc[src, dst] += 1
    return matrix


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
    game_aoi = next(a for a in cfg["aoi"] if a["name"] == "game_area")
    panel_aois = [a for a in cfg["aoi"] if a["name"] != "game_area"]
    panel_names = [a["name"] for a in panel_aois]
    trans_types = list(OBJECT_TYPES) + panel_names + ["other"]
    trial_field = cfg["xdf"]["trial_field"]

    feat_rows, trans_rows, fix_rows = [], [], []

    with pd.HDFStore(str(h5_path), mode="r") as store:
        for sid in [str(s) for s in cfg.get("sub", [])]:
            print(f"Processing {sid}")
            try:
                streams = _load_streams(sid, cfg)
            except FileNotFoundError:
                print(f"  XDF not found for {sid}, skipping")
                continue

            game_stream = _get_stream(streams, cfg["xdf"]["game_stream"])
            if game_stream is None:
                print(f"  No game stream for {sid}, skipping")
                continue

            trial_grids = _extract_trial_grids(game_stream, trial_field)
            trials_cfg = [str(t) for t in cfg.get("trials", [])]

            for eye_key in [k for k in store.keys() if f"/{sid}/" in k and k.endswith("/eye_tracking")]:
                game_key = eye_key.replace("/eye_tracking", "/game")
                if game_key not in store:
                    continue

                parts = eye_key.strip("/").split("/")
                trial_h5, run_num = parts[1], int(parts[2].replace("run_", ""))

                trial_match = next((t for t in trials_cfg if t in trial_h5), None)
                if trial_match is None:
                    continue

                grid_info = next((v for tid, v in trial_grids.items() if trial_match in tid), None)
                if grid_info is None:
                    print(f"  No grid for {sid}/{trial_match}, skipping")
                    continue

                eye_df = store[eye_key]
                game_df = store[game_key]
                fix_df = run_eyetracking(eye_df, cfg)["fixations"]

                meta = {"subject": sid, "trial": trial_match, "run": run_num}
                total_dur = float(fix_df["duration_ms"].sum()) if not fix_df.empty else 0.0

                labeled = label_fixations_dynamic(
                    fix_df, game_df, eye_df, grid_info["grid"], game_aoi, panel_aois
                )

                feat_rows.append({
                    **meta,
                    "n_fixations_total": len(fix_df),
                    **{k: v for t in OBJECT_TYPES for k, v in _type_features(labeled, total_dur, t).items()},
                    **{k: v for p in panel_aois for k, v in _panel_features(labeled, total_dur, p["name"]).items()},
                })

                # Save grid layout once per trial (same map for all subjects).
                grid_path = ROOT / cfg["paths"]["processed"] / "grids" / f"grid_{trial_match}.json"
                if not grid_path.exists():
                    grid_path.parent.mkdir(exist_ok=True)
                    with open(grid_path, "w") as _f:
                        json.dump(grid_info["grid"], _f)

                if not labeled.empty:
                    lf = labeled[["start_ms", "end_ms", "duration_ms", "x", "y", "grid_x", "grid_y", "aoi", "obj_type"]].copy()
                    for k, v in meta.items():
                        lf[k] = v
                    fix_rows.append(lf)

                matrix = _transition_matrix(labeled, trans_types)
                trans_rows.append({
                    **meta,
                    "n_fixations_total": len(fix_df),
                    **{f"trans_{s}_to_{d}": int(matrix.loc[s, d]) for s in trans_types for d in trans_types},
                })

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


if __name__ == "__main__":
    with open(ROOT / "configs" / "analysis.yml") as f:
        cfg = yaml.safe_load(f)
    run_object_aoi(cfg)
