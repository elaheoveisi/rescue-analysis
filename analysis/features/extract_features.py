from pathlib import Path

import pandas as pd

from .aoi_fixation import build_aoi_features, run_aoi
from .eye_tracking_features import build_eye_features, run_eyetracking
from .game_features import extract_game_features


def load_data_from_h5(cfg: dict) -> dict:
    """Read data.h5 and return the same nested dict as split_all_data_by_trial."""
    h5_path = Path(cfg["paths"]["processed"]) / "data.h5"
    data: dict = {}
    with pd.HDFStore(str(h5_path), mode="r") as store:
        for key in store.keys():
            _, sid, trial, run_dir, stream = key.split("/")
            run_num = int(run_dir.replace("run_", ""))
            data.setdefault(sid, {}).setdefault(trial, {}).setdefault(run_num, {})[
                stream
            ] = store[key]
    return data


def extract_features(cfg: dict) -> pd.DataFrame:
    """Extract per-trial features for all subjects and trials.

    Args:
        data: {sub: {trial_id: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}}
        cfg:  Full config dict.

    Returns:
        DataFrame with one row per run.
    """
    all_aois = cfg.get("aoi", [])
    # Static AOIs only — dynamic (game_area) is handled by victim_aoi.py.
    aois = [a for a in all_aois if a.get("type", "static") == "static"]
    offscreen_label = cfg.get("analysis", {}).get("offscreen_label", "offscreen")
    eye_cfg = cfg.get("eyetracker", {})
    expertise = cfg.get("expertise", {})
    processed_dir = Path(cfg["paths"]["processed"])

    # Read the data.h5 here
    data = load_data_from_h5(cfg)

    rows = []
    for sub, trials in data.items():
        for trial_id, runs in trials.items():
            for run_num, streams in runs.items():
                game_data = streams["game"]
                eye_data = streams["eye_tracking"]

                et = run_eyetracking(eye_data, cfg)
                aoi = run_aoi(et["fixations"], aois)

                eye_features = build_eye_features(
                    et["fixations"], et["saccades"], eye_data, eye_cfg
                )
                aoi_features = build_aoi_features(
                    aoi["fix_aoi"], aoi["transitions"], aois, offscreen_label
                )
                game_features = extract_game_features(game_data)

                row = {
                    "participant": sub,
                    "trial": trial_id,
                    "run": run_num,
                    "expertise": expertise.get(sub, "unknown"),
                    **eye_features,
                    **aoi_features,
                    **game_features,
                }

                print(
                    f"  {sub}  {trial_id}  run{run_num}  "
                    f"victims={row.get('saved_victims', '?')}  "
                    f"fixations={row.get('n_fixations', '?')}  "
                    f"saccades={row.get('n_saccades', '?')}"
                )

                csv_dir = processed_dir / sub / trial_id
                csv_dir.mkdir(parents=True, exist_ok=True)
                et["fixations"].to_csv(
                    csv_dir / f"run_{run_num}_fixations.csv", index=False
                )
                et["saccades"].to_csv(
                    csv_dir / f"run_{run_num}_saccades.csv", index=False
                )
                pd.DataFrame([{**eye_features, **aoi_features}]).to_csv(
                    csv_dir / f"run_{run_num}_eye_features.csv", index=False
                )
                pd.DataFrame([game_features]).to_csv(
                    csv_dir / f"run_{run_num}_game_features.csv", index=False
                )

                rows.append(row)

    return pd.DataFrame(rows)
