"""analysis/prepare_data/data.py — Subject processing and trial iteration."""

from pathlib import Path

import pyxdf


from .h5 import open_store
from .parse import get_stream, split_streams_by_trial, xdf_path


def process_subject(subject_id: str, cfg: dict) -> dict[str, dict[int, dict]]:
    """Parse one subject's XDF and return trial data split by episode.

    Returns:
        {trial_id: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}
    """
    streams, _ = pyxdf.load_xdf(str(xdf_path(subject_id, cfg)))
    game_stream = get_stream(streams, cfg["xdf"]["game_stream"])
    eye_stream = get_stream(streams, cfg["xdf"]["eye_stream"])
    return split_streams_by_trial(game_stream, eye_stream, cfg)


def split_all_data_by_trial(cfg: dict) -> dict:
    """Process all subjects and return a nested dict saved to a single HDF5 file.

    Structure: {subject_id: {trial: {run_num: {"game": DataFrame, "eye_tracking": DataFrame}}}}
    Saved to cfg["paths"]["processed"]/data.h5 with keys
    /{subject_id}/{trial}/run_{n}/game and /{subject_id}/{trial}/run_{n}/eye_tracking.
    """
    subjects = [str(s) for s in cfg.get("sub", [])]
    trials = [str(t) for t in cfg.get("trials", [])]

    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    h5_path = processed_dir / "data.h5"

    result: dict = {}

    with open_store(h5_path) as store:
        for sid in subjects:
            print(f"Reading data of {sid}")
            result[sid] = {}

            for trial_id, runs in process_subject(sid, cfg).items():
                trial = next((t for t in trials if t in trial_id), None)
                if trial is None:
                    continue
                result[sid][trial] = runs
                for run_num, streams in runs.items():
                    store[f"/{sid}/{trial}/run_{run_num}/game"] = streams["game"]
                    store[f"/{sid}/{trial}/run_{run_num}/eye_tracking"] = streams[
                        "eye_tracking"
                    ]

    print(f"Saving the data at {h5_path}")
    return result
