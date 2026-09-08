from pathlib import Path

import pandas as pd
import yaml
from features.extract_features import extract_features
from features.gaze_entropy import run_entropy_grouped
from features.general_gaze_features import build_features_dataset as build_general_gaze_features
from features.victim_aoi import run_object_aoi
from features.vs_mot_kmeans import run_vs_mot_kmeans
from model.glmm import run_all as run_glmmsecond
from model.help_vs_auto import build_features_dataset as build_help_vs_auto_features
from model.help_vs_auto import run_models as run_help_vs_auto_models
from model.power_analysis import run as run_power_analysis
from prepare_data.data import split_all_data_by_trial
from utils import skip_run
from validate_aoi_fixations import run_validation

with open("configs/analysis.yml") as f:
    cfg = yaml.safe_load(f)


with skip_run("skip", "split_data_by_trial") as check, check():
    split_all_data_by_trial(cfg)

with skip_run("skip", "vs_mot_classification") as check, check():
    run_vs_mot_kmeans(Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"]))

with skip_run("skip", "victim_aoi") as check, check():
    run_object_aoi(cfg)

with skip_run("skip", "validate_victim_aoi") as check, check():
    run_validation(cfg)

with skip_run("skip", "gaze_entropy") as check, check():
    run_entropy_grouped(cfg)

with skip_run("skip", "extract_features") as check, check():
    df = extract_features(cfg)
    processed_dir = Path(cfg["paths"]["processed"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / "features_all_subjects.csv"
    df.to_csv(out, index=False)
    print(f"\nfeatures -> {out}")

with skip_run("skip", "mixed_effect_model") as check, check():
    # Read the df
    df = pd.read_csv("data/processed/features_all_subjects.csv")
    glmm_results = run_glmmsecond(cfg, dataframes={"best_features": df})
    if glmm_results is not None and not glmm_results.empty:
        processed_dir = Path(cfg["paths"]["processed"])
        filename = cfg.get("glmm2", {}).get("output_file", "glmm2_results.csv")
        out = processed_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        glmm_results.to_csv(out, index=False)
        print(f"\nglmmsecond -> {out}")

with skip_run("skip", "power_analysis") as check, check():
    run_power_analysis(cfg)

with skip_run("skip", "general_gaze_features") as check, check():
    build_general_gaze_features(cfg)

# with skip_run("skip", "help_vs_auto_features") as check, check():
#     build_help_vs_auto_features(cfg)

with skip_run("run"
"", "help_vs_auto_models") as check, check():
    run_help_vs_auto_models(cfg)
