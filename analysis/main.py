from pathlib import Path

import pandas as pd
import yaml
from features.extract_features import extract_features
from features.vs_mot_kmeans import run_vs_mot_kmeans
from model.glmm import run_all as run_glmmsecond
from model.power_analysis import run as run_power_analysis
from prepare_data.data import split_all_data_by_trial
from utils import skip_run

with open("configs/analysis.yml") as f:
    cfg = yaml.safe_load(f)


with skip_run("skip", "split_data_by_trial") as check, check():
    split_all_data_by_trial(cfg)

with skip_run("skip", "vs_mot_classification") as check, check():
    run_vs_mot_kmeans(Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"]))

with skip_run("run", "extract_features") as check, check():
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
