import pandas as pd
from classification.features import build_features, build_pause_events
from classification.stepwin import build_interval, classify_interval


def run(cfg: dict) -> pd.DataFrame:
    """Pause windows: gaze in the first tau seconds after an action, request (the pause ends in an Alt press) vs non-request."""
    features_df = build_interval(cfg, "pause", build_pause_events, build_features)
    return classify_interval(cfg, cfg["info_request"]["auto_interval_steps"], "pause", features_df, win_col="window_s")
