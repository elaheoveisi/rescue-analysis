import pandas as pd

from .game_steps import action_timestamps


def extract_game_features(game_data) -> dict:
    """Observed actions/rewards; missing terminal or inter-frame actions stay unknown."""
    action_timestamps(game_data)
    frame = game_data.sort_values("timestamp", kind="stable").copy()
    frame["total_steps"] = pd.to_numeric(frame["total_steps"])
    actions = frame.loc[frame["action"].notna() & frame["total_steps"].gt(0)]
    actions = actions.drop_duplicates("total_steps", keep="first")
    rewards = pd.to_numeric(actions.get("reward", pd.Series(dtype=float)), errors="coerce")
    saved = pd.to_numeric(frame["saved_victims"], errors="coerce").max()
    n_actions = len(actions)
    return {
        "n_actions": n_actions,
        "n_llm_calls": int(frame["llm_request_id"].nunique()) if "llm_request_id" in frame else None,
        "saved_victims": int(saved) if pd.notna(saved) else None,
        "mean_reward": float(rewards.mean()) if rewards.notna().any() else None,
        "total_reward": float(rewards.sum()) if rewards.notna().any() else None,
        "victims_per_step": saved / n_actions if n_actions and pd.notna(saved) else None,
    }
