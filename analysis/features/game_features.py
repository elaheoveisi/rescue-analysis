import pandas as pd


def extract_game_features(game_data) -> dict:
    if "reward" not in game_data.columns or "step_count" not in game_data.columns:
        deduped_reward = pd.Series(dtype=float)
    else:
        reward_df = game_data[["step_count", "reward"]].copy()
        reward_df["reward"] = pd.to_numeric(reward_df["reward"], errors="coerce")
        reward_df["step_count"] = pd.to_numeric(
            reward_df["step_count"], errors="coerce"
        )
        reward_df = reward_df.dropna(subset=["step_count"])

        def one_reward_per_step(step_rewards: pd.Series) -> float:
            nonzero = step_rewards[step_rewards.ne(0)].dropna()
            if not nonzero.empty:
                return float(nonzero.iloc[-1])
            return float(step_rewards.dropna().iloc[-1])

        deduped_reward = reward_df.groupby("step_count", sort=False)["reward"].apply(
            one_reward_per_step
        )

    max_steps = game_data["step_count"].max()
    saved_victims = int(game_data["saved_victims"].max())

    return {
        "n_actions": int(game_data["action"].notna().sum()),
        "n_llm_calls": int(game_data["llm_response"].notna().sum())
        if "llm_response" in game_data.columns
        else None,
        "saved_victims": saved_victims,
        "mean_reward": float(deduped_reward.mean())
        if not deduped_reward.empty
        else None,
        "total_reward": float(deduped_reward.sum())
        if not deduped_reward.empty
        else None,
        "victims_per_step": (saved_victims / max_steps) if max_steps else None,
    }
