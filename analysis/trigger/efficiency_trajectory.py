"""Within-run trajectories and knee-point characterization for two candidate
proficiency signals:

- ``efficiency``: cumulative saved_victims(t) / t. Blind to wrong picks.
- ``pick_accuracy``: cumulative correct picks / cumulative pick attempts,
  where an "attempt" is any reward event (+10 real victim, -10 light penalty,
  -20 heavy penalty). This folds in the false-victim / error dimension that
  ``efficiency`` misses entirely, and two runs with identical saved_victims
  can have very different pick_accuracy depending on how many wrong picks
  they made along the way.

For each run and each metric we record the normalized step after which the
metric never again drops below a fraction of its own end-of-run value
(settling time), so we can see empirically where a natural "knee" sits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from features.extract_features import load_data_from_h5
from visualization.visualize_efficiency_trajectory import plot_knee_curve

DEFAULT_FRACTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def build_trajectory(game_data: pd.DataFrame) -> pd.DataFrame:
    """Per-step cumulative trajectory of both proficiency signals across a run."""
    d = game_data[["step_count", "saved_victims", "reward"]].copy()
    d["step_count"] = pd.to_numeric(d["step_count"], errors="coerce")
    d["saved_victims"] = pd.to_numeric(d["saved_victims"], errors="coerce")
    d["reward"] = pd.to_numeric(d["reward"], errors="coerce")
    d = d.dropna(subset=["step_count", "saved_victims"]).sort_values("step_count")
    if d.empty:
        return pd.DataFrame(
            columns=["step", "saved_victims", "efficiency", "pick_accuracy"]
        )

    # one reward event per step, matching game_features.extract_game_features
    def one_reward_per_step(step_rewards: pd.Series) -> float:
        nonzero = step_rewards[step_rewards.ne(0)].dropna()
        if not nonzero.empty:
            return float(nonzero.iloc[-1])
        rest = step_rewards.dropna()
        return float(rest.iloc[-1]) if not rest.empty else 0.0

    saved_per_step = d.groupby("step_count")["saved_victims"].max()
    reward_per_step = d.groupby("step_count")["reward"].apply(one_reward_per_step)

    start = max(int(saved_per_step.index.min()), 1)
    full_index = np.arange(start, int(saved_per_step.index.max()) + 1)
    saved_per_step = saved_per_step.reindex(full_index).ffill()
    reward_per_step = reward_per_step.reindex(full_index).fillna(0.0)

    pos10 = (reward_per_step == 10).astype(int).cumsum()
    neg10 = (reward_per_step == -10).astype(int).cumsum()
    neg20 = (reward_per_step == -20).astype(int).cumsum()
    attempts = pos10 + neg10 + neg20

    steps = full_index
    attempts_arr = attempts.to_numpy()
    pick_accuracy = np.full(attempts_arr.shape, np.nan)
    has_attempts = attempts_arr > 0
    pick_accuracy[has_attempts] = (
        pos10.to_numpy()[has_attempts] / attempts_arr[has_attempts]
    )

    return pd.DataFrame(
        {
            "step": steps,
            "saved_victims": saved_per_step.to_numpy(),
            "efficiency": saved_per_step.to_numpy() / steps,
            "pick_accuracy": pick_accuracy,
            "net_reward_rate": (10 * pos10 - 10 * neg10 - 20 * neg20).to_numpy()
            / steps,
        }
    )


def settling_fracs(
    steps: np.ndarray, values: np.ndarray, fractions: list[float]
) -> dict[float, float | None]:
    """For each fraction of the metric's final value, the normalized step after
    which it never again drops below that fraction (settling time, not first
    transient crossing).

    Raw first-crossing is unusable here: these cumulative ratios peak early
    whenever the first event happens quickly, then decay as steps accumulate
    without a repeat event. So "first time >= threshold" mostly fires at that
    early spike rather than at a genuine stabilization point.
    """
    if len(values) == 0 or not np.isfinite(values[-1]) or values[-1] <= 0:
        return dict.fromkeys(fractions, None)

    max_step = steps.max()
    final = values[-1]
    result: dict[float, float | None] = {}
    for frac in fractions:
        below = values < frac * final
        settle_step = steps[below][-1] + 1 if below.any() else steps[0]
        result[frac] = float(settle_step / max_step)
    return result


def summarize_run(
    participant: str,
    trial: str,
    run: int,
    game_data: pd.DataFrame,
    fractions: list[float],
) -> dict | None:
    trajectory = build_trajectory(game_data)
    if trajectory.empty:
        return None

    row = {
        "participant": participant,
        "trial": trial,
        "run": run,
        "max_step": int(trajectory["step"].max()),
        "e_max": float(trajectory["efficiency"].iloc[-1]),
        "acc_max": float(trajectory["pick_accuracy"].iloc[-1])
        if pd.notna(trajectory["pick_accuracy"].iloc[-1])
        else None,
        "net_reward_rate_final": float(trajectory["net_reward_rate"].iloc[-1]),
    }

    eff_settled = settling_fracs(
        trajectory["step"].to_numpy(), trajectory["efficiency"].to_numpy(), fractions
    )
    row.update({f"eff_frac_step_to_{int(f * 100)}": eff_settled[f] for f in fractions})

    acc_trajectory = trajectory.dropna(subset=["pick_accuracy"])
    acc_settled = settling_fracs(
        acc_trajectory["step"].to_numpy(),
        acc_trajectory["pick_accuracy"].to_numpy(),
        fractions,
    )
    row.update({f"acc_frac_step_to_{int(f * 100)}": acc_settled[f] for f in fractions})

    return row


def run_efficiency_trajectory_analysis(
    cfg: dict, fractions: list[float] = DEFAULT_FRACTIONS
) -> pd.DataFrame:
    data = load_data_from_h5(cfg)
    processed_dir = Path(cfg["paths"]["processed"])

    rows = []
    for participant, trials in data.items():
        for trial, runs in trials.items():
            for run_num, streams in runs.items():
                row = summarize_run(
                    participant, trial, run_num, streams["game"], fractions
                )
                if row is not None:
                    rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    out = processed_dir / "efficiency_trajectory_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nefficiency trajectory summary -> {out}")

    plot_knee_curve(
        summary,
        fractions,
        prefix="eff_frac_step_to_",
        ylabel="Normalized settling step (step / max_step)",
        title="Search-efficiency knee (victims / step)",
        output_path=processed_dir / "efficiency_knee.png",
    )
    plot_knee_curve(
        summary,
        fractions,
        prefix="acc_frac_step_to_",
        ylabel="Normalized settling step (step / max_step)",
        title="Pick-accuracy knee (correct / attempted picks)",
        output_path=processed_dir / "pick_accuracy_knee.png",
    )
    return summary
