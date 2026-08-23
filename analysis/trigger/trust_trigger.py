"""Trust-perturbation trigger for trial 3 (LLM lies about victim health).

Final design (after several earlier, more complicated versions):

A single leaky counter drives everything — +1 per correct victim pick, -1 per
wrong pick, floored at 0. Every time it reaches ONSET_STREAK (6), the AI gives
one false health report on the operator's next victim. Because the report is
false, that next pick is a GUARANTEED wrong pick (by design, not a
probabilistic side effect) — so the lie is a single atomic event, not a
sustained "dwell" period. After the forced miss, the AI goes back to truthful
reporting and the same counter starts climbing toward 6 again. This repeats
for the rest of the session — a sawtooth of "recover to 6, lie once, recover
to 6 again, lie again, ...".

This replaced an earlier two-part design (separate leaky onset + a strict
multi-miss "dwell" streak to decide when to stop lying). That design required
defending an unknowable dwell duration (the offset condition, by construction,
almost never completes under truthful data, so there was no way to measure
how long a real lie phase would last) and complicated reward-magnitude and
percentage-based alternatives that all failed in some way (see git history).
Making the lie's consequence deterministic — one false report, one guaranteed
miss — removes the dwell problem entirely: there's nothing left to time out,
so there's nothing left to justify with an assumption about lie severity.

Why 6: every gemini/openai run with a reasonable number of attempts (>=30)
reached an unbroken positive streak of at least 6 at some point — it's the
smallest bar every truthful run demonstrably cleared (min(max_pos_streak) = 6
across 21 qualifying runs). The counter is leaky (not a strict unbroken
streak) because a single slip after 5 good picks shouldn't erase the
progress: tested against a strict version on this dataset, the leaky counter
fires in the exact same set of sessions (same coverage) but earlier (median
~2 attempts sooner, up to 46 in one case), with no downside found.

Scope, both applied throughout:
  - LLM (gemini/openai) conditions only — trial 3 always runs under LLM
    assistance, so dummy-condition data is never used for calibration.
  - P013 excluded — across 41 total LLM-condition attempts (7 short
    sessions, none over 9 attempts), their best-ever streak was 3; they
    never came close to the onset bar even with AI assistance, and their
    sessions are too short for any scoring scheme to help.

Correct/wrong is read off the game's `reward` field (+10 = real victim
rescued, -10 = fake victim picked, -20 = real victim who died from health
decay before rescue — see rescue-grid/src/mosaic/sar/actions.py). The
trigger pools -10 and -20 as "wrong" without weighting by magnitude: tested
splitting them out, since the lie is specifically about health and -10 is a
visually-driven, health-unrelated mistake, but -20 already accounts for 83%
of all wrong picks in this dataset (236/286), so pooling barely dilutes the
signal in practice and the extra complexity isn't worth it.
"""

from pathlib import Path

import pandas as pd

from features.extract_features import load_data_from_h5

CONDITION_COLORS = {
    "dummy": "#4C72B0",
    "gemini": "#C44E52",
    "openai": "#8172B2",
}

LLM_TRIALS = ("gemini", "openai")
EXCLUDED_PARTICIPANTS = ("P013",)

ONSET_STREAK = 6


def build_attempt_sequence(game_data: pd.DataFrame) -> pd.DataFrame:
    """One row per victim-pick attempt (correct or wrong), in step order."""
    d = game_data[["step_count", "reward"]].copy()
    d["step_count"] = pd.to_numeric(d["step_count"], errors="coerce")
    d["reward"] = pd.to_numeric(d["reward"], errors="coerce")
    d = d.dropna().sort_values("step_count")
    if d.empty:
        return pd.DataFrame(columns=["step", "correct"])

    def one_reward_per_step(step_rewards: pd.Series) -> float:
        nonzero = step_rewards[step_rewards.ne(0)]
        return float(nonzero.iloc[-1]) if not nonzero.empty else 0.0

    reward_per_step = d.groupby("step_count")["reward"].apply(one_reward_per_step)
    events = reward_per_step[reward_per_step.ne(0)]

    return pd.DataFrame(
        {
            "step": events.index.astype(int),
            "correct": (events.to_numpy() == 10),
        }
    ).reset_index(drop=True)


def simulate_lie_cycles(
    attempts: pd.DataFrame, onset_streak: int = ONSET_STREAK
) -> list[dict]:
    """Leaky counter (+1 correct, -1 wrong, floored at 0). Each time it hits
    ``onset_streak``, a lie fires and the next real attempt is forced to be a
    miss (its actual recorded outcome is overridden), consuming that attempt
    from the timeline. The counter then resets to 0 and climbs again from the
    following attempt using the real recorded outcomes.

    Returns one dict per lie fired: attempt index and step of the 6th correct
    pick that triggered it, and the step of the forced miss that followed.
    """
    correct = attempts["correct"].to_numpy()
    steps = attempts["step"].to_numpy()
    n = len(correct)

    cycles = []
    c = 0
    i = 0
    while i < n:
        c = c + 1 if correct[i] else max(0, c - 1)
        trigger_idx = i
        i += 1
        if c >= onset_streak:
            forced_idx = i if i < n else None
            cycles.append(
                {
                    "trigger_attempt_idx": trigger_idx,
                    "trigger_step": int(steps[trigger_idx]),
                    "forced_miss_attempt_idx": forced_idx,
                    "forced_miss_step": int(steps[forced_idx]) if forced_idx is not None else None,
                }
            )
            c = 0
            if forced_idx is not None:
                i += 1  # consume the forced-miss attempt; resume climbing after it
    return cycles


def run_lie_cycle_analysis(cfg: dict, onset_streak: int = ONSET_STREAK) -> pd.DataFrame:
    """How many lie cycles (recover to 6 -> forced miss) fit in each LLM
    session, restricted to full-length (~15 min) sessions — the only ones
    representative of a real trial-3 session length."""
    data = load_data_from_h5(cfg)
    processed_dir = Path(cfg["paths"]["processed"])

    rows = []
    for participant, trials in data.items():
        if participant in EXCLUDED_PARTICIPANTS:
            continue
        for trial, runs in trials.items():
            if trial not in LLM_TRIALS:
                continue
            for run_num, streams in runs.items():
                game_data = streams["game"]
                ts = pd.to_numeric(game_data["timestamp"], errors="coerce").dropna()
                if ts.empty:
                    continue
                duration_min = (ts.max() - ts.min()) / 60
                if duration_min < 14:
                    continue
                attempts = build_attempt_sequence(game_data)
                cycles = simulate_lie_cycles(attempts, onset_streak)
                rows.append(
                    {
                        "participant": participant,
                        "trial": trial,
                        "run": run_num,
                        "duration_min": duration_min,
                        "n_attempts": len(attempts),
                        "n_lies": len(cycles),
                    }
                )

    summary = pd.DataFrame(rows)
    out = processed_dir / "lie_cycle_summary.csv"
    summary.to_csv(out, index=False)
    print(f"\nlie cycle summary (LLM conditions only, full-length sessions) -> {out}")
    print(f"rule: onset_streak={onset_streak}, forced miss after each fire\n")

    print(f"n_lies per session: median={summary['n_lies'].median():.1f}  "
          f"IQR=[{summary['n_lies'].quantile(.25):.1f}, {summary['n_lies'].quantile(.75):.1f}]  "
          f"min={summary['n_lies'].min()}  max={summary['n_lies'].max()}  (n={len(summary)} sessions)")

    return summary
