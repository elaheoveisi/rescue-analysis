"""Automatic vs manual AI help per participant, read straight from each XDF.

Participants: the sub: list in configs/analysis.yml. Only the LLM trials
(help_vs_auto.llm_trials) count; automatic calls are reconstructed every
help_vs_auto.auto_interval_steps actions since the last AI call.
Does not touch data.h5. Run from the repo root: python analysis/steps.py
"""

from pathlib import Path

import pandas as pd
import yaml
from features.game_steps import event_timeline, join_trial_games
from prepare_data.data import process_subject

with open("configs/analysis.yml") as f:
    cfg = yaml.safe_load(f)

trials = [str(t) for t in cfg["trials"]]
llm_trials = cfg["help_vs_auto"]["llm_trials"]
interval = cfg["help_vs_auto"]["auto_interval_steps"]

rows, failed = [], []
for sid in [str(s) for s in cfg["sub"]]:
    print(f"Reading {sid}")
    try:
        split = process_subject(sid, cfg)
    except Exception as e:
        failed.append((sid, str(e)))
        continue

    auto = manual = 0
    for trial_id, runs in split.items():
        trial = next((t for t in trials if t in trial_id), None)
        if trial not in llm_trials:
            continue
        timeline = event_timeline(join_trial_games([r["game"] for r in runs.values()]), interval)
        auto += int((timeline["label"] == 0).sum())
        manual += int((timeline["label"] == 1).sum())
    rows.append({"subject": sid, "automatic": auto, "manual": manual, "total": auto + manual})

counts = pd.DataFrame(rows)
total = counts[["automatic", "manual", "total"]].sum()
counts = pd.concat([counts, pd.DataFrame([{"subject": "Total", **total}])], ignore_index=True)

print(f"\nAI help per participant (trials: {', '.join(llm_trials)}; auto interval = {interval} actions)")
print(counts.to_string(index=False))
for sid, reason in failed:
    print(f"Skipped {sid}: {reason}")

out = Path(cfg["paths"]["processed"]) / "help_counts.csv"
counts.to_csv(out, index=False)
print(f"\nSaved {out}")
