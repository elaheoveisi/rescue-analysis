"""Was the picked-up victim the right one to pick up?

For every real-victim pickup (reward +10 or -20), reconstructs the room
state one raw sample before the pickup and asks the same question the
LLM's own prompt-building logic (mosaic.llm.pathfinding /
mosaic.llm.process_prompts) already answers when giving advice: among all
reachable real victims, which one should the operator have gone for —
the nearest currently-saveable one, or (if none are saveable) the one
with the best remaining margin?

Two things this depends on that are easy to get wrong, both found the
hard way while building this:

1. ``step_count`` in the raw stream is not one-action-per-value — the game
   logs continuously (at the eye-tracker's rate), not on every player
   action, so several distinct actions (e.g. a move immediately followed
   by a pickup) can share one ``step_count``. Keying off ``step_count``
   drops most events and silently reuses stale positions for the rest.
   The fix is to walk the raw, time-ordered sample sequence directly and
   watch for the exact sample where ``reward`` first flips to a pickup
   outcome, using the immediately preceding raw sample (not the previous
   *step*) as the pre-pickup state.

2. A single raw XDF file holds every trial condition for a subject
   back-to-back in one continuous stream, and ``step_count`` restarts
   near zero at the start of each. Splitting only on ``step_count``
   without also detecting these restarts silently merges unrelated
   episodes together (a pickup's "previous state" ends up borrowed from
   a different trial's room layout entirely).

``grid`` and ``victim_health`` are dict/list-valued fields that
``prepare_data.parse.parse_game`` deliberately drops when building the
processed per-run DataFrames (see its list/dict filter) — this is why
this analysis reads the raw XDF directly rather than ``data.h5``.
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd
import pyxdf

DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
TRIAL_MAP = {
    "trial_dummy": "dummy",
    "trial_detailed_gemini": "gemini",
    "trial_detailed_openai": "openai",
}


def _load_mosaic(mosaic_src: str):
    """mosaic (the game engine repo) is a sibling repo, not a declared
    dependency of this one — its pathfinding/observation helpers are the
    only reliable way to reconstruct reachability and margin exactly the
    way the game's own LLM prompt does, so we reuse them directly rather
    than reimplementing the BFS."""
    if mosaic_src not in sys.path:
        sys.path.insert(0, mosaic_src)
    try:
        from mosaic.llm.pathfinding import query_all_objects
        from mosaic.sar.observations import cam_bounds
    except ImportError as e:
        raise ImportError(
            f"Could not import mosaic from {mosaic_src!r}. Set "
            "cfg['paths']['mosaic_src'] to the mosaic repo's 'src' directory."
        ) from e
    return query_all_objects, cam_bounds


def split_episodes(raw: list[dict]) -> list[list[dict]]:
    """Split a raw SARGame sample sequence into per-episode chunks.

    A single XDF file holds every trial back-to-back; ``step_count``
    restarts near zero at the start of each, which is the only boundary
    signal available in the raw stream.
    """
    episodes, cur_ep, last = [], [], None
    for r in raw:
        if last is not None and r["step_count"] < last:
            episodes.append(cur_ep)
            cur_ep = []
        cur_ep.append(r)
        last = r["step_count"]
    if cur_ep:
        episodes.append(cur_ep)
    return episodes


def find_optimal_pickups(
    episode: list[dict], query_all_objects, cam_bounds
) -> list[dict]:
    """One row per real-victim pickup in this episode: what was picked,
    what should have been picked, and whether they match."""
    max_steps = episode[0].get("max_steps") or 1000
    deplete_per_step = 17.5 / max_steps

    rows = []
    prev_reward = 0.0
    for i in range(1, len(episode)):
        cur = episode[i]
        is_pickup = cur["reward"] in (10.0, -20.0) and prev_reward not in (10.0, -20.0)
        prev_reward = cur["reward"]
        if not is_pickup:
            continue

        pre = episode[i - 1]
        ax, ay, adir = pre["agent_x"], pre["agent_y"], pre["agent_dir"]
        dx, dy = DIR_VEC[adir]
        fwd = (ax + dx, ay + dy)
        if pre["grid"][fwd[1]][fwd[0]] != 5:  # not a real victim -> unreliable reconstruction
            continue

        vh = pre["victim_health"]
        obs = {"grid": pre["grid"], "agent_x": ax, "agent_y": ay, "carrying": pre.get("carrying")}
        paths = query_all_objects(obs)
        cx0, cy0, cx1, cy1 = cam_bounds(obs)

        candidates = []
        for key, health in vh.items():
            vx, vy = (int(v) for v in key.split(","))
            pr = paths.get((vx, vy))
            if pr is None or not pr.reachable:
                continue
            is_visible = cx0 <= vx < cx1 and cy0 <= vy < cy1
            steps_until_death = health / deplete_per_step if deplete_per_step > 0 else float("inf")
            margin = steps_until_death - pr.path_length if is_visible else None
            saveable = (margin is not None and margin >= 0) if health > 0 else False
            candidates.append(
                {"pos": (vx, vy), "health": health, "path_length": pr.path_length,
                 "margin": margin, "saveable": saveable}
            )

        saveable_candidates = [c for c in candidates if c["saveable"]]
        best = min(saveable_candidates, key=lambda c: c["path_length"]) if saveable_candidates else None
        if best is None:
            alive = [c for c in candidates if c["health"] > 0]
            best = max(alive, key=lambda c: c["margin"] if c["margin"] is not None else -1e9) if alive else None

        rows.append(
            {
                "step_count": cur["step_count"],
                "picked_health": vh.get(f"{fwd[0]},{fwd[1]}"),
                "reward": cur["reward"],
                "n_candidates": len(candidates),
                "n_saveable": len(saveable_candidates),
                "was_optimal": (best["pos"] == fwd) if best is not None else None,
            }
        )
    return rows


def run_optimal_pickup_analysis(cfg: dict) -> pd.DataFrame:
    query_all_objects, cam_bounds = _load_mosaic(
        cfg.get("paths", {}).get("mosaic_src", "../mosaic/src")
    )

    raw_dir = Path(cfg["paths"]["raw"])
    processed_dir = Path(cfg["paths"]["processed"])
    expertise_map = cfg.get("expertise", {})
    subjects = set(cfg.get("sub") or [])

    files = sorted(glob.glob(str(raw_dir / "sub-*/ses-S001/sarmissiong/sub-*_sarmissiong.xdf")))
    files = [f for f in files if "_old" not in f]
    if subjects:
        files = [f for f in files if any(f"sub-{s}/" in f for s in subjects)]

    all_rows = []
    for fpath in files:
        pid = fpath.split("/sub-")[1].split("/")[0]
        print(f"optimal pickup: processing {pid} ...")
        streams, _ = pyxdf.load_xdf(fpath)
        game = next(s for s in streams if s["info"]["name"][0] == "SARGame")
        raw = [json.loads(x[0]) for x in game["time_series"]]

        run_counters: dict[str, int] = {}
        for episode in split_episodes(raw):
            trial = TRIAL_MAP.get(episode[0].get("trial_id", ""))
            if trial is None:
                continue
            run_counters[trial] = run_counters.get(trial, 0) + 1
            run_num = run_counters[trial]

            for row in find_optimal_pickups(episode, query_all_objects, cam_bounds):
                all_rows.append({"participant": pid, "trial": trial, "run": run_num, **row})

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["expertise"] = df["participant"].map(expertise_map)

    out = processed_dir / "optimal_pickup_analysis.csv"
    df.to_csv(out, index=False)
    print(f"\noptimal pickup analysis -> {out}")
    print(f"total pickup events: {len(df)}")
    print()
    print("optimal-pick rate by trial:")
    print(df.groupby("trial")["was_optimal"].agg(["count", "mean"]).round(3))
    print()
    print("optimal-pick rate by trial x expertise:")
    print(df.groupby(["trial", "expertise"])["was_optimal"].agg(["count", "mean"]).round(3))

    return df
