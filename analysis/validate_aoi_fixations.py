from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

<<<<<<< HEAD
N_SAMPLES = 6   # how many fixations to show
=======
with open(root / "configs" / "analysis.yml") as f:
    cfg = yaml.safe_load(f)

aois = {a["name"]: a for a in cfg["aoi"]}
screen_w = cfg["eyetracker"]["screen_w"]
screen_h = cfg["eyetracker"]["screen_h"]

n_samples = 6  # how many fixations to show
>>>>>>> fdabb0aa26122d117290d0a8bf6d2a116eab47f0


def draw_screen(ax, aois: dict, screen_w: int, screen_h: int):
    """Draw the static screen layout (game area + panels) as background."""
    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)  # y increases downward like screen coords
    ax.set_aspect("equal")

    ax.add_patch(
        patches.Rectangle((0, 0), screen_w, screen_h, linewidth=0, facecolor="#e0e0e0")
    )

    for name, color, edge in [
        ("game_area", "#ddeeff", "#4488cc"),
        ("info_panel", "#f5f5f5", "#888888"),
        ("chat_panel", "#f5f5f5", "#888888"),
    ]:
        a = aois[name]
        ax.add_patch(
            patches.Rectangle(
                (a["x_min"], a["y_min"]),
                a["x_max"] - a["x_min"],
                a["y_max"] - a["y_min"],
                linewidth=1,
                edgecolor=edge,
                facecolor=color,
                alpha=0.5,
                label=name.replace("_", " "),
            )
        )

    ax.set_xticks([])
    ax.set_yticks([])


def plot_validation(
<<<<<<< HEAD
    df: pd.DataFrame,
    aois: dict,
    screen_w: int,
    screen_h: int,
    obj_type_filter: str = "victim",
    n: int = N_SAMPLES,
    show: bool = True,
=======
    df: pd.DataFrame, obj_type_filter: str = "victim", n: int = n_samples
>>>>>>> fdabb0aa26122d117290d0a8bf6d2a116eab47f0
):
    mask = (df["obj_type"] == obj_type_filter) & df["tile_pixel_x_min"].notna()
    victims = df[mask].sample(min(n, mask.sum()), random_state=42)

    if victims.empty:
        print(f"No fixations found for obj_type='{obj_type_filter}'")
        return

    ncols = 3
    nrows = len(victims) // ncols  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 7, nrows * 4.5))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, victims.iterrows()):
        draw_screen(ax, aois, screen_w, screen_h)

        # Green box = the tile the code labeled as victim
        tx0 = row["tile_pixel_x_min"]
        tx1 = row["tile_pixel_x_max"]
        ty0 = row["tile_pixel_y_min"]
        ty1 = row["tile_pixel_y_max"]
        ax.add_patch(
            patches.Rectangle(
                (tx0, ty0),
                tx1 - tx0,
                ty1 - ty0,
                linewidth=2,
                edgecolor="green",
                facecolor="limegreen",
                alpha=0.5,
                label="victim tile box",
            )
        )

        # Red dot = where the eye was looking
        ax.plot(row["x"], row["y"], "ro", markersize=8, label="fixation")

        inside = (tx0 <= row["x"] <= tx1) and (ty0 <= row["y"] <= ty1)
        color = "green" if inside else "red"
        status = "INSIDE ✓" if inside else "OUTSIDE ✗"

        ax.set_title(
            f"{row['subject']} | {row['trial']} | run {row['run']}\n"
            f"AOI: {row['aoi']}   [{status}]",
            color=color,
            fontsize=9,
        )

    for ax in axes[len(victims) :]:
        ax.set_visible(False)

    fig.suptitle(
        f"Validation: does the fixation (red dot) land inside the {obj_type_filter} tile (green box)?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    out = ROOT / "data" / "processed" / f"validate_{obj_type_filter}_fixations.png"
    plt.savefig(out, dpi=150)
    print(f"Saved → {out}")
    if show:
        plt.show()


def run_validation(cfg: dict, obj_type_filter: str = "victim", n: int = N_SAMPLES, show: bool = True):
    """Validate victim AOI fixation labeling against the tile bounding boxes.

    Reads {processed}/object_aoi_fixations.csv (written by ``run_object_aoi``)
    and saves a sanity-check figure to {processed}/validate_{obj_type_filter}_fixations.png.
    """
    fix_csv = ROOT / cfg["paths"]["processed"] / "object_aoi_fixations.csv"
    aois = {a["name"]: a for a in cfg["aoi"]}
    screen_w = cfg["eyetracker"]["screen_w"]
    screen_h = cfg["eyetracker"]["screen_h"]

    df = pd.read_csv(fix_csv)
    print(f"Loaded {len(df)} fixations")
    print(f"obj_type counts:\n{df['obj_type'].value_counts()}\n")
    plot_validation(df, aois, screen_w, screen_h, obj_type_filter=obj_type_filter, n=n, show=show)


if __name__ == "__main__":
    with open(ROOT / "configs" / "analysis.yml") as f:
        cfg = yaml.safe_load(f)
    run_validation(cfg)
