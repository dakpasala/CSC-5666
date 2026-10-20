"""Plot example diaries (minute-level vs binned) and an aggregate time-of-day activity profile.

Usage:
    python scripts/plot_examples.py --year 2022

Examples are drawn at random from the TRAIN split only. Figures go to outputs/checkpoint1/<year>/
(git-ignored, because the example panels show individual respondent records).

Colors: the eight most common categories get fixed hues; the five rarest are folded into one gray
"other (folded)" band so the categorical palette stays distinguishable. Full detail stays in the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize import minute_timelines  # noqa: E402

# Fixed hue per category (reference categorical palette, light mode); never reassigned by rank.
COLORS = {
    "sleep": "#2a78d6", "leisure_social": "#eb6834", "work": "#1baf7a", "household": "#eda100",
    "eating_drinking": "#e87ba4", "personal_care": "#008300", "travel": "#4a3aa7", "caregiving": "#e34948",
}
FOLDED_COLOR = "#b5b4ae"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def palette(categories: list[str]) -> tuple[ListedColormap, list[Patch]]:
    cols = [COLORS.get(c, FOLDED_COLOR) for c in categories]
    folded = [c for c in categories if c not in COLORS]
    handles = [Patch(color=COLORS[c], label=c.replace("_", " ")) for c in COLORS]
    handles.append(Patch(color=FOLDED_COLOR, label="folded:\n" + "\n".join(f.replace("_", " ") for f in folded)))
    return ListedColormap(cols), handles


def hour_axis(ax, n: int, start_clock: int) -> None:
    ticks = np.arange(0, n + 1, n // 8)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{((start_clock + t * 1440 // n) // 60) % 24:02d}:00" for t in ticks], color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2022)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/data.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    proc = ROOT / cfg["processed_dir"] / str(args.year)
    out = ROOT / "outputs" / "checkpoint1" / str(args.year)
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(proc / "sequences.npz", allow_pickle=False)
    cats = z["categories"].tolist()
    start_clock = int(z["diary_start_clock_minute"])
    cmap, handles = palette(cats)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9,
                         "text.color": INK, "axes.labelcolor": MUTED})

    # --- Example diaries: raw minute timeline above its binned version --------------------------
    train = np.flatnonzero(z["split"] == ("train" if args.year == cfg["train_year"] else "test"))
    rng = np.random.default_rng(cfg["plots"]["seed"])
    pick = np.sort(rng.choice(train, size=min(cfg["plots"]["n_examples"], len(train)), replace=False))
    ids = z["case_ids"][pick].tolist()
    ep = pd.read_parquet(proc / "episodes.parquet")
    minutes = minute_timelines(ep, ids, cfg["diary_minutes"])
    seq = z["seq"][pick]
    w = int(z["bin_minutes"])

    fig, axes = plt.subplots(len(ids), 1, figsize=(10, 0.55 * len(ids) + 1.4), sharex=True)
    for i, ax in enumerate(np.atleast_1d(axes)):
        img = np.vstack([minutes[i], np.repeat(seq[i], w)])
        ax.imshow(img, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(cats) - 0.5, interpolation="nearest")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["minute", f"{w}-min"], color=MUTED, fontsize=7)
        ax.set_ylabel(f"#{i + 1}", rotation=0, ha="right", va="center", color=INK)
        hour_axis(ax, cfg["diary_minutes"], start_clock)
    fig.suptitle(f"ATUS {args.year}: {len(ids)} random training diaries, raw vs binned "
                 f"(window starts {start_clock // 60:02d}:00)", x=0.01, ha="left", color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    fig.savefig(out / "example_diaries.png", dpi=150)
    plt.close(fig)

    # --- Aggregate profile: share of diaries in each category at each bin (train split) ------
    s = z["seq"][train]
    share = np.stack([(s == k).mean(axis=0) for k in range(len(cats))])
    order = [cats.index(c) for c in COLORS] + [k for k, c in enumerate(cats) if c not in COLORS]
    stacked = np.vstack([share[order[:len(COLORS)]], share[order[len(COLORS):]].sum(axis=0, keepdims=True)])
    x = np.arange(s.shape[1])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.stackplot(x, stacked, colors=[COLORS[c] for c in COLORS] + [FOLDED_COLOR],
                 edgecolor=SURFACE, linewidth=0.6)
    ax.set_xlim(0, s.shape[1] - 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of diaries")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    hour_axis(ax, s.shape[1], start_clock)
    ax.tick_params(axis="y", colors=MUTED)
    ax.set_title(f"ATUS {args.year} train split (n={len(train)}, unweighted): activity by time of day",
                 loc="left", color=INK)
    ax.legend(handles=handles[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "time_of_day_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
