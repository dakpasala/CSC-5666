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

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize import minute_timelines  # noqa: E402
from src.evaluation.plotting import INK, MUTED, hour_axis, palette, plt, stacked_profile  # noqa: E402


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
    fig, ax = plt.subplots(figsize=(10, 4))
    stacked_profile(ax, s, cats, start_clock)
    ax.set_ylabel("share of diaries")
    ax.set_title(f"ATUS {args.year} train split (n={len(train)}, unweighted): activity by time of day",
                 loc="left", color=INK)
    ax.legend(handles=handles[::-1], loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "time_of_day_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
