"""Shared plotting style for diary timelines and time-of-day profiles.

The eight most common categories get fixed hues (reference categorical palette, light mode); the
rarest five fold into one gray band so hues stay distinguishable. Color follows the category, never rank.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COLORS = {
    "sleep": "#2a78d6", "leisure_social": "#eb6834", "work": "#1baf7a", "household": "#eda100",
    "eating_drinking": "#e87ba4", "personal_care": "#008300", "travel": "#4a3aa7", "caregiving": "#e34948",
}
FOLDED_COLOR = "#b5b4ae"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "font.size": 9,
                     "text.color": INK, "axes.labelcolor": MUTED})


def palette(categories: list[str]) -> tuple[ListedColormap, list[Patch]]:
    cols = [COLORS.get(c, FOLDED_COLOR) for c in categories]
    folded = [c for c in categories if c not in COLORS]
    handles = [Patch(color=COLORS[c], label=c.replace("_", " ")) for c in COLORS]
    handles.append(Patch(color=FOLDED_COLOR, label="folded:\n" + "\n".join(f.replace("_", " ") for f in folded)))
    return ListedColormap(cols), handles


def hour_axis(ax, n: int, start_clock: int, every: int | None = None) -> None:
    step = every or n // 8
    ticks = np.arange(0, n + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{((start_clock + t * 1440 // n) // 60) % 24:02d}:00" for t in ticks], color=MUTED)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def stacked_profile(ax, seq: np.ndarray, categories: list[str], start_clock: int) -> None:
    """Share of diaries in each category at each bin; folded categories summed into one gray band."""
    share = np.stack([(seq == k).mean(axis=0) for k in range(len(categories))])
    main = [categories.index(c) for c in COLORS]
    rest = [k for k, c in enumerate(categories) if c not in COLORS]
    stacked = np.vstack([share[main], share[rest].sum(axis=0, keepdims=True)])
    x = np.arange(seq.shape[1])
    ax.stackplot(x, stacked, colors=[*COLORS.values(), FOLDED_COLOR], edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlim(0, seq.shape[1] - 1)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    hour_axis(ax, seq.shape[1], start_clock)
    ax.tick_params(axis="y", colors=MUTED)


def timeline_rows(ax, rows: np.ndarray, categories: list[str], start_clock: int, labels: list[str]) -> None:
    cmap, _ = palette(categories)
    ax.imshow(rows, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(categories) - 0.5, interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, color=MUTED, fontsize=7)
    hour_axis(ax, rows.shape[1], start_clock)
