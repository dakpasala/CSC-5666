"""Figures for the report and presentation, written to visuals/. 2022 validation data only (no test year is read).

Usage:
    OMP_NUM_THREADS=1 python scripts/make_visuals.py            # all figures
    OMP_NUM_THREADS=1 python scripts/make_visuals.py --only 2 5  # a subset (figure numbers)

Models: v3 = the 40-epoch v1-design model (seed 42 for single-model figures); final = M_big2 (1.09M parameters,
trained on 2022 train + 2023 + 2024), seeds 42-44 (ensemble A) and 45-47 (ensemble B). Probes, PCA and k-means are
fitted on 2022 training days; everything shown is computed on 2022 validation days.

Colors: the eight most common activities take the reference categorical palette in fixed order; the remaining
five fold into a gray "other". Sequential encodings use the single-hue blue ramp.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation import v3_metrics as M  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402
from src.models.vae_v2 import exact_budget_decode, integer_budgets  # noqa: E402

OUT = ROOT / "visuals"
V3 = "outputs/checkpoint5_robustness/guided_l0.3_s42-*"
FINAL = {s: f"outputs/v3/M_big2_s{s}-*" for s in (42, 43, 44, 45, 46, 47)}
ENS = {"A": (42, 43, 44), "B": (45, 46, 47)}

# Reference palette (light mode): categorical slots 1-8 in fixed order, text inks, surface, sequential blue ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER_GRAY = "#b9b8b3"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
SHOWN = ["sleep", "leisure_social", "work", "household", "eating_drinking", "travel", "personal_care", "caregiving"]
LABEL = {"sleep": "Sleep", "leisure_social": "Leisure & social", "work": "Work", "household": "Household",
         "eating_drinking": "Eating & drinking", "travel": "Travel", "personal_care": "Personal care",
         "caregiving": "Caregiving", "other_folded": "Other (5 categories)"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "legend.frameon": False})


def latest(pat: str) -> Path:
    runs = sorted(p for p in glob.glob(str(ROOT / pat)) if Path(p, "best.pt").exists())
    if not runs:
        raise FileNotFoundError(pat)
    return Path(runs[-1])


class Data:
    def __init__(self) -> None:
        z = np.load(ROOT / "data/processed/2022/sequences.npz")
        self.cats = [str(c) for c in z["categories"]]
        self.seq, self.split = z["seq"].astype(np.int64), z["split"].astype(str)
        self.tr, self.va = self.split == "train", self.split == "val"
        self.start = int(z["diary_start_clock_minute"])
        # Fold to 9 display classes: the 8 shown activities, then "other".
        fold = np.full(len(self.cats), len(SHOWN))
        for i, c in enumerate(SHOWN):
            fold[self.cats.index(c)] = i
        self.fold = fold
        self.cmap = ListedColormap(SERIES + [OTHER_GRAY])


def time_axis(ax, d: Data, n: int = 288, labels: bool = True) -> None:
    ticks = np.arange(0, n + 1, 48)
    ax.set_xticks(ticks - 0.5)
    ax.set_xticklabels([f"{((d.start + t * 5) // 60) % 24:d}:00" for t in ticks] if labels else [])
    ax.tick_params(length=0)


def legend_handles(d: Data) -> list:
    return [Patch(color=c, label=LABEL[a]) for c, a in zip(SERIES, SHOWN)] + [Patch(color=OTHER_GRAY, label=LABEL["other_folded"])]


def sleep_midpoint(seq: np.ndarray, sleep_idx: int, start: int) -> np.ndarray:
    m = seq == sleep_idx
    ang = 2 * np.pi * ((start + 5 * np.arange(seq.shape[1]) + 2.5) % 1440) / 1440
    return np.degrees(np.arctan2((m * np.sin(ang)).sum(1), (m * np.cos(ang)).sum(1))) % 360 / 360 * 24


@torch.no_grad()
def decode_days(model, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = model.encode(torch.from_numpy(seq).long())
    logp = torch.log_softmax(model.decode(q["amount_mu"], q["timing_mu"]), -1).numpy()
    return logp.argmax(-1), exact_budget_decode(logp, integer_budgets(np.exp(logp).mean(1), logp.shape[1]))


def save(fig, name: str) -> None:
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote visuals/{name}", flush=True)


# ----------------------------------------------------------------------------------------------- figure 1
def fig_architecture(_d: Data) -> None:
    fig, ax = plt.subplots(figsize=(12.3, 6))
    ax.set_xlim(0, 123)
    ax.set_ylim(0, 60)
    ax.axis("off")
    orange_fill, aqua_fill, blue_fill = "#fdeee7", "#e6f7f0", "#eef4fd"

    def box(x, y, w, h, title, sub, fill=blue_fill, edge=SERIES[0]):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fill, ec=edge, lw=1.2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 4.2) / 2, sub, ha="center", va="center", fontsize=7, color=INK2, linespacing=1.3)

    def arrow(pts, text=None, at=None):
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            last = (x1, y1) == pts[-1]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>" if last else "-", mutation_scale=11,
                                         lw=1.2, color=INK2, shrinkA=0, shrinkB=0))
        if text:
            ax.text(*at, text, ha="center", va="top", fontsize=6.8, color=MUTED)

    ax.text(1, 57, "Encoder: reads the whole day", fontsize=9.5, fontweight="bold", color=SERIES[0])
    w, h, y = 19, 11, 41
    xs = [1, 25, 49, 73]
    box(xs[0], y, w, h, "Input day", "288 five-minute slots\n4 a.m. to 4 a.m.\none activity id per slot")
    box(xs[1], y, w, h, "Embedding + clock", "32 learned numbers\nper activity, plus sin\nand cos of time of day")
    box(xs[2], y, w, h, "Bidirectional GRU", "2 layers, 192 per\ndirection: 384 numbers\nper slot, in context")
    box(xs[3], y, w, h, "Average over slots", "one 384-number\nsummary of the\nwhole day")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow([(a + w + 0.6, y + h / 2), (b - 0.6, y + h / 2)])

    box(98, 47, 21, 9, "Amount head", "linear: mean + spread\n8 numbers", fill=orange_fill, edge=SERIES[1])
    box(98, 35, 21, 9, "Timing head", "linear: mean + spread\n16 numbers", fill=aqua_fill, edge=SERIES[2])
    arrow([(xs[3] + w + 0.6, y + h / 2 + 1), (97.4, 51.5)])
    arrow([(xs[3] + w + 0.6, y + h / 2 - 1), (97.4, 39.5)])

    ax.text(1, 29, "Decoder: rebuilds the day from the two vectors only", fontsize=9.5, fontweight="bold", color=SERIES[0])
    yd = 13
    box(98, yd, 21, 11, "Vectors z", "training: μ + σ·ε (sampled)\nafter training: the means\n8 amount + 16 timing")
    # Both heads feed the vectors: join their right edges and run down into the vector box.
    arrow([(119.6, 51.5), (121.5, 51.5), (121.5, yd + 5.5), (119.8, yd + 5.5)])
    ax.plot([119.6, 121.5], [39.5, 39.5], color=INK2, lw=1.2)
    box(73, yd, 19, 11, "Decoder, per slot", "3 layers × 256; the same\n24 numbers at every slot\n+ 16 clock features")
    arrow([(97.4, yd + 5.5), (92.6, yd + 5.5)])
    box(49, yd, 19, 11, "Probabilities", "13 activities\nfor each of the\n288 slots")
    arrow([(72.4, yd + 5.5), (68.6, yd + 5.5)])
    box(25, yd, 19, 11, "Budgeted decoding", "expected slots per\nactivity, then exact\nassignment")
    arrow([(48.4, yd + 5.5), (44.6, yd + 5.5)])
    box(1, yd, 19, 11, "Rebuilt day", "288 slots, short\nactivities keep\ntheir time")
    arrow([(24.4, yd + 5.5), (20.6, yd + 5.5)])

    box(98, 0.5, 21, 9, "Amount-head MLP", "reads the 8 amount numbers\n8 → 32 → 13 shares", fill=orange_fill, edge=SERIES[1])
    arrow([(108.5, yd - 0.6), (108.5, 10.1)])

    ax.text(1, 5, "Training loss = reconstruction + β·(KL amount + KL timing)/288 + 0.3 · pair rule + 1.0 · amount-head error\n"
                  "Pair rule: each training day is paired with a shifted or block-swapped copy with identical totals,\n"
                  "and the distance between their amount means is penalized.", fontsize=7.3, color=INK2, va="center",
            linespacing=1.4)
    ax.set_title("Final model architecture (about 1.09 million parameters)", loc="left")
    save(fig, "fig1_architecture.png")


# ----------------------------------------------------------------------------------------------- figure 2
def fig_reconstructions(d: Data) -> None:
    v3, _ = load_checkpoint(latest(V3) / "best.pt")
    fin, _ = load_checkpoint(latest(FINAL[42]) / "best.pt")
    va_idx = np.flatnonzero(d.va)
    cnt = category_counts(d.seq[va_idx], len(d.cats))
    work = cnt[:, d.cats.index("work")] * 5
    rng = np.random.default_rng(3)
    # Two work days and two non-work days, chosen at random (not cherry-picked by error).
    pick = np.concatenate([rng.choice(va_idx[work >= 360], 2, replace=False), rng.choice(va_idx[work == 0], 2, replace=False)])
    days = d.seq[pick]
    v3_arg, v3_bud = decode_days(v3, days)
    _, fin_bud = decode_days(fin, days)
    rows = ["Real day", "v3, most likely activity", "v3, budgeted", "Final model, budgeted"]
    fig, axes = plt.subplots(len(pick), 1, figsize=(10, 1.25 * len(pick) * 1.45), sharex=True)
    for k, ax in enumerate(axes):
        block = np.stack([days[k], v3_arg[k], v3_bud[k], fin_bud[k]])
        ax.imshow(d.fold[block], aspect="auto", cmap=d.cmap, vmin=-0.5, vmax=len(SHOWN) + 0.5, interpolation="nearest")
        ax.set_yticks(range(4))
        ax.set_yticklabels(rows, fontsize=7.5)
        for y in np.arange(0.5, 3.5):
            ax.axhline(y, color=SURFACE, lw=2)
        ax.spines[:].set_visible(False)
        ax.set_title(f"Validation day {k + 1} ({'work day' if k < 2 else 'non-work day'})", loc="left", fontsize=8.5)
        time_axis(ax, d, labels=k == len(pick) - 1)
    fig.legend(handles=legend_handles(d), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.suptitle("Rebuilding held-out days: v3 versus the final model", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    save(fig, "fig2_reconstructions.png")


# ----------------------------------------------------------------------------------------------- figure 3
def fig_pair_test(d: Data) -> None:
    # Validation means over six seeds, from docs/v3_final_val.md and docs/v3_mbig2_final_val.md.
    vals = {"v3": {"Shift": (0.019, 0.370), "Block swap": (0.029, 0.290)},
            "Final model": {"Shift": (0.015, 0.406), "Block swap": (0.025, 0.425)}}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    w = 0.34
    for ax, (model, v) in zip(axes, vals.items()):
        x = np.arange(len(v))
        a = [v[t][0] for t in v]
        t = [v[t][1] for t in v]
        ax.bar(x - w / 2 - 0.01, a, w, color=SERIES[1], label="Amount vector")
        ax.bar(x + w / 2 + 0.01, t, w, color=SERIES[2], label="Timing vector")
        for xi, ai, ti in zip(x, a, t):
            ax.text(xi - w / 2 - 0.01, ai + 0.01, f"{ai:.3f}", ha="center", fontsize=7.5, color=INK)
            ax.text(xi + w / 2 + 0.01, ti + 0.01, f"{ti:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(list(v))
        ax.set_title(model, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Movement\n(1.0 = gap between two random days)")
    axes[0].set_ylim(0, 0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.9), fontsize=8)
    fig.suptitle("Rearranging a day without changing its totals moves the timing vector, not the amount vector",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "fig3_pair_test.png")


# ----------------------------------------------------------------------------------------------- shared embeddings
def final_embeddings(d: Data) -> dict[int, dict[str, np.ndarray]]:
    out = {}
    for s, pat in FINAL.items():
        model, _ = load_checkpoint(latest(pat) / "best.pt")
        out[s] = posterior_means(model, torch.from_numpy(d.seq).long())
        print(f"  encoded with final model seed {s}", flush=True)
    return out


# ----------------------------------------------------------------------------------------------- figure 4
def fig_embeddings(d: Data, emb: dict) -> None:
    e = emb[42]
    mid = sleep_midpoint(d.seq, d.cats.index("sleep"), d.start)
    mid = (mid + 12) % 24 - 12          # hours relative to midnight: negative = before midnight
    work = category_counts(d.seq, len(d.cats))[:, d.cats.index("work")] * 5 / 60
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, colour, clabel, title in (
            (axes[0], "timing_mu", np.clip(mid, 0, 8), "Sleep midpoint (hours after midnight)", "Timing vector"),
            (axes[1], "amount_mu", np.clip(work, 0, 12), "Hours of work that day", "Amount vector")):
        p = PCA(2, random_state=0).fit(e[key][d.tr])
        xy = p.transform(e[key][d.va])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=colour[d.va], cmap=SEQ, s=9, lw=0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(clabel, color=INK2)
        cb.outline.set_visible(False)
        ax.set_title(f"{title}: first two principal components", loc="left")
        ax.set_xlabel(f"PC1 ({p.explained_variance_ratio_[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({p.explained_variance_ratio_[1]:.0%})")
        ax.tick_params(length=0)
    fig.suptitle("What each vector encodes (final model, seed 42, 1,627 validation days)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig4_embeddings.png")


# ----------------------------------------------------------------------------------------------- figures 5 and 6
def ensemble_labels(d: Data, emb: dict) -> tuple[dict, dict]:
    labels, reps = {}, {}
    for name, seeds in ENS.items():
        R = np.hstack([emb[s]["timing_mu"] for s in seeds])
        reps[name] = R
        labels[name] = M.kmeans_labels(R[d.tr], R[d.va], 6, 0, scale=False)
    # Relabel B's groups to best match A's, for display only (ARI is unaffected by names).
    ct = np.zeros((6, 6))
    np.add.at(ct, (labels["A"], labels["B"]), 1)
    r, c = linear_sum_assignment(-ct)
    remap = np.empty(6, int)
    remap[c] = r
    labels["B"] = remap[labels["B"]]
    # Order groups by sleep midpoint of ensemble A's groups (earliest first) for a readable figure.
    mid = sleep_midpoint(d.seq[d.va], d.cats.index("sleep"), d.start)
    mid = (mid + 12) % 24 - 12
    order = np.argsort([mid[labels["A"] == g].mean() for g in range(6)])
    rank = np.empty(6, int)
    rank[order] = np.arange(6)
    return {k: rank[v] for k, v in labels.items()}, reps


def profile(d: Data, days: np.ndarray) -> np.ndarray:
    f = d.fold[days]
    return np.stack([(f == k).mean(0) for k in range(len(SHOWN) + 1)])


def fig_groups(d: Data, labels: dict) -> None:
    sv = d.seq[d.va]
    ari = adjusted_rand_score(labels["A"], labels["B"])
    fig, axes = plt.subplots(6, 2, figsize=(10, 9.5), sharex=True, sharey=True)
    x = np.arange(288)
    for j, name in enumerate(("A", "B")):
        for g in range(6):
            ax = axes[g, j]
            m = labels[name] == g
            ax.stackplot(x, profile(d, sv[m]), colors=SERIES + [OTHER_GRAY], lw=0)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
            ax.text(-4, 0.5, f"G{g + 1}\n{m.mean():.0%}", ha="right", va="center", fontsize=7.5, color=INK2)
            time_axis(ax, d, labels=g == 5)
        axes[0, j].set_title(f"Ensemble {name} (seeds {', '.join(map(str, ENS[name]))})", loc="left")
    fig.legend(handles=legend_handles(d), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=7.5)
    fig.suptitle(f"Six groups from two independent ensembles of the final model (agreement ARI = {ari:.2f})\n"
                 "Each panel: share of the group's days in each activity, by time of day; groups sorted by sleep timing",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.94))
    save(fig, "fig5_ensemble_groups.png")

    # Overlap of the two groupings: which share of each A group lands in each B group.
    ct = np.zeros((6, 6))
    np.add.at(ct, (labels["A"], labels["B"]), 1)
    share = ct / ct.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(share, cmap=SEQ, vmin=0, vmax=1)
    for i in range(6):
        for k in range(6):
            ax.text(k, i, f"{share[i, k]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if share[i, k] > 0.55 else INK)
    ax.set_xticks(range(6), [f"G{k + 1}" for k in range(6)])
    ax.set_yticks(range(6), [f"G{k + 1}" for k in range(6)])
    ax.set_xlabel("Ensemble B group")
    ax.set_ylabel("Ensemble A group")
    ax.tick_params(length=0)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.set_label("Share of the A group's days", color=INK2)
    ax.set_title(f"Where each A group's days land in B\n(ARI = {ari:.2f}; a perfect match would be a diagonal)",
                 loc="left", fontsize=9)
    save(fig, "fig5b_group_overlap.png")


def rhythm_scores(d: Data, emb: dict) -> dict[str, np.ndarray]:
    """Early-to-late score per diary for each ensemble, as in scripts/continuum_axes.py: per-model PCA (3 axes) of the
    timing means fitted on training days, scores standardized by their training SD, rotated onto seed 42's axes by
    orthogonal Procrustes (fitted on training days), then averaged within each ensemble. Axis 1 is oriented so that a
    higher score means a later sleep midpoint (on training days)."""
    from scipy.linalg import orthogonal_procrustes
    sc, sc_tr = {}, {}
    for sd, e in emb.items():
        pca = PCA(3, random_state=0).fit(e["timing_mu"][d.tr])
        full = pca.transform(e["timing_mu"])
        full = full / full[d.tr].std(0, keepdims=True)
        sc[sd] = full
    ref_tr = sc[42][d.tr]
    for sd in sc:
        if sd != 42:
            R, _ = orthogonal_procrustes(sc[sd][d.tr], ref_tr)
            sc[sd] = sc[sd] @ R
    mid = sleep_midpoint(d.seq, d.cats.index("sleep"), d.start)
    mid = (mid + 12) % 24 - 12
    out = {}
    for name, seeds in ENS.items():
        a1 = np.mean([sc[sd][:, 0] for sd in seeds], axis=0)
        if spearmanr(a1[d.tr], mid[d.tr])[0] < 0:
            a1 = -a1
        out[name] = a1
    return out


def fig_rhythm_axis(d: Data, emb: dict) -> None:
    mid = sleep_midpoint(d.seq, d.cats.index("sleep"), d.start)
    mid = (mid + 12) % 24 - 12
    full = rhythm_scores(d, emb)
    score = {k: v[d.va] for k, v in full.items()}
    rho = spearmanr(score["A"], score["B"])[0]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(score["A"], score["B"], s=9, lw=0, color=SERIES[0], alpha=0.6)
    ax.set_xlabel("Ensemble A: early-to-late score")
    ax.set_ylabel("Ensemble B: early-to-late score")
    ax.set_title(f"Early-to-late score from two independent ensembles\n(Spearman ρ = {rho:.2f}, 1,627 validation days)",
                 loc="left", fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    save(fig, "fig6_rhythm_axis_agreement.png")

    sv = d.seq[d.va]
    cuts = np.quantile(full["A"][d.tr], [1 / 3, 2 / 3])   # band cut points fixed on training days
    band = np.digitize(score["A"], cuts)
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.2), sharex=True)
    x = np.arange(288)
    for b, (ax, name) in enumerate(zip(axes, ("Earliest third", "Middle third", "Latest third"))):
        m = band == b
        ax.stackplot(x, profile(d, sv[m]), colors=SERIES + [OTHER_GRAY], lw=0)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.spines[:].set_visible(False)
        mh = np.median(mid[d.va][m])
        ax.set_title(f"{name} of days (median sleep midpoint {int(mh) % 24}:{int(round(60 * (mh % 1))):02d} a.m.)",
                     loc="left", fontsize=8.5)
        time_axis(ax, d, labels=b == 2)
    fig.legend(handles=legend_handles(d), ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=7.5)
    fig.suptitle("Days ordered by the early-to-late score (final model, ensemble A)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save(fig, "fig6b_rhythm_bands.png")


# ----------------------------------------------------------------------------------------------- figure 7
def fig_model_selection(_d: Data) -> None:
    # 2022 validation cross-entropy, mean over 3 seeds (docs/v3_capacity_results.md, docs/v3_round*_results.md).
    size_series = [(61786, 0.540, "v3 (62K)"), (581290, 0.421, "581K"), (1093546, 0.385, "1.09M"), (1802410, 0.384, "1.80M")]
    others = [(1082026, 0.395, "GRU decoder"), (581290, 0.372, "581K, 80 epochs"),
              (1093546, 0.314, "1.09M, trained on 2022 to 2024 (final)")]
    fig, ax = plt.subplots(figsize=(7, 4))
    xs, ys, _ = zip(*size_series)
    ax.plot(xs, ys, color=SERIES[0], lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label="Larger network, 2022 data only")
    for x, y, t in size_series:
        below = t == "1.09M"          # the GRU-decoder marker sits just above this point
        ax.text(x, y - 0.02 if below else y + 0.012, t, ha="center", fontsize=7.5, color=INK2)
    for (x, y, t), c, mk in zip(others, (SERIES[3], SERIES[4], SERIES[1]), ("s", "D", "*")):
        ax.plot([x], [y], marker=mk, ms=17 if mk == "*" else 9, color=c, mec=SURFACE, mew=1.5, ls="none", label=t)
    ax.text(1093546 * 1.08, 0.314, "final model", va="center", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Validation reconstruction error (lower is better)")
    ax.set_ylim(0.28, 0.58)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("Model size helps until about 1M parameters; more data helps most", loc="left")
    save(fig, "fig7_model_selection.png")


FIGS = {1: fig_architecture, 2: fig_reconstructions, 3: fig_pair_test, 7: fig_model_selection}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    args = ap.parse_args()
    torch.set_num_threads(1)
    d = Data()
    for k in args.only:
        if k in FIGS:
            FIGS[k](d)
    if {4, 5, 6} & set(args.only):
        emb = final_embeddings(d)
        if 4 in args.only:
            fig_embeddings(d, emb)
        if {5, 6} & set(args.only):
            labels, reps = ensemble_labels(d, emb)
            if 5 in args.only:
                fig_groups(d, labels)
            if 6 in args.only:
                fig_rhythm_axis(d, emb)


if __name__ == "__main__":
    main()
