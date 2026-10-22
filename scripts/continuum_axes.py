"""Describe daily rhythm as a continuum: reproducible axes from the v1 timing latent (2022 validation; development).

Usage:
    python scripts/continuum_axes.py

Uses the nine frozen 80-epoch v1 models (seeds 42-44: outputs/checkpoint5_final/guided_s*, seeds 45-50:
outputs/v3/base80_s*). For each model, PCA is fitted on 2022 train-split timing posterior means. Its leading
axes are aligned to seed 42's by an orthogonal Procrustes rotation fitted on TRAIN scores only (axes with
similar variance can rotate into each other between models, so one-to-one matching understates stability;
the matching-only result is reported for comparison). Then:

- reproducibility per axis: Spearman correlation of validation-day scores across all model pairs, and
  agreement of tercile bands (low / middle / high) across models (share of days and Cohen's kappa);
- ensemble stability: consensus score = mean over 3 models; three disjoint consensus sets (42-44, 45-47,
  48-50) compared with each other;
- interpretation: correlation of the consensus axes with plain rhythm measures (sleep midpoint, wake time,
  bedtime, work centre, transitions) and with activity totals (an amount check);
- figures: average day for the low / middle / high band of each axis.

Writes outputs/continuum/ (per-day scores and bands, figures; git-ignored) and docs/continuum_axes.md.
"""

from __future__ import annotations

import glob
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation.plotting import INK, MUTED, palette, plt, stacked_profile  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

N_AXES = 3
MODELS = {**{s: f"outputs/checkpoint5_final/guided_s{s}-*" for s in (42, 43, 44)},
          **{s: f"outputs/v3/base80_s{s}-*" for s in (45, 46, 47, 48, 49, 50)}}
CONSENSUS_SETS = {"A": (42, 43, 44), "B": (45, 46, 47), "C": (48, 49, 50)}
START_CLOCK, W = 240, 5


def clock_hours(bins: np.ndarray) -> np.ndarray:
    return ((START_CLOCK + W * bins) % 1440) / 60.0


def rhythm_measures(seq: np.ndarray, cats: list[str]) -> pd.DataFrame:
    """Plain, interpretable per-day measures (hours are clock time; 'unwrapped' past midnight where noted)."""
    sl, wk = cats.index("sleep"), cats.index("work")
    T = seq.shape[1]
    is_sleep = seq == sl
    ang = 2 * np.pi * ((START_CLOCK + W * np.arange(T) + W / 2) % 1440) / 1440
    mid = np.degrees(np.arctan2((is_sleep * np.sin(ang)).sum(1), (is_sleep * np.cos(ang)).sum(1))) % 360 / 15
    mid_unwrapped = np.where(mid < 16, mid + 24, mid)                     # e.g. 03:00 -> 27.0
    # Wake = end of the first sleep run if the diary starts asleep; bedtime = start of the last sleep run if it
    # ends asleep (window 04:00-04:00). NaN otherwise.
    first_awake = np.where(is_sleep[:, 0], np.argmax(~is_sleep, axis=1), -1)
    wake = np.where(first_awake > 0, clock_hours(first_awake), np.nan)
    last_awake = T - 1 - np.argmax(~is_sleep[:, ::-1], axis=1)
    bed = np.where(is_sleep[:, -1] & (last_awake < T - 1), clock_hours(last_awake + 1), np.nan)
    bed = np.where(bed < 12, bed + 24, bed)                                # after midnight -> 24+
    w = seq == wk
    has_w = w.any(1)
    wc = np.degrees(np.arctan2((w * np.sin(ang)).sum(1), (w * np.cos(ang)).sum(1))) % 360 / 15
    cnt = category_counts(seq, len(cats)) * W
    return pd.DataFrame({
        "sleep_midpoint_h": mid_unwrapped, "wake_h": wake, "bedtime_h": bed,
        "work_centre_h": np.where(has_w, wc, np.nan),
        "transitions": (np.diff(seq, axis=1) != 0).sum(1),
        "sleep_min": cnt[:, sl], "work_min": cnt[:, wk], "leisure_min": cnt[:, cats.index("leisure_social")],
    })


def align(ref: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Reorder and sign-flip columns of S to best match ref (|Pearson correlation|, Hungarian)."""
    c = np.corrcoef(ref.T, S.T)[: ref.shape[1], ref.shape[1]:]
    r, col = linear_sum_assignment(-np.abs(c))
    out = S[:, col].copy()
    out *= np.sign(c[r, col])[None, :]
    return out


def bands(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.digitize(x, cuts)                                             # 0 low, 1 middle, 2 high


def main() -> None:
    out_dir = ROOT / "outputs/continuum"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    tr, va = z["split"] == "train", z["split"] == "val"
    seq = z["seq"]
    sv = seq[va]
    rm = rhythm_measures(sv, cats)
    meta = pd.read_parquet(ROOT / "data/processed/2022/metadata.parquet").set_index("case_id")
    rm["weekend"] = meta.loc[z["case_ids"][va], "TUDIARYDAY"].isin([1, 7]).to_numpy().astype(float)

    scores, scores_tr, var_expl = {}, {}, {}
    for s, pat in MODELS.items():
        e = posterior_means(load_checkpoint(Path(sorted(glob.glob(str(ROOT / pat)))[-1]) / "best.pt")[0],
                            torch.from_numpy(seq).long())["timing_mu"]
        p = PCA(N_AXES, random_state=0).fit(e[tr])
        scores[s], scores_tr[s] = p.transform(e[va]), p.transform(e[tr])
        var_expl[s] = p.explained_variance_ratio_
    # Standardise each model's PC scores by train SD so the rotation is not dominated by scale differences.
    for s in MODELS:
        sd = scores_tr[s].std(0, keepdims=True)
        scores[s], scores_tr[s] = scores[s] / sd, scores_tr[s] / sd
    matched = {}
    ref, ref_tr = scores[42], scores_tr[42]
    for s in MODELS:
        if s == 42:
            matched[s] = ref
            continue
        # (a) one-to-one matching (fitted on train scores), for comparison only
        c = np.corrcoef(ref_tr.T, scores_tr[s].T)[:N_AXES, N_AXES:]
        r, col = linear_sum_assignment(-np.abs(c))
        matched[s] = scores[s][:, col] * np.sign(c[r, col])[None, :]
        # (b) orthogonal Procrustes rotation onto seed 42's axes, fitted on train scores, applied to both splits
        R, _ = orthogonal_procrustes(scores_tr[s], ref_tr)
        scores[s], scores_tr[s] = scores[s] @ R, scores_tr[s] @ R
    # Tercile cut points fitted on TRAIN scores of each model (so bands are defined without validation data).
    cuts = {s: [np.quantile(scores_tr[s][:, a], [1 / 3, 2 / 3]) for a in range(N_AXES)] for s in MODELS}

    seeds = list(MODELS)
    rep = []
    for a in range(N_AXES):
        rho = [abs(spearmanr(scores[x][:, a], scores[y][:, a])[0]) for x, y in combinations(seeds, 2)]
        bx = {s: bands(scores[s][:, a], cuts[s][a]) for s in seeds}
        same = [np.mean(bx[x] == bx[y]) for x, y in combinations(seeds, 2)]
        kap = [cohen_kappa_score(bx[x], bx[y]) for x, y in combinations(seeds, 2)]
        cons = {n: np.mean([scores[s][:, a] for s in ss], axis=0) for n, ss in CONSENSUS_SETS.items()}
        cons_tr = {n: np.mean([scores_tr[s][:, a] for s in ss], axis=0) for n, ss in CONSENSUS_SETS.items()}
        cb = {n: bands(cons[n], np.quantile(cons_tr[n], [1 / 3, 2 / 3])) for n in cons}
        rho_matched = [abs(spearmanr(matched[x][:, a], matched[y][:, a])[0]) for x, y in combinations(seeds, 2)]
        rep.append({"axis": a + 1, "var_explained_mean": float(np.mean([var_expl[s][a] for s in seeds])),
                    "matching_only_spearman_mean": float(np.mean(rho_matched)),
                    "single_spearman_mean": float(np.mean(rho)), "single_spearman_min": float(np.min(rho)),
                    "single_band_agreement": float(np.mean(same)), "single_band_kappa": float(np.mean(kap)),
                    "consensus_spearman_mean": float(np.mean([abs(spearmanr(cons[x], cons[y])[0]) for x, y in combinations(cons, 2)])),
                    "consensus_band_agreement": float(np.mean([np.mean(cb[x] == cb[y]) for x, y in combinations(cb, 2)])),
                    "consensus_band_kappa": float(np.mean([cohen_kappa_score(cb[x], cb[y]) for x, y in combinations(cb, 2)]))})
    stab = pd.DataFrame(rep)

    # Final consensus over all nine models (scores standardised by train SD), bands from train terciles.
    allc = np.stack([np.mean([scores[s][:, a] for s in seeds], axis=0) for a in range(N_AXES)], 1)
    allc_tr = np.stack([np.mean([scores_tr[s][:, a] for s in seeds], axis=0) for a in range(N_AXES)], 1)
    # Re-express the consensus space in its own principal axes (fitted on train) so axis 1 is the dominant
    # consensus direction; sign each so that "high" correlates positively with sleep midpoint (later).
    cp = PCA(N_AXES, random_state=0).fit(allc_tr)
    allc, allc_tr = cp.transform(allc), cp.transform(allc_tr)
    for a in range(N_AXES):
        sgn = np.sign(spearmanr(allc[:, a], rm["sleep_midpoint_h"])[0]) or 1.0
        allc[:, a] *= sgn
        allc_tr[:, a] *= sgn
    final_bands = np.stack([bands(allc[:, a], np.quantile(allc_tr[:, a], [1 / 3, 2 / 3])) for a in range(N_AXES)], 1)
    corr = pd.DataFrame({f"axis{a + 1}": [spearmanr(allc[:, a], rm[c], nan_policy="omit")[0] for c in rm.columns]
                         for a in range(N_AXES)}, index=rm.columns)
    # Orient each axis so that "high" means a later sleep midpoint (axis 1) or simply the positive direction.
    pd.DataFrame({"case_id": z["case_ids"][va], **{f"axis{a + 1}": allc[:, a] for a in range(N_AXES)},
                  **{f"band{a + 1}": final_bands[:, a] for a in range(N_AXES)}}).to_csv(out_dir / "axes_val_2022.csv", index=False)
    band_means = {}
    for a in range(N_AXES):
        band_means[a] = rm.groupby(final_bands[:, a]).mean(numeric_only=True)

    # Figures: average day per band for each axis.
    fig, axes = plt.subplots(3, N_AXES, figsize=(4.2 * N_AXES, 6.2), sharex=True, sharey=True)
    names = ["low", "middle", "high"]
    for a in range(N_AXES):
        for b in range(3):
            ax = axes[b, a]
            stacked_profile(ax, sv[final_bands[:, a] == b], cats, START_CLOCK)
            ax.set_xticks(ax.get_xticks()[::2])
            if a == 0:
                ax.set_ylabel(names[b], rotation=0, ha="right", va="center", color=INK)
            if b == 0:
                ax.set_title(f"Axis {a + 1}", color=INK, fontsize=10)
            ax.set_yticks([])
    fig.suptitle("Average day by tercile band on each rhythm axis (9-model consensus, 2022 validation)", x=0.01,
                 ha="left", color=INK)
    fig.legend(handles=palette(cats)[1], loc="lower center", ncol=9, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(out_dir / "axes_band_profiles.png", dpi=130)
    plt.close(fig)

    fmt = lambda d: d.to_markdown(floatfmt=".3f")  # noqa: E731
    lines = ["# Daily rhythm as a continuum: reproducible axes (2022 validation; development result)", "",
             "Generated by `scripts/continuum_axes.py`. Nine frozen 80-epoch v1 models (seeds 42-50). Per model: PCA of",
             "the timing posterior means fitted on 2022 train rows; each model's 3-axis space rotated onto seed 42's",
             "(orthogonal Procrustes, fitted on train scores only). Consensus axes = principal axes of the 9-model",
             "average, signed so that 'high' means a later sleep midpoint. Bands = terciles of train-set scores.", "",
             "## Reproducibility of each axis across models", "",
             "- `matching_only_*`: axes matched one-to-one without rotation, for comparison.",
             "- `single_*`: after Procrustes alignment, across all 36 pairs of single models (Spearman |ρ| of day",
             "  scores; same band; Cohen's κ).",
             "- `consensus_*`: across 3 disjoint 3-model consensus sets (42-44, 45-47, 48-50).",
             "- For reference: hard k-means groups gave ARI ≈ 0.2 (single) and 0.25-0.41 (3-model ensembles).", "",
             fmt(stab.set_index("axis")), "",
             "## What the axes mean (Spearman correlation of the 9-model consensus axis with plain measures)", "",
             "Clock measures are in hours (sleep midpoint and bedtime run past 24 for after-midnight times). `*_min` are",
             "activity totals: the timing latent is built to carry timing beyond totals, so these should be weak.", "",
             fmt(corr), "",
             "## Band averages (validation days)", ""]
    for a in range(N_AXES):
        bm = band_means[a].copy()
        bm.index = [names[i] for i in bm.index]
        lines += [f"### Axis {a + 1}", "", fmt(bm[["sleep_midpoint_h", "wake_h", "bedtime_h", "work_centre_h",
                                                     "transitions", "sleep_min", "work_min", "weekend"]]), ""]
    lines += ["Figure (git-ignored): `outputs/continuum/axes_band_profiles.png`. Per-day consensus scores and bands:",
              "`outputs/continuum/axes_val_2022.csv`.", ""]
    (ROOT / "docs/continuum_axes.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
