"""Two-level schedule types (docs/two_level_types_plan.md): composition types, then timing cores within each.

Usage:
    python scripts/two_level_types.py

2022 data only (development). Uses the nine frozen 80-epoch v1 models; no training.
Writes outputs/two_level/ (per-day labels, figures; git-ignored) and docs/two_level_types.md.
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import diptest
import numpy as np
import pandas as pd
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import TimingFeaturizer  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation.clustering import ClusterPipeline  # noqa: E402
from src.evaluation.plotting import INK, palette, plt, stacked_profile  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("ca", "scripts/continuum_axes.py")
KS = range(3, 9)
N_BOOT = 10
SETS = {"A": (42, 43, 44), "B": (45, 46, 47), "C": (48, 49, 50)}
MIN_TYPE = 200
N_NULL = 5


def composition_k(P: np.ndarray, tr: np.ndarray, va: np.ndarray) -> tuple[int, pd.DataFrame]:
    rows = []
    rng = np.random.default_rng(0)
    idx_tr = np.flatnonzero(tr)
    for k in KS:
        labs = []
        for b in range(N_BOOT):
            boot = rng.choice(idx_tr, size=len(idx_tr), replace=True)
            labs.append(ClusterPipeline(k, seed=b, n_init=50).fit(P[boot]).predict(P[va]))
        aris = [adjusted_rand_score(a, c) for a, c in combinations(labs, 2)]
        full = ClusterPipeline(k, seed=0, n_init=50).fit(P[tr]).predict(P)
        rows.append({"k": k, "bootstrap_ARI": float(np.mean(aris)), "bootstrap_ARI_sd": float(np.std(aris)),
                     "smallest_type": float(np.bincount(full, minlength=k).min() / len(full))})
        print(f"composition k={k}: ARI {rows[-1]['bootstrap_ARI']:.3f}, smallest {rows[-1]['smallest_type']:.1%}", flush=True)
    t = pd.DataFrame(rows)
    ok = t[(t["bootstrap_ARI"] >= 0.8) & (t["smallest_type"] >= 0.03)]
    return (int(ok["k"].max()) if len(ok) else int(t.sort_values("bootstrap_ARI").iloc[-1]["k"])), t


def set_axes(seq: np.ndarray, tr: np.ndarray) -> dict[str, np.ndarray]:
    """Per model set: mean of Procrustes-aligned 3-axis PCA scores of the timing latent (all rows)."""
    scores, scores_tr = {}, {}
    for s, pat in ca.MODELS.items():
        e = posterior_means(load_checkpoint(Path(sorted(glob.glob(str(ROOT / pat)))[-1]) / "best.pt")[0],
                            torch.from_numpy(seq).long())["timing_mu"]
        p = PCA(3, random_state=0).fit(e[tr])
        sc = p.transform(e)
        sc /= sc[tr].std(0, keepdims=True)
        scores[s] = sc
    ref = scores[42]
    for s in scores:
        if s != 42:
            R, _ = orthogonal_procrustes(scores[s][tr], ref[tr])
            scores[s] = scores[s] @ R
    return {n: np.mean([scores[s] for s in ss], axis=0) for n, ss in SETS.items()}


def hdb(X: np.ndarray, n_type: int) -> np.ndarray:
    return HDBSCAN(min_cluster_size=max(20, int(0.05 * n_type)), min_samples=10).fit_predict(X)


def core_stats(lab: np.ndarray) -> tuple[int, float]:
    return int(len(set(lab) - {-1})), float((lab >= 0).mean())


def main() -> None:
    out = ROOT / "outputs/two_level"
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    C, seq, ids, clock = len(cats), z["seq"], z["case_ids"], z["clock_features"]
    tr, va = z["split"] == "train", z["split"] == "val"
    meta = pd.read_parquet(ROOT / "data/processed/2022/metadata.parquet").set_index("case_id").loc[ids]
    weekend = meta["TUDIARYDAY"].isin([1, 7]).to_numpy()
    employed = meta["TELFS"].isin([1, 2]).to_numpy()
    cnt = category_counts(seq, C)
    P = StandardScaler().fit((cnt / cnt.sum(1, keepdims=True))[tr]).transform(cnt / cnt.sum(1, keepdims=True))

    # ---- Level 1 -----------------------------------------------------------------------------------------
    k, ktab = composition_k(P, tr, va)
    comp = ClusterPipeline(k, seed=0, n_init=50).fit(P[tr]).predict(P)
    mins = pd.DataFrame(cnt * 5, columns=cats)
    prof = mins.groupby(comp).mean().round(0)
    prof.insert(0, "share_%", (pd.Series(comp).value_counts(normalize=True).sort_index() * 100).round(1))
    prof.insert(1, "weekend_%", (pd.Series(weekend).groupby(comp).mean() * 100).round(1))
    prof.insert(2, "employed_%", (pd.Series(employed).groupby(comp).mean() * 100).round(1))
    # Name each type by its most over-represented activity relative to the overall mean (for readability only).
    over = (mins.groupby(comp).mean() / mins.mean()).drop(columns=["sleep", "unknown"])
    prof.insert(0, "label", [f"{over.loc[t].idxmax()}-heavy" for t in prof.index])

    # ---- Level 2 -----------------------------------------------------------------------------------------
    axes = set_axes(seq, tr)
    tf = TimingFeaturizer(cats, clock).fit(seq[tr])
    E = tf.transform(seq)
    rm = ca.rhythm_measures(seq, cats)
    rows, profiles = [], []
    for t in range(k):
        m = comp == t
        n = int(m.sum())
        if n < MIN_TYPE:
            rows.append({"type": t, "label": prof.loc[t, "label"], "n_days": n, "note": "too small"})
            continue
        labs = {name: hdb(A[m], n) for name, A in axes.items()}
        vm = va[m]
        ari_all = [adjusted_rand_score(labs[a][vm], labs[b][vm]) for a, b in combinations(labs, 2)]
        ari_core = []
        for a, b in combinations(labs, 2):
            both = vm & (labs[a] >= 0) & (labs[b] >= 0)
            if both.sum() >= 20 and len(set(labs[a][both])) > 1:
                ari_core.append(adjusted_rand_score(labs[a][both], labs[b][both]))
        nc, frac = zip(*(core_stats(l) for l in labs.values()))
        # Gaussian null matched to set A's representation within the type.
        XA = axes["A"][m]
        rng = np.random.default_rng(t)
        null = [core_stats(hdb(rng.multivariate_normal(XA.mean(0), np.cov(XA.T), size=n), n)) for _ in range(N_NULL)]
        # Engineered timing features within the type (deterministic; for comparison).
        Et = PCA(3, random_state=0).fit_transform(StandardScaler().fit_transform(E[m]))
        eng = core_stats(hdb(Et, n))
        eng_null = [core_stats(hdb(rng.multivariate_normal(Et.mean(0), np.cov(Et.T), size=n), n)) for _ in range(N_NULL)]
        dips = [diptest.diptest(XA[:, a])[1] for a in range(3)]
        rows.append({"type": t, "label": prof.loc[t, "label"], "n_days": n,
                     "VAE_cores_mean": float(np.mean(nc)), "VAE_frac_in_cores": float(np.mean(frac)),
                     "VAE_null_cores": float(np.mean([x[0] for x in null])), "VAE_null_frac": float(np.mean([x[1] for x in null])),
                     "VAE_ARI_all": float(np.mean(ari_all)),
                     "VAE_ARI_cores": float(np.mean(ari_core)) if ari_core else np.nan,
                     "eng_cores": eng[0], "eng_frac_in_cores": eng[1],
                     "eng_null_cores": float(np.mean([x[0] for x in eng_null])), "eng_null_frac": float(np.mean([x[1] for x in eng_null])),
                     "dip_p_axis1": dips[0], "dip_p_axis2": dips[1], "dip_p_axis3": dips[2]})
        print(rows[-1], flush=True)
        # Profiles of set-A cores (if any).
        la = labs["A"]
        for c in sorted(set(la) - {-1}):
            mm = np.flatnonzero(m)[la == c]
            profiles.append({"type": t, "label": prof.loc[t, "label"], "core": int(c), "share_of_type_%": round(100 * len(mm) / n, 1),
                             **rm.iloc[mm][["wake_h", "sleep_midpoint_h", "bedtime_h", "work_centre_h", "transitions"]].mean().round(2).to_dict(),
                             "work_min": round(float(rm.iloc[mm]["work_min"].mean()), 0), "weekend_%": round(100 * weekend[mm].mean(), 1)})
    lv2 = pd.DataFrame(rows)
    cores = pd.DataFrame(profiles)

    # ---- Shift-work check ---------------------------------------------------------------------------------
    work_types = [t for t in range(k) if prof.loc[t, "work"] >= 300]
    wm = np.isin(comp, work_types)
    wc = rm["work_centre_h"].to_numpy()[wm]
    wc = wc[~np.isnan(wc)]
    wc_shift = np.where(wc < 4, wc + 24, wc)           # treat 00:00-04:00 as late night (24-28 h)
    bins = {"day (centre 06-15 h)": ((wc_shift >= 6) & (wc_shift < 15)).mean(),
            "evening (15-21 h)": ((wc_shift >= 15) & (wc_shift < 21)).mean(),
            "night (21-06 h)": ((wc_shift >= 21) | (wc_shift < 6)).mean()}
    dip_wc = diptest.diptest(wc_shift)[1]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(wc_shift, bins=np.arange(4, 28.5, 0.5), color="#1baf7a", edgecolor="#fcfcfb")
    ax.set_xlabel("work centre (clock hour; 24-28 = after midnight)")
    ax.set_ylabel("days")
    ax.set_title("Work-centre time on workday-type days (2022)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(out / "work_centre_hist.png", dpi=130)
    plt.close(fig)

    # Profiles figure: average day per composition type.
    fig, axs = plt.subplots(k, 1, figsize=(9, 1.45 * k + 1), sharex=True)
    for t, axx in enumerate(np.atleast_1d(axs)):
        stacked_profile(axx, seq[comp == t], cats, 240)
        axx.set_ylabel(f"T{t}\n{prof.loc[t, 'share_%']:.0f}%", rotation=0, ha="right", va="center", color=INK)
        axx.set_yticks([])
    fig.suptitle("Composition types (2022, all days)", x=0.01, ha="left", color=INK)
    fig.legend(handles=palette(cats)[1], loc="lower center", ncol=9, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(out / "composition_types.png", dpi=130)
    plt.close(fig)
    pd.DataFrame({"case_id": ids, "split": z["split"], "composition_type": comp}).to_csv(out / "composition_types.csv", index=False)

    fmtd = lambda d: d.to_markdown(index=False, floatfmt=".3f")  # noqa: E731
    lines = ["# Two-level schedule types (2022; development result)", "",
             "Generated by `scripts/two_level_types.py`; plan in `docs/two_level_types_plan.md`.", "",
             "## Level 1: composition types", "", f"Chosen k = {k} (rule: largest k with bootstrap ARI ≥ 0.8 and smallest type ≥ 3%).", "",
             fmtd(ktab), "", prof.reset_index(names="type").to_markdown(index=False, floatfmt=".0f"), "",
             "## Level 2: timing cores within each composition type", "",
             "`VAE_*`: HDBSCAN on the three model sets' rhythm axes (mean over sets); `*_null_*`: same method on a matched",
             "Gaussian (no clumps by construction); `VAE_ARI_*`: agreement between model sets on validation days (all",
             "days with noise as a label / days in a core in both sets); `dip_p_*`: Hartigan dip test (small p = not",
             "unimodal).", "", fmtd(lv2), "",
             "### Cores found (model set A)", "", (fmtd(cores) if len(cores) else "No cores found."), "",
             "## Shift-work check (workday-type days)", "",
             f"Workday types: {work_types} (mean work ≥ 300 min). Work-centre distribution: " +
             ", ".join(f"{k_}: {v:.1%}" for k_, v in bins.items()) + f". Dip test p = {dip_wc:.3g}.", "",
             "Figures (git-ignored): `outputs/two_level/composition_types.png`, `outputs/two_level/work_centre_hist.png`.", ""]
    (ROOT / "docs/two_level_types.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
