"""How many schedule groups? k sweep for the frozen v3 ensembles (2022 validation only; development result).

Usage:
    python scripts/k_sweep_v3.py

For k = 2..10, using the frozen v3 models (ensemble A = seeds 42-44, B = 45-47; concatenated timing posterior means,
k-means n_init=50 fitted on 2022 train rows) and, for reference, engineered timing features:
- reproducibility: ensemble A vs B ARI (all days) and on "core" days (both ensembles confident);
- distinctness: share of variance in engineered clock-timing features explained by the groups (eta², timing)
  vs share of variance in activity totals explained (eta², totals) — groups should differ mainly in timing;
- meaning: among pairs with near-identical totals (≤ 60 min), sleep-midpoint difference of split vs kept pairs;
- silhouette and smallest group share.
Writes outputs/v3/k_sweep/k_sweep.csv and docs/v3_k_sweep.md.
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import FeatureSet, TimingFeaturizer, totals_features  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation.clustering import ClusterPipeline  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

_spec = importlib.util.spec_from_file_location("v3f", ROOT / "scripts/evaluate_v3_final.py")
v3f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v3f)

KS = list(range(2, 11))
CORE_SHARE = 0.5   # "core" = days whose assignment margin is in the top half for BOTH ensembles


def eta2(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean over columns of between-group variance / total variance."""
    out = []
    for j in range(X.shape[1]):
        x = X[:, j]
        if x.std() == 0:
            continue
        means = np.array([x[labels == g].mean() for g in np.unique(labels)])
        sizes = np.array([(labels == g).sum() for g in np.unique(labels)])
        out.append((sizes * (means - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())
    return float(np.mean(out))


def fit(R_tr: np.ndarray, R_ev: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels and confidence margin (2nd-nearest minus nearest centroid distance, relative to nearest)."""
    p = ClusterPipeline(k, seed=seed, scale=False, n_init=50).fit(R_tr)
    d = p.km_.transform(p.embed(R_ev))
    srt = np.sort(d, axis=1)
    return d.argmin(1), (srt[:, 1] - srt[:, 0]) / (srt[:, 0] + 1e-9)


def main() -> None:
    out_dir = ROOT / "outputs/v3/k_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    C, clock, seq = len(cats), z["clock_features"], z["seq"]
    tr, va = z["split"] == "train", z["split"] == "val"
    sv = seq[va]
    tot, _ = totals_features(category_counts(seq, C), cats)
    tf = TimingFeaturizer(cats, clock).fit(seq[tr])
    X, names = tf.transform(seq), tf.names_
    ccols = [i for i, n in enumerate(names) if n.split("_")[0] in ("center", "first", "last")]
    Tclock = StandardScaler().fit(X[tr][:, ccols]).transform(X[va][:, ccols])
    Ttot = StandardScaler().fit(tot[tr]).transform(tot[va])
    pairs = matched_totals_pairs(category_counts(sv, C), 60, 5)
    mid = v3f.sleep_midpoint_hours(sv, cats.index("sleep"))

    emb = {s: posterior_means(load_checkpoint(v3f.latest(p) / "best.pt")[0], torch.from_numpy(seq).long())["timing_mu"]
           for s, p in v3f.MODELS.items()}
    reps = {name: np.hstack([emb[s] for s in seeds]) for name, seeds in v3f.ENSEMBLES.items()}
    fs = FeatureSet("eng", False, True, cats, clock).fit(seq[tr])
    E = fs.transform(seq, category_counts(seq, C))
    E = StandardScaler().fit(E[tr]).transform(E)

    rows = []
    for k in KS:
        la, ma = fit(reps["A"][tr], reps["A"][va], k, 0)
        lb, mb = fit(reps["B"][tr], reps["B"][va], k, 0)
        core = (ma >= np.quantile(ma, 1 - CORE_SHARE)) & (mb >= np.quantile(mb, 1 - CORE_SHARE))
        mpa, mpb = v3f.midpoint_ratio(la, pairs, mid), v3f.midpoint_ratio(lb, pairs, mid)
        le = [fit(E[tr], E[va], k, s)[0] for s in (42, 43)]
        mpe = v3f.midpoint_ratio(le[0], pairs, mid)
        rows.append({
            "k": k,
            "ARI_A_vs_B": adjusted_rand_score(la, lb),
            "ARI_core_days": adjusted_rand_score(la[core], lb[core]), "core_share": float(core.mean()),
            "eta2_timing": (eta2(Tclock, la) + eta2(Tclock, lb)) / 2,
            "eta2_totals": (eta2(Ttot, la) + eta2(Ttot, lb)) / 2,
            "midpoint_ratio": (mpa["ratio"] + mpb["ratio"]) / 2,
            "split_share_same_totals": (mpa["share_split"] + mpb["share_split"]) / 2,
            "silhouette": (silhouette_score(reps["A"][va], la, sample_size=None, random_state=0)
                           + silhouette_score(reps["B"][va], lb, random_state=0)) / 2,
            "smallest_group": min(np.bincount(la).min(), np.bincount(lb).min()) / len(la),
            "eng_ARI_inits": adjusted_rand_score(le[0], le[1]),
            "eng_eta2_timing": eta2(Tclock, le[0]), "eng_eta2_totals": eta2(Ttot, le[0]),
            "eng_midpoint_ratio": mpe["ratio"],
        })
        r = rows[-1]
        print(f"k={k}: ARI {r['ARI_A_vs_B']:.3f} (core {r['ARI_core_days']:.3f}), eta2 timing {r['eta2_timing']:.3f} "
              f"totals {r['eta2_totals']:.3f}, midpoint ratio {r['midpoint_ratio']:.2f}, smallest {r['smallest_group']:.1%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "k_sweep.csv", index=False)
    lines = ["# How many schedule groups? (v3 ensembles, 2022 validation — development result)", "",
             "Generated by `scripts/k_sweep_v3.py`. Frozen v3 models; ensemble A = seeds 42-44, B = 45-47; k-means n_init=50",
             "fitted on 2022 train rows; 1,627 validation diaries; 179 near-identical-totals pairs (≤ 60 min).", "",
             "- `ARI_A_vs_B`: agreement between two independent model sets (all days); `ARI_core_days`: on days both sets",
             "  assign confidently (top half of assignment margin in both).",
             "- `eta2_timing` / `eta2_totals`: share of variance in clock-timing features / activity totals explained by the",
             "  groups. Timing groups should explain timing much more than totals.",
             "- `midpoint_ratio`: among same-total pairs, sleep-midpoint difference of pairs split by the groups vs kept together.",
             "- `eng_*`: the same for engineered timing features (k-means initialisations 42 vs 43).", "",
             df.to_markdown(index=False, floatfmt=".3f"), ""]
    (ROOT / "docs/v3_k_sweep.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
