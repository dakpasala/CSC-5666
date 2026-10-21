"""Longer training + more ensembles: group reproducibility vs k, with error bars (2022 validation; development).

Usage:
    python scripts/k_sweep_ensembles.py

80-epoch v1 models (identical settings to the 40-epoch v3 models apart from epochs):
  A = outputs/checkpoint5_final/guided_s42-44, B = outputs/v3/base80_s45-47, C = outputs/v3/base80_s48-50.
40-epoch reference: A40 = guided_l0.3_s42-44, B40 = base_s45-47 (one comparison only).
Each ensemble: concatenated timing posterior means -> k-means (n_init=50) fitted on 2022 train rows.

For k = 2..10 reports, per training length: mean ± SD over ensemble pairs of ARI (all days) and ARI on
"core" days (confident in both ensembles of a pair: top half of assignment margin); share of totals and
clock-timing variance explained by groups; same-total sleep-midpoint ratio; smallest group; and the
single-model ARI for reference. Also writes per-day confidence labels for the 80-epoch ensemble A at k=4.
Outputs: outputs/v3/k_sweep_ensembles/, docs/v3_k_sweep_ensembles.md.
"""

from __future__ import annotations

import importlib.util
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import TimingFeaturizer, totals_features  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v3f = _load("v3f", "scripts/evaluate_v3_final.py")
ks3 = _load("ks3", "scripts/k_sweep_v3.py")

KS = list(range(2, 11))
SETS = {
    "80": {"A": {s: f"outputs/checkpoint5_final/guided_s{s}-*" for s in (42, 43, 44)},
           "B": {s: f"outputs/v3/base80_s{s}-*" for s in (45, 46, 47)},
           "C": {s: f"outputs/v3/base80_s{s}-*" for s in (48, 49, 50)}},
    "40": {"A": {s: f"outputs/checkpoint5_robustness/guided_l0.3_s{s}-*" for s in (42, 43, 44)},
           "B": {s: f"outputs/v3/base_s{s}-*" for s in (45, 46, 47)}},
}


def main() -> None:
    out_dir = ROOT / "outputs/v3/k_sweep_ensembles"
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

    rows = []
    for length, ens in SETS.items():
        emb = {}
        for name, models in ens.items():
            for s, pat in models.items():
                emb[(name, s)] = posterior_means(load_checkpoint(v3f.latest(pat) / "best.pt")[0],
                                                 torch.from_numpy(seq).long())["timing_mu"]
        reps = {name: np.hstack([emb[(name, s)] for s in models]) for name, models in ens.items()}
        singles = list(emb)
        for k in KS:
            fits = {name: ks3.fit(R[tr], R[va], k, 0) for name, R in reps.items()}
            aris, cores = [], []
            for a, b in combinations(reps, 2):
                (la, ma), (lb, mb) = fits[a], fits[b]
                core = (ma >= np.quantile(ma, 0.5)) & (mb >= np.quantile(mb, 0.5))
                aris.append(adjusted_rand_score(la, lb))
                cores.append(adjusted_rand_score(la[core], lb[core]))
            single_l = {key: ks3.fit(E[tr], E[va], k, key[1])[0] for key, E in emb.items()}
            single_ari = np.mean([adjusted_rand_score(single_l[a], single_l[b]) for a, b in combinations(singles, 2)])
            labs = [fits[n][0] for n in reps]
            rows.append({
                "epochs": length, "k": k, "n_ensemble_pairs": len(aris),
                "ARI_mean": float(np.mean(aris)), "ARI_sd": float(np.std(aris)),
                "ARI_core_mean": float(np.mean(cores)), "ARI_core_sd": float(np.std(cores)),
                "single_model_ARI": float(single_ari),
                "eta2_timing": float(np.mean([ks3.eta2(Tclock, l) for l in labs])),
                "eta2_totals": float(np.mean([ks3.eta2(Ttot, l) for l in labs])),
                "midpoint_ratio": float(np.mean([v3f.midpoint_ratio(l, pairs, mid)["ratio"] for l in labs])),
                "smallest_group": float(min(np.bincount(l, minlength=k).min() for l in labs) / len(sv)),
            })
            r = rows[-1]
            print(f"{length} ep k={k}: ARI {r['ARI_mean']:.3f} ± {r['ARI_sd']:.3f} (core {r['ARI_core_mean']:.3f}), "
                  f"single {r['single_model_ARI']:.3f}, midpoint {r['midpoint_ratio']:.2f}, smallest {r['smallest_group']:.1%}",
                  flush=True)
        if length == "80":
            # Per-day confidence labels at k = 4 for ensemble A: group, margin, and "confident" flag (top half).
            la, ma = ks3.fit(reps["A"][tr], reps["A"][va], 4, 0)
            pd.DataFrame({"case_id": z["case_ids"][va], "group": la, "margin": ma,
                          "confident": ma >= np.quantile(ma, 0.5)}).to_csv(out_dir / "confidence_k4_ensembleA_80ep.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "k_sweep_ensembles.csv", index=False)
    lines = ["# Longer training and more ensembles: reproducibility vs k (2022 validation; development)", "",
             "Generated by `scripts/k_sweep_ensembles.py`. 80 epochs: three disjoint 3-model ensembles (A = 42-44,",
             "B = 45-47, C = 48-50) → 3 pairwise comparisons (mean ± SD). 40 epochs: two ensembles → 1 comparison.",
             "`ARI_core`: days confidently assigned by both ensembles of a pair (top half of margin in each).", "",
             df.to_markdown(index=False, floatfmt=".3f"), ""]
    (ROOT / "docs/v3_k_sweep_ensembles.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
