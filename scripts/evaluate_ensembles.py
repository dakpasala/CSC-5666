"""Round 2 (docs/v3_plan.md): are seed-ensemble groups reliable across independent model sets? 2022 validation.

Usage:
    python scripts/evaluate_ensembles.py

For each model family (v1 baseline, E2), ensemble A = seeds 42-44 and ensemble B = seeds 45-47.
Primary: concatenated timing posterior means -> k-means (k=6, n_init=50, fitted on train rows).
Secondary: co-association consensus of the members' k-means labels (average linkage, 6 groups).
Writes outputs/v3/eval/ensembles.json and docs/v3_ensembles.md.
"""

from __future__ import annotations

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import TimingFeaturizer, totals_features  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation import v3_metrics as M  # noqa: E402
from src.evaluation.separation import PROBE_TARGETS  # noqa: E402
from src.evaluation.timing_checks import matched_totals_pairs  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

K = 6
FAMILIES = {
    "v1_base": {42: "outputs/checkpoint5_robustness/guided_l0.3_s42-*", 43: "outputs/checkpoint5_robustness/guided_l0.3_s43-*",
                44: "outputs/checkpoint5_robustness/guided_l0.3_s44-*", 45: "outputs/v3/base_s45-*",
                46: "outputs/v3/base_s46-*", 47: "outputs/v3/base_s47-*"},
    "E2_betaT2": {s: f"outputs/v3/E2_betaT2_s{s}-*" for s in (42, 43, 44, 45, 46, 47)},
}
SETS = {"A": (42, 43, 44), "B": (45, 46, 47)}


def coassoc_consensus(labels: list[np.ndarray], k: int) -> np.ndarray:
    n = len(labels[0])
    co = np.mean([(l[:, None] == l[None, :]) for l in labels], axis=0)
    d = 1.0 - co
    np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(d, checks=False), method="average"), k, criterion="maxclust")


def main() -> None:
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    C, clock, seq = len(cats), z["clock_features"], z["seq"]
    tr, va = z["split"] == "train", z["split"] == "val"
    tot, _ = totals_features(category_counts(seq, C), cats)
    tf = TimingFeaturizer(cats, clock).fit(seq[tr])
    X, names = tf.transform(seq), tf.names_
    clock_cols = [names.index(t) for t in PROBE_TARGETS["clock"]]
    ccols = [i for i, n in enumerate(names) if n.split("_")[0] in ("center", "first", "last")]
    Tc = StandardScaler().fit(X[tr][:, ccols]).transform(X[va][:, ccols])
    pairs = matched_totals_pairs(category_counts(seq[va], C), 120, 5)
    pdist = np.linalg.norm(Tc[pairs[:, 0]] - Tc[pairs[:, 1]], axis=1)

    results = {}
    for fam, pats in FAMILIES.items():
        emb = {}
        for s, pat in pats.items():
            runs = sorted(glob.glob(str(ROOT / pat)))
            if not runs:
                print(f"missing {fam} seed {s}")
                continue
            emb[s] = posterior_means(load_checkpoint(Path(runs[-1]) / "best.pt")[0], torch.from_numpy(seq).long())["timing_mu"]
        single = {s: M.kmeans_labels(E[tr], E[va], K, s, scale=False) for s, E in emb.items()}
        sp = list(combinations(sorted(emb), 2))
        res = {"single_models": {
            "ARI_mean_over_pairs": float(np.mean([adjusted_rand_score(single[a], single[b]) for a, b in sp])),
            "knn20_overlap_mean_over_pairs": float(np.mean([M.knn_overlap(emb[a][va], emb[b][va]) for a, b in sp])),
            "timing_clock_gain_mean": float(np.mean([M.gain_beyond_totals(emb[s], tot, X, clock_cols, tr, va) for s in emb])),
            "timing_split_AUC_mean": float(np.mean([M.timing_split_auc(single[s], pairs, pdist) for s in emb])),
            "n_models": len(emb)}}
        ens_rep, ens_lab, cons_lab = {}, {}, {}
        for name, seeds in SETS.items():
            if not all(s in emb for s in seeds):
                continue
            R = np.hstack([emb[s] for s in seeds])
            ens_rep[name] = R
            ens_lab[name] = M.kmeans_labels(R[tr], R[va], K, 0, scale=False)
            cons_lab[name] = coassoc_consensus([single[s] for s in seeds], K)
        if len(ens_rep) == 2:
            res["ensemble_concat"] = {
                "ARI_A_vs_B": float(adjusted_rand_score(ens_lab["A"], ens_lab["B"])),
                "knn20_overlap_A_vs_B": float(M.knn_overlap(ens_rep["A"][va], ens_rep["B"][va])),
                "timing_clock_gain_A": float(M.gain_beyond_totals(ens_rep["A"], tot, X, clock_cols, tr, va)),
                "timing_split_AUC_A": float(M.timing_split_auc(ens_lab["A"], pairs, pdist)),
                "timing_split_AUC_B": float(M.timing_split_auc(ens_lab["B"], pairs, pdist)),
            }
            res["ensemble_coassoc"] = {"ARI_A_vs_B": float(adjusted_rand_score(cons_lab["A"], cons_lab["B"])),
                                       "smallest_group_share_A": float(np.bincount(cons_lab["A"]).min() / len(cons_lab["A"]))}
        results[fam] = res
        print(fam, json.dumps(res, indent=1), flush=True)

    (ROOT / "outputs/v3/eval/ensembles.json").write_text(json.dumps(results, indent=2))
    lines = ["# v3 round 2: seed ensembles (2022 validation)", "",
             "Generated by `scripts/evaluate_ensembles.py`. Ensemble A = seeds 42-44, B = seeds 45-47 (disjoint models).",
             "Concatenated timing posterior means → k-means (k=6, n_init=50, fitted on 2022 train rows); co-association",
             "consensus as a secondary method. Single-model values are means over all pairs of the 6 models.", "",
             "| Family | Single models: ARI | Single models: kNN-20 overlap | Ensemble A vs B: ARI (concat) | Ensemble A vs B: kNN-20 overlap | Ensemble A vs B: ARI (co-association) | Timing clock gain (single → ensemble) | Timing-split AUC (single → ensemble) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for fam, r in results.items():
        s, e, c = r["single_models"], r.get("ensemble_concat", {}), r.get("ensemble_coassoc", {})
        lines.append(f"| {fam} | {s['ARI_mean_over_pairs']:.3f} | {s['knn20_overlap_mean_over_pairs']:.3f} | "
                     f"{e.get('ARI_A_vs_B', float('nan')):.3f} | {e.get('knn20_overlap_A_vs_B', float('nan')):.3f} | "
                     f"{c.get('ARI_A_vs_B', float('nan')):.3f} | {s['timing_clock_gain_mean']:.3f} → {e.get('timing_clock_gain_A', float('nan')):.3f} | "
                     f"{s['timing_split_AUC_mean']:.3f} → {np.mean([e.get('timing_split_AUC_A', np.nan), e.get('timing_split_AUC_B', np.nan)]):.3f} |")
    lines.append("")
    lines.append("Engineered timing features (round 1 reference): k-means ARI across initialisations 0.819, timing-split AUC 0.732.")
    (ROOT / "docs/v3_ensembles.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
