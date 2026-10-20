"""Checks that the evaluation can see temporal organization when activity totals match.

1. Transform sensitivity: total-preserving circular shifts and block swaps of validation diaries.
   A totals-only representation must not move at all; a timing-aware one should.
2. Matched-totals pairs: real validation diaries whose nearest neighbour by totals is within a few
   minutes. Do timing features still separate them, and do clusterings put them together?
3. Totals -> timing predictability: how much of each timing feature a linear model of totals explains.
   R^2 well below 1 means timing carries information that totals do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data.transforms import block_swap, circular_shift


def make_transformed(seq: np.ndarray, kind: str, n_categories: int, rng: np.random.Generator,
                     shift: int | None = None) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Transform each diary; returns (row indices kept, transformed seqs, params). Rejections dropped."""
    keep, out, params = [], [], []
    for i, s in enumerate(seq):
        t, p = circular_shift(s, shift, n_categories) if kind == "shift" else block_swap(s, rng, n_categories)
        if t is not None:
            keep.append(i)
            out.append(t)
            params.append(p)
    return np.array(keep), np.array(out, dtype=seq.dtype), params


def median_pair_distance(Z: np.ndarray, rng: np.random.Generator, n: int = 5000) -> float:
    a, b = rng.integers(0, len(Z), n), rng.integers(0, len(Z), n)
    m = a != b
    return float(np.median(np.linalg.norm(Z[a[m]] - Z[b[m]], axis=1)))


def transform_sensitivity(baselines: dict, seq: np.ndarray, counts: np.ndarray, n_categories: int,
                          shifts: list[int], seed: int) -> pd.DataFrame:
    """`baselines`: name -> (FeatureSet, ClusterPipeline). Distances are in each pipeline's scaled space,
    expressed relative to the median distance between random pairs of untouched diaries."""
    rng = np.random.default_rng(seed)
    cases = [("shift", s) for s in shifts] + [("swap", None)]
    rows = []
    for kind, s in cases:
        idx, tseq, params = make_transformed(seq, kind, n_categories, np.random.default_rng(seed), s)
        tcounts = counts[idx]  # unchanged by construction (asserted inside each transform)
        for name, (fs, pipe) in baselines.items():
            Z0 = pipe.embed(fs.transform(seq, counts))
            Z1 = pipe.embed(fs.transform(tseq, tcounts))
            ref = median_pair_distance(Z0, rng)
            d = np.linalg.norm(Z0[idx] - Z1, axis=1) / ref
            l0, l1 = pipe.predict(fs.transform(seq[idx], tcounts)), pipe.predict(fs.transform(tseq, tcounts))
            rows.append({
                "transform": kind if s is None else f"shift_{s:+d}",
                "baseline": name, "n_pairs": len(idx), "n_rejected": len(seq) - len(idx),
                "rel_distance_median": float(np.median(d)), "rel_distance_p90": float(np.quantile(d, 0.9)),
                "share_moved_at_all": float((d > 1e-9).mean()),
                "cluster_reassignment_rate": float((l0 != l1).mean()),
            })
    return pd.DataFrame(rows)


def matched_totals_pairs(counts: np.ndarray, max_reassigned_minutes: float, bin_minutes: int) -> np.ndarray:
    """Mutual-or-not nearest neighbours by totals within the threshold. Returns (P, 2) index pairs."""
    nn = NearestNeighbors(n_neighbors=2, metric="manhattan").fit(counts)
    dist, ind = nn.kneighbors(counts)
    reassigned = dist[:, 1] * bin_minutes / 2
    i = np.flatnonzero(reassigned <= max_reassigned_minutes)
    pairs = np.sort(np.stack([i, ind[i, 1]], axis=1), axis=1)
    return np.unique(pairs, axis=0)


def matched_pairs_report(pairs: np.ndarray, labels: dict[str, np.ndarray], timing_Z: np.ndarray,
                         seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(timing_Z)
    ra, rb = rng.integers(0, n, 20000), rng.integers(0, n, 20000)
    m = ra != rb
    ref = np.median(np.linalg.norm(timing_Z[ra[m]] - timing_Z[rb[m]], axis=1))
    d = np.linalg.norm(timing_Z[pairs[:, 0]] - timing_Z[pairs[:, 1]], axis=1) / ref
    return {
        "n_pairs": int(len(pairs)),
        "timing_distance_rel_to_random_pairs": {"median": float(np.median(d)), "p10": float(np.quantile(d, 0.1)),
                                                "p90": float(np.quantile(d, 0.9))},
        "same_cluster_rate": {k: float((v[pairs[:, 0]] == v[pairs[:, 1]]).mean()) for k, v in labels.items()},
        "same_cluster_rate_random_pairs": {k: float((v[ra[m]] == v[rb[m]]).mean()) for k, v in labels.items()},
    }


def totals_to_timing_r2(Xtot_tr: np.ndarray, Xtim_tr: np.ndarray, Xtot_va: np.ndarray, Xtim_va: np.ndarray,
                        names: list[str]) -> pd.DataFrame:
    """Validation R^2 of a ridge model predicting each (standardized) timing feature from totals."""
    sx, sy = StandardScaler().fit(Xtot_tr), StandardScaler().fit(Xtim_tr)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(sx.transform(Xtot_tr), sy.transform(Xtim_tr))
    pred = model.predict(sx.transform(Xtot_va))
    yv = sy.transform(Xtim_va)
    r2 = [r2_score(yv[:, j], pred[:, j]) if yv[:, j].std() > 0 else np.nan for j in range(yv.shape[1])]
    df = pd.DataFrame({"feature": names, "val_r2": r2})
    df["group"] = df["feature"].str.split("_").str[0]
    return df
