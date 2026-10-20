"""Scaling -> optional PCA -> k-means, all fitted on training rows only; k chosen on validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusterPipeline:
    k: int
    seed: int = 42
    scale: bool = True
    pca_variance: float | None = None
    n_init: int = 10

    def fit(self, X: np.ndarray) -> "ClusterPipeline":
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        Z = self._scale(X)
        self.pca_ = PCA(n_components=self.pca_variance, random_state=self.seed).fit(Z) if self.pca_variance else None
        Z = self.pca_.transform(Z) if self.pca_ is not None else Z
        self.km_ = KMeans(self.k, n_init=self.n_init, random_state=self.seed).fit(Z)
        return self

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return self.scaler_.transform(X) if self.scaler_ is not None else X

    def embed(self, X: np.ndarray) -> np.ndarray:
        Z = self._scale(X)
        return self.pca_.transform(Z) if self.pca_ is not None else Z

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.km_.predict(self.embed(X))

    def centroid_distance(self, X: np.ndarray) -> np.ndarray:
        return self.km_.transform(self.embed(X)).min(axis=1)


def bootstrap_stability(X_tr: np.ndarray, X_va: np.ndarray, k: int, n_boot: int, seed: int,
                        **kw) -> tuple[float, float]:
    """Refit on bootstrap resamples of train; mean/sd pairwise ARI of the SAME validation diaries."""
    rng = np.random.default_rng(seed)
    labels = []
    for b in range(n_boot):
        idx = rng.integers(0, len(X_tr), len(X_tr))
        labels.append(ClusterPipeline(k, seed=seed + b, **kw).fit(X_tr[idx]).predict(X_va))
    aris = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
    return float(np.mean(aris)), float(np.std(aris))


def select_k(X_tr: np.ndarray, X_va: np.ndarray, ks: list[int], n_boot: int, seed: int,
             silhouette_sample: int | None = None, **kw) -> pd.DataFrame:
    rows = []
    for k in ks:
        p = ClusterPipeline(k, seed=seed, **kw).fit(X_tr)
        Z_va, lab = p.embed(X_va), p.predict(X_va)
        ari_mean, ari_sd = bootstrap_stability(X_tr, X_va, k, n_boot, seed, **kw)
        rows.append({
            "k": k,
            "train_inertia": float(p.km_.inertia_),
            "val_silhouette": float(silhouette_score(Z_va, lab, sample_size=silhouette_sample, random_state=seed)),
            "val_davies_bouldin": float(davies_bouldin_score(Z_va, lab)),
            "bootstrap_ari_mean": ari_mean,
            "bootstrap_ari_sd": ari_sd,
            "smallest_val_cluster_share": float(np.bincount(lab, minlength=k).min() / len(lab)),
        })
    return pd.DataFrame(rows)


def choose_k(table: pd.DataFrame, min_stability: float, min_cluster_share: float) -> int:
    """Provisional rule: best validation silhouette among stable k with no tiny clusters; ties -> smaller k.

    Falls back to the most stable k if none qualify.
    """
    ok = table[(table["bootstrap_ari_mean"] >= min_stability) & (table["smallest_val_cluster_share"] >= min_cluster_share)]
    if ok.empty:
        return int(table.sort_values(["bootstrap_ari_mean", "k"], ascending=[False, True]).iloc[0]["k"])
    return int(ok.sort_values(["val_silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
