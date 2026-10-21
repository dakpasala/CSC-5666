"""Metrics for the v3 experiments (docs/v3_plan.md). 2022 train/validation only.

All probe models, scalers and k-means centroids are fitted on train-split rows and scored on validation rows.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, f1_score, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from src.data.pairs import PairConfig, make_pairs
from src.data.rasterize import category_counts
from src.evaluation.clustering import ClusterPipeline
from src.evaluation.reconstruction import latent_diagnostics
from src.models.train import posterior_means
from src.models.vae_v2 import exact_budget_decode, integer_budgets

MAIN = ["sleep", "work", "leisure_social", "household", "eating_drinking", "travel", "personal_care", "caregiving"]
TRANSFORMS = {
    "global_shift": (PairConfig(p_shift=1.0), "circular_shift"),
    "far_swap": (PairConfig(p_shift=0.0), "block_swap"),
    "split_merge": (PairConfig(p_split_merge=1.0), "split_merge"),
    "local_shift": (PairConfig(family="local", p_local_shift=1.0), "local_shift"),
}
EVAL_SEED = 2024


def tree_r2(Xtr, ytr, Xva, yva) -> float:
    if ytr.std() == 0:
        return float("nan")
    return float(r2_score(yva, HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr).predict(Xva)))


# ---------------------------------------------------------------- information in each latent ----
def totals_r2(Z, tot, tr, va, cats) -> float:
    return float(np.nanmean([tree_r2(Z[tr], tot[tr, cats.index(c)], Z[va], tot[va, cats.index(c)]) for c in MAIN]))


def gain_beyond_totals(Z, tot, X, cols, tr, va) -> float:
    both = np.hstack([tot, Z])
    g = [tree_r2(both[tr], X[tr, j], both[va], X[va, j]) - tree_r2(tot[tr], X[tr, j], tot[va], X[va, j]) for j in cols]
    return float(np.nanmean(g))


# ---------------------------------------------------------------- transforms ---------------------
def transform_sensitivity(model, seq_va: np.ndarray, emb_va: dict, C: int) -> dict:
    """Median distance between a diary's posterior mean and its transform's, relative to the median
    distance between random validation diaries (per latent)."""
    rng = np.random.default_rng(EVAL_SEED)
    i, j = rng.integers(0, len(seq_va), 5000), rng.integers(0, len(seq_va), 5000)
    ok = i != j
    out = {}
    for name, (pc, kind) in TRANSFORMS.items():
        t, params = make_pairs(seq_va, np.random.default_rng(EVAL_SEED), pc, C)
        keep = np.array([p["type"] == kind for p in params])
        e_t = posterior_means(model, torch.from_numpy(t[keep]).long())
        for lat in ("amount_mu", "timing_mu"):
            ref = emb_va[lat]
            if ref.shape[1] == 0:
                continue
            scale = np.median(np.linalg.norm(ref[i[ok]] - ref[j[ok]], axis=1))
            d = np.linalg.norm(ref[keep] - e_t[lat], axis=1) / scale
            out[f"{lat[:-3]}_{name}"] = float(np.median(d))
    return out


# ---------------------------------------------------------------- decoder dependence ------------
@torch.no_grad()
def decoder_dependence(model, seq_va: np.ndarray) -> dict:
    t = torch.from_numpy(seq_va).long()
    q = model.encode(t)
    perm = torch.randperm(len(t), generator=torch.Generator().manual_seed(0))

    def ce(za, zt):
        return float(-torch.log_softmax(model.decode(za, zt), -1).gather(-1, t[..., None]).mean())
    out = {"CE_own": ce(q["amount_mu"], q["timing_mu"]), "CE_timing_from_other": ce(q["amount_mu"], q["timing_mu"][perm])}
    if q["amount_mu"].shape[1]:
        out["CE_amount_from_other"] = ce(q["amount_mu"][perm], q["timing_mu"])
    return out


# ---------------------------------------------------------------- reconstruction -----------------
def tolerant_f1(true: np.ndarray, pred: np.ndarray, c: int, tol: int) -> float:
    t, p = true == c, pred == c
    st = np.ones((1, 2 * tol + 1), bool)
    prec = (p & binary_dilation(t, structure=st)).sum() / max(p.sum(), 1)
    rec = (t & binary_dilation(p, structure=st)).sum() / max(t.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))


@torch.no_grad()
def reconstruction(model, seq_va: np.ndarray, cats: list[str], w: int = 5) -> dict:
    """Argmax decoding vs self-budgeted decoding (integer budgets from the decoder's own expected minutes,
    then exact assignment). Neither uses the observed totals."""
    C = len(cats)
    t = torch.from_numpy(seq_va).long()
    q = model.encode(t)
    logp = torch.log_softmax(model.decode(q["amount_mu"], q["timing_mu"]), -1).numpy()
    out = {"CE": float(-np.take_along_axis(logp, seq_va[..., None].astype(np.int64), -1).mean())}
    argmax = logp.argmax(-1).astype(np.int8)
    budget = integer_budgets(np.exp(logp).mean(1), logp.shape[1])
    selfb = exact_budget_decode(logp, budget)
    true_cnt = category_counts(seq_va, C)
    for name, pred in (("argmax", argmax), ("selfbudget", selfb)):
        cnt = category_counts(pred, C)
        out[f"{name}_acc"] = float((pred == seq_va).mean())
        out[f"{name}_macroF1"] = float(f1_score(seq_va.ravel(), pred.ravel(), labels=np.arange(C), average="macro", zero_division=0))
        out[f"{name}_transitions"] = float((np.diff(pred, axis=1) != 0).sum(1).mean())
        out[f"{name}_dur_err_min"] = float(np.abs(cnt.astype(int) - true_cnt.astype(int)).mean() * w)
        for c in ("eating_drinking", "travel"):
            out[f"{name}_F1_{c}"] = tolerant_f1(seq_va, pred, cats.index(c), 0)
            out[f"{name}_F1_{c}_15min"] = tolerant_f1(seq_va, pred, cats.index(c), 3)
    out["true_transitions"] = float((np.diff(seq_va, axis=1) != 0).sum(1).mean())
    return out


# ---------------------------------------------------------------- stability & groups ------------
def knn_overlap(A: np.ndarray, B: np.ndarray, k: int = 20) -> float:
    na = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]
    nb = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(na, nb)]))


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def kmeans_labels(R_tr: np.ndarray, R_va: np.ndarray, k: int, seed: int, scale: bool) -> np.ndarray:
    return ClusterPipeline(k, seed=seed, scale=scale, n_init=50).fit(R_tr).predict(R_va)


def timing_split_auc(labels: np.ndarray, pairs: np.ndarray, pair_dist: np.ndarray) -> float:
    split = labels[pairs[:, 0]] != labels[pairs[:, 1]]
    return float(roc_auc_score(split, pair_dist)) if 0 < split.sum() < len(split) else float("nan")


def cross_seed(reps_va: dict[int, np.ndarray], labels: dict[int, np.ndarray]) -> dict:
    ss = sorted(reps_va)
    pairs = list(combinations(ss, 2))
    return {"knn20_overlap": float(np.mean([knn_overlap(reps_va[a], reps_va[b]) for a, b in pairs])),
            "linear_CKA": float(np.mean([linear_cka(reps_va[a], reps_va[b]) for a, b in pairs])),
            "kmeans_ARI": float(np.mean([adjusted_rand_score(labels[a], labels[b]) for a, b in pairs]))}


def latent_summary(emb_va: dict) -> dict:
    out = {}
    for lat in ("amount", "timing"):
        mu, lv = emb_va[f"{lat}_mu"], emb_va[f"{lat}_logvar"]
        if mu.shape[1]:
            d = latent_diagnostics(mu, lv)
            out[f"{lat}_KL"], out[f"{lat}_active"] = d["kl_total_mean"], d["active_dims"]
    return out
