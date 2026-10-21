"""Diagnose model v1's weaknesses on 2022 validation (no training).

Usage:
    python scripts/diagnose_v1.py

Uses the frozen v1 final models (outputs/checkpoint5_final, 80 epochs, seeds 42/43/44): guided (λ=0.3),
joint, plus engineered features. Answers the five diagnostic questions in docs/v3_plan.md and writes
outputs/v3/diagnosis/diagnosis.json and docs/v3_diagnosis_tables.md. 2022 data only.
"""

from __future__ import annotations

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import adjusted_rand_score, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.features import FeatureSet, TimingFeaturizer, totals_features  # noqa: E402
from src.data.pairs import PairConfig, make_pairs  # noqa: E402
from src.data.rasterize import category_counts  # noqa: E402
from src.evaluation.clustering import ClusterPipeline  # noqa: E402
from src.evaluation.reconstruction import latent_diagnostics  # noqa: E402
from src.models.train import load_checkpoint, posterior_means  # noqa: E402

SEEDS = [42, 43, 44]
K = 6
OUT = ROOT / "outputs/v3/diagnosis"


def run_dir(prefix: str, seed: int) -> str:
    return sorted(glob.glob(str(ROOT / f"outputs/checkpoint5_final/{prefix}_s{seed}-*")))[-1]


def tree_r2(Xtr, ytr, Xva, yva) -> float:
    return r2_score(yva, HistGradientBoostingRegressor(random_state=0).fit(Xtr, ytr).predict(Xva))


def multi_tree_r2(Xtr, Ytr, Xva, Yva) -> np.ndarray:
    return np.array([tree_r2(Xtr, Ytr[:, j], Xva, Yva[:, j]) if Ytr[:, j].std() > 0 else np.nan
                     for j in range(Ytr.shape[1])])


def knn_overlap(A: np.ndarray, B: np.ndarray, k: int = 20) -> float:
    """Mean Jaccard overlap of k-nearest-neighbour sets of the same diaries in two representations."""
    na = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A, return_distance=False)[:, 1:]
    nb = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(na, nb)]))


def linear_cka(A: np.ndarray, B: np.ndarray) -> float:
    A, B = A - A.mean(0), B - B.mean(0)
    return float(np.linalg.norm(A.T @ B) ** 2 / (np.linalg.norm(A.T @ A) * np.linalg.norm(B.T @ B)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    cats = z["categories"].tolist()
    C, clock = len(cats), z["clock_features"]
    tr, va = z["split"] == "train", z["split"] == "val"
    seq = z["seq"]
    tot, _ = totals_features(category_counts(seq, C), cats)
    tf = TimingFeaturizer(cats, clock).fit(seq[tr])
    X = tf.transform(seq)
    names = tf.names_
    clock_cols = [i for i, n in enumerate(names) if n.split("_")[0] in ("center", "first", "last")]
    frag_cols = [i for i, n in enumerate(names) if n.split("_")[0] in ("bouts", "transitions")]
    D: dict = {}

    emb = {}
    models = {}
    for prefix in ("guided", "joint"):
        for s in SEEDS:
            m, _ = load_checkpoint(Path(run_dir(prefix, s)) / "best.pt")
            models[(prefix, s)] = m
            emb[(prefix, s)] = posterior_means(m, torch.from_numpy(seq).long())

    # ---- Q1: totals information in each latent, against the natural amount-timing correlation -----
    main_cats = ["sleep", "work", "leisure_social", "household", "eating_drinking", "travel", "personal_care", "caregiving"]
    mc = [cats.index(c) for c in main_cats]
    q1 = {}
    reps = {"v1 amount latent": emb[("guided", 42)]["amount_mu"], "v1 timing latent": emb[("guided", 42)]["timing_mu"],
            "joint latent": emb[("joint", 42)]["timing_mu"],
            "engineered clock-only timing features": X[:, clock_cols],
            "engineered clock + fragmentation features": X[:, clock_cols + frag_cols]}
    for name, R in reps.items():
        r2 = multi_tree_r2(R[tr], tot[tr][:, mc], R[va], tot[va][:, mc])
        q1[name] = {"mean": float(np.nanmean(r2)), **dict(zip(main_cats, np.round(r2, 3).tolist()))}
    D["q1_totals_R2_tree"] = q1

    # ---- Q2: collapse and decoder dependence -------------------------------------------------------
    q2 = {}
    for s in SEEDS:
        e, m = emb[("guided", s)], models[("guided", s)]
        la = latent_diagnostics(e["amount_mu"][va], e["amount_logvar"][va])
        lt = latent_diagnostics(e["timing_mu"][va], e["timing_logvar"][va])
        t = torch.from_numpy(seq[va]).long()
        with torch.no_grad():
            q = m.encode(t)
            base = m.decode(q["amount_mu"], q["timing_mu"])
            perm = torch.randperm(len(t), generator=torch.Generator().manual_seed(0))
            swap_a = m.decode(q["amount_mu"][perm], q["timing_mu"])   # amount from another diary
            swap_t = m.decode(q["amount_mu"], q["timing_mu"][perm])   # timing from another diary
        nll = lambda lg: float(-torch.log_softmax(lg, -1).gather(-1, t[..., None]).mean())  # noqa: E731
        q2[s] = {"amount_KL": la["kl_total_mean"], "amount_active": la["active_dims"],
                 "timing_KL": lt["kl_total_mean"], "timing_active": lt["active_dims"],
                 "CE_both_own": nll(base), "CE_amount_from_other_diary": nll(swap_a),
                 "CE_timing_from_other_diary": nll(swap_t)}
        # What each latent carries beyond totals (tree probes): clock and fragmentation.
        for lat in ("amount_mu", "timing_mu"):
            both = np.hstack([tot, e[lat]])
            for grp, cols in (("clock", clock_cols[:12]), ("frag", frag_cols)):
                g = []
                for j in cols:
                    if X[tr][:, j].std() == 0:
                        continue
                    g.append(tree_r2(both[tr], X[tr][:, j], both[va], X[va][:, j]) - tree_r2(tot[tr], X[tr][:, j], tot[va], X[va][:, j]))
                q2[s][f"{lat[:-3]}_{grp}_gain_tree"] = float(np.mean(g))
        if s != 42:
            for k in [k for k in q2[s] if k.endswith("_tree")]:
                pass
    D["q2_collapse_and_dependence"] = q2

    # ---- Q3: transformations — plausibility and what they change -----------------------------------
    marg = np.stack([(seq[tr] == k).mean(0) for k in range(C)], 1) + 1e-4          # (T, C) time-of-day marginal
    logm = np.log(marg / marg.sum(1, keepdims=True))
    def plaus(s_):
        return np.take_along_axis(logm[None].repeat(len(s_), 0), s_[..., None].astype(int), 2)[..., 0].mean(1)
    sv = seq[va]
    base_pl = plaus(sv)
    q3 = {"original": {"mean_logp_per_slot": float(base_pl.mean())}}
    Xv = X[va]
    for name, pc in {"shift": PairConfig(p_shift=1.0), "swap": PairConfig(p_shift=0.0),
                     "split_merge": PairConfig(p_split_merge=1.0)}.items():
        out, params = make_pairs(sv, np.random.default_rng(0), pc, C)
        Xt = tf.transform(out)
        pl = plaus(out)
        wraps_sleep = [p for p in params if p["type"] == "circular_shift"]
        q3[name] = {
            "mean_logp_per_slot": float(pl.mean()),
            "share_less_plausible_than_original": float((pl < base_pl).mean()),
            "mean_abs_change_clock_features": float(np.abs(Xt[:, clock_cols] - Xv[:, clock_cols]).mean()),
            "mean_abs_change_bouts": float(np.abs(Xt[:, [i for i in frag_cols if names[i].startswith("bouts")]]
                                                  - Xv[:, [i for i in frag_cols if names[i].startswith("bouts")]]).sum(1).mean()),
            "mean_abs_change_transitions": float(np.abs(Xt[:, names.index("transitions")] - Xv[:, names.index("transitions")]).mean()),
        }
        if wraps_sleep:
            q3[name]["share_shifts_wrapping_non_sleep"] = float(np.mean([any(c != cats.index("sleep") for c in p.get("wrapped_categories", [])) for p in wraps_sleep]))
    D["q3_transforms"] = q3

    # ---- Q4: oversmoothing — argmax decoding vs what the probabilities know --------------------------
    q4 = {}
    m = models[("guided", 42)]
    with torch.no_grad():
        q = m.encode(torch.from_numpy(sv).long())
        prob = torch.softmax(m.decode(q["amount_mu"], q["timing_mu"]), -1).numpy()       # (N, T, C)
    argmax = prob.argmax(-1)
    rng = np.random.default_rng(0)
    cum = prob.cumsum(-1)
    sampled = (rng.random(prob.shape[:2])[..., None] > cum).sum(-1).clip(max=C - 1)
    true_min = category_counts(sv, C) * 5
    exp_min = prob.sum(1) * 5
    arg_min = category_counts(argmax, C) * 5
    for c in ("eating_drinking", "travel", "personal_care", "household", "work"):
        k = cats.index(c)
        q4[c] = {"true_mean_min": float(true_min[:, k].mean()),
                 "argmax_mean_min": float(arg_min[:, k].mean()),
                 "expected_mean_min_from_probs": float(exp_min[:, k].mean()),
                 "abs_err_argmax": float(np.abs(arg_min[:, k] - true_min[:, k]).mean()),
                 "abs_err_expected": float(np.abs(exp_min[:, k] - true_min[:, k]).mean()),
                 "max_prob_mean_where_true": float(prob[..., k][sv == k].mean())}
    q4["transitions"] = {"true": float((np.diff(sv, axis=1) != 0).sum(1).mean()),
                         "argmax": float((np.diff(argmax, axis=1) != 0).sum(1).mean()),
                         "sampled": float((np.diff(sampled, axis=1) != 0).sum(1).mean())}
    D["q4_oversmoothing"] = q4

    # ---- Q5: cluster instability — k-means vs representation -----------------------------------------
    q5 = {}
    reps5 = {f"v1 timing s{s}": emb[("guided", s)]["timing_mu"] for s in SEEDS}
    reps5.update({f"joint s{s}": emb[("joint", s)]["timing_mu"] for s in SEEDS})
    fs = FeatureSet("eng", False, True, cats, clock).fit(seq[tr])
    E = fs.transform(seq, category_counts(seq, C))
    E = StandardScaler().fit(E[tr]).transform(E)
    reps5["engineered timing"] = E
    for name, R in reps5.items():
        labs1 = [ClusterPipeline(K, seed=s, scale=False, n_init=1).fit(R[tr]).predict(R[va]) for s in range(10)]
        labs50 = [ClusterPipeline(K, seed=s, scale=False, n_init=50).fit(R[tr]).predict(R[va]) for s in range(3)]
        q5[name] = {"kmeans_init_ARI_n_init1": float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labs1, 2)])),
                    "kmeans_init_ARI_n_init50": float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(labs50, 2)]))}
        q5[name]["_labels50"] = labs50[0]
    for model in ("v1 timing", "joint"):
        R = {s: reps5[f"{model} s{s}"][va] for s in SEEDS}
        L = {s: q5[f"{model} s{s}"]["_labels50"] for s in SEEDS}
        q5[f"{model} across model seeds"] = {
            "ARI_n_init50": float(np.mean([adjusted_rand_score(L[a], L[b]) for a, b in combinations(SEEDS, 2)])),
            "knn20_overlap": float(np.mean([knn_overlap(R[a], R[b]) for a, b in combinations(SEEDS, 2)])),
            "linear_CKA": float(np.mean([linear_cka(R[a], R[b]) for a, b in combinations(SEEDS, 2)])),
        }
    # Engineered features with feature-level bootstrap noise is not comparable; report kNN overlap with v1 instead.
    q5["v1 timing s42 vs engineered timing: knn20_overlap"] = knn_overlap(reps5["v1 timing s42"][va], E[va])
    for v in q5.values():
        if isinstance(v, dict):
            v.pop("_labels50", None)
    D["q5_cluster_instability"] = q5

    (OUT / "diagnosis.json").write_text(json.dumps(D, indent=2, default=float))
    print(json.dumps(D, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
