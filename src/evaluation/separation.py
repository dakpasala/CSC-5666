"""Separation diagnostics for two-latent (or joint) schedule VAEs.

All probes are linear (ridge, alpha by internal CV) and are fitted on TRAIN-split embeddings and scored
on VALIDATION embeddings. Evaluation pairs are generated with a fixed seed from the validation split, so
every model is scored on identical pairs.

1. Pair sensitivity: distance between a diary's posterior mean and its transform's, divided by the
   median distance between random validation diaries (scale-free). Ideal pattern: small for amount,
   larger for timing.
2. Probes: R^2 for predicting true totals and engineered timing features from each latent separately.
3. Timing beyond totals: R^2 gain on timing features from adding a latent to the TRUE totals.
4. Mix-and-match decoding: decode (z_amount of A, z_timing of B). Do decoded totals follow A and decoded
   timing follow B? A diagnostic, not proof of identifiable factors.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src.baselines.features import TimingFeaturizer, totals_features
from src.data.pairs import PairConfig, make_pairs
from src.data.rasterize import category_counts
from src.models.train import posterior_means

EVAL_PAIR_SEED = 2024
ALPHAS = np.logspace(-3, 3, 13)


def _rel_dist(a: np.ndarray, b: np.ndarray, ref: np.ndarray, rng: np.random.Generator) -> dict | None:
    if a.shape[1] == 0:
        return None
    i, j = rng.integers(0, len(ref), 5000), rng.integers(0, len(ref), 5000)
    m = i != j
    scale = np.median(np.linalg.norm(ref[i[m]] - ref[j[m]], axis=1))
    d = np.linalg.norm(a - b, axis=1) / scale
    return {"median": float(np.median(d)), "mean": float(d.mean()), "p90": float(np.quantile(d, 0.9))}


def pair_sensitivity(model, seq_va: np.ndarray, emb_va: dict, n_categories: int) -> dict:
    out = {}
    for name, pcfg in {"shift": PairConfig(p_shift=1.0), "swap": PairConfig(p_shift=0.0)}.items():
        rng = np.random.default_rng(EVAL_PAIR_SEED)
        seq_t, params = make_pairs(seq_va, rng, pcfg, n_categories)
        keep = np.array([p["type"] == ("circular_shift" if name == "shift" else "block_swap") for p in params])
        e_t = posterior_means(model, torch.from_numpy(seq_t[keep]).long())
        rng = np.random.default_rng(EVAL_PAIR_SEED)
        out[name] = {
            "n_pairs": int(keep.sum()),
            "amount": _rel_dist(emb_va["amount_mu"][keep], e_t["amount_mu"], emb_va["amount_mu"], rng),
            "timing": _rel_dist(emb_va["timing_mu"][keep], e_t["timing_mu"], emb_va["timing_mu"], rng),
        }
    return out


def ridge_r2(X_tr: np.ndarray, Y_tr: np.ndarray, X_va: np.ndarray, Y_va: np.ndarray) -> np.ndarray:
    """Per-target validation R^2 of a standardized ridge model; NaN for constant targets."""
    if X_tr.shape[1] == 0:
        return np.full(Y_tr.shape[1], np.nan)
    sx, sy = StandardScaler().fit(X_tr), StandardScaler().fit(Y_tr)
    model = RidgeCV(alphas=ALPHAS).fit(sx.transform(X_tr), sy.transform(Y_tr))
    pred, yv = model.predict(sx.transform(X_va)), sy.transform(Y_va)
    return np.array([r2_score(yv[:, k], pred[:, k]) if Y_tr[:, k].std() > 0 and yv[:, k].std() > 0 else np.nan
                     for k in range(Y_tr.shape[1])])


def probes(emb: dict, seq: np.ndarray, split: np.ndarray, categories: list[str], clock: np.ndarray) -> dict:
    tr, va = split == "train", split == "val"
    counts = category_counts(seq, len(categories))
    tot, _ = totals_features(counts, categories)
    tf = TimingFeaturizer(categories, clock).fit(seq[tr])
    tim = tf.transform(seq)
    # Clock-position features only (centre/first/last), where totals have little linear information.
    clock_cols = np.array([n.split("_")[0] in ("center", "first", "last") for n in tf.names_])

    def summary(r2: np.ndarray) -> float:
        return float(np.nanmean(r2))

    out = {"inputs": {}}
    base_tim = ridge_r2(tot[tr], tim[tr], tot[va], tim[va])
    out["totals_only_baseline"] = {"timing_r2": summary(base_tim), "timing_clock_r2": summary(base_tim[clock_cols])}
    latents = {"amount": emb["amount_mu"], "timing": emb["timing_mu"],
               "both": np.hstack([emb["amount_mu"], emb["timing_mu"]])}
    for name, Z in latents.items():
        if Z.shape[1] == 0:
            out["inputs"][name] = None
            continue
        r_tot = ridge_r2(Z[tr], tot[tr], Z[va], tot[va])
        r_tim = ridge_r2(Z[tr], tim[tr], Z[va], tim[va])
        both = np.hstack([tot, Z])
        r_gain = ridge_r2(both[tr], tim[tr], both[va], tim[va])
        out["inputs"][name] = {
            "totals_r2": summary(r_tot),
            "totals_r2_per_category": dict(zip(categories, np.round(r_tot, 3).tolist())),
            "timing_r2": summary(r_tim),
            "timing_clock_r2": summary(r_tim[clock_cols]),
            "timing_gain_over_true_totals": summary(r_gain) - out["totals_only_baseline"]["timing_r2"],
            "timing_clock_gain_over_true_totals": summary(r_gain[clock_cols]) - out["totals_only_baseline"]["timing_clock_r2"],
        }
    return out


@torch.no_grad()
def mix_and_match(model, seq_va: np.ndarray, emb_va: dict, categories: list[str], clock: np.ndarray,
                  seq_tr: np.ndarray, n: int = 1000) -> dict | None:
    if emb_va["amount_mu"].shape[1] == 0:
        return None
    rng = np.random.default_rng(EVAL_PAIR_SEED)
    a, b = rng.integers(0, len(seq_va), n), rng.integers(0, len(seq_va), n)
    keep = a != b
    a, b = a[keep], b[keep]
    za, zt = torch.from_numpy(emb_va["amount_mu"]), torch.from_numpy(emb_va["timing_mu"])
    dec = model.decode(za[a], zt[b]).argmax(-1).numpy()
    C = len(categories)
    props = lambda s: category_counts(s, C) / s.shape[1]   # noqa: E731
    pd_, pa, pb = props(dec), props(seq_va[a]), props(seq_va[b])
    tf = TimingFeaturizer(categories, clock).fit(seq_tr)
    sc = StandardScaler().fit(tf.transform(seq_tr))
    clock_cols = np.array([nm.split("_")[0] in ("center", "first", "last") for nm in tf.names_])
    td, ta, tb = (sc.transform(tf.transform(x))[:, clock_cols] for x in (dec, seq_va[a], seq_va[b]))
    tot_a, tot_b = np.abs(pd_ - pa).sum(1), np.abs(pd_ - pb).sum(1)
    tim_a, tim_b = np.linalg.norm(td - ta, axis=1), np.linalg.norm(td - tb, axis=1)
    return {
        "n_pairs": int(len(a)),
        "totals_closer_to_amount_source": float((tot_a < tot_b).mean()),
        "timing_closer_to_timing_source": float((tim_b < tim_a).mean()),
        "note": "0.5 = no preference. Ideal separation: both high. Decoded with posterior means, argmax.",
    }


@torch.no_grad()
def mix_and_match_matched_totals(model, seq_va: np.ndarray, emb_va: dict, categories: list[str],
                                 clock: np.ndarray, seq_tr: np.ndarray, max_reassigned_minutes: float = 120,
                                 bin_minutes: int = 5) -> dict | None:
    """Mix-and-match restricted to validation pairs with near-identical totals.

    In random pairs, which activities are present at all follows the totals (and so the amount source),
    which confounds the timing comparison. Here A and B share nearly the same totals, so the question is
    only whether *when* things happen follows the timing latent's source (B). Both directions are used.
    """
    if emb_va["amount_mu"].shape[1] == 0:
        return None
    from src.evaluation.timing_checks import matched_totals_pairs
    C = len(categories)
    pairs = matched_totals_pairs(category_counts(seq_va, C), max_reassigned_minutes, bin_minutes)
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    za, zt = torch.from_numpy(emb_va["amount_mu"]), torch.from_numpy(emb_va["timing_mu"])
    dec = model.decode(za[a], zt[b]).argmax(-1).numpy()
    tf = TimingFeaturizer(categories, clock).fit(seq_tr)
    sc = StandardScaler().fit(tf.transform(seq_tr))
    clock_cols = np.array([nm.split("_")[0] in ("center", "first", "last") for nm in tf.names_])
    td, ta, tb = (sc.transform(tf.transform(x))[:, clock_cols] for x in (dec, seq_va[a], seq_va[b]))
    tim_a, tim_b = np.linalg.norm(td - ta, axis=1), np.linalg.norm(td - tb, axis=1)
    slot_a, slot_b = (dec == seq_va[a]).mean(1), (dec == seq_va[b]).mean(1)
    return {
        "n_directed_pairs": int(len(a)), "max_reassigned_minutes": max_reassigned_minutes,
        "timing_closer_to_timing_source": float((tim_b < tim_a).mean()),
        "slot_agreement_with_timing_source": float(slot_b.mean()),
        "slot_agreement_with_amount_source": float(slot_a.mean()),
        "note": "0.5 = no preference; pairs have near-identical totals, so presence is not a confound.",
    }


def evaluate_separation(model, seq: np.ndarray, split: np.ndarray, categories: list[str], clock: np.ndarray,
                        emb: dict) -> dict:
    va = split == "val"
    emb_va = {k: v[va] for k, v in emb.items()}
    return {
        "pair_sensitivity_val": pair_sensitivity(model, seq[va], emb_va, len(categories)),
        "probes": probes(emb, seq, split, categories, clock),
        "mix_and_match_val": mix_and_match(model, seq[va], emb_va, categories, clock, seq[split == "train"]),
        "mix_and_match_matched_totals_val": mix_and_match_matched_totals(model, seq[va], emb_va, categories, clock,
                                                                         seq[split == "train"]),
    }
