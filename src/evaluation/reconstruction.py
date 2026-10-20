"""Reconstruction and latent diagnostics for sequence models.

Accuracy alone is inflated by long majority activities (sleep), so we also report macro-F1, per-category
duration error in minutes, transition-count error (exposes oversmoothing), and a position-majority
reference that predicts the most common training activity at each time of day.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from src.data.rasterize import category_counts


def reconstruction_metrics(true: np.ndarray, pred: np.ndarray, n_categories: int, bin_minutes: int,
                           logp: np.ndarray | None = None) -> dict:
    t, p = true.reshape(-1), pred.reshape(-1)
    ct, cp = category_counts(true, n_categories), category_counts(pred, n_categories)
    dur_err = np.abs(cp.astype(int) - ct.astype(int)) * bin_minutes                # (N, C) minutes
    tr_true = (np.diff(true, axis=1) != 0).sum(1)
    tr_pred = (np.diff(pred, axis=1) != 0).sum(1)
    out = {
        "accuracy": float((t == p).mean()),
        "macro_f1": float(f1_score(t, p, labels=np.arange(n_categories), average="macro", zero_division=0)),
        "per_category_f1": f1_score(t, p, labels=np.arange(n_categories), average=None, zero_division=0).round(4).tolist(),
        "duration_abs_error_min_mean": float(dur_err.mean()),
        "duration_abs_error_min_per_category": dur_err.mean(axis=0).round(2).tolist(),
        "reassigned_minutes_per_diary_mean": float(dur_err.sum(1).mean() / 2),
        "transitions_true_mean": float(tr_true.mean()),
        "transitions_pred_mean": float(tr_pred.mean()),
        "transitions_abs_error_mean": float(np.abs(tr_pred - tr_true).mean()),
    }
    if logp is not None:
        nll = -np.take_along_axis(logp, true[..., None].astype(np.int64), axis=-1)[..., 0]
        out["cross_entropy"] = float(nll.mean())
    return out


def position_majority(train_seq: np.ndarray, n_categories: int) -> np.ndarray:
    """Most common training category at each position: (T,)."""
    counts = np.stack([(train_seq == k).sum(0) for k in range(n_categories)])
    return counts.argmax(0).astype(np.int8)


def latent_diagnostics(mu: np.ndarray, logvar: np.ndarray, active_threshold: float = 0.01) -> dict:
    """Per-dimension KL to N(0,1) (mean over diaries) and variance of posterior means across diaries.

    A dimension whose posterior mean barely varies across diaries is inactive (carries no information).
    """
    kl = 0.5 * (mu ** 2 + np.exp(logvar) - 1 - logvar)
    var = mu.var(axis=0)
    return {
        "kl_per_dim_mean": kl.mean(0).round(4).tolist(),
        "kl_total_mean": float(kl.sum(1).mean()),
        "posterior_mean_variance_per_dim": var.round(4).tolist(),
        "active_dims": int((var > active_threshold).sum()),
        "n_dims": int(mu.shape[1]),
        "mean_posterior_std": float(np.exp(0.5 * logvar).mean()),
    }
