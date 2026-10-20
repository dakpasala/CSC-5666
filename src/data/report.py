"""Discretization-error and boundary-censoring diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rasterize import n_runs


def discretization_report(ep: pd.DataFrame, case_ids: list[str], minutes: np.ndarray,
                          seq: np.ndarray, bin_minutes: int, categories: list[str]) -> dict:
    """Compare raw minute totals/runs with the binned sequence."""
    C = len(categories)
    raw = (minutes[..., None] == np.arange(C)).sum(axis=1)                  # minutes
    binned = (seq[..., None] == np.arange(C)).sum(axis=1) * bin_minutes     # minutes
    abs_err = np.abs(binned - raw)
    # Half the L1 distance = minutes reassigned to a different category.
    reassigned = abs_err.sum(axis=1) / 2

    # Episodes whose category is absent from every bin they overlap ("lost" short episodes).
    order = {c: i for i, c in enumerate(case_ids)}
    sub = ep[ep["case_id"].isin(order)]
    rows = sub["case_id"].map(order).to_numpy()
    b0 = sub["start_min"].to_numpy() // bin_minutes
    b1 = (sub["end_min"].to_numpy() - 1) // bin_minutes
    cat = sub["category_idx"].to_numpy()
    lost = np.fromiter(
        (not (seq[r, s:e + 1] == c).any() for r, s, e, c in zip(rows, b0, b1, cat)),
        dtype=bool, count=len(sub),
    )
    short = sub["duration_min"].to_numpy() < bin_minutes

    raw_runs = n_runs(minutes)
    bin_runs = n_runs(seq)
    present = raw > 0
    return {
        "n_diaries": len(case_ids),
        "bin_minutes": bin_minutes,
        "reassigned_minutes_per_diary": _describe(reassigned),
        "share_of_day_reassigned_mean": float(reassigned.mean() / minutes.shape[1]),
        "per_category_abs_error_minutes_mean": dict(zip(categories, abs_err.mean(axis=0).round(2).tolist())),
        "per_category_signed_error_minutes_mean": dict(zip(categories, (binned - raw).mean(axis=0).round(2).tolist())),
        "per_category_raw_minutes_mean": dict(zip(categories, raw.mean(axis=0).round(1).tolist())),
        "categories_dropped_entirely": {
            c: int((present[:, i] & (binned[:, i] == 0)).sum()) for i, c in enumerate(categories)
        },
        "episodes_total": int(len(sub)),
        "episodes_shorter_than_bin": int(short.sum()),
        "episodes_lost_after_binning": int(lost.sum()),
        "episodes_lost_by_category": sub.loc[lost, "category"].value_counts().to_dict(),
        "runs_per_diary_raw_minute": _describe(raw_runs),
        "runs_per_diary_binned": _describe(bin_runs),
    }


def boundary_report(ep: pd.DataFrame, valid_ids: list[str], seq: np.ndarray, sleep_idx: int) -> dict:
    """Left/right censoring by the 04:00-04:00 window and within-window sleep run counts."""
    sub = ep[ep["case_id"].isin(set(valid_ids))]
    g = sub.groupby("case_id")
    first_cat = g["category"].first()
    last_cat = g["category"].last()
    last_trunc = g.apply(lambda x: x["duration_untruncated_min"].iloc[-1] > x["duration_min"].iloc[-1],
                         include_groups=False)
    is_sleep = seq == sleep_idx
    sleep_runs = (np.diff(is_sleep.astype(np.int8), axis=1) == 1).sum(axis=1) + is_sleep[:, 0]
    return {
        "first_episode_category_share": first_cat.value_counts(normalize=True).round(4).to_dict(),
        "last_episode_category_share": last_cat.value_counts(normalize=True).round(4).to_dict(),
        "share_starting_in_sleep": float(is_sleep[:, 0].mean()),
        "share_ending_in_sleep": float(is_sleep[:, -1].mean()),
        "share_sleep_at_both_boundaries": float((is_sleep[:, 0] & is_sleep[:, -1]).mean()),
        "share_last_episode_truncated_at_window_end": float(last_trunc.mean()),
        "within_window_sleep_runs_binned": pd.Series(sleep_runs).value_counts().sort_index().to_dict(),
        "note": ("Sleep runs are counted within the 04:00-04:00 window. Sleep touching a boundary is "
                 "censored and is not evidence of fragmented sleep."),
    }


def _describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    q = np.percentile(x, [50, 90, 99])
    return {"mean": round(float(x.mean()), 2), "median": float(q[0]), "p90": float(q[1]),
            "p99": float(q[2]), "max": float(x.max())}
