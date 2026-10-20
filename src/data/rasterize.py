"""Minute-level timelines and fixed-width bins.

Bin b covers diary-relative minutes [b*w, (b+1)*w); minute 0 is the diary start clock (04:00 in
ATUS). Each bin takes the category with the largest overlap. Ties go to the tied category that
appears earliest within the bin, which is deterministic and follows the diary's own order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def minute_timelines(ep: pd.DataFrame, case_ids: list[str], diary_minutes: int = 1440) -> np.ndarray:
    """(N, diary_minutes) array of category indices. Episodes must already be validated."""
    sub = ep[ep["case_id"].isin(set(case_ids))]
    order = {c: i for i, c in enumerate(case_ids)}
    sub = sub.assign(_row=sub["case_id"].map(order)).sort_values(["_row", "episode_id"])
    flat = np.repeat(sub["category_idx"].to_numpy(), sub["duration_min"].to_numpy())
    if flat.size != len(case_ids) * diary_minutes:
        raise ValueError("timelines do not tile the diary window; validate diaries first")
    return flat.reshape(len(case_ids), diary_minutes).astype(np.int16)


def bin_timelines(minutes: np.ndarray, bin_minutes: int, n_categories: int) -> np.ndarray:
    """Majority-overlap binning with earliest-occurrence tie break. Returns (N, n_bins) int8."""
    n, m = minutes.shape
    if m % bin_minutes:
        raise ValueError("bin width must divide the diary length")
    nb = m // bin_minutes
    blocks = minutes.reshape(n, nb, bin_minutes)
    onehot = blocks[..., None] == np.arange(n_categories)              # (n, nb, w, C)
    counts = onehot.sum(axis=2)                                         # (n, nb, C)
    first_pos = np.where(onehot.any(axis=2), onehot.argmax(axis=2), bin_minutes)
    is_max = counts == counts.max(axis=2, keepdims=True)
    return np.where(is_max, first_pos, bin_minutes + 1).argmin(axis=2).astype(np.int8)


def category_counts(seq: np.ndarray, n_categories: int) -> np.ndarray:
    """Per-diary count of steps in each category: (N, C)."""
    return (seq[..., None] == np.arange(n_categories)).sum(axis=1).astype(np.int16)


def run_lengths(seq: np.ndarray) -> list[list[tuple[int, int, int]]]:
    """Per diary, list of (category, start, length) maximal runs."""
    out = []
    for row in seq:
        change = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate([[0], change])
        lens = np.diff(np.concatenate([starts, [len(row)]]))
        out.append(list(zip(row[starts].tolist(), starts.tolist(), lens.tolist())))
    return out


def n_runs(seq: np.ndarray) -> np.ndarray:
    return 1 + (np.diff(seq, axis=1) != 0).sum(axis=1)


def clock_features(n_bins: int, bin_minutes: int, diary_start_clock_minute: int) -> np.ndarray:
    """(n_bins, 2) sin/cos of clock time at each bin's centre (period 24h)."""
    centre = diary_start_clock_minute + (np.arange(n_bins) + 0.5) * bin_minutes
    ang = 2 * np.pi * (centre % 1440) / 1440
    return np.stack([np.sin(ang), np.cos(ang)], axis=1).astype(np.float32)
