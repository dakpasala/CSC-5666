"""Total-preserving schedule transformations.

Both transforms keep per-category bin counts exactly (asserted on every call) while changing the
arrangement. They are diagnostic and training-guidance cases, not realistic diaries.

- circular_shift: rotates the whole sequence by `shift` bins. Bins pushed past 04:00 wrap to the other
  end of the window, which is an artificial boundary effect; the wrapped categories are recorded.
- block_swap: exchanges two whole runs of different categories, moving everything between them.
  Runs touching the window boundaries are never chosen, to avoid moving censored sleep.
"""

from __future__ import annotations

import numpy as np


def assert_counts_preserved(a: np.ndarray, b: np.ndarray, n_categories: int) -> None:
    ca = np.bincount(a, minlength=n_categories)
    cb = np.bincount(b, minlength=n_categories)
    if not np.array_equal(ca, cb):
        raise AssertionError("transformation changed category counts")


def circular_shift(seq: np.ndarray, shift: int, n_categories: int) -> tuple[np.ndarray | None, dict]:
    """Return (shifted, params); shifted is None if the result is unchanged."""
    out = np.roll(seq, shift)
    if np.array_equal(out, seq):
        return None, {"type": "circular_shift", "shift": shift, "rejected": "unchanged"}
    assert_counts_preserved(seq, out, n_categories)
    wrapped = seq[-shift:] if shift > 0 else seq[:-shift]
    return out, {"type": "circular_shift", "shift": int(shift),
                 "wrapped_categories": sorted(set(np.unique(wrapped).tolist()))}


def _runs(seq: np.ndarray) -> list[tuple[int, int, int]]:
    change = np.flatnonzero(np.diff(seq)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(seq)]])
    return [(int(seq[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def block_swap(seq: np.ndarray, rng: np.random.Generator, n_categories: int,
               max_tries: int = 20) -> tuple[np.ndarray | None, dict]:
    """Swap two non-boundary runs of different categories. Returns (None, params) if impossible."""
    runs = _runs(seq)[1:-1]  # exclude the runs touching 04:00 at either end
    if len(runs) < 2:
        return None, {"type": "block_swap", "rejected": "too_few_runs"}
    for _ in range(max_tries):
        i, j = sorted(rng.choice(len(runs), size=2, replace=False))
        (ci, si, ei), (cj, sj, ej) = runs[i], runs[j]
        if ci == cj:
            continue
        out = np.concatenate([seq[:si], seq[sj:ej], seq[ei:sj], seq[si:ei], seq[ej:]])
        if np.array_equal(out, seq):
            continue
        assert_counts_preserved(seq, out, n_categories)
        return out, {"type": "block_swap", "run_a": [ci, si, ei], "run_b": [cj, sj, ej]}
    return None, {"type": "block_swap", "rejected": "no_valid_pair"}
