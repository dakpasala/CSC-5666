"""Total-preserving schedule transformations.

Every transform keeps per-category bin counts exactly (asserted on every call) while changing the
arrangement. They are diagnostic and training-guidance cases, not realistic diaries.

- circular_shift: rotates the whole sequence by `shift` bins. Bins pushed past 04:00 wrap to the other
  end of the window, which is an artificial boundary effect; the wrapped categories are recorded.
- block_swap: exchanges two whole runs of different categories, moving everything between them.
  Runs touching the window boundaries are never chosen, to avoid moving censored sleep.
- local_shift (v3): moves one interior block earlier or later by up to `max_shift` bins, staying strictly
  inside the interior (never past the first or last run), so clock timing changes locally without wrapping.
- adjacent_swap (v3): exchanges two neighbouring interior runs (a local order change).
- split_merge: changes fragmentation. "split" moves a chunk of another activity into the middle of a run
  (one more bout of that activity); "merge" moves the segment separating two same-activity runs to just
  after them (one fewer bout). Boundary runs are never split, merged or moved.
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


def n_bouts(seq: np.ndarray, category: int) -> int:
    m = seq == category
    return int(m[0]) + int((m[1:] & ~m[:-1]).sum())


def _split(seq: np.ndarray, rng: np.random.Generator, max_chunk: int, max_tries: int) -> tuple[np.ndarray | None, dict]:
    runs = _runs(seq)
    interior = runs[1:-1]
    hosts = [r for r in interior if r[2] - r[1] >= 2]
    if not hosts or len(interior) < 2:
        return None, {"rejected": "no_host"}
    for _ in range(max_tries):
        hc, hs, he = hosts[rng.integers(len(hosts))]
        donors = [r for r in interior if r[0] != hc]
        if not donors:
            return None, {"rejected": "no_donor"}
        dc, ds, de = donors[rng.integers(len(donors))]
        k = int(rng.integers(1, min(de - ds, max_chunk) + 1))
        cut = hs + int(rng.integers(1, he - hs))              # strictly inside the host run
        chunk = seq[ds:ds + k]
        rest = np.delete(seq, np.arange(ds, ds + k))
        pos = cut - k if ds < hs else cut
        out = np.insert(rest, pos, chunk)
        if n_bouts(out, hc) == n_bouts(seq, hc) + 1:
            return out, {"host": [hc, hs, he], "donor_chunk": [dc, ds, ds + k], "cut": cut}
    return None, {"rejected": "no_valid_split"}


def _merge(seq: np.ndarray, rng: np.random.Generator, max_gap: int, max_tries: int) -> tuple[np.ndarray | None, dict]:
    runs = _runs(seq)
    interior = runs[1:-1]
    pairs = [(a, b) for i, a in enumerate(interior) for b in interior[i + 2:]
             if a[0] == b[0] and 0 < b[1] - a[2] <= max_gap]
    if not pairs:
        return None, {"rejected": "no_pair"}
    for _ in range(max_tries):
        (c, s1, e1), (_, s2, e2) = pairs[rng.integers(len(pairs))]
        out = np.concatenate([seq[:s1], seq[s1:e1], seq[s2:e2], seq[e1:s2], seq[e2:]])
        if n_bouts(out, c) == n_bouts(seq, c) - 1:
            return out, {"category": c, "run_a": [s1, e1], "run_b": [s2, e2]}
    return None, {"rejected": "no_valid_merge"}


def split_merge(seq: np.ndarray, rng: np.random.Generator, n_categories: int, max_chunk: int = 6,
                max_gap: int = 24, max_tries: int = 20) -> tuple[np.ndarray | None, dict]:
    """Change the number of bouts of one activity while keeping all totals. Tries split or merge (random
    order) and falls back to the other. Returns (None, params) if neither is possible."""
    order = ["split", "merge"] if rng.random() < 0.5 else ["merge", "split"]
    for op in order:
        out, p = _split(seq, rng, max_chunk, max_tries) if op == "split" else _merge(seq, rng, max_gap, max_tries)
        if out is not None:
            assert_counts_preserved(seq, out, n_categories)
            return out, {"type": "split_merge", "op": op, **p}
    return None, {"type": "split_merge", "rejected": "no_valid_split_or_merge"}


def local_shift(seq: np.ndarray, rng: np.random.Generator, n_categories: int, max_shift: int = 12,
                max_tries: int = 20) -> tuple[np.ndarray | None, dict]:
    """Move one interior run by d bins (1 <= |d| <= max_shift) within the interior region."""
    runs = _runs(seq)
    if len(runs) < 4:
        return None, {"type": "local_shift", "rejected": "too_few_runs"}
    lo, hi = runs[0][2], runs[-1][1]                     # interior region [lo, hi)
    interior = runs[1:-1]
    for _ in range(max_tries):
        c, s, e = interior[rng.integers(len(interior))]
        d = int(rng.integers(1, max_shift + 1)) * int(rng.choice([-1, 1]))
        block = seq[s:e]
        rest = np.concatenate([seq[:s], seq[e:]])
        pos = s + d
        if pos < lo or pos + (e - s) > hi:
            continue
        out = np.concatenate([rest[:pos], block, rest[pos:]])
        if np.array_equal(out, seq):
            continue
        assert_counts_preserved(seq, out, n_categories)
        return out, {"type": "local_shift", "run": [c, s, e], "shift": d}
    return None, {"type": "local_shift", "rejected": "no_valid_move"}


def adjacent_swap(seq: np.ndarray, rng: np.random.Generator, n_categories: int,
                  max_tries: int = 20) -> tuple[np.ndarray | None, dict]:
    """Swap two neighbouring interior runs of different activities."""
    runs = _runs(seq)[1:-1]
    if len(runs) < 2:
        return None, {"type": "adjacent_swap", "rejected": "too_few_runs"}
    for _ in range(max_tries):
        i = int(rng.integers(len(runs) - 1))
        (ca, sa, ea), (cb, sb, eb) = runs[i], runs[i + 1]
        out = np.concatenate([seq[:sa], seq[sb:eb], seq[sa:ea], seq[eb:]])
        if np.array_equal(out, seq):
            continue
        assert_counts_preserved(seq, out, n_categories)
        return out, {"type": "adjacent_swap", "runs": [[ca, sa, ea], [cb, sb, eb]]}
    return None, {"type": "adjacent_swap", "rejected": "no_valid_pair"}
