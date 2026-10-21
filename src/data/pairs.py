"""Total-preserving pairs for training guidance and evaluation.

Each diary is paired with one transformed copy of itself: a bounded circular shift (30 min to 2 h by
default, either direction), a swap of two non-boundary activity blocks, or (v2, optional) a split/merge
that changes fragmentation. Pairs are always made within a split, after splitting, so a diary and its
transform never cross splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .transforms import adjacent_swap, block_swap, circular_shift, local_shift, split_merge


@dataclass
class PairConfig:
    p_shift: float = 0.5        # probability of a shift (otherwise a block swap)
    shift_min_bins: int = 6     # 30 minutes at 5-minute bins
    shift_max_bins: int = 24    # 2 hours
    p_split_merge: float = 0.0  # v2: probability of a split/merge; 0 keeps v1 sampling exactly unchanged
    family: str = "global"      # "global" (v1: wrap-shift / far swap) or "local" (v3: local shift, adjacent swap, split/merge)
    local_max_shift_bins: int = 12
    p_local_shift: float = 0.5  # local family: else adjacent swap or split/merge (equal split of the remainder)

    def to_dict(self) -> dict:
        return asdict(self)


def transform_one(seq: np.ndarray, rng: np.random.Generator, cfg: PairConfig,
                  n_categories: int) -> tuple[np.ndarray, dict]:
    """Return (transformed, params). Falls back to the other transform if the first is rejected; if both
    are rejected, returns the original with params['type'] == 'identity' (invariance is then trivially 0)."""
    if cfg.family == "local":
        r = rng.random()
        rest = (1 - cfg.p_local_shift) / 2
        order = (["local", "adjacent", "split_merge"] if r < cfg.p_local_shift else
                 ["adjacent", "local", "split_merge"] if r < cfg.p_local_shift + rest else
                 ["split_merge", "local", "adjacent"])
        for kind in order:
            if kind == "local":
                out, p = local_shift(seq, rng, n_categories, cfg.local_max_shift_bins)
            elif kind == "adjacent":
                out, p = adjacent_swap(seq, rng, n_categories)
            else:
                out, p = split_merge(seq, rng, n_categories)
            if out is not None:
                return out, p
        return seq.copy(), {"type": "identity"}
    if cfg.p_split_merge > 0 and rng.random() < cfg.p_split_merge:
        out, p = split_merge(seq, rng, n_categories)
        if out is not None:
            return out, p
    kinds = ["shift", "swap"] if rng.random() < cfg.p_shift else ["swap", "shift"]
    for kind in kinds:
        if kind == "shift":
            s = int(rng.integers(cfg.shift_min_bins, cfg.shift_max_bins + 1)) * int(rng.choice([-1, 1]))
            out, p = circular_shift(seq, s, n_categories)
        else:
            out, p = block_swap(seq, rng, n_categories)
        if out is not None:
            return out, p
    return seq.copy(), {"type": "identity"}


def make_pairs(batch: np.ndarray, rng: np.random.Generator, cfg: PairConfig,
               n_categories: int) -> tuple[np.ndarray, list[dict]]:
    outs, params = zip(*(transform_one(s, rng, cfg, n_categories) for s in batch))
    return np.stack(outs).astype(batch.dtype), list(params)
