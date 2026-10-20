"""Deterministic respondent-level split. Each ATUS respondent contributes exactly one diary."""

from __future__ import annotations

import numpy as np
import pandas as pd


def respondent_split(case_ids: list[str], val_fraction: float, seed: int) -> pd.DataFrame:
    ids = np.array(sorted(set(case_ids)))
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = int(round(val_fraction * len(ids)))
    split = np.full(len(ids), "train", dtype=object)
    split[perm[:n_val]] = "val"
    return pd.DataFrame({"case_id": ids, "split": split})
