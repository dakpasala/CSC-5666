"""Hand-built baseline features from binned sequences.

Totals: per-category share of the day (counts / n_bins).

Timing, for each category except `unknown`:
  present      1 if the category occurs at all
  center_sin,  circular mean of clock time over the category's bins (mean resultant vector, so its
  center_cos   length also reflects spread)
  first, last  diary-relative position (0-1) of the first onset and last offset
  bouts        number of separate within-window runs (0 when absent)
plus the total number of category transitions in the day.

Clock-time features are undefined for an absent category. They are filled with the training-set mean
over diaries where the category is present (so after scaling they sit near 0) and the `present` flag
carries the absence; an absent meal is never treated as a meal at midnight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNDEFINED_WHEN_ABSENT = ("center_sin", "center_cos", "first", "last")


def totals_features(counts: np.ndarray, categories: list[str]) -> tuple[np.ndarray, list[str]]:
    X = counts / counts.sum(axis=1, keepdims=True)
    return X.astype(np.float64), [f"share_{c}" for c in categories]


def raw_timing(seq: np.ndarray, categories: list[str], clock: np.ndarray,
               exclude: tuple[str, ...] = ("unknown",)) -> tuple[dict[str, np.ndarray], list[str]]:
    """Per-category timing arrays (N, C') with NaN where undefined, plus kept category names."""
    keep = [k for k, c in enumerate(categories) if c not in exclude]
    names = [categories[k] for k in keep]
    T = seq.shape[1]
    oh = seq[..., None] == np.array(keep)                         # (N, T, C')
    n = oh.sum(axis=1)
    present = n > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        cs = (oh * clock[None, :, 0, None]).sum(axis=1) / n
        cc = (oh * clock[None, :, 1, None]).sum(axis=1) / n
    first = oh.argmax(axis=1) / T
    last = (T - oh[:, ::-1].argmax(axis=1)) / T
    bouts = oh[:, 0].astype(int) + (oh[:, 1:] & ~oh[:, :-1]).sum(axis=1)
    feats = {"present": present.astype(float), "center_sin": cs, "center_cos": cc,
             "first": first, "last": last, "bouts": bouts.astype(float)}
    for k in UNDEFINED_WHEN_ABSENT:
        feats[k] = np.where(present, feats[k], np.nan)
    feats["transitions"] = (np.diff(seq, axis=1) != 0).sum(axis=1).astype(float)[:, None]
    return feats, names


@dataclass
class TimingFeaturizer:
    categories: list[str]
    clock: np.ndarray
    exclude: tuple[str, ...] = ("unknown",)
    fill_: dict[str, np.ndarray] = field(default_factory=dict)
    names_: list[str] = field(default_factory=list)

    def fit(self, seq_train: np.ndarray) -> "TimingFeaturizer":
        feats, cats = raw_timing(seq_train, self.categories, self.clock, self.exclude)
        for k in UNDEFINED_WHEN_ABSENT:
            m = np.nanmean(feats[k], axis=0)
            self.fill_[k] = np.where(np.isnan(m), 0.0, m)   # category never present in train -> 0
        self.names_ = [f"{k}_{c}" for k in ("present", *UNDEFINED_WHEN_ABSENT, "bouts") for c in cats]
        self.names_.append("transitions")
        return self

    def transform(self, seq: np.ndarray) -> np.ndarray:
        feats, _ = raw_timing(seq, self.categories, self.clock, self.exclude)
        for k in UNDEFINED_WHEN_ABSENT:
            feats[k] = np.where(np.isnan(feats[k]), self.fill_[k][None, :], feats[k])
        parts = [feats[k] for k in ("present", *UNDEFINED_WHEN_ABSENT, "bouts")] + [feats["transitions"]]
        return np.hstack(parts)


@dataclass
class FeatureSet:
    """Builds one baseline's feature matrix from (seq, counts)."""
    name: str
    use_totals: bool
    use_timing: bool
    categories: list[str]
    clock: np.ndarray
    timing: TimingFeaturizer | None = None

    def fit(self, seq: np.ndarray) -> "FeatureSet":
        if self.use_timing:
            self.timing = TimingFeaturizer(self.categories, self.clock).fit(seq)
        return self

    def transform(self, seq: np.ndarray, counts: np.ndarray) -> np.ndarray:
        parts = []
        if self.use_totals:
            parts.append(totals_features(counts, self.categories)[0])
        if self.use_timing:
            parts.append(self.timing.transform(seq))
        return np.hstack(parts)

    @property
    def names(self) -> list[str]:
        out = []
        if self.use_totals:
            out += [f"share_{c}" for c in self.categories]
        if self.use_timing:
            out += self.timing.names_
        return out
