"""Invariants for transforms, baseline features and clustering. Synthetic sequences are fixtures only."""

import numpy as np
import pytest

from src.baselines.features import FeatureSet, TimingFeaturizer
from src.data.rasterize import category_counts, clock_features
from src.data.transforms import block_swap, circular_shift
from src.evaluation.clustering import ClusterPipeline

CATS = ["sleep", "work", "eating_drinking", "leisure_social", "unknown"]
CLOCK = clock_features(288, 5, 240)


def day(blocks):
    """[(category_idx, n_bins), ...] -> 288-bin sequence."""
    s = np.concatenate([np.full(n, c, dtype=np.int8) for c, n in blocks])
    assert len(s) == 288
    return s


A = day([(0, 36), (1, 96), (2, 12), (1, 48), (3, 60), (0, 36)])
B = day([(0, 36), (3, 60), (2, 12), (1, 144), (0, 36)])   # same totals as A, different order


def test_transforms_preserve_counts_and_reject_unchanged():
    rng = np.random.default_rng(0)
    for s in (A, B):
        for shift in (-24, 6, 24):
            out, p = circular_shift(s, shift, len(CATS))
            assert np.array_equal(np.bincount(out, minlength=5), np.bincount(s, minlength=5))
            assert p["shift"] == shift
        out, p = block_swap(s, rng, len(CATS))
        assert out is not None and not np.array_equal(out, s)
        assert np.array_equal(np.bincount(out, minlength=5), np.bincount(s, minlength=5))
    flat = np.zeros(288, dtype=np.int8)
    assert circular_shift(flat, 12, 5)[0] is None
    assert block_swap(flat, rng, 5)[0] is None


def test_totals_identical_but_timing_differs_for_reordered_day():
    seq = np.stack([A, B])
    counts = category_counts(seq, len(CATS))
    tot = FeatureSet("t", True, False, CATS, CLOCK).fit(seq)
    tim = FeatureSet("m", False, True, CATS, CLOCK).fit(seq)
    Xt, Xm = tot.transform(seq, counts), tim.transform(seq, counts)
    assert np.allclose(Xt[0], Xt[1])
    assert not np.allclose(Xm[0], Xm[1])
    assert len(tim.names) == Xm.shape[1] and not any(n.endswith("_unknown") for n in tim.names)


def test_absent_activity_uses_train_fill_not_midnight():
    no_eat = day([(0, 96), (3, 156), (0, 36)])
    tf = TimingFeaturizer(CATS, CLOCK).fit(np.stack([A, B]))
    X = tf.transform(no_eat[None])
    names = tf.names_
    assert X[0, names.index("present_eating_drinking")] == 0
    for f in ("center_sin", "center_cos", "first", "last"):
        assert X[0, names.index(f"{f}_eating_drinking")] == pytest.approx(tf.fill_[f][CATS.index("eating_drinking")])
    assert np.isfinite(X).all()


def test_cluster_pipeline_deterministic():
    X = np.random.default_rng(1).normal(size=(200, 5))
    a = ClusterPipeline(3, seed=42).fit(X).predict(X)
    b = ClusterPipeline(3, seed=42).fit(X).predict(X)
    assert np.array_equal(a, b)


def test_pairs_preserve_counts_and_record_params():
    from src.data.pairs import PairConfig, make_pairs
    seq = np.stack([A, B, A])
    out, params = make_pairs(seq, np.random.default_rng(3), PairConfig(), len(CATS))
    assert out.shape == seq.shape and len(params) == 3
    for s, t, p in zip(seq, out, params):
        assert np.array_equal(np.bincount(s, minlength=5), np.bincount(t, minlength=5))
        assert p["type"] in ("circular_shift", "block_swap") and not np.array_equal(s, t)
        if p["type"] == "circular_shift":
            assert 6 <= abs(p["shift"]) <= 24
