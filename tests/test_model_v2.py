"""Model v2 invariants. Synthetic sequences are software fixtures; overfitting a tiny batch is a pipeline check."""

import numpy as np
import pytest
import torch

from src.data.rasterize import clock_features
from src.data.transforms import n_bouts, split_merge
from src.models.losses import LossWeights, paired_loss, schedule_loss
from src.models.train import build_model, load_checkpoint, posterior_means, save_checkpoint, train
from src.models.vae_v2 import exact_budget_decode, integer_budgets, sinkhorn

C, T = 5, 288
CLOCK = torch.from_numpy(clock_features(T, 5, 240))


def toy_days(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    out = np.zeros((n, T), dtype=np.int64)
    for i in range(n):
        wake, mid, end = rng.integers(24, 60), rng.integers(100, 160), rng.integers(220, 260)
        out[i, wake:mid] = rng.integers(1, C)
        out[i, mid:end] = rng.integers(1, C)
        meal = rng.integers(wake + 2, mid - 6)
        out[i, meal:meal + 4] = 4
    return torch.from_numpy(out)


def v2(**kw):
    torch.manual_seed(0)
    cfg = {"model_type": "v2", "n_categories": C, "hidden_dim": 32, "decoder_hidden": 64, **kw}
    return build_model(cfg, CLOCK)


def test_sinkhorn_marginals():
    torch.manual_seed(1)
    scores = torch.randn(3, T, C)
    budget = torch.softmax(torch.randn(3, C), -1)
    log_p = sinkhorn(scores, torch.log(budget * T), iters=50)
    p = log_p.exp()
    assert torch.allclose(p.sum(-1), torch.ones(3, T), atol=1e-5)          # each slot is a distribution
    assert torch.allclose(p.sum(1), budget * T, rtol=1e-2, atol=0.05)     # each activity uses its budget


def test_amount_latent_ignores_order():
    m, x = v2(), toy_days(4)
    perm = x[:, torch.randperm(T)]                                        # same totals, scrambled order
    a, b = m.encode(x), m.encode(perm)
    assert torch.allclose(a["amount_mu"], b["amount_mu"]) and torch.allclose(a["amount_logvar"], b["amount_logvar"])
    assert not torch.allclose(a["timing_mu"], b["timing_mu"])


def test_shapes_losses_and_gradients():
    m, x = v2(conditional_prior=True), toy_days(6)
    out = m(x)
    assert out["logits"].shape == (6, T, C) and out["amount_mu"].shape == (6, 8) and out["timing_mu"].shape == (6, 16)
    assert torch.allclose(out["logits"].exp().sum(-1), torch.ones(6, T), atol=1e-5)
    terms = paired_loss(m, x, x.roll(12, dims=1), LossWeights(lambda_amount=1.0))
    assert all(torch.isfinite(v) for v in terms.values())
    assert terms["amount_invariance"].item() == pytest.approx(0.0, abs=1e-10)   # invariant by construction
    terms["loss"].backward()
    params = dict(m.named_parameters())
    for name in ("amount_encoder.mu.weight", "timing_encoder.mu.weight", "timing_encoder.rnn.weight_ih_l0",
                 "scorer.0.weight", "amount_head.0.weight", "prior_net.0.weight"):
        assert params[name].grad is not None and params[name].grad.abs().sum() > 0, name


def test_exact_budget_decoding_matches_budgets():
    rng = np.random.default_rng(0)
    props = rng.dirichlet(np.ones(C), size=4)
    b = integer_budgets(props, T)
    assert (b.sum(1) == T).all() and (np.abs(b - props * T) < 1).all()
    dec = exact_budget_decode(rng.normal(size=(4, T, C)), b)
    for i in range(4):
        assert np.array_equal(np.bincount(dec[i], minlength=C), b[i])
    m, x = v2(), toy_days(3)
    q = m.encode(x)
    d = m.decode_exact(q["amount_mu"], q["timing_mu"], budget_props=q["props"])
    assert np.array_equal(np.stack([np.bincount(r, minlength=C) for r in d]),
                          np.stack([np.bincount(r, minlength=C) for r in x.numpy()]))


def test_split_merge_changes_fragmentation_only():
    rng = np.random.default_rng(0)
    for s in toy_days(20).numpy():
        out, p = split_merge(s, rng, C)
        if out is None:
            continue
        assert np.array_equal(np.bincount(out, minlength=C), np.bincount(s, minlength=C))
        if p["op"] == "split":
            assert n_bouts(out, p["host"][0]) == n_bouts(s, p["host"][0]) + 1
        else:
            assert n_bouts(out, p["category"]) == n_bouts(s, p["category"]) - 1


def test_v2_tiny_batch_overfit_and_reload(tmp_path):
    torch.manual_seed(0)
    m, x = v2(budget_source="observed"), toy_days(8)
    train(m, x, x, LossWeights(beta_amount=0.01, beta_timing=0.01, lambda_amount=1.0), epochs=200, batch_size=8,
          lr=3e-3, warmup_epochs=0, seed=0, ckpt_dir=tmp_path, log_every=0)
    with torch.no_grad():
        acc = (m(x, sample=False)["logits"].argmax(-1) == x).float().mean().item()
    assert acc > 0.9, acc
    save_checkpoint(tmp_path / "v2.pt", m)
    r, _ = load_checkpoint(tmp_path / "v2.pt")
    a, b = posterior_means(m, x), posterior_means(r, x)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)
    assert type(r).__name__ == "StructuredScheduleVAE"


def test_v1_checkpoints_still_load(tmp_path):
    m = build_model({"n_categories": C, "hidden_dim": 32}, CLOCK)
    save_checkpoint(tmp_path / "v1.pt", m)
    r, _ = load_checkpoint(tmp_path / "v1.pt")
    assert type(r).__name__ == "ScheduleVAE"
    out = schedule_loss(r(toy_days(2)), toy_days(2), LossWeights())
    assert torch.isfinite(out["loss"])
