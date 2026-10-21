"""Model invariants: shapes, finite losses, gradient flow, tiny-batch overfit, checkpoint reload.

Synthetic sequences are software fixtures. Overfitting a tiny batch is a pipeline check, not a result.
"""

import numpy as np
import torch

from src.data.rasterize import clock_features
from src.models.losses import LossWeights, schedule_loss
from src.models.train import load_checkpoint, posterior_means, save_checkpoint, train
from src.models.vae import ScheduleVAE, VAEConfig

C, T = 5, 288
CLOCK = torch.from_numpy(clock_features(T, 5, 240))


def toy_days(n: int, seed: int = 0) -> torch.Tensor:
    """Sleep / work-or-leisure / sleep days with random wake and work times."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, T), dtype=np.int64)
    for i in range(n):
        wake, mid, end = rng.integers(24, 60), rng.integers(100, 160), rng.integers(220, 260)
        out[i, wake:mid] = rng.integers(1, C)
        out[i, mid:end] = rng.integers(1, C)
    return torch.from_numpy(out)


def small_model(**kw) -> ScheduleVAE:
    torch.manual_seed(0)
    return ScheduleVAE(VAEConfig(n_categories=C, hidden_dim=32, decoder_hidden=64, **kw), CLOCK)


def test_shapes_and_finite_losses():
    m, x = small_model(), toy_days(6)
    out = m(x)
    assert out["logits"].shape == (6, T, C)
    assert out["amount_mu"].shape == (6, 8) and out["timing_mu"].shape == (6, 16)
    assert out["z_amount"].shape == (6, 8) and out["amount_logits"].shape == (6, C)
    terms = schedule_loss(out, x, LossWeights(lambda_amount=1.0, lambda_invariance=1.0), out_pair=m(x.flip(1)))
    assert all(torch.isfinite(v) for v in terms.values())


def test_gradients_reach_both_heads_decoder_and_amount_head():
    m, x = small_model(), toy_days(6)
    terms = schedule_loss(m(x), x, LossWeights(lambda_amount=1.0))
    terms["loss"].backward()
    for name in ("encoder.amount_mu.weight", "encoder.timing_mu.weight", "encoder.amount_logvar.weight",
                 "encoder.timing_logvar.weight", "encoder.rnn.weight_ih_l0", "decoder.mlp.0.weight",
                 "amount_head.0.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and g.abs().sum() > 0, name


def test_tiny_batch_can_be_overfit(tmp_path):
    torch.manual_seed(0)
    m, x = small_model(), toy_days(8)
    train(m, x, x, LossWeights(beta_amount=0.01, beta_timing=0.01), epochs=250, batch_size=8, lr=3e-3,
          warmup_epochs=0, seed=0, ckpt_dir=tmp_path, log_every=0)
    with torch.no_grad():
        acc = (m(x, sample=False)["logits"].argmax(-1) == x).float().mean().item()
    assert acc > 0.9, acc


def test_checkpoint_reload_reproduces_posterior_means(tmp_path):
    m, x = small_model(), toy_days(10)
    save_checkpoint(tmp_path / "m.pt", m)
    r, _ = load_checkpoint(tmp_path / "m.pt")
    a, b = posterior_means(m, x), posterior_means(r, x)
    for k in a:
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)


def test_paired_loss_invariance_and_joint_model():
    from src.models.losses import paired_loss
    m, x = small_model(), toy_days(4)
    t = paired_loss(m, x, x.roll(12, dims=1), LossWeights(lambda_invariance=1.0, lambda_amount=1.0))
    assert torch.isfinite(t["loss"]) and t["amount_invariance"] > 0
    same = paired_loss(m, x, x, LossWeights(lambda_invariance=1.0), sample=False)
    assert same["amount_invariance"].item() == 0.0
    j = small_model(z_amount=0, z_timing=24)
    out = j(x)
    assert out["amount_mu"].shape == (4, 0) and out["logits"].shape == (4, T, C)
    assert paired_loss(j, x, x.roll(12, dims=1), LossWeights(lambda_invariance=1.0))["amount_invariance"] == 0


def test_v3_timing_head_and_local_transforms():
    from src.data.pairs import PairConfig, make_pairs
    from src.models.vae import timing_targets
    m, x = small_model(timing_head=True), toy_days(6)
    terms = schedule_loss(m(x), x, LossWeights(lambda_timing=1.0))
    assert torch.isfinite(terms["timing_supervision"]) and terms["timing_supervision"] > 0
    terms["loss"].backward()
    assert m.timing_head[0].weight.grad.abs().sum() > 0 and m.encoder.timing_mu.weight.grad.abs().sum() > 0
    assert m.encoder.amount_mu.weight.grad is not None
    t, mask = timing_targets(x, CLOCK, C)
    assert t.shape == (6, 4 * (C - 1)) and ((t != 0) <= (mask > 0)).all()
    out, params = make_pairs(x.numpy(), np.random.default_rng(0), PairConfig(family="local"), C)
    for s, o, p in zip(x.numpy(), out, params):
        assert np.array_equal(np.bincount(s, minlength=C), np.bincount(o, minlength=C))
        assert p["type"] in ("local_shift", "adjacent_swap", "split_merge", "identity")
        if p["type"] == "local_shift":
            assert 1 <= abs(p["shift"]) <= 12
