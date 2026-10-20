"""Training loop, deterministic data subsets, and checkpoint save/load."""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .losses import LossWeights, paired_loss, schedule_loss
from .vae import ScheduleVAE, VAEConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def subset_indices(split: np.ndarray, which: str, n: int | None, seed: int) -> np.ndarray:
    """Deterministic random subset of a split (sorted indices). n=None -> whole split."""
    idx = np.flatnonzero(split == which)
    if n is None or n >= len(idx):
        return idx
    return np.sort(np.random.default_rng(seed).choice(idx, size=n, replace=False))


def kl_scale_at(epoch: int, warmup_epochs: int) -> float:
    return 1.0 if warmup_epochs <= 0 else min(1.0, (epoch + 1) / warmup_epochs)


@torch.no_grad()
def evaluate_loss(model: ScheduleVAE, seq: torch.Tensor, w: LossWeights, batch_size: int = 512,
                  seq_pair: torch.Tensor | None = None) -> dict[str, float]:
    """Full-weight loss terms on posterior means (no sampling), averaged over diaries.
    With `seq_pair`, uses the paired objective on fixed validation pairs."""
    model.eval()
    sums, n = {}, 0
    for i in range(0, len(seq), batch_size):
        b = seq[i:i + batch_size]
        if seq_pair is None:
            terms = schedule_loss(model(b, sample=False), b, w)
        else:
            terms = paired_loss(model, b, seq_pair[i:i + batch_size], w, sample=False)
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + float(v) * len(b)
        n += len(b)
    return {k: v / n for k, v in sums.items()}


def train(model: ScheduleVAE, seq_tr: torch.Tensor, seq_va: torch.Tensor, w: LossWeights, epochs: int,
          batch_size: int, lr: float, warmup_epochs: int, seed: int, ckpt_dir: Path,
          log_every: int = 10, pair_fn: Callable[[np.ndarray], tuple[np.ndarray, list[dict]]] | None = None,
          seq_va_pair: torch.Tensor | None = None) -> list[dict]:
    """Adam training. Saves `best.pt` (lowest validation loss at full KL weight) and `last.pt`.

    With `pair_fn`, every training batch is paired with fresh total-preserving transforms and the paired
    objective is used; validation then uses the fixed pairs in `seq_va_pair`. Transform types are counted.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    history, best = [], float("inf")
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        scale = kl_scale_at(epoch, warmup_epochs)
        perm = torch.randperm(len(seq_tr), generator=gen)
        sums, n, kinds = {}, 0, Counter()
        for i in range(0, len(seq_tr), batch_size):
            b = seq_tr[perm[i:i + batch_size]]
            if pair_fn is None:
                terms = schedule_loss(model(b), b, w, kl_scale=scale)
            else:
                bt, params = pair_fn(b.cpu().numpy())
                kinds.update(p["type"] for p in params)
                terms = paired_loss(model, b, torch.from_numpy(bt).long().to(b.device), w, kl_scale=scale)
            opt.zero_grad()
            terms["loss"].backward()
            opt.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + v.item() * len(b)
            n += len(b)
        row = {"epoch": epoch + 1, "kl_scale": scale, "seconds": round(time.time() - t0, 3),
               **{f"train_{k}": v / n for k, v in sums.items()},
               **{f"val_{k}": v for k, v in evaluate_loss(model, seq_va, w, seq_pair=seq_va_pair).items()},
               **{f"pairs_{k}": c for k, c in kinds.items()}}
        history.append(row)
        if row["val_loss"] < best and scale == 1.0:
            best = row["val_loss"]
            save_checkpoint(ckpt_dir / "best.pt", model, extra={"epoch": epoch + 1, "val_loss": best})
        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(f"epoch {epoch + 1:4d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  "
                  f"recon {row['val_recon']:.4f}  KLa {row['val_kl_amount']:.4f}  KLt {row['val_kl_timing']:.4f}  "
                  f"inv {row['val_amount_invariance']:.4f}  amt {row['val_amount_prediction']:.4f}", flush=True)
    save_checkpoint(ckpt_dir / "last.pt", model, extra={"epoch": epochs})
    return history


def save_checkpoint(path: Path, model: ScheduleVAE, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "clock": model.clock.cpu(),
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, **(extra or {})}, path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[ScheduleVAE, dict]:
    """Rebuild the model from the checkpoint alone: config and clock features are stored inside it."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = ScheduleVAE(VAEConfig(**ck["config"]), ck["clock"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def posterior_means(model: ScheduleVAE, seq: torch.Tensor, batch_size: int = 512) -> dict[str, np.ndarray]:
    model.eval()
    out = {"amount_mu": [], "timing_mu": [], "amount_logvar": [], "timing_logvar": []}
    for i in range(0, len(seq), batch_size):
        q = model.encode(seq[i:i + batch_size])
        for k in out:
            out[k].append(q[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def reconstruct(model: ScheduleVAE, seq: torch.Tensor, batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Decode posterior means. Returns (argmax sequence (N, T), log-probabilities (N, T, C))."""
    model.eval()
    preds, logps = [], []
    for i in range(0, len(seq), batch_size):
        logits = model(seq[i:i + batch_size], sample=False)["logits"]
        logp = torch.log_softmax(logits, dim=-1)
        preds.append(logp.argmax(-1).cpu().numpy())
        logps.append(logp.cpu().numpy())
    return np.concatenate(preds).astype(np.int8), np.concatenate(logps)
