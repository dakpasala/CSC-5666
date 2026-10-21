"""Null dataset for the v4 fake-groups check (docs/v4_mixture_plan.md).

Usage:
    python scripts/make_null_data.py

Generates synthetic diaries from a frozen 80-epoch v1 model by sampling BOTH latents from N(0, I), so the
timing latent of the generating process is a single Gaussian with no discrete groups, then decodes them with
self-budgeted decoding. The same number of diaries and the same train/val labels as 2022 are used. Written to
data/processed/null2022/ (git-ignored). Synthetic data: never presented as ATUS results.
"""

from __future__ import annotations

import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.rasterize import category_counts  # noqa: E402
from src.models.train import load_checkpoint  # noqa: E402
from src.models.vae_v2 import exact_budget_decode, integer_budgets  # noqa: E402

GENERATOR = "outputs/checkpoint5_final/guided_s42-*"
SEED = 2026


@torch.no_grad()
def main() -> None:
    z = np.load(ROOT / "data/processed/2022/sequences.npz")
    model, _ = load_checkpoint(Path(sorted(glob.glob(str(ROOT / GENERATOR)))[-1]) / "best.pt")
    n = len(z["seq"])
    g = torch.Generator().manual_seed(SEED)
    za = torch.randn(n, model.cfg.z_amount, generator=g)
    zt = torch.randn(n, model.cfg.z_timing, generator=g)
    seqs = []
    for i in range(0, n, 512):
        logp = torch.log_softmax(model.decode(za[i:i + 512], zt[i:i + 512]), -1).numpy()
        seqs.append(exact_budget_decode(logp, integer_budgets(np.exp(logp).mean(1), logp.shape[1])))
    seq = np.concatenate(seqs).astype(np.int8)
    out = ROOT / "data/processed/null2022"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "sequences.npz", case_ids=np.array([f"null{i:06d}" for i in range(n)]), split=z["split"],
                        seq=seq, counts=category_counts(seq, len(z["categories"])), clock_features=z["clock_features"],
                        categories=z["categories"], bin_minutes=z["bin_minutes"],
                        diary_start_clock_minute=z["diary_start_clock_minute"])
    shutil.copy(ROOT / "data/processed/2022/category_mapping.yaml", out / "category_mapping.yaml")
    real = z["seq"]
    print(f"wrote {n} synthetic diaries to {out}")
    print(f"transitions/day: synthetic {(np.diff(seq, axis=1) != 0).sum(1).mean():.1f} vs real {(np.diff(real, axis=1) != 0).sum(1).mean():.1f}")
    for c in ("sleep", "work", "leisure_social", "eating_drinking", "travel"):
        k = z["categories"].tolist().index(c)
        print(f"  {c:16s} min/day synthetic {(seq == k).sum(1).mean() * 5:6.1f}  real {(real == k).sum(1).mean() * 5:6.1f}")


if __name__ == "__main__":
    main()
