"""Training objective with every term reported separately.

loss = reconstruction + beta_amount * KL_amount + beta_timing * KL_timing
       + lambda_invariance * amount_invariance + lambda_amount * amount_prediction

Reduction convention (fixed so the scale does not drift with sequence length):
- reconstruction: categorical cross-entropy averaged over positions and diaries (nats per position);
- KL terms: summed over latent dimensions, averaged over diaries, then DIVIDED BY the number of positions.
  With both betas = 1 the total is the negative ELBO divided by T (nats per position).
- amount_prediction: cross-entropy between true category proportions and the amount head's softmax.
- amount_invariance: mean squared distance between amount posterior means of a diary and its
  total-preserving transform (needs pairs; Checkpoint 4). Zero when no pair batch is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .vae import gaussian_kl


@dataclass
class LossWeights:
    beta_amount: float = 1.0
    beta_timing: float = 1.0
    lambda_invariance: float = 0.0
    lambda_amount: float = 0.0


def schedule_loss(out: dict[str, torch.Tensor], seq: torch.Tensor, w: LossWeights, kl_scale: float = 1.0,
                  out_pair: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    T = seq.shape[1]
    C = out["logits"].shape[-1]
    recon = F.cross_entropy(out["logits"].reshape(-1, C), seq.reshape(-1).long())
    kl_a = gaussian_kl(out["amount_mu"], out["amount_logvar"]).sum(1).mean() / T
    kl_t = gaussian_kl(out["timing_mu"], out["timing_logvar"]).sum(1).mean() / T
    props = F.one_hot(seq.long(), C).float().mean(dim=1)
    amount_pred = -(props * F.log_softmax(out["amount_logits"], dim=-1)).sum(1).mean()
    if out_pair is not None:
        invariance = (out["amount_mu"] - out_pair["amount_mu"]).pow(2).sum(1).mean()
    else:
        invariance = torch.zeros((), device=seq.device)
    total = (recon + kl_scale * (w.beta_amount * kl_a + w.beta_timing * kl_t)
             + w.lambda_invariance * invariance + w.lambda_amount * amount_pred)
    return {"loss": total, "recon": recon, "kl_amount": kl_a, "kl_timing": kl_t,
            "amount_invariance": invariance, "amount_prediction": amount_pred}
