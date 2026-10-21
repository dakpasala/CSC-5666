"""Training objective with every term reported separately.

loss = reconstruction + beta_amount * KL_amount + beta_timing * KL_timing
       + lambda_invariance * amount_invariance + lambda_amount * amount_prediction

Reduction convention (fixed so the scale does not drift with sequence length):
- reconstruction: categorical cross-entropy averaged over positions and diaries (nats per position);
- KL terms (to N(0, I), or to the learned conditional prior for z_timing in v2): summed over latent dimensions, averaged over diaries, then DIVIDED BY the number of positions.
  With both betas = 1 the total is the negative ELBO divided by T (nats per position).
- amount_prediction: cross-entropy between true category proportions and the amount head's softmax.
- amount_invariance: mean squared distance between amount posterior means of a diary and its
  total-preserving transform (needs pairs; Checkpoint 4). Zero when no pair batch is given.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .vae import gaussian_kl, gaussian_kl_general


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
    if "timing_prior_mu" in out:   # v2 conditional prior p(z_timing | z_amount)
        kl_t = gaussian_kl_general(out["timing_mu"], out["timing_logvar"], out["timing_prior_mu"],
                                   out["timing_prior_logvar"]).sum(1).mean() / T
    else:
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


def paired_loss(model, x: torch.Tensor, x_t: torch.Tensor, w: LossWeights, kl_scale: float = 1.0,
                sample: bool = True) -> dict[str, torch.Tensor]:
    """Loss on a diary batch and its total-preserving transforms.

    Reconstruction, KL and amount prediction are averaged over originals and transforms together;
    invariance compares the amount posterior means of each diary with its own transform.
    """
    B = x.shape[0]
    out = model(torch.cat([x, x_t]), sample=sample)
    terms = schedule_loss(out, torch.cat([x, x_t]), w, kl_scale)
    inv = (out["amount_mu"][:B] - out["amount_mu"][B:]).pow(2).sum(1).mean()
    terms["amount_invariance"] = inv
    terms["loss"] = terms["loss"] + w.lambda_invariance * inv
    return terms
