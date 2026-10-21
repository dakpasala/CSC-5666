"""Two-latent sequence VAE for daily schedules.

Encoder: activity embedding + clock features -> bidirectional GRU -> mean-pooled day summary ->
separate Gaussian heads for an `amount` latent and a `timing` latent.
Decoder: at each position, an MLP sees [z_amount, z_timing, position features] and outputs category
logits. It never sees the input sequence, so everything it reconstructs must pass through the latents.
Amount head: optional auxiliary MLP predicting per-category proportions from z_amount (Checkpoint 4).

Separate heads alone do not make the latents mean "amount" and "timing"; that requires the guidance
added in Checkpoint 4 (paired transforms + invariance + amount prediction).
"""

from __future__ import annotations

import math

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    n_categories: int
    n_positions: int = 288
    emb_dim: int = 16
    hidden_dim: int = 64          # per direction
    num_layers: int = 1
    z_amount: int = 8
    z_timing: int = 16
    decoder_hidden: int = 128
    decoder_layers: int = 2
    n_harmonics: int = 4          # Fourier features of clock time given to the decoder
    amount_head_hidden: int = 32
    dropout: float = 0.0
    timing_head: bool = False     # v3 E3: auxiliary head predicting engineered clock features from z_timing
    timing_prior: str = "standard"  # v4: "standard" N(0, I) or "mixture" (learned Gaussian mixture, VaDE-style)
    mixture_k: int = 6

    def to_dict(self) -> dict:
        return asdict(self)


def position_features(clock: torch.Tensor, n_harmonics: int) -> torch.Tensor:
    """clock: (T, 2) sin/cos of time of day -> (T, 2*n_harmonics) sin/cos of k * angle, k=1..n."""
    angle = torch.atan2(clock[:, 0], clock[:, 1])
    k = torch.arange(1, n_harmonics + 1, dtype=clock.dtype, device=clock.device)
    a = angle[:, None] * k[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_categories, cfg.emb_dim)
        self.rnn = nn.GRU(cfg.emb_dim + 2, cfg.hidden_dim, num_layers=cfg.num_layers, batch_first=True,
                          bidirectional=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        d = 2 * cfg.hidden_dim
        # z_amount == 0 gives the joint-latent baseline: a single day-level latent (named "timing" internally).
        self.z_amount = cfg.z_amount
        if cfg.z_amount:
            self.amount_mu, self.amount_logvar = nn.Linear(d, cfg.z_amount), nn.Linear(d, cfg.z_amount)
        self.timing_mu, self.timing_logvar = nn.Linear(d, cfg.z_timing), nn.Linear(d, cfg.z_timing)

    def forward(self, seq: torch.Tensor, clock: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.cat([self.embed(seq), clock.expand(seq.shape[0], -1, -1)], dim=-1)
        h, _ = self.rnn(x)
        pooled = h.mean(dim=1)
        if self.z_amount:
            a_mu, a_lv = self.amount_mu(pooled), self.amount_logvar(pooled)
        else:
            a_mu = a_lv = pooled.new_zeros(pooled.shape[0], 0)
        return {"amount_mu": a_mu, "amount_logvar": a_lv,
                "timing_mu": self.timing_mu(pooled), "timing_logvar": self.timing_logvar(pooled)}


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        layers, d = [], cfg.z_amount + cfg.z_timing + 2 * cfg.n_harmonics
        for _ in range(cfg.decoder_layers):
            layers += [nn.Linear(d, cfg.decoder_hidden), nn.ReLU()]
            d = cfg.decoder_hidden
        layers.append(nn.Linear(d, cfg.n_categories))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z_amount: torch.Tensor, z_timing: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        B, T = z_amount.shape[0], pos.shape[0]
        z = torch.cat([z_amount, z_timing], dim=1)[:, None, :].expand(B, T, -1)
        return self.mlp(torch.cat([z, pos[None].expand(B, -1, -1)], dim=-1))   # (B, T, C)


class ScheduleVAE(nn.Module):
    def __init__(self, cfg: VAEConfig, clock: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("clock", clock.float())                                   # (T, 2)
        self.register_buffer("pos", position_features(clock.float(), cfg.n_harmonics))  # (T, 2H)
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        # Auxiliary proportions head reads z_amount (or the single latent in the joint baseline).
        self.amount_head = nn.Sequential(nn.Linear(cfg.z_amount or cfg.z_timing, cfg.amount_head_hidden), nn.ReLU(),
                                         nn.Linear(cfg.amount_head_hidden, cfg.n_categories))
        # v4 mixture prior on z_timing: logits of π, component means and log-variances. `mixture_active` is
        # switched off during pretraining (standard prior), then on after GMM initialisation.
        self.mixture_active = cfg.timing_prior == "mixture"
        if cfg.timing_prior == "mixture":
            self.mix_logits = nn.Parameter(torch.zeros(cfg.mixture_k))
            self.mix_mu = nn.Parameter(torch.randn(cfg.mixture_k, cfg.z_timing))
            self.mix_logvar = nn.Parameter(torch.zeros(cfg.mixture_k, cfg.z_timing))
        if cfg.timing_head:
            # Targets: (sin, cos, first, last) for every category except the last (`unknown`); see timing_targets().
            self.timing_head = nn.Sequential(nn.Linear(cfg.z_timing, cfg.amount_head_hidden), nn.ReLU(),
                                             nn.Linear(cfg.amount_head_hidden, 4 * (cfg.n_categories - 1)))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encoder(seq.long(), self.clock)

    def decode(self, z_amount: torch.Tensor, z_timing: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_amount, z_timing, self.pos)

    def forward(self, seq: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        q = self.encode(seq)
        if sample:
            za = self.reparameterize(q["amount_mu"], q["amount_logvar"])
            zt = self.reparameterize(q["timing_mu"], q["timing_logvar"])
        else:
            za, zt = q["amount_mu"], q["timing_mu"]
        out = {**q, "z_amount": za, "z_timing": zt, "logits": self.decode(za, zt),
               "amount_logits": self.amount_head(za if self.cfg.z_amount else zt)}
        if self.cfg.timing_head:
            out["timing_pred"] = self.timing_head(zt)
            out["clock"] = self.clock
        if self.cfg.timing_prior == "mixture" and self.mixture_active:
            out["mix_logits"], out["mix_mu"], out["mix_logvar"] = self.mix_logits, self.mix_mu, self.mix_logvar
        return out

    def mixture_responsibilities(self, z_timing: torch.Tensor) -> torch.Tensor:
        """q(c | x) evaluated at z_timing: (B, K). Only for mixture-prior models."""
        return mixture_gamma(z_timing, self.mix_logits, self.mix_mu, self.mix_logvar)


def timing_targets(seq: torch.Tensor, clock: torch.Tensor, n_categories: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Engineered clock features per category (excluding the last, `unknown`), computed from a batch.

    For each category: circular mean of clock time over its slots (sin, cos; mean resultant, so length also
    reflects spread), first onset and last offset as fractions of the window. Returns (targets, mask) of
    shape (B, 4 * (C - 1)); mask is 0 where the category is absent (undefined targets are never trained on).
    """
    B, T = seq.shape
    oh = F.one_hot(seq.long(), n_categories)[..., :-1].float()            # (B, T, C-1)
    n = oh.sum(1)
    present = (n > 0).float()
    denom = n.clamp_min(1.0)
    sin = (oh * clock[None, :, 0, None]).sum(1) / denom
    cos = (oh * clock[None, :, 1, None]).sum(1) / denom
    pos = torch.arange(T, device=seq.device, dtype=torch.float32)[None, :, None]
    first = torch.where(oh > 0, pos, torch.full_like(oh, float(T))).amin(1) / T
    last = (torch.where(oh > 0, pos, torch.full_like(oh, -1.0)).amax(1) + 1) / T
    targets = torch.cat([sin, cos, first, last], dim=1)
    mask = present.repeat(1, 4)
    return targets * mask, mask


def mixture_gamma(z: torch.Tensor, logits: torch.Tensor, mu_c: torch.Tensor, logvar_c: torch.Tensor) -> torch.Tensor:
    """Responsibilities γ_c ∝ π_c N(z; μ_c, σ_c²) for z (B, D), components (K, D): returns (B, K)."""
    log_n = -0.5 * (math.log(2 * math.pi) + logvar_c[None] + (z[:, None, :] - mu_c[None]).pow(2) / logvar_c.exp()[None]).sum(-1)
    return torch.softmax(torch.log_softmax(logits, -1)[None] + log_n, dim=-1)


def mixture_kl(mu: torch.Tensor, logvar: torch.Tensor, z: torch.Tensor, logits: torch.Tensor, mu_c: torch.Tensor,
               logvar_c: torch.Tensor) -> torch.Tensor:
    """VaDE-style KL of q(z|x) q(c|x) to the mixture prior, per diary: (B,).

    Σ_c γ_c KL(N(mu, e^logvar) ‖ N(μ_c, σ_c²)) + KL(γ ‖ π), with γ = q(c|x) computed at the sampled z.
    With one component at N(0, I) this equals the standard Gaussian KL.
    """
    gamma = mixture_gamma(z, logits, mu_c, logvar_c)                                        # (B, K)
    kl_c = 0.5 * (logvar_c[None] - logvar[:, None] + (logvar.exp()[:, None] + (mu[:, None] - mu_c[None]).pow(2))
                  / logvar_c.exp()[None] - 1.0).sum(-1)                                     # (B, K)
    log_pi = torch.log_softmax(logits, -1)[None]
    return (gamma * kl_c).sum(-1) + (gamma * (torch.log(gamma.clamp_min(1e-10)) - log_pi)).sum(-1)


def gaussian_kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)) per sample and dimension: (B, D)."""
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)


def gaussian_kl_general(mu: torch.Tensor, logvar: torch.Tensor, p_mu: torch.Tensor,
                        p_logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, s^2) || N(p_mu, p_s^2)) per sample and dimension: (B, D)."""
    return 0.5 * (p_logvar - logvar + (logvar.exp() + (mu - p_mu).pow(2)) / p_logvar.exp() - 1.0)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

