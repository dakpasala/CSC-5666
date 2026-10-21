"""Model v2: amount/timing separation built into the architecture (see docs/model_v2_proposal.md).

- Amount encoder sees only category proportions (no order), so z_amount cannot carry timing.
- Timing encoder sees the full sequence plus the proportions, so z_timing is charged (via KL) only for
  information beyond totals. Optional learned conditional prior p(z_timing | z_amount).
- Decoder: z_timing (+ clock position) scores every (slot, activity). Sinkhorn balancing then forces each
  slot to be a distribution and each activity column to sum to its budget (minutes as slots). Budgets come
  from the amount head (or, as a diagnostic, from the observed counts). "Amount says how much, timing says
  when" holds by construction.

Exposes the same interface as ScheduleVAE (encode / decode / forward returning amount_mu, timing_mu,
logits, amount_logits, ...), so training and evaluation code is shared. `logits` are row-normalised
log-probabilities, so log_softmax leaves them unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from .vae import position_features


@dataclass
class VAEv2Config:
    n_categories: int
    n_positions: int = 288
    emb_dim: int = 16
    hidden_dim: int = 64
    num_layers: int = 1
    z_amount: int = 8
    z_timing: int = 16
    amount_hidden: int = 64
    decoder_hidden: int = 128
    decoder_layers: int = 2
    n_harmonics: int = 4
    amount_head_hidden: int = 32
    sinkhorn_iters: int = 20
    sinkhorn_tau: float = 1.0          # temperature on decoder scores before balancing
    conditional_prior: bool = False    # learned p(z_timing | z_amount) instead of N(0, I)
    budget_source: str = "predicted"   # "predicted" (from z_amount) or "observed" (diagnostic / teacher)
    dropout: float = 0.0
    model_type: str = "v2"

    def to_dict(self) -> dict:
        return asdict(self)


def proportions(seq: torch.Tensor, n_categories: int) -> torch.Tensor:
    return F.one_hot(seq.long(), n_categories).float().mean(dim=1)


def sinkhorn(log_scores: torch.Tensor, log_budget: torch.Tensor, iters: int) -> torch.Tensor:
    """Balance (B, T, C) log-scores so each row sums to 1 and column c sums to budget_c (sum_c budget = T).

    Log-domain Sinkhorn (Cuturi, 2013). Ends with a row normalisation, so rows are exact distributions and
    column sums are approximately the budgets.
    """
    log_p = log_scores
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, dim=1, keepdim=True) + log_budget[:, None, :]   # columns
        log_p = log_p - torch.logsumexp(log_p, dim=2, keepdim=True)                            # rows
    return log_p


def integer_budgets(props: np.ndarray, n_positions: int) -> np.ndarray:
    """Largest-remainder rounding of proportions to integer slot counts summing to n_positions."""
    raw = props * n_positions
    base = np.floor(raw).astype(int)
    short = n_positions - base.sum(axis=1)
    order = np.argsort(-(raw - base), axis=1)
    for i, k in enumerate(short):
        base[i, order[i, :k]] += 1
    return base


def exact_budget_decode(log_p: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """Assign slots to activities maximising total log-probability with exactly `budgets` slots each.

    Solved as a linear assignment between slots and budget-expanded activity columns (one problem per day).
    """
    N, T, C = log_p.shape
    out = np.empty((N, T), dtype=np.int8)
    for i in range(N):
        cols = np.repeat(np.arange(C), budgets[i])
        cost = -log_p[i][:, cols]
        rows, assigned = linear_sum_assignment(cost)
        out[i, rows] = cols[assigned]
    return out


class AmountEncoder(nn.Module):
    def __init__(self, cfg: VAEv2Config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_categories, cfg.amount_hidden), nn.ReLU(),
                                 nn.Linear(cfg.amount_hidden, cfg.amount_hidden), nn.ReLU())
        self.mu = nn.Linear(cfg.amount_hidden, cfg.z_amount)
        self.logvar = nn.Linear(cfg.amount_hidden, cfg.z_amount)

    def forward(self, props: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(props)
        return self.mu(h), self.logvar(h)


class TimingEncoder(nn.Module):
    def __init__(self, cfg: VAEv2Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.n_categories, cfg.emb_dim)
        self.rnn = nn.GRU(cfg.emb_dim + 2 + cfg.n_categories, cfg.hidden_dim, num_layers=cfg.num_layers,
                          batch_first=True, bidirectional=True,
                          dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        d = 2 * cfg.hidden_dim
        self.mu, self.logvar = nn.Linear(d, cfg.z_timing), nn.Linear(d, cfg.z_timing)

    def forward(self, seq: torch.Tensor, clock: torch.Tensor, props: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = seq.shape
        x = torch.cat([self.embed(seq), clock.expand(B, -1, -1), props[:, None, :].expand(B, T, -1)], dim=-1)
        h, _ = self.rnn(x)
        pooled = h.mean(dim=1)
        return self.mu(pooled), self.logvar(pooled)


class StructuredScheduleVAE(nn.Module):
    def __init__(self, cfg: VAEv2Config, clock: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("clock", clock.float())
        self.register_buffer("pos", position_features(clock.float(), cfg.n_harmonics))
        self.amount_encoder = AmountEncoder(cfg)
        self.timing_encoder = TimingEncoder(cfg)
        self.amount_head = nn.Sequential(nn.Linear(cfg.z_amount, cfg.amount_head_hidden), nn.ReLU(),
                                         nn.Linear(cfg.amount_head_hidden, cfg.n_categories))
        layers, d = [], cfg.z_timing + 2 * cfg.n_harmonics
        for _ in range(cfg.decoder_layers):
            layers += [nn.Linear(d, cfg.decoder_hidden), nn.ReLU()]
            d = cfg.decoder_hidden
        layers.append(nn.Linear(d, cfg.n_categories))
        self.scorer = nn.Sequential(*layers)
        if cfg.conditional_prior:
            self.prior_net = nn.Sequential(nn.Linear(cfg.z_amount, cfg.amount_hidden), nn.ReLU(),
                                           nn.Linear(cfg.amount_hidden, 2 * cfg.z_timing))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def encode(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = seq.long()
        props = proportions(seq, self.cfg.n_categories)
        a_mu, a_lv = self.amount_encoder(props)
        t_mu, t_lv = self.timing_encoder(seq, self.clock, props)
        return {"amount_mu": a_mu, "amount_logvar": a_lv, "timing_mu": t_mu, "timing_logvar": t_lv,
                "props": props}

    def scores(self, z_timing: torch.Tensor) -> torch.Tensor:
        B, T = z_timing.shape[0], self.pos.shape[0]
        x = torch.cat([z_timing[:, None, :].expand(B, T, -1), self.pos[None].expand(B, -1, -1)], dim=-1)
        return self.scorer(x)                                              # (B, T, C)

    def decode(self, z_amount: torch.Tensor, z_timing: torch.Tensor,
               budget_props: torch.Tensor | None = None) -> torch.Tensor:
        """Row-normalised log-probabilities (B, T, C). Budgets from z_amount unless given explicitly."""
        if budget_props is None:
            budget_props = torch.softmax(self.amount_head(z_amount), dim=-1)
        T = self.pos.shape[0]
        log_budget = torch.log(budget_props.clamp_min(1e-6) * T)
        return sinkhorn(self.scores(z_timing) / self.cfg.sinkhorn_tau, log_budget, self.cfg.sinkhorn_iters)

    def forward(self, seq: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        q = self.encode(seq)
        if sample:
            za = self.reparameterize(q["amount_mu"], q["amount_logvar"])
            zt = self.reparameterize(q["timing_mu"], q["timing_logvar"])
        else:
            za, zt = q["amount_mu"], q["timing_mu"]
        amount_logits = self.amount_head(za)
        budget = q["props"] if self.cfg.budget_source == "observed" else torch.softmax(amount_logits, dim=-1)
        out = {**q, "z_amount": za, "z_timing": zt, "amount_logits": amount_logits,
               "logits": self.decode(za, zt, budget_props=budget)}
        if self.cfg.conditional_prior:
            p = self.prior_net(za)
            out["timing_prior_mu"], out["timing_prior_logvar"] = p.chunk(2, dim=-1)
        return out

    @torch.no_grad()
    def decode_exact(self, z_amount: torch.Tensor, z_timing: torch.Tensor,
                     budget_props: torch.Tensor | None = None) -> np.ndarray:
        """Hard decoding with exactly-budgeted slot counts per activity (int8, (B, T))."""
        if budget_props is None:
            budget_props = torch.softmax(self.amount_head(z_amount), dim=-1)
        log_p = self.decode(z_amount, z_timing, budget_props).cpu().numpy()
        return exact_budget_decode(log_p, integer_budgets(budget_props.cpu().numpy(), log_p.shape[1]))
