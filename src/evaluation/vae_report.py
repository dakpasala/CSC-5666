"""Evaluate a saved VAE checkpoint: reconstruction, latent diagnostics, embeddings export, plots.

Everything is rebuilt from the run directory (checkpoint + resolved config + subset IDs), so it doubles
as the reload check: no preprocessing is refitted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.plotting import GRID, INK, MUTED, palette, plt, timeline_rows
from src.evaluation.reconstruction import latent_diagnostics, position_majority, reconstruction_metrics
from src.models.train import load_checkpoint, posterior_means, reconstruct

ROOT = Path(__file__).resolve().parents[2]


def load_run_data(run: Path) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    ids = json.loads((run / "subset_ids.json").read_text())
    z = np.load(ROOT / cfg["sequences_path"])
    return cfg, ids, z


def rows_for(z, ids: list[str]) -> np.ndarray:
    pos = {c: i for i, c in enumerate(z["case_ids"].tolist())}
    return np.array([pos[c] for c in ids])


def evaluate_run(run: Path, which: str = "best.pt") -> dict:
    cfg, ids, z = load_run_data(run)
    cats = z["categories"].tolist()
    C, w = len(cats), int(z["bin_minutes"])
    model, ck = load_checkpoint(run / which, "cpu")

    seq_all = z["seq"]
    tr_rows, va_rows = rows_for(z, ids["train"]), rows_for(z, ids["val"])
    seq_tr, seq_va = seq_all[tr_rows], seq_all[va_rows]
    t_tr, t_va = torch.from_numpy(seq_tr).long(), torch.from_numpy(seq_va).long()

    pred_va, logp_va = reconstruct(model, t_va)
    pred_tr, logp_tr = reconstruct(model, t_tr)
    maj = position_majority(z["seq"][z["split"] == "train"], C)       # reference fitted on train split only
    maj_va = np.broadcast_to(maj, seq_va.shape)
    metrics = {
        "checkpoint": which, "checkpoint_epoch": ck.get("epoch"),
        "n_train_subset": len(tr_rows), "n_val": len(va_rows),
        "val": reconstruction_metrics(seq_va, pred_va, C, w, logp_va),
        "train_subset": reconstruction_metrics(seq_tr, pred_tr, C, w, logp_tr),
        "val_position_majority_reference": reconstruction_metrics(seq_va, maj_va, C, w),
        "categories": cats,
    }

    # Posterior-mean embeddings for every 2022 diary (train + val), for later probes and clustering.
    emb = posterior_means(model, torch.from_numpy(seq_all).long())
    np.savez_compressed(run / f"embeddings_{Path(which).stem}.npz", case_ids=z["case_ids"], split=z["split"],
                        in_train_subset=np.isin(z["case_ids"], ids["train"]), **emb)
    va_mask = z["split"] == "val"
    metrics["latents_val"] = {
        "amount": latent_diagnostics(emb["amount_mu"][va_mask], emb["amount_logvar"][va_mask]),
        "timing": latent_diagnostics(emb["timing_mu"][va_mask], emb["timing_logvar"][va_mask]),
    }

    # Decoder reliance on each latent: replace one latent with its validation mean and re-decode.
    with torch.no_grad():
        q = model.encode(t_va)
        base = model.decode(q["amount_mu"], q["timing_mu"])
        fixed_a = model.decode(q["amount_mu"].mean(0, keepdim=True).expand_as(q["amount_mu"]), q["timing_mu"])
        fixed_t = model.decode(q["amount_mu"], q["timing_mu"].mean(0, keepdim=True).expand_as(q["timing_mu"]))
    def acc(logits):
        return float((logits.argmax(-1).numpy() == seq_va).mean())
    metrics["decoder_reliance_val_accuracy"] = {
        "both_latents": acc(base), "amount_fixed_to_mean": acc(fixed_a), "timing_fixed_to_mean": acc(fixed_t),
        "note": "Diagnostic only. Drops show what the decoder reads from each latent; not proof of meaning.",
    }

    (run / f"eval_{Path(which).stem}.json").write_text(json.dumps(metrics, indent=2))
    plot_reconstructions(run, cfg, seq_va, pred_va, cats, int(z["diary_start_clock_minute"]), w)
    plot_history(run)
    return metrics


def plot_reconstructions(run: Path, cfg: dict, seq_va: np.ndarray, pred_va: np.ndarray, cats: list[str],
                         start_clock: int, w: int) -> None:
    """Random (not best) validation diaries: original row above its reconstruction."""
    rng = np.random.default_rng(cfg["eval"]["plot_seed"])
    pick = np.sort(rng.choice(len(seq_va), size=min(cfg["eval"]["n_plot_examples"], len(seq_va)), replace=False))
    rows, labels = [], []
    for j, i in enumerate(pick):
        acc = (seq_va[i] == pred_va[i]).mean()
        rows += [seq_va[i], pred_va[i]]
        labels += [f"#{j + 1} original", f"#{j + 1} recon ({acc:.0%})"]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(rows) + 1.3))
    timeline_rows(ax, np.array(rows), cats, start_clock, labels)
    for y in np.arange(1.5, len(rows) - 1, 2):
        ax.axhline(y, color="white", linewidth=3)
    ax.set_title(f"{cfg['name']}: random validation diaries, original vs reconstruction from posterior means",
                 loc="left", color=INK)
    fig.legend(handles=palette(cats)[1], loc="lower center", ncol=9, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(run / "reconstructions_val.png", dpi=130)
    plt.close(fig)


def plot_history(run: Path) -> None:
    h = pd.read_csv(run / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    series = [("recon", "#2a78d6"), ("kl_amount", "#eb6834"), ("kl_timing", "#1baf7a")]
    for ax, split in zip(axes, ("train", "val")):
        for name, col in series:
            ax.plot(h["epoch"], h[f"{split}_{name}"], color=col, linewidth=2, label=name.replace("_", " "))
        ax.set_title(f"{split} loss terms (nats per position)", loc="left", color=INK, fontsize=9)
        ax.set_xlabel("epoch")
        ax.grid(color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "training_curves.png", dpi=130)
    plt.close(fig)
