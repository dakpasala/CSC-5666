"""Train the two-latent schedule VAE, then reload its checkpoint from disk and evaluate it.

Usage:
    python scripts/train_vae.py --config configs/vae_smoke.yaml        # seconds; pipeline check only
    python scripts/train_vae.py --config configs/vae_checkpoint3.yaml  # Checkpoint 3 prototype

Creates outputs/<output_dir>/<name>-<timestamp>/ containing the resolved config, subset IDs, category
mapping, history, best/last checkpoints, evaluation JSON, posterior-mean embeddings and plots.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.vae_report import evaluate_run  # noqa: E402
from src.models.losses import LossWeights  # noqa: E402
from src.models.train import (  # noqa: E402
    load_checkpoint, pick_device, posterior_means, set_seed, subset_indices, train,
)
from src.models.vae import ScheduleVAE, VAEConfig, parameter_count  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/vae_checkpoint3.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    dcfg = yaml.safe_load((ROOT / cfg["data_config"]).read_text())
    seed = cfg["seed"]
    set_seed(seed)
    device = pick_device(cfg["device"])

    npz_path = ROOT / dcfg["processed_dir"] / str(cfg["year"]) / "sequences.npz"
    z = np.load(npz_path)
    tr_idx = subset_indices(z["split"], "train", cfg["data"]["n_train"], seed)
    va_idx = subset_indices(z["split"], "val", cfg["data"]["n_val"], seed)
    assert not set(z["case_ids"][tr_idx]) & set(z["case_ids"][va_idx])

    run = ROOT / cfg["output_dir"] / f"{cfg['name']}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run.mkdir(parents=True, exist_ok=False)
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    resolved = {**cfg, "data_settings": dcfg, "sequences_path": str(npz_path.relative_to(ROOT)),
                "sequences_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
                "git_revision": rev, "git_dirty": dirty, "device_used": str(device),
                "torch_version": str(torch.__version__), "n_train_used": int(len(tr_idx)), "n_val_used": int(len(va_idx))}
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
    (run / "subset_ids.json").write_text(json.dumps({"train": z["case_ids"][tr_idx].tolist(),
                                                     "val": z["case_ids"][va_idx].tolist()}))
    shutil.copy(npz_path.parent / "category_mapping.yaml", run / "category_mapping.yaml")

    mcfg = VAEConfig(n_categories=len(z["categories"]), n_positions=z["seq"].shape[1], **cfg["model"])
    model = ScheduleVAE(mcfg, torch.from_numpy(z["clock_features"])).to(device)
    print(f"run {run.relative_to(ROOT)}  device={device}  params={parameter_count(model):,}  "
          f"train={len(tr_idx)}  val={len(va_idx)}")

    seq_tr = torch.from_numpy(z["seq"][tr_idx]).long().to(device)
    seq_va = torch.from_numpy(z["seq"][va_idx]).long().to(device)
    w = LossWeights(**cfg["loss"])
    t = cfg["train"]
    history = train(model, seq_tr, seq_va, w, t["epochs"], t["batch_size"], t["lr"], t["kl_warmup_epochs"],
                    seed, run, t["log_every"])
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)

    # Reload check: the model still in memory (final weights) vs the same weights reloaded from last.pt.
    best = run / "best.pt" if (run / "best.pt").exists() else run / "last.pt"
    reloaded, _ = load_checkpoint(run / "last.pt", "cpu")
    a = posterior_means(model.cpu(), seq_va.cpu())
    b = posterior_means(reloaded, seq_va.cpu())
    max_diff = max(float(np.abs(a[k] - b[k]).max()) for k in a)
    assert max_diff < 1e-5, f"reload mismatch {max_diff}"

    metrics = evaluate_run(run, best.name)
    v, ref = metrics["val"], metrics["val_position_majority_reference"]
    print(f"[{best.name} @ epoch {metrics['checkpoint_epoch']}] val accuracy {v['accuracy']:.3f} "
          f"(position-majority {ref['accuracy']:.3f}), macro-F1 {v['macro_f1']:.3f} ({ref['macro_f1']:.3f}), "
          f"CE {v['cross_entropy']:.3f}, transitions true/pred {v['transitions_true_mean']:.1f}/"
          f"{v['transitions_pred_mean']:.1f}")
    la, lt = metrics["latents_val"]["amount"], metrics["latents_val"]["timing"]
    print(f"active dims: amount {la['active_dims']}/{la['n_dims']}, timing {lt['active_dims']}/{lt['n_dims']}; "
          f"reload max |diff| {max_diff:.2e}")
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
