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

from src.data.pairs import PairConfig, make_pairs  # noqa: E402
from src.evaluation.vae_report import evaluate_run  # noqa: E402
from src.models.losses import LossWeights  # noqa: E402
from src.models.train import (  # noqa: E402
    build_model, load_checkpoint, pick_device, posterior_means, set_seed, subset_indices, train,
)
from src.models.vae import parameter_count  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/vae_checkpoint3.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. name=foo loss.lambda_invariance=10 model.z_amount=0")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
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

    model = build_model({"n_categories": len(z["categories"]), "n_positions": z["seq"].shape[1], **cfg["model"]},
                        torch.from_numpy(z["clock_features"])).to(device)
    print(f"run {run.relative_to(ROOT)}  device={device}  params={parameter_count(model):,}  "
          f"train={len(tr_idx)}  val={len(va_idx)}")

    seq_tr = torch.from_numpy(z["seq"][tr_idx]).long().to(device)
    seq_va = torch.from_numpy(z["seq"][va_idx]).long().to(device)
    w = LossWeights(**cfg["loss"])
    t = cfg["train"]
    pair_fn, seq_va_pair = None, None
    pcfg_raw = cfg.get("pairs") or {}
    if pcfg_raw.get("enabled"):
        # Pairs are built inside each split after splitting: train pairs fresh every batch, validation
        # pairs fixed once (seeded) for model selection. Transform types are logged per epoch.
        pcfg = PairConfig(**{k: v for k, v in pcfg_raw.items() if k != "enabled"})
        C = len(z["categories"])
        pair_rng = np.random.default_rng(seed + 1)
        pair_fn = lambda b: make_pairs(b, pair_rng, pcfg, C)  # noqa: E731
        va_t, va_params = make_pairs(z["seq"][va_idx], np.random.default_rng(seed + 2), pcfg, C)
        seq_va_pair = torch.from_numpy(va_t).long().to(device)
        (run / "val_pairs_params.json").write_text(json.dumps(
            {"pair_config": pcfg.to_dict(), "case_ids": z["case_ids"][va_idx].tolist(), "params": va_params}))
    epoch_hook = None
    if cfg["model"].get("timing_prior") == "mixture":
        # v4: standard prior for `pretrain_epochs`, then initialise the mixture from a diagonal GMM fitted to the
        # train-split timing posterior means and switch it on (VaDE-style initialisation).
        pre = int((cfg.get("mixture") or {}).get("pretrain_epochs", 20))
        model.mixture_active = pre == 0

        def epoch_hook(epoch: int, m) -> bool:
            if epoch != pre or pre == 0:
                return False
            from sklearn.mixture import GaussianMixture
            pm = posterior_means(m, seq_tr)
            mu = pm["timing_mu"]
            gm = GaussianMixture(m.cfg.mixture_k, covariance_type="diag", random_state=seed, n_init=5).fit(mu)
            # Component variance = spread of posterior means within the component + average posterior variance,
            # so each component matches the aggregate posterior (fitting means alone makes components far too
            # narrow and the KL jumps by orders of magnitude at the switch).
            var = gm.covariances_ + np.exp(pm["timing_logvar"]).mean(0, keepdims=True)
            with torch.no_grad():
                m.mix_mu.copy_(torch.as_tensor(gm.means_, dtype=torch.float32))
                m.mix_logvar.copy_(torch.as_tensor(np.log(var), dtype=torch.float32))
                m.mix_logits.copy_(torch.as_tensor(np.log(gm.weights_), dtype=torch.float32))
            m.mixture_active = True
            print(f"epoch {epoch + 1}: mixture prior on (GMM init, weights {np.round(gm.weights_, 3).tolist()})", flush=True)
            return True
    history = train(model, seq_tr, seq_va, w, t["epochs"], t["batch_size"], t["lr"], t["kl_warmup_epochs"],
                    seed, run, t["log_every"], pair_fn=pair_fn, seq_va_pair=seq_va_pair, epoch_hook=epoch_hook)
    pd.DataFrame(history).to_csv(run / "history.csv", index=False)

    # Reload check: the model still in memory (final weights) vs the same weights reloaded from last.pt.
    best = run / "best.pt" if (run / "best.pt").exists() else run / "last.pt"
    reloaded, _ = load_checkpoint(run / "last.pt", "cpu")
    a = posterior_means(model.cpu(), seq_va.cpu())
    b = posterior_means(reloaded, seq_va.cpu())
    max_diff = max(float(np.abs(a[k] - b[k]).max()) for k in a if a[k].size)
    assert max_diff < 1e-5, f"reload mismatch {max_diff}"

    metrics = evaluate_run(run, best.name)
    v, ref = metrics["val"], metrics["val_position_majority_reference"]
    print(f"[{best.name} @ epoch {metrics['checkpoint_epoch']}] val accuracy {v['accuracy']:.3f} "
          f"(position-majority {ref['accuracy']:.3f}), macro-F1 {v['macro_f1']:.3f} ({ref['macro_f1']:.3f}), "
          f"CE {v['cross_entropy']:.3f}, transitions true/pred {v['transitions_true_mean']:.1f}/"
          f"{v['transitions_pred_mean']:.1f}")
    la, lt = metrics["latents_val"]["amount"], metrics["latents_val"]["timing"]
    amount_str = f"{la['active_dims']}/{la['n_dims']}" if la else "none (joint)"
    print(f"active dims: amount {amount_str}, timing {lt['active_dims']}/{lt['n_dims']}; "
          f"reload max |diff| {max_diff:.2e}")
    print(f"wrote {run}")


if __name__ == "__main__":
    main()
