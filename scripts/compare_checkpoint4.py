"""Compare Checkpoint 4 variants side by side and apply the pre-registered lambda_invariance rule.

Usage:
    python scripts/compare_checkpoint4.py [--cp3-run outputs/checkpoint3/<run>]

Uses the latest run directory for each variant in outputs/checkpoint4/. Adds two reference rows:
- `cp3_no_pairs`: the Checkpoint 3 model (no pairs, no guidance), if a run is given or found.
- `untrained`: the same architecture with random initial weights. Because the encoder mean-pools over the
  day, even untrained latents carry linear information about totals; probe numbers must beat this row.
Writes outputs/checkpoint4/comparison.csv and docs/checkpoint4_report.md (aggregate only, validation data).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.separation import evaluate_separation  # noqa: E402
from src.models.train import posterior_means, set_seed  # noqa: E402
from src.models.vae import ScheduleVAE, VAEConfig  # noqa: E402

VARIANTS = ["guided_inv0.1", "guided_inv1", "guided_inv10", "no_invariance", "invariance_only", "unguided", "joint"]


def latest_run(base: Path, name: str) -> Path | None:
    runs = sorted(p for p in base.glob(f"{name}-*") if (p / "eval_best.json").exists())
    return runs[-1] if runs else None


def untrained_reference(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    dcfg = yaml.safe_load((ROOT / cfg["data_config"]).read_text())
    z = np.load(ROOT / dcfg["processed_dir"] / str(cfg["year"]) / "sequences.npz")
    set_seed(cfg["seed"])
    model = ScheduleVAE(VAEConfig(n_categories=len(z["categories"]), **cfg["model"]),
                        torch.from_numpy(z["clock_features"])).eval()
    emb = posterior_means(model, torch.from_numpy(z["seq"]).long())
    return {"separation": evaluate_separation(model, z["seq"], z["split"], z["categories"].tolist(),
                                              z["clock_features"], emb)}


def row(name: str, m: dict, cfg: dict | None) -> dict:
    s = m["separation"]
    ps, pr, mm = s["pair_sensitivity_val"], s["probes"]["inputs"], s["mix_and_match_val"]
    get = lambda d, *ks: (None if d is None else (get(d.get(ks[0]), *ks[1:]) if ks else d))  # noqa: E731
    r = {"variant": name}
    if cfg:
        r.update({"lambda_inv": cfg["loss"]["lambda_invariance"], "lambda_amt": cfg["loss"]["lambda_amount"],
                  "pairs": bool((cfg.get("pairs") or {}).get("enabled")), "z_amount": cfg["model"]["z_amount"],
                  "z_timing": cfg["model"]["z_timing"]})
    if "val" in m:
        v = m["val"]
        r.update({"val_CE": v["cross_entropy"], "val_acc": v["accuracy"], "val_macroF1": v["macro_f1"],
                  "transitions_pred": v["transitions_pred_mean"], "best_epoch": m["checkpoint_epoch"]})
        la, lt = m["latents_val"]["amount"], m["latents_val"]["timing"]
        r.update({"amount_active": la and la["active_dims"], "amount_KL": la and la["kl_total_mean"],
                  "timing_active": lt["active_dims"], "timing_KL": lt["kl_total_mean"]})
        dr = m["decoder_reliance_val_accuracy"]
        r.update({"acc_amount_fixed": dr["amount_fixed_to_mean"], "acc_timing_fixed": dr["timing_fixed_to_mean"]})
    for t in ("shift", "swap"):
        r[f"amount_sens_{t}"] = get(ps, t, "amount", "median")
        r[f"timing_sens_{t}"] = get(ps, t, "timing", "median")
    for lat in ("amount", "timing"):
        r[f"totalsR2_from_{lat}"] = get(pr, lat, "totals_r2")
        r[f"clockR2_from_{lat}"] = get(pr, lat, "timing_clock_r2")
        r[f"clock_gain_{lat}"] = get(pr, lat, "timing_clock_gain_over_true_totals")
    r["mix_totals_follow_amount"] = get(mm, "totals_closer_to_amount_source")
    r["mix_timing_follow_timing"] = get(mm, "timing_closer_to_timing_source")
    mt = s.get("mix_and_match_matched_totals_val")
    r["matched_mix_timing_follow_timing"] = get(mt, "timing_closer_to_timing_source")
    r["matched_mix_slots_timing_src"] = get(mt, "slot_agreement_with_timing_source")
    r["matched_mix_slots_amount_src"] = get(mt, "slot_agreement_with_amount_source")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=ROOT / "outputs/checkpoint4")
    ap.add_argument("--cp3-run", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/vae_checkpoint4.yaml")
    args = ap.parse_args()

    rows, runs = [], {}
    for name in VARIANTS:
        run = latest_run(args.base, name)
        if run is None:
            print(f"missing run for {name}")
            continue
        runs[name] = run
        rows.append(row(name, json.loads((run / "eval_best.json").read_text()),
                        yaml.safe_load((run / "resolved_config.yaml").read_text())))
    cp3 = args.cp3_run or latest_run(ROOT / "outputs/checkpoint3", "vae_cp3")
    if cp3:
        runs["cp3_no_pairs"] = cp3
        rows.append(row("cp3_no_pairs", json.loads((cp3 / "eval_best.json").read_text()),
                        yaml.safe_load((cp3 / "resolved_config.yaml").read_text())))
    ref_path = args.base / "untrained_reference.json"
    if not ref_path.exists():
        ref_path.write_text(json.dumps(untrained_reference(args.config)))
    rows.append(row("untrained", json.loads(ref_path.read_text()), None))
    df = pd.DataFrame(rows)
    df.to_csv(args.base / "comparison.csv", index=False)

    # Pre-registered selection of lambda_invariance (validation only).
    guided = df[df["variant"].str.startswith("guided_inv")].copy()
    ref_ce = float(df.loc[df["variant"] == "no_invariance", "val_CE"].iloc[0])
    guided["amount_sens"] = guided[["amount_sens_shift", "amount_sens_swap"]].mean(axis=1)
    ok = guided[guided["val_CE"] <= 1.05 * ref_ce]
    selected = ok.sort_values("amount_sens").iloc[0]["variant"] if len(ok) else None
    write_report(df, runs, selected, ref_ce)
    print(df.round(3).to_string())
    print(f"selected: {selected}")


def write_report(df: pd.DataFrame, runs: dict, selected: str | None, ref_ce: float) -> None:
    fmt = lambda d: d.to_markdown(index=False, floatfmt=".3f")  # noqa: E731
    recon_cols = ["variant", "lambda_inv", "lambda_amt", "val_CE", "val_acc", "val_macroF1", "transitions_pred",
                  "best_epoch", "amount_active", "timing_active", "amount_KL", "timing_KL"]
    sens_cols = ["variant", "amount_sens_shift", "amount_sens_swap", "timing_sens_shift", "timing_sens_swap"]
    probe_cols = ["variant", "totalsR2_from_amount", "totalsR2_from_timing", "clockR2_from_amount",
                  "clockR2_from_timing", "clock_gain_amount", "clock_gain_timing"]
    mix_cols = ["variant", "mix_totals_follow_amount", "mix_timing_follow_timing", "matched_mix_timing_follow_timing",
                "matched_mix_slots_timing_src", "matched_mix_slots_amount_src", "acc_amount_fixed", "acc_timing_fixed"]
    lines = [
        "# Checkpoint 4 comparison report",
        "",
        "Generated by `scripts/compare_checkpoint4.py`. 2022 data only: each model trained on the same",
        "400-diary train subset; all numbers on the 1,627-diary validation split (probes fitted on train-split",
        "embeddings). Single seed (42). Engineering-scale evidence, not final findings. 2023 untouched.",
        "",
        "Runs: " + ", ".join(f"`{k}` → `{v.relative_to(ROOT)}`" for k, v in runs.items()),
        "",
        f"**Pre-registered λ_invariance choice:** {selected} (lowest mean amount pair-sensitivity among guided runs"
        f" with validation CE ≤ 1.05 × no_invariance CE = {1.05 * ref_ce:.3f}).",
        "",
        "## Reconstruction and collapse diagnostics",
        "",
        "CE in nats per 5-minute slot. Active = latent dims whose posterior mean varies across diaries (var > 0.01).",
        "KL in nats per diary.",
        "",
        fmt(df[[c for c in recon_cols if c in df]]),
        "",
        "## Pair sensitivity (validation pairs, identical for all models)",
        "",
        "Median distance between a diary's posterior mean and that of its total-preserving transform, divided by",
        "the median distance between two random validation diaries. Desired: amount small, timing larger.",
        "",
        fmt(df[sens_cols]),
        "",
        "## Linear probes (fit on train embeddings, R² on validation)",
        "",
        "`totalsR2_from_*`: predicting true category shares. `clockR2_from_*`: predicting engineered clock-position",
        "timing features (centre, first onset, last offset). `clock_gain_*`: R² gain on those timing features from",
        "adding the latent to the true totals (timing information beyond totals).",
        "The `untrained` row is the floor: mean-pooling makes even random-weight latents linear in totals.",
        "",
        fmt(df[probe_cols]),
        "",
        "## Mix-and-match decoding and decoder reliance",
        "",
        "Decode (amount latent of diary A, timing latent of diary B) for 1,000 random validation pairs: share where",
        "decoded totals are closer to A, and decoded clock timing closer to B (0.5 = no preference). Reliance =",
        "validation accuracy when one latent is replaced by its mean (both-latent accuracy is `val_acc`).",
        "Random pairs confound timing with activity presence (presence follows totals, i.e. the amount source).",
        "`matched_mix_*` repeats the test on validation pairs whose totals differ by <= 120 reassigned minutes",
        "(both directions): share with clock timing closer to the timing source, and mean per-slot agreement of",
        "the decoded day with the timing source vs the amount source.",
        "",
        fmt(df[[c for c in mix_cols if c in df]]),
        "",
    ]
    (ROOT / "docs/checkpoint4_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
