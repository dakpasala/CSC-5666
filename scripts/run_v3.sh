#!/usr/bin/env bash
# v3 one-factor experiments (docs/v3_plan.md): experiments x seeds 42-44, 40 epochs, 6 runs at a time on CPU.
# Finished runs are skipped, so the script can be re-launched. JOBS sets parallel runs (default 6). Pass experiment names to run a subset;
# SEEDS="45 46 47" overrides the seeds.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_v3.yaml
LOG=outputs/v3/logs
mkdir -p "$LOG"
export PY CFG LOG
sets_for() {
  case "$1" in
    base)        echo "" ;;                                 # v1 baseline settings (for extra seeds)
    base80)      echo "train.epochs=80" ;;                  # v1 baseline trained 80 epochs (longer-training test)
    E1_local)    echo "pairs.family=local" ;;
    E2_betaT2)   echo "loss.beta_timing=2.0" ;;
    E3_tsup)     echo "model.timing_head=true loss.lambda_timing=0.3" ;;
    C1_combined) echo "${C1_SETS:?set C1_SETS for the combined run}" ;;
    # Capacity sweep (2022 validation only): one size changed per run, latent sizes unchanged (8 + 16).
    K_emb32)     echo "model.emb_dim=32" ;;
    K_gru128)    echo "model.hidden_dim=128" ;;
    K_gru2L)     echo "model.num_layers=2" ;;
    K_dec256)    echo "model.decoder_hidden=256 model.decoder_layers=3" ;;
    K_harm8)     echo "model.n_harmonics=8" ;;
    K_big)       echo "model.emb_dim=32 model.hidden_dim=128 model.num_layers=2 model.decoder_hidden=256 model.decoder_layers=3 model.n_harmonics=8" ;;
    # Latent-size sweep (v3 network unchanged). Bigger latents give probes more dimensions; compare with care.
    Z_t32)       echo "model.z_timing=32" ;;
    Z_a12)       echo "model.z_amount=12" ;;
    Z_t32a12)    echo "model.z_timing=32 model.z_amount=12" ;;
    # Larger networks (K_big was not overfitting and still improving at 40 epochs). Latents stay 8 + 16.
    K_big_e80)   echo "model.emb_dim=32 model.hidden_dim=128 model.num_layers=2 model.decoder_hidden=256 model.decoder_layers=3 model.n_harmonics=8 train.epochs=80" ;;
    K_big2)      echo "model.emb_dim=32 model.hidden_dim=192 model.num_layers=2 model.decoder_hidden=256 model.decoder_layers=3 model.n_harmonics=8" ;;
    K_big3)      echo "model.emb_dim=32 model.hidden_dim=256 model.num_layers=2 model.decoder_hidden=256 model.decoder_layers=3 model.n_harmonics=8" ;;
    # K_big2 encoder with a bidirectional GRU decoder (not autoregressive); 1.08M params vs K_big2's 1.09M.
    D_gru)       echo "model.emb_dim=32 model.hidden_dim=192 model.num_layers=2 model.decoder_type=gru model.decoder_hidden=128 model.decoder_layers=1 model.n_harmonics=8" ;;
    KZ_big_t32)  echo "model.emb_dim=32 model.hidden_dim=128 model.num_layers=2 model.decoder_hidden=256 model.decoder_layers=3 model.n_harmonics=8 model.z_timing=32" ;;
    *) echo "unknown experiment $1" >&2; exit 1 ;;
  esac
}
if [ $# -eq 0 ]; then set -- E1_local E2_betaT2 E3_tsup; fi
for exp in "$@"; do
  extra=$(sets_for "$exp")
  for seed in ${SEEDS:-42 43 44}; do
    name="${exp}_s$seed"
    if ls -d outputs/v3/"$name"-*/eval_best.json >/dev/null 2>&1; then echo "skip $name" >&2; continue; fi
    echo "$name seed=$seed${extra:+ $extra}"   # no trailing blank: xargs -L treats it as a continuation
  done
done | xargs -P "${JOBS:-6}" -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "v3 runs finished"
