#!/usr/bin/env bash
# Final frozen models (Checkpoint 5): 4 variants x seeds 42-44, 80 epochs, 6 runs at a time on CPU.
# lambda_invariance for `guided` comes from configs/vae_final.yaml (set by the full-data sweep rule).
# Then: python scripts/evaluate_final.py --target val   (and, after approval, --target test --confirm-preregistered)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_final.yaml
LOG=outputs/checkpoint5_final/logs
mkdir -p "$LOG"
export PY CFG LOG
for seed in 42 43 44; do
  echo "guided_s$seed seed=$seed"
  echo "no_invariance_s$seed seed=$seed loss.lambda_invariance=0.0"
  echo "unguided_s$seed seed=$seed loss.lambda_invariance=0.0 loss.lambda_amount=0.0"
  echo "joint_s$seed seed=$seed model.z_amount=0 model.z_timing=24 loss.lambda_invariance=0.0 loss.lambda_amount=0.0"
done | xargs -P 6 -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "final runs finished"
