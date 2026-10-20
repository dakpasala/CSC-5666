#!/usr/bin/env bash
# Checkpoint 5 robustness: 4 variants x 3 seeds on the full 2022 train split, 6 runs at a time on CPU.
# Logs: outputs/checkpoint5_robustness/logs/. Then: python scripts/compare_robustness.py
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_checkpoint5_robustness.yaml
LOG=outputs/checkpoint5_robustness/logs
mkdir -p "$LOG"
jobs_list() {
  for seed in 42 43 44; do
    echo "guided_s$seed seed=$seed"
    echo "no_invariance_s$seed seed=$seed loss.lambda_invariance=0.0"
    echo "unguided_s$seed seed=$seed loss.lambda_invariance=0.0 loss.lambda_amount=0.0"
    echo "joint_s$seed seed=$seed model.z_amount=0 model.z_timing=24 loss.lambda_invariance=0.0 loss.lambda_amount=0.0"
  done
}
export PY CFG LOG
jobs_list | xargs -P 6 -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "all robustness runs finished"
