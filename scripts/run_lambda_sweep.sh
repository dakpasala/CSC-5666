#!/usr/bin/env bash
# Full-data lambda_invariance sweep (Checkpoint 5): lambda in {0.1, 0.3} x seeds 42-44, same 40-epoch budget
# as the robustness runs so the pre-registered rule (CE within 5% of no_invariance) compares like with like.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_checkpoint5_robustness.yaml
LOG=outputs/checkpoint5_robustness/logs
mkdir -p "$LOG"
export PY CFG LOG
for lam in 0.1 0.3; do for seed in 42 43 44; do echo "guided_l${lam}_s$seed seed=$seed loss.lambda_invariance=$lam"; done; done |
  xargs -P 6 -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "lambda sweep finished"
