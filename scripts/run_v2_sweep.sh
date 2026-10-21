#!/usr/bin/env bash
# Model v2 settings sweep (docs/v2_validation_plan.md): 6 runs, seed 42, 30 epochs, 6 at a time on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_v2.yaml
LOG=outputs/v2_sweep/logs
mkdir -p "$LOG"
export PY CFG LOG
COMMON="output_dir=outputs/v2_sweep train.epochs=30"
{
  echo "sw_base $COMMON"
  echo "sw_observed $COMMON model.budget_source=observed"
  echo "sw_beta0.3 $COMMON loss.beta_amount=0.3 loss.beta_timing=0.3"
  echo "sw_beta0.1 $COMMON loss.beta_amount=0.1 loss.beta_timing=0.1"
  echo "sw_condprior $COMMON model.conditional_prior=true"
  echo "sw_nopairs $COMMON pairs.enabled=false"
} | xargs -P 6 -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "v2 sweep finished"
