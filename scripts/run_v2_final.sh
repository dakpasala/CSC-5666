#!/usr/bin/env bash
# Final model v2 runs: chosen sweep setting (configs/vae_v2.yaml, beta 0.3) x seeds 42-44, 80 epochs, in parallel.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_v2.yaml
LOG=outputs/v2/logs
mkdir -p "$LOG"
export PY CFG LOG
for seed in 42 43 44; do
  name="v2_s$seed"
  if ls -d outputs/v2/"$name"-*/eval_best.json >/dev/null 2>&1; then echo "skip $name (already finished)" >&2; continue; fi
  echo "$name seed=$seed"
done | xargs -P 3 -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "v2 final runs finished"
