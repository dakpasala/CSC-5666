#!/usr/bin/env bash
# v4 mixture-prior runs (docs/v4_mixture_plan.md): real K=4, real K=6, null K=6 x seeds 42-44, 80 epochs
# (20 standard-prior + 60 mixture), 6 at a time. Finished runs are skipped. Run detached:
#   nohup caffeinate -i ./scripts/run_v4.sh > outputs/v4/logs/launcher.log 2>&1 < /dev/null & disown
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_v3.yaml
LOG=outputs/v4/logs
mkdir -p "$LOG"
export PY CFG LOG
COMMON="output_dir=outputs/v4 train.epochs=80 model.timing_prior=mixture mixture.pretrain_epochs=20"
for seed in 42 43 44; do
  for spec in "gm_k4:model.mixture_k=4" "gm_k6:model.mixture_k=6" "null_k6:model.mixture_k=6 year=null2022"; do
    exp=${spec%%:*}; extra=${spec#*:}; name="${exp}_s$seed"
    if ls -d outputs/v4/"$name"-*/eval_best.json >/dev/null 2>&1; then echo "skip $name" >&2; continue; fi
    echo "$name seed=$seed $COMMON $extra"
  done
done | xargs -P 6 -L 1 bash -c 'name=$0; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "$LOG/$name.log" 2>&1 || echo "FAILED $name"'
echo "v4 runs finished"
