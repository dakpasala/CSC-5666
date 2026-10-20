#!/usr/bin/env bash
# Launch all Checkpoint 4 variants in parallel (one CPU thread each). Logs: outputs/checkpoint4/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
CFG=configs/vae_checkpoint4.yaml
mkdir -p outputs/checkpoint4/logs
run() { local name=$1; shift; "$PY" scripts/train_vae.py --config "$CFG" --set "name=$name" "$@" > "outputs/checkpoint4/logs/$name.log" 2>&1 & }
run guided_inv0.1   loss.lambda_invariance=0.1
run guided_inv1     loss.lambda_invariance=1.0
run guided_inv10    loss.lambda_invariance=10.0
run no_invariance   loss.lambda_invariance=0.0
run invariance_only loss.lambda_amount=0.0
run unguided        loss.lambda_invariance=0.0 loss.lambda_amount=0.0
run joint           model.z_amount=0 model.z_timing=24 loss.lambda_invariance=0.0 loss.lambda_amount=0.0
wait
echo "all Checkpoint 4 runs finished"
