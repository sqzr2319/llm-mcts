#!/usr/bin/env sh

# Lightweight wrapper to run with profiling enabled.
# Usage:
#   sh profile_run.sh --model_type vllm --n_iters 4 --output_tree_vis

set -eu

LOG_DIR=${LOG_DIR:-"./output"}
EXPN=${EXPN:-"profile_exp"}

python run.py \
  --entry mcts \
  --model_type ${MODEL_TYPE:-vllm} \
  --n_iters ${N_ITERS:-4} \
  --output_tree_vis \
  --profile \
  --save_dir "$LOG_DIR" \
  --exp_name "$EXPN" "$@"

echo "Run complete. Find timings under the created save_dir (timings.jsonl + summary)."
