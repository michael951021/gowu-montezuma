#!/usr/bin/env bash
# Two LR-GoWU runs side by side, one per GPU, then a summary.
# usage: scripts/lr_pair.sh OUT HOURS "--set overrides for run a" "--set overrides for run b"
set -u
OUT=$1; HOURS=$2; A=$3; B=$4
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_lr_gowu.py --config configs/lr_gowu_mr.yaml \
  --set out_dir="$OUT/a" max_hours="$HOURS" n_workers=14 $A > "$OUT/a.log" 2>&1 &
PA=$!
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_lr_gowu.py --config configs/lr_gowu_mr.yaml \
  --set out_dir="$OUT/b" max_hours="$HOURS" n_workers=14 $B > "$OUT/b.log" 2>&1 &
PB=$!
wait $PA; EA=$?
wait $PB; EB=$?
.venv/bin/python scripts/summarize_lr.py "$OUT"
exit $(( EA || EB ))
