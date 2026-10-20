#!/usr/bin/env bash
# Phase I over many seeds, two at a time (one per GPU, 7 env workers each on the 16-core box).
#   bash scripts/phase1_seeds.sh 0 9          # seeds 0..9 -> runs/gowu_mr_s{seed}
set -euo pipefail
PY=${PY:-.venv/bin/python}
FIRST=${1:-0}; LAST=${2:-9}; WORKERS=${WORKERS:-7}
run_seed() {
  local s=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_gowu.py --config configs/gowu_mr.yaml \
    --set seed=$s n_workers=$WORKERS out_dir=runs/gowu_mr_s$s > runs/gowu_mr_s$s.out 2>&1
}
mkdir -p runs
s=$FIRST
while [ $s -le $LAST ]; do
  run_seed $s 0 &
  if [ $((s+1)) -le $LAST ]; then run_seed $((s+1)) 1 & fi
  wait
  s=$((s+2))
done
$PY scripts/summarize.py runs/gowu_mr_s*
