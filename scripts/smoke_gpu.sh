#!/usr/bin/env bash
# Directional check on the 2x3090 box (~2-3 h total). Run from the repo root:  bash scripts/smoke_gpu.sh
#  1. unit tests
#  2. tiny GoWU phase I: 20M frames (~10-30 min). Healthy: best_R climbs past 400 (key + door) and rooms > 3.
#  3. tiny phase II on that demo: 100M frames on GPU0 with the background evaluator on GPU1.
#     Healthy: `solved` climbs for the first segments, eval score > 0 (first key from reset).
set -euo pipefail
PY=${PY:-.venv/bin/python}
$PY -m pytest -q tests/
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_gowu.py --config configs/gowu_mr_tiny.yaml
$PY scripts/summarize.py runs/gowu_mr_tiny
CUDA_VISIBLE_DEVICES=1 $PY scripts/eval_watch.py --run runs/backward_mr_tiny --poll 60 > runs/backward_mr_tiny.eval.out 2>&1 &
EVAL_PID=$!
trap 'kill $EVAL_PID 2>/dev/null || true' EXIT
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_backward.py --config configs/backward_mr_tiny.yaml
sleep 90
$PY scripts/summarize.py runs/backward_mr_tiny
