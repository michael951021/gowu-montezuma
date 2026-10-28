#!/usr/bin/env bash
# Two phase-I seeds at once (one per GPU, 7 env workers each) with extra --set overrides.
#   bash scripts/phase1_pair.sh OUT SEED_A SEED_B "key=value ..."   -> OUT/seed{A,B}
set -uo pipefail
PY=${PY:-.venv/bin/python}
OUT=${1:?out}; A=${2:?}; B=${3:?}; SETS=${4:-}; WORKERS=${WORKERS:-7}
mkdir -p "$OUT"
run() {
  CUDA_VISIBLE_DEVICES=$2 $PY scripts/run_gowu.py --config configs/gowu_mr.yaml \
    --set seed=$1 n_workers=$WORKERS out_dir="$OUT/seed$1" $SETS > "$OUT/seed$1.out" 2>&1 || echo "FAILED seed$1" >> "$OUT/failures"
}
run "$A" 0 & run "$B" 1 & wait
[ ! -f "$OUT/failures" ]
