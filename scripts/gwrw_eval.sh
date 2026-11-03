#!/usr/bin/env bash
# Frozen test of the two GWRW runs from scripts/gwrw_pair.sh, one per GPU: weights fixed, no saves, every game targets the
# deepest reproducible route and the explorers play on after arrival; GAMES games each, equal per env, fresh env seeds.
# Reads the final checkpoint through hard-linked copies (prefix gwrw_<r>_eval), so the training files are never touched.
# usage: scripts/gwrw_eval.sh OUT GAMES      -> OUT/gwrw_<r>_eval_eval.json
set -u
OUT=$1; GAMES=$2
G=third_party/montezuma_gwrw
ln -sfn "$(realpath --relative-to="$G/gwrw" "$OUT")" "$G/gwrw/out"
PIDS=()
for i in 0 1; do
  r=$([ $i = 0 ] && echo a || echo b)
  for f in .pt _demos.pkl _bdemos.pkl _front.pkl _lmref.pkl; do ln -f "$OUT/gwrw_$r$f" "$OUT/gwrw_${r}_eval$f"; done
  cp "$OUT/gwrw_${r}_live.json" "$OUT/gwrw_${r}_eval_live.json"
  CUDA_VISIBLE_DEVICES=$i RUN_NAME=gwrw_${r}_eval DEVICE=cuda:0 ENVS_PER_PROC=8 N_PROC=12 EXPLORER_ARCH=trav MAX_FRAMES=1000000000000 \
    EVAL_GAMES=$GAMES OMP_NUM_THREADS=2 .venv/bin/python -u "$G/gwrw/train_gwrw_comp_cd.py" > "$OUT/gwrw_${r}_eval_stdout.log" 2>&1 &
  PIDS+=($!)
done
EA=0; EB=0; wait "${PIDS[0]}" || EA=$?; wait "${PIDS[1]}" || EB=$?
grep -h "\[eval\] done" "$OUT"/gwrw_?_eval_stdout.log
exit $(( EA || EB ))
