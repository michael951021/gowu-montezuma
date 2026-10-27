#!/usr/bin/env bash
# Phase I throughput per particle-policy setting on one GPU (diagnostic; ~90 s each).
# Usage: bash scripts/bench_throughput.sh OUT_DIR
set -u
out=${1:?out dir}; mkdir -p "$out"
run() {  # name, overrides...
  name=$1; shift
  CUDA_VISIBLE_DEVICES=0 timeout 100 .venv/bin/python scripts/run_gowu.py --config configs/gowu_mr.yaml \
    --set seed=9 n_workers=7 total_frames=1000000000 out_dir="$out/$name" save_every_sec=9999 "$@" > "$out/$name.out" 2>&1
  python3 -c "import json,sys;L=[json.loads(l) for l in open('$out/$name/log.jsonl')];r=L[-1];print('$name', r['frames'], r['fps'], 'ens', r.get('ens_updates_total'))"
}
run hold_random
run ens_noupdate particle_policy=ensemble ens_updates_per_1k_steps=0
run ens_r0.25 particle_policy=ensemble ens_updates_per_1k_steps=0.25
run ens_r0.5 particle_policy=ensemble ens_updates_per_1k_steps=0.5
run ens_r1 particle_policy=ensemble ens_updates_per_1k_steps=1
