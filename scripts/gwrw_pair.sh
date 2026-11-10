#!/usr/bin/env bash
# Two fresh GWRW runs (third_party/montezuma_gwrw, the gwrw_cd_v1 code package), one per GPU, each to
# FRAMES emulator frames. Re-running the same command resumes both from their last checkpoint.
# usage: scripts/gwrw_pair.sh OUT FRAMES [PATCH_A] [PATCH_B]
# PATCH_A / PATCH_B: optional JSON objects merged into run a's / run b's copy of the shipped live json on first
# start (e.g. '{"entropy_coef": 0.03}'), so the two GPUs can run two different variants.
set -u
OUT=$1; FRAMES=$2; PATCH_A=${3:-}; PATCH_B=${4:-}
G=third_party/montezuma_gwrw
mkdir -p "$OUT"
ln -sfn "$(realpath --relative-to="$G/gwrw" "$OUT")" "$G/gwrw/out"   # the trainer writes to gwrw/out
PIDS=()
for i in 0 1; do
  r=$([ $i = 0 ] && echo a || echo b)
  if [ ! -f "$OUT/gwrw_${r}_live.json" ]; then
    cp "$G/configs/gwrw_cd_v1_live.json" "$OUT/gwrw_${r}_live.json"
    PATCH=$([ $i = 0 ] && echo "$PATCH_A" || echo "$PATCH_B")
    [ -n "$PATCH" ] && .venv/bin/python -I -c 'import json,sys; f=sys.argv[1]; d=json.load(open(f)); d.update(json.loads(sys.argv[2])); json.dump(d,open(f,"w"),indent=1)' "$OUT/gwrw_${r}_live.json" "$PATCH"
  fi
  CUDA_VISIBLE_DEVICES=$i RUN_NAME=gwrw_$r DEVICE=cuda:0 ENVS_PER_PROC=8 N_PROC=12 EXPLORER_ARCH=trav MAX_FRAMES=$FRAMES \
    OMP_NUM_THREADS=2 .venv/bin/python -u "$G/gwrw/train_gwrw_comp_cd.py" >> "$OUT/gwrw_${r}_stdout.log" 2>&1 &
  PIDS+=($!); echo $! > "$OUT/gwrw_$r.pid"
done
trap 'kill -INT "${PIDS[@]}" 2>/dev/null' INT TERM   # SIGINT = save a checkpoint and exit
while kill -0 "${PIDS[0]}" 2>/dev/null || kill -0 "${PIDS[1]}" 2>/dev/null; do   # a progress line every 10 min keeps the job log alive
  sleep 600 & wait $!
  for r in a b; do echo "$(date '+%F %T') $r $(grep -o 'update=[0-9]* frames=[0-9]* fps=[0-9]* nodes=[0-9]* max_depth=[0-9]* scenes=[0-9]*' "$OUT/gwrw_${r}_stdout.log" | tail -1)"; done
done
EA=0; EB=0; wait "${PIDS[0]}" || EA=$?; wait "${PIDS[1]}" || EB=$?
exit $(( EA || EB ))
