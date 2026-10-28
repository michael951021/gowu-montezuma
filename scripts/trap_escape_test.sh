#!/usr/bin/env bash
# Room-8 trap escape test: start the whole population just outside room 8 (room 9) from saved demos of
# trapped runs, then run phase I with the confinement-trap fix off vs on. Two runs at a time (one per GPU).
#   bash scripts/trap_escape_test.sh OUT [FRAMES] [WINDOW]
set -uo pipefail
PY=${PY:-.venv/bin/python}
OUT=${1:?out dir}; FRAMES=${2:-20000000}; WIN=${3:-100}; WORKERS=${WORKERS:-7}
FIX=(
  "l2s6 runs/big_v4_400m_20261027/seed6/demo_best.pkl.gz 30556"
  "l2s7 runs/big_v4_400m_20261027/seed7/demo_best.pkl.gz 16404"
  "l1s3 runs/phase1_seeds23_50m_20261026/seed3/demo_best.pkl.gz 5631"
  "l1v1 runs/phase1_validation/seed1/demo_best.pkl.gz 6030"
)
mkdir -p "$OUT"
run() {  # name demo node window gpu [tag extra-sets...]
  local d="$OUT/$1-w$4${6:-}"
  CUDA_VISIBLE_DEVICES=$5 $PY scripts/run_gowu.py --config configs/gowu_mr.yaml --set seed=0 n_workers=$WORKERS \
    total_frames=$FRAMES out_dir="$d" start_demo="$2" start_node=$3 trap_window=$4 track_room_landmarks=false ${@:7} \
    > "$d.out" 2>&1 || echo "FAILED $d" >> "$OUT/failures"
}
for f in "${FIX[@]}"; do
  set -- $f
  run $1 $2 $3 0 0 &
  run $1 $2 $3 $WIN 1 &
  wait
done
# Does a longer rollout range (paper: T between 3 and 20 across games; MR uses [5, 15]) get out without the fix?
set -- ${FIX[0]}; run $1 $2 $3 0 0 -T3-20 "inner_steps=[3,20]" &
set -- ${FIX[2]}; run $1 $2 $3 0 1 -T3-20 "inner_steps=[3,20]" &
wait
$PY - "$OUT" <<'EOF'
import json, sys, glob, os
rows = []
for d in sorted(glob.glob(sys.argv[1] + "/*-w*/")):
    recs = [json.loads(l) for l in open(d + "log.jsonl")]
    first, last = recs[0], recs[-1]
    rooms = sorted({tuple(r["gw_ram"][:2]) for r in recs})
    tr = [json.loads(l) for l in open(d + "traps.jsonl")] if os.path.exists(d + "traps.jsonl") else []
    rows.append(dict(run=os.path.basename(d.rstrip("/")), frames=last["frames"], best_R=last["best_R"],
                     gw_R_end=last["global_winner_R"], gw_room_end=last["gw_ram"][:2], gw_rooms=rooms,
                     max_gw_R=max(r["global_winner_R"] for r in recs), rooms_seen=last["rooms"], traps=len(tr),
                     trap_Rs=sorted({t["R"] for t in tr})))
    print(json.dumps(rows[-1]))
json.dump(rows, open(sys.argv[1] + "/summary.json", "w"), indent=1)
EOF
[ ! -f "$OUT/failures" ]
