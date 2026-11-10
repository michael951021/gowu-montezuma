#!/usr/bin/env bash
# Start (or resume) the Montezuma GWRW run. With an empty gwrw/out/ this is a FRESH START:
# no run14 checkpoint, no pool seed, no saved states. Re-running the same command resumes from gwrw/out/$RUN_NAME.pt.
# Stop it gracefully with:  kill -INT $(cat gwrw/out/$RUN_NAME.pid)   (checkpoint is saved at the end of the update; Ctrl-C in the foreground also works)
set -e
cd "$(dirname "$0")"
export RUN_NAME=${RUN_NAME:-gwrw_cd_v1}
export DEVICE=${DEVICE:-cuda:0}
export ENVS_PER_PROC=${ENVS_PER_PROC:-8}          # x N_PROC (default 12) = 96 parallel games
export EXPLORER_ARCH=${EXPLORER_ARCH:-trav}
export MAX_FRAMES=${MAX_FRAMES:-4800000000}       # TOTAL frame cap, including frames of a resumed run
PY=${PYTHON:-python3}
mkdir -p gwrw/out
[ -f "gwrw/out/${RUN_NAME}_live.json" ] || cp configs/gwrw_cd_v1_live.json "gwrw/out/${RUN_NAME}_live.json"
echo "run $RUN_NAME on $DEVICE, $((${N_PROC:-12} * ENVS_PER_PROC)) games; log: gwrw/out/${RUN_NAME}_stdout.log"
$PY -u gwrw/train_gwrw_comp_cd.py >> "gwrw/out/${RUN_NAME}_stdout.log" 2>&1 &
echo $! > "gwrw/out/${RUN_NAME}.pid"
wait $!
