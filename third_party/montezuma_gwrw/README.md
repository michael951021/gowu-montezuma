# Montezuma's Revenge: GWRW (go-with-the-winner route walker), code only

This is the code of the `gwrw_cd_v1` Montezuma's Revenge run, with no checkpoints and no run data. Starting it in an empty
`gwrw/out/` is a fresh start from scratch.

## What it does

Everything the algorithm uses comes from the screen and the game reward. Emulator RAM (room, x/y, lives) is read only into
`diag_*` log fields and never reaches a policy, reward, archive or target.

* **Landmarks**: the n-th positive game reward of a game (`r n`), and the first entry into a screen background (`s id`;
  pixel scene-change detector, codebook grows online; `gwrw/landmarks.py`). Frontier entries (`f uid`) are stored novelty
  windows that serve as extra targets.
* **Archive**: every distinct ordered list of landmarks some game has produced is a node; depth = list length.
* **Route controller `mu`**: a goal-conditioned policy that follows a target's landmark list. It imitates (behaviour cloning,
  loop-erased) the fastest self-generated demos of each leg, kept per node.
* **Targets** are drawn per game from the archive (deepest / hard / spread / frontier wings; see `pick_target`).
* **Takeover explorer**: on arrival, wrong landmark or leg timeout, the explorer plays on to game over. It is the traversal
  net (`EXPLORER_ARCH=trav`) with three agents `a`/`b`/`c` (per-agent actor and critic heads; RND novelty + game reward + progress
  streams; a count-deficit competition between the agents; with the shipped json `a` is a non-learning uniform-random walker and `b`/`c`
  learn by PPO). Anything new it reaches becomes a node and so a future target.

**A longer explanation (game life cycle, targets, route controller, explorers, frontier entries, hyper-parameter table, output
formats, what is not known) is in `EXPLAINER.md`.**

## Contents

```
EXPLAINER.md                  how the algorithm and the code work (read this first)
run_fresh.sh                  launcher (fresh start or resume)
requirements.txt
configs/gwrw_cd_v1_live.json  the run's live hyper-parameters (overlaid on LIVE_DEFAULT in the trainer)
gwrw/train_gwrw_comp_cd.py    the trainer (the run's trainer, plus the fresh-start patches below)
gwrw/train_gwrw.py            shared pieces: vector env, scene tracker (older GWRW v2 trainer)
gwrw/landmarks.py  gwrw/xworker.py  gwrw/train_route.py   landmark detector, env worker, env construction
algorithms/rnd.py  models/*.py  env/*.py                  RND, networks, Atari wrappers, RAM diagnostics
CHANGES_FROM_RESEARCH_CODE.diff
```

## Setup

Python 3.12, one CUDA GPU (the run used a 12 GB card), about 12 CPU cores (12 worker processes by default).

```
pip install torch --index-url https://download.pytorch.org/whl/cu121     # match your CUDA; >= 2.1
pip install -r requirements.txt                                          # ale-py bundles the Atari ROMs
```

## Fresh start

```
./run_fresh.sh                       # RUN_NAME=gwrw_cd_v1, 12 procs x 8 envs = 96 games, EXPLORER_ARCH=trav
tail -f gwrw/out/gwrw_cd_v1_stdout.log
```

Environment overrides: `RUN_NAME`, `DEVICE` (`cuda:0`), `ENVS_PER_PROC` (8), `N_PROC` (12), `MAX_FRAMES` (4.8e9, a TOTAL cap
including resumed frames), `SAVE_EVERY` (50 updates), `PYTHON` (interpreter for the launcher), `GOAL_GAIN` (0.01),
`LEG_TIMEOUT` (1500), `BC_WARMUP` (400 updates), `ROLLOUT` (128). `ALE_ROMS_DIR` is only needed for a ROM directory outside ale-py.
`DEBUG_FAKE_R=0.002` injects fake rewards and is for smoke tests only.

**Stop** with SIGINT to the python process: `kill -INT $(cat gwrw/out/gwrw_cd_v1.pid)` (Ctrl-C in the foreground also works).
The trainer finishes the update, saves `gwrw/out/<RUN>.pt` and exits. SIGTERM / kill -9 lose everything since the last save.
**Resume** by running `./run_fresh.sh` again with the same `RUN_NAME` (raise `MAX_FRAMES` first if the cap was hit).

**Live tuning**: `gwrw/out/<RUN>_live.json` is re-read every update; edit it while the run is going, no restart needed. The effective
hyper-parameters are `LIVE_DEFAULT` (in the trainer) overlaid by that file. `run_fresh.sh` copies `configs/gwrw_cd_v1_live.json`
there on the first launch, so the defaults used are those of the research run. Note that this file holds the values the research run had at
update 90,900 (e.g. `entropy_coef` 1.0, `p_spread` 1.0, `bc_hard` 4.0), tuned for a mature run, not for its first updates; retune it for a fresh start if needed.

## What "fresh start" means here

The research run started from two inputs that are NOT in this package:

1. the RND predictor/target, PopArt and advantage-scale state of the three explorer agents, from an earlier run's checkpoint (`CKPT14`);
2. the initial archive, scene codebook and demos, seeded from 800 episodes of that earlier agent (`pool800_*` files, `SEED_FROM_POOL=1`).

Here both are optional. With neither present (the default), the three agents get freshly initialised RND models and PopArt / advantage
scales, and the archive, scene codebook and demo set start empty. The first games have no target (the explorer plays from the start)
and the archive grows from what they reach. If you do have those two inputs, set `CKPT14` and put the `pool800_*` files into
`gwrw/out/`; the original seeded start is still there.

Checked: imports and syntax, a fresh start from an empty directory, checkpoint save, SIGINT/limit stop, resume from a fresh-start
checkpoint, and a run with fake rewards that exercised node creation, demos, frontier admission and BC. Not done: a full-length fresh run.
How fast a fresh start discovers the first rewards and rooms is not known; the research run reached about 3.9e9 frames (90,900 updates) from its
seeded start.

Resources grow over a long run: all demos are kept in RAM and pickled at every save (the research run's `_demos.pkl` reached about 25 GB and the
trainer's RSS about 55 GB at update 90,900), and the episode recorders (`<RUN>_deep/`, `<RUN>_deep_rare/`) write frames to disk up to fixed caps.
Plan for 64 GB+ RAM and tens of GB of disk for a multi-billion-frame run.

## Outputs (`gwrw/out/`, all prefixed with `<RUN>`)

`.pt` model/optimiser/agent state and archive; `_demos.pkl`, `_bdemos.pkl`, `_front.pkl`, `_lmref.pkl` demo and frontier stores;
`_log.jsonl` one row per update (frames, nodes, max_depth, scenes, losses, per-agent stats); `_episodes.jsonl` one row per finished game;
`_events.jsonl`, `_front.jsonl`, `_lmroom.json`; `_live.json`; `_stdout.log`; `_deep/`, `_deep_rare/` recorder episodes.

## Changes from the research code

Only the trainer start-up and the ROM path were touched (`CHANGES_FROM_RESEARCH_CODE.diff`). The algorithm is unchanged.

* `CKPT14` optional (fresh RND / PopArt / advantage scales when absent); `Ag.__init__` handles `ck=None`.
* Pool seeding is skipped when `pool800_files.json` is absent; the traversal net is initialised from `mu` in both cases (previously only in the pool branch, which failed with `EXPLORER_ARCH=trav`: `strict=False` with the actor/critic heads exempt).
* `EXPLORER_ARCH` defaults to `trav` (the research run's setting; `run14`/`mu` need the earlier checkpoint). `gwrw/out/` is created if missing.
* `train_route.py`: the hard-coded ROM directory is replaced by the optional `ALE_ROMS_DIR`.
