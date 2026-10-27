# Phase I validation protocol — 2026-10-20

The first pair stopped near 100M frames with rewards 14,100 / 14,000. The 2026-10-21
CPU diagnosis reproduces a room-8 doomed-jump trap and identifies a replay data mismatch:
the held-action branch stored post-action images, excluding restored starting states.
It now stores pre-action states, consistent with Sec. 5.1 and the ensemble branch.
`scripts/diagnose_phase1.py` recreates the physics probes and writes `runs/diagnosis/REPORT.md`.
A fresh 100M-frame pair (same seeds 0,1, same hyperparameters) tests that single correction;
it is a diagnostic, not the four-seed 400M reproduction. The launcher now accepts explicit
`--seeds` and `--frames-per-seed`, records them, and uses the same early-stop rules.

**2026-10-22 follow-up:** that diagnostic completed with rewards 13,600 / 43,400
(mean 28,500), compared with the original pair's 14,100 / 14,000. Seed 1 reached
level 2; seed 0 remained on level 1. The user authorized the full-budget check
on the corrected code, without changing hyperparameters or replacing seeds.
Queue job `gowu-validation:phase1-full-replay-20261022` uses a fresh output
directory, `runs/phase1_full_replay_20261022/`, preserving both earlier campaigns.
The broader RL design and milestone list are in [PLAN_LR_GOWU.md](PLAN_LR_GOWU.md).

**Full-check outcome:** `phase1_full_replay_20261022` stopped at its prescribed
100M gate after about 26 minutes. Seeds 0/1 scored 14,400/10,600 (mean 12,500,
below 20,000); final frame counts were 104.6M/102.0M. Neither completed the 400M
budget; seeds 2/3 were not started. This was an intentional scientific stop,
not a GPU failure or a successful reproduction. The central queue released
both cards and restored the model server. See the
[summary](../runs/phase1_full_replay_20261022/summary.json).

Validate exploration only. Four predetermined seeds (0, 1, 2, 3), each at 400M game frames,
two simultaneously, seven emulator workers and one RTX 3090 per seed. No phase II, policy
evaluator, replacement of poor seeds, or post-hoc best-seed selection. The held-action policy
is supported explicitly by Remark 5.2 and Fig. 6b of the supplied research PDF.

## Fidelity fixes and limits

Preprocessing now max-pools the last two **RGB** images, converts to grayscale using the
standard luminosity coefficients (OpenCV RGB2GRAY), then resizes bilinearly to 84x84,
following App. C.1. Early terminal frames never mix with stale pre-restore buffers.

The default exploit mode is `middle_room`, replacing the generic fourth-repeat death rule.
It rejects the first positive-reward event that consumes a key at the upper right door of
room 5, near the rope/skull platform. A normal door opening below the platform is allowed;
repeated treasure rewards are not generically capped during exploration. The old `loop`
mode remains available explicitly for historical experiments.

This is a **RAM surrogate**, not a reproduction of the authors' visual detector. Their room
crops and motion thresholds are not supplied. Geometry is grounded in the original
[OCAtari RAM extractor](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/montezumarevenge.py):
room 5, right rope at x=125, skull platform y=130, right door at (100,135), player screen
position `(RAM[42]-1, 308-RAM[43])`, and inventory key bits 1..4 in RAM[65]. App. H / Fig. 7
provides the bug mechanism. The detector requires a same-room/same-level positive-reward key
loss, both endpoint x coordinates in [95,130], pre-event screen y<135 and post-event y<140.
These margins are our explicit implementation choices, not published thresholds.

Semantic regression tests cover first-event rejection and normal opening/repetition cases.
They do not prove sensitivity on real exploit trajectories. Each seed saves the first 64
room-5 key-consumption events, **accepted and rejected**, into `exploit_audit/`: four-frame
PNG strips and compressed parent/endpoint checkpoints plus RAM/action metadata. Inspect
these before claiming detector equivalence or interpreting near-paper scores as definitive.

RND still uses one frame and eight updates per 1000 agent steps. The paper does not specify
the learner-to-data ratio or clearly identify RND's input channel count. Both choices stay
fixed across all four seeds and are recorded in the manifest; do not tune to the four results.

## Comparison and stopping rules

The target is the best discovered trajectory's cumulative reward (`best_R`), not policy
evaluation reward. Table 2 reports mean **98,752.5 at 400M frames across 100 seeds**.
Fig. 6 visually gives approximately 20–30k at 50M, 45–60k at 100M and 75–85k at 200M.
These are population means and the shaded bands are standard deviations, not confidence
intervals or individual-seed pass bounds. Report all four curves, endpoint mean, median and
sample standard deviation, plus room/level progress and exploit audits. Four seeds give
directional evidence; they cannot establish tight statistical equivalence with 100 seeds.

A pair averaging below 10k at 50M gets a warning. A pair averaging below **20k at 100M**
stops the campaign for diagnosis. That is a deliberately loose gross-failure screen, not a
claim that 20k matches the paper. Crashes, non-finite metrics, or 15 minutes without training
log progress also stop the campaign. The first failing pair prevents later pairs starting.
Training code/configuration are frozen into the run directory and hashed before seed 0.

## Queue and monitoring

Submit `scripts/validate_phase1.py` as a native `(gpu)` job through the-box hub. The hub
owns acquisition of both cards, graceful agent draining and model-server restoration.
`AGENT_JOB_GPU=1` is required and exactly two CUDA devices must be exposed.

An independent monitor process checks every 30 seconds and also reports at 50M, 100M,
200M and 400M. The original campaign used hourly reports (one third of the initial
3-hour-per-seed estimate). The 2026-10-22 job passes `--monitor-interval 1800` for
30-minute reports, roughly one third of the measured 1.5–2-hour seed runtime.
Crashes and stalls therefore do not wait for the periodic review. Reports go to `STATUS.md`,
`status.json`, `monitor.jsonl` and `monitor.out`. Final output adds `summary.json` and
`curves.png`. Actual speed is visible in every report; the estimate is not a GPU benchmark.
The orchestrator fails if its monitor exits unexpectedly and cleans up its own children.

Completed output is reused; partial output is never silently restarted because phase I
has no full training-resume implementation. A retry of a stopped campaign fails fast.
A designed early stop (the pair-mean gate) exits 0 with the report intact; a crash, stall,
or partial run exits non-zero so the queue flags it. Either way, a retry never re-arms
the GPUs: a stopped campaign is terminal and needs a new output directory.

Preparation/tests run before reserving GPUs. The 100M probe took about 26 minutes for a
pair; extrapolating gives about 3.5 hours for two pairs at 400M. Budget **4–6 hours**
for training, allowing for growing trees, plus queue wait, with a 12-hour job wall-clock
ceiling. A ceiling hit is an interrupted experiment, not a
successful reproduction. The queue automatically restores its prior agent state afterward.

The full-check queue command is:

```bash
cd experiment && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/validate_phase1.py --seeds 0 1 2 3 \
  --frames-per-seed 400000000 --monitor-interval 1800 \
  --output runs/phase1_full_replay_20261022
```
