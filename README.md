# gowu-montezuma

Reproduction of **GoWU** (Go-With-Uncertainty, Mhammedi & Cohan 2026) on Montezuma's Revenge, the
baseline for **LR-GoWU**, a no-reset, learned-return variant.

- `docs/PLAN_LR_GOWU.md` — design plan for the new algorithm (options, variables, baselines, compute)
- `docs/GOWU_REPRO.md` — fidelity table vs. the paper, deviations, throughput numbers

> ⚠️ **Evaluation cap:** our evaluator defaults to the standard 108k-frame ALE cap. **GoWU reports
> uncapped scores.** Use `--uncapped` for any comparison with the paper.

## Setup (GPU box)
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# CUDA torch if the default wheel is CPU-only:
# uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pytest -q tests/
```
ROMs come with `ale-py` ≥ 0.9.

## Run
```bash
bash scripts/smoke_gpu.sh                    # ~2-3 h directional check: tiny phase I -> tiny phase II
bash scripts/phase1_seeds.sh 0 9             # phase I, 10 seeds x 400M frames, 2 at a time
python scripts/run_backward.py --config configs/backward_mr.yaml          # phase II (edit demos:)
CUDA_VISIBLE_DEVICES=1 python scripts/eval_watch.py --run runs/backward_mr  # background evaluator
python scripts/eval_final.py --run runs/backward_mr [--uncapped]          # final 500-rollout eval
python scripts/summarize.py runs/gowu_mr_s* --plot phase1.png
```
Any config field can be overridden: `--set seed=3 particle_policy=ensemble curriculum.window=10`.

For the four-seed phase-I-only validation, see `docs/PHASE1_VALIDATION.md`. Its queued entry point is
`scripts/validate_phase1.py`; it includes an independent watchdog, early failure gates, frozen source
and periodic reports. The default middle-room exploit filter is a targeted RAM surrogate for the
paper's unreleased visual detector, not a claim of exact detector equivalence.

## What healthy looks like
- **Phase I** (`runs/*/log.jsonl`, stdout): `best_R` hits 100 (key) within ~1M frames and 400+
  (doors) soon after. `rooms` and `max_level` keep growing. `death_rate` stays around 0.3–0.5.
  `rollbacks` grows slowly. Paper reference: ~98.7k at 400M frames.
- **Phase II**: `progress` rises, `n_solved` counts up segment by segment, `eval_watch.jsonl` mean > 0
  once the first segment is solved from the start.

## Layout
```
gowu/atari.py        env: sticky actions, preprocessing, checkpoint/restore, MR RAM info
gowu/vec.py          process pool (shared-memory obs; worker-side inner-step rollouts)
gowu/tree.py         lineage tree with refcount pruning
gowu/nets.py         RND, zero-target particle ensemble, IMPALA actor-critic
gowu/explore.py      GoWU phase I (Algorithms 1 + 2)
gowu/exploit_guard.py  MR exploit guard (RAM-based stand-in for App. H)
gowu/backward/       phase II: curriculum, PPO trainer, evaluator
```
