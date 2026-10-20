# GoWU reproduction notes

This is a reimplementation of *Decoupling Exploration and Policy Optimization: Uncertainty Guided Tree
Search for Hard Exploration* (Mhammedi & Cohan, arXiv 2603.22273). It was written from the paper alone,
because no official code is public as of 2026-10.

> ⚠️ **EVAL CAP.** `gowu/backward/evaluate.py` defaults to the standard **108k-frame** cap. GoWU's
> numbers are **uncapped** (episodes end only on game over or loitering). Run
> `scripts/eval_final.py --uncapped` before comparing against 182,671.

## Targets (Montezuma's Revenge, sticky actions p = 0.25)

| phase | paper | budget in paper |
|---|---|---|
| I: cumulative reward of the best discovered trajectory | **98,752.5** (mean of 100 seeds) | 400M frames / seed |
| II: robustified policy score | **182,671.8** (mean of 10 runs; worst 89,172) | 15–20B frames / seed × 5 seeds × 10 runs |
| baselines | Go-Explore 43,791 · BYOL-Hindsight ≈14,517 · MEME 9,429 · RND 8,152 | |

## Fidelity: paper vs. this code

| component | paper | here | status |
|---|---|---|---|
| Preprocessing | max-pool last 2 of 4 frames, gray, 84×84 bilinear, stack 4 | same (ALE grayscale palette, then max-pool) | ✅ |
| Sticky actions / action set | 0.25 / full 18 | same | ✅ |
| Dead state | life loss (discount 0) | `ale.lives()` decrease, or game over | ✅ |
| Lineage tree | node = (checkpoint, R, parent); prune unreferenced | same, with refcounts; nodes also store rollout actions + RAM | ✅ (+) |
| Alg. 1 (outer/inner steps, halt on reward, rollback, winner = lex-max (R, U), prune dead → winner, group consolidation) | | implemented line by line in `GoWU.outer_step` / `consolidate` | ✅ |
| Alg. 2 (M groups, global sync if group best R < global winner R, global-winner update) | | `GoWU.iterate` | ✅ |
| Hyperparameters (Table 1) | M = 4, N = 32, K ∈ [10, 20], T ∈ [5, 15], rollback ∈ [1, 3] | same (`configs/gowu_mr.yaml`) | ✅ |
| RND | Nature torso + 1024-ReLU-1024, orthogonal init, input normalisation clipped to [−5, 5], Adam 3e-4, batch 128, trained on all groups' buffers | same; input = last frame | ✅ (input channel count not stated) |
| Learner/actor ratio | asynchronous, not given | `rnd_updates_per_1k_steps = 8`. Training overlaps with the rollouts | ⚠️ free parameter |
| Uncertainty freshness | live estimator | every winner comparison re-scores U with the current RND | ✅ |
| Particle policy | bootstrapped zero-target ensembles, ε = 0.2, Muon 1e-4 | `ensemble` implemented (Adam, because Muon needs 2-D weights). **Default `hold_random`** (one random action held per inner-step block), which the paper's Fig. 6b shows is "on par" | ⚠️ default = ablation variant |
| Replay | 128k / group, bootstrap masks | same | ✅ |
| Middle-room bug guard | visual detector (unreleased reference crops) | RAM-based: `loop` guard (same (level, room, reward) > 3 times in a lineage → dead) + explicit `exploit_rules`. All reward events are logged to `reward_events.jsonl` to find the exact signature | ⚠️ approximation |
| Demo extraction | highest-reward leaf → root path | best-R node (earliest), checkpoints every 5–15 steps | ✅ |
| Phase II segmentation | ≤ 10 segments at reward events, curriculum may go past the segment start | same | ✅ |
| Curriculum | Δ = 25, dec (0.25, 0.75), inc (0.5, 1.0), S_req 0.2, begin 0.95, N_update 32, ε_tol 0, μ = 2, b = 500, EMA 0.9, extra ≈ e^U(0,7) steps | same. Δ is in **demo-node** units (unit not given) | ⚠️ unit assumption |
| Segment sampling | "train on all segments simultaneously" | uniform over unsolved, 10% rehearsal of solved | ⚠️ unspecified |
| PPO | Table 6 (lr 1e-4, γ 0.999, λ 0.95, unroll 128, batch 128, 1 epoch, 8 minibatches, ent 1e-3, vf 0.5, clip 0.1, grad 0.5) | same; synchronous, bf16 autocast | ✅ |
| Network | IMPALA ResNet, separate π/V encoders, dense 1024, heads 512 | same | ✅ |
| 5th channel | frame at the last reward | same (stored in every checkpoint) | ✅ |
| Training episode end | life loss | same | ✅ |
| Evaluator | 10 rollouts per eval, top-100 checkpoints, final 500 rollouts, loitering 10k, treasure bug ≥ 30 × 1000 | `eval_watch.py`, `eval_final.py` | ✅ |

## Throughput (measured, Apple M-series laptop core)

- Raw ALE on MR: ~20k frames/s/core. With preprocessing and sticky actions: **~4.2k agent steps/s/core**.
- Phase I, env only, 6 workers: ~31k frames/s. Coordinator overhead is small (`cProfile`). On CPU the
  hot spot is RND scoring, which is negligible on a GPU.

Expected on the 16-core / 2× 3090 box (estimates):
- Phase I: 14 workers → ~50–100k frames/s → **400M frames in ~1.5–3 h per seed**. Two seeds at
  once (7 workers each, one GPU each) gives about the same total throughput.
- Phase II: probably ~20–50k frames/s per GPU (env-bound with 14 workers) → 5B frames ≈ 1–3 days.

## Known open questions

1. Whether GoWU blocks the **treasure-room** loop during exploration. Our `loop` guard does (> 3 repeats).
2. The middle-room bug's exact RAM signature: check `reward_events.jsonl` of a long run for repeated
   `(level, room, 300.0)` events with `exploit: true`, then add them to `exploit_rules` if wanted.
3. The paper does not say whether the backward window Δ counts env steps or demo nodes.
