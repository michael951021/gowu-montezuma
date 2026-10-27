# Phase I implementation vs. the paper — checkpoint 2026-10-26

Paper: Mhammedi & Cohan 2026, *Decoupling Exploration and Policy Optimization: Uncertainty Guided Tree Search
for Hard Exploration* (`GOWU_Research_paper.pdf`). This maps every Phase I element of the paper to the code
as of this checkpoint. "Same" means it was checked line by line. Deviations are listed separately at the end
and in `AUDIT_2026-10-26.md`.

Status at this checkpoint: 8 seed-runs at 50M frames score 10.6k–14.9k (one outlier 34.8k); the paper's Fig. 6
mean at 50M is ~20–30k. The code matches Algorithms 1–2. The remaining gap is attributed to the deviations below.

## Algorithm 1 (EvolveGroup) — `gowu/explore.py`

| Paper | Code | |
|---|---|---|
| l.1 K ~ Unif(K_min, K_max), per iteration | `iterate`, per group: `K = [rng.integers(10, 20+1)]` | same |
| l.4 T_i ~ Unif(T_min, T_max), per particle, per outer step | `_items` / `_rollout_ensemble`: `rng.integers(5, 15+1)` per particle | same |
| l.6–8 step; break if dead or R increased | `vec.py` rollout loop: break on life loss / game over, break on `r > 0` | same |
| l.10–11 AddChild for alive particles | `outer_step`: `tree.add_child(parent, ckpt, R, ...)` for non-dead results | same |
| l.14–17 all dead → each particle resets to its k_i-th ancestor, k_i ~ Unif(k_min, k_max) | `outer_step`: per particle `ancestor(nodes[p], k)`, k ∈ [1, 3] | same (indexing: see D3) |
| l.20 winner = lex-argmax over alive (R, U) | `lexmax([(R, U_new[p], p) for alive])`, U scored by the current RND | same |
| l.21–23 dead → reset to winner; alive non-winners continue | only `p not in alive_new` are moved to the winner | same |
| l.29–30 consolidation: whole group → lex-argmax (R, U) | `consolidate(m)` after the group's K-th outer step | same |

## Algorithm 2 (GowU) — `GoWU.iterate`

| Paper | Code | |
|---|---|---|
| l.1–4 tree with root s0; M groups × N particles at s0; shared U | `__init__`: one root, `P = M·N` particles acquire it, one RND | same |
| l.8–10 group best (lex-max) has R < global winner R → whole group reset to global winner | global sync loop at the top of `iterate`, strict `<` | same |
| l.13 EvolveGroup for each group | groups run in lockstep; a group that finished its K idles until the slowest group finishes | same result (groups are independent within an iteration) |
| l.16 global winner = lex-argmax over {group winners, old global winner} | `cands = winners + [gw]`, re-scored with the current U | same |

## Lineage tree (Sec. 4.1, 5.1) — `gowu/tree.py`

Node = (checkpoint, cumulative R, parent pointer). Nodes are freed once no particle occupies them and no
descendant references them (ref counts). Demo = path from root to the highest-reward node. Same. Extra fields
(actions, RAM, obs) are bookkeeping only and never influence decisions.

## Uncertainty: RND (Sec. 5.2) — `gowu/nets.py`, `explore.py::_train`

| Paper | Code | |
|---|---|---|
| target + predictor: Nature-CNN torso + 1024–ReLU–1024 | `RNDNet`: torso → Linear(3136, 1024) → ReLU → Linear(1024, 1024) | same |
| orthogonal init | `ortho_` (gain √2) | same |
| inputs normalised by running mean/std, clipped [−5, 5] | `RunningNorm`, per pixel | same |
| MSE to the target embedding | `error()` = mean squared difference | same |
| trained on all M groups' replay buffers | `sample_across(self.replays, ...)` | same |
| Adam 3e-4, batch 128 | config `rnd_lr`, `rnd_batch` | same |
| asynchronous learner, rate not given | 8 updates / 1k agent steps, overlapped with rollouts | **D2** |

## Replay (Sec. 5.1) — `gowu/replay.py`

128k per group, FIFO, entries (s, a, m) with the bootstrap mask drawn at collection time; s is the state where
the action was chosen (corrected 2026-10-21). Same.

## Particle policy (Sec. 5.3, App. E.1, Remark 5.2)

| Paper | Code | |
|---|---|---|
| headline: per-particle zero-target network f(s, a), Nature torso + 512, masked MSE to 0, argmax with ε = 0.2, ties random; member = particle index mod N; per-group data only; Muon 1e-4, batch 128 | `particle_policy: ensemble`: `EnsembleQ` (grouped convs), same loss, ε = 0.2, ties random, member p mod N, per-group replay, **Adam** 1e-4, batch 128 | **D1** (not the default so far) |
| ablation (Fig. 6b, "on par"): one random action per particle held for the inner-step block | `particle_policy: hold_random` (default in all runs so far) | **D1** |

## Environment (Sec. 6.1, App. C.1) — `gowu/atari.py`

Full 18 actions; sticky actions p = 0.25 per emulator frame; action repeat 4; max-pool over the last 2 RGB frames
→ luminosity grayscale → bilinear 84×84; stack 4; no episode frame cap during exploration; restore = ALE state +
frame stack; dead = life loss. Same. Checked 2026-10-26: restore reproduces the frame stack exactly,
deterministic replay with sticky off, clones diverge with sticky on, life loss registers on the landing step.

## Exploit guard (App. H) — `gowu/exploit_guard.py`

The paper kills a particle whose reward comes from the middle-room rope→skull-platform key bug, detected
visually with unreleased reference crops. Ours is a RAM surrogate of the same event (room 5, key consumed,
upper-right door geometry). **D5**.

## Hyperparameters (Table 1, Montezuma)

M = 4, N = 32, K ∈ [10, 20], T ∈ [5, 15], rollback ∈ [1, 3], sticky 0.25, replay 128k, 400M frames. Same.

## Deviations (candidates, in test order)

- **D1** particle policy: all runs used the ablation `hold_random`; the headline uses the ensemble (Adam, not Muon).
- **D2** RND update rate: unspecified in the paper; ours is a fixed 8 updates / 1k steps.
- **D3** rollback indexing: our k = 1 is the parent of the dead particle's rollout start node; the paper's
  "k generations prior" may mean k = 1 = that start node.
- **D4** RND normaliser warm-up: 8 copies of the start frame; standard RND initialises from random-agent steps.
- **D5** exploit guard is a RAM surrogate.
