# LR-GoWU: Learned-Return Go-With-Uncertainty (design plan)

Working name for the new algorithm. GoWU (Mhammedi & Cohan, 2026) explores by **restoring** emulator
snapshots of "winner" states. LR-GoWU keeps GoWU's winner logic (highest reward first, then highest
RND uncertainty), but it is not allowed to restore. Instead it trains a goal-conditioned **return policy**
π_R that walks from the true initial state s0 to a chosen landmark. It then hands control to an
**explore policy** π_E. This is a "learned replay": π_R plays the role of `restoreState`.

## Current execution status — 2026-10-22

**Full-budget check stopped early; reproduction remains unvalidated.** The queued
`phase1-full-replay-20261022` campaign ran for about 26 minutes, then its prescribed
100M-frame gate stopped both seeds: seed 0 scored 14,400, seed 1 scored 10,600
(pair mean 12,500, below the 20,000 floor). Final frame counts were 104.6M/102.0M;
neither seed reached 400M, and seeds 2/3 never started. The queue marked the job
done because an intentional early stop successfully produces its report; this
does not mean the scientific validation passed. GPUs were released and the
model server restored. See the [full-check summary](../runs/phase1_full_replay_20261022/summary.json).
The earlier diagnostic's 43,400 score did not recur in this fresh run; the
algorithm's plateau and run-to-run reproducibility need further diagnosis.

The active milestone is **M0 / B0a: validate GoWU phase I**, before implementing or
evaluating LR-GoWU. The controlled replay-fix diagnostic finished at 100M frames:
seeds 0/1 scored 13,600/43,400, mean 28,500; only seed 1 reached level 2. This is
partial improvement, not a successful reproduction. See the
[diagnostic comparison](../runs/diagnosis/COMPARISON.md).

The user authorized a fresh full-budget check: **seeds 0–3 × 400M game frames**,
two at a time on the two RTX 3090s, using the corrected pre-action replay and
unchanged hyperparameters. Central queue project/job:
`gowu-validation:phase1-full-replay-20261022`. Output:
`runs/phase1_full_replay_20261022/`. Health checks run every 30 seconds, with
reports every 30 minutes and at frame milestones. Existing scientific and
technical stop rules remain in force. Estimated training time is 4–6 hours
after GPU acquisition, with a 12-hour ceiling. See the
[phase-I validation protocol](PHASE1_VALIDATION.md) for gates and limitations.
Later milestones remain planned; this job runs phase I only.

> ⚠️ **EPISODE CAP NOTE — READ ME.** All LR-GoWU evaluation in this plan uses the standard ALE protocol:
> sticky actions p = 0.25, the full 18-action set, all lives, and a **108,000-frame (27,000 agent step) cap**.
> **GoWU's reported numbers do NOT use this cap.** Their evaluator lets episodes run past the normal
> length and only cuts off "loitering" (10k steps with no reward). We must re-run our evaluation
> **uncapped** (`eval.max_episode_frames: null`) before comparing against GoWU's 182k. Every eval config
> has this switch, and every results table must say which cap it used.

---

## 0. Decisions made so far (from Q&A)

| # | Decision |
|---|---|
| Resets | **No `restoreState` during training.** Every training episode starts from `env.reset()`. Storing trajectories (observations, actions, rewards) is allowed. |
| RAM | RAM reads are allowed **for now**. Every component that uses RAM is marked **[RAM]** so it can be swapped out later for a "no domain knowledge" claim. |
| Eval | Sticky 0.25, 18 actions, all lives, 108k-frame cap, **plus an uncapped re-run** (see the note above). |
| Landmarks | **One landmark per room** (key = `(level, room)` [RAM]). Within a room the landmark is the state with the highest (R, U), compared lexicographically as in GoWU: highest reward wins, RND uncertainty breaks ties. Landmarks are **merged/replaced, never removed**. Optional sub-landmarks. |
| Reward in landmarks | Yes. R (cumulative extrinsic reward on the way to the state) comes first, then uncertainty. |
| Goal embedding | A conv (+MLP) embedding of the goal frame stack. Several options below. |
| Return task | Always **s0 → landmark g**. We never train S_i → S_{i+1} with S_i as a start state, because we cannot start there. |
| Landmark choice | Default mirrors GoWU's group/winner/consolidation structure. All alternatives are implemented as switches (§5). |
| BC + PPO | **Alternating** is the default; joint-loss and pretrain-then-finetune are switches (§4.4). |
| π_R reward | Goal reward **plus extrinsic** game reward. |
| π_E | All three are implemented: held random actions (GoWU), RND-PPO, and π_R with exploration goals. |
| Handoff | When the target landmark is reached, **or** when return fails (timeout or stall) → π_E. |
| Death | Default: life loss ends the episode (GoWU-like). Baseline B2 runs with **infinite lives**. |
| Deployable policy | **π_R** (the smart policy). |
| Hardware | 2× RTX 3090 (24 GB), 16-core CPU, 128 GB DDR5. |

---

## 1. How LR-GoWU relates to prior work (novelty boundary)

| | GoWU | Policy-based Go-Explore (Ecoffet+ 2021) | DTSIL (Guo+ 2019) | **LR-GoWU** |
|---|---|---|---|---|
| Return mechanism | restoreState | goal-conditioned PPO + SIL | trajectory-conditioned policy + SIL | goal-conditioned PPO + BC |
| What a state/goal is | none (tree of snapshots) | domain-knowledge cell (level, room, x, y, keys) | domain-knowledge embedding clusters | room [RAM] + learned conv embedding of a landmark frame |
| Return target | n/a | **next cell along the archived trajectory** (soft-order chain) | chunks of the trajectory | **the final goal only** (s0 → g), with optional ancestor context |
| Selection | max R, then RND uncertainty; population cloning | count-based cell weights | count-based | **max R, then RND uncertainty** + virtual population (§5) |
| Recurrence | none (snapshots) | GRU | LSTM/attention | **none** (frame stack + context channels) |
| Score on MR | 98.7k explore / 182k policy | 97,728 at 12B frames (sticky) | lower, no sticky | **?** |

LR-GoWU's claims must be phrased against policy-based Go-Explore. The differences are: (1) GoWU's
uncertainty-driven winner selection instead of count weights; (2) a direct s0→g goal with no subgoal chain;
(3) a feedforward network; (4) hindsight-relabelled goal-conditioned BC instead of SIL; (5) a separate explore policy;
(6) eventually, no RAM.
**Risk:** (2) is *exactly* what Ecoffet et al. reported "did not perform well in preliminary experiments".
That is why §4.2 provides ancestor context and shaping as switches. They keep the goal fixed at g, so they
do not violate the s0→g rule, but they give the policy a sense of progress.

---

## 2. Objects

### 2.1 Episode structure (the core loop)
```
env.reset()                                   # s0, always (no restore)
g  ← SelectLandmark(archive, env_slot)        # §5
# ---- RETURN phase: π_R(a | s, g, ctx) ----
while not reached(s, g) and not return_failed:
    a ~ π_R(·|s, g, ctx); s, r ← env.step(a)
    UpdateArchive(s)  # states seen while returning can also create or replace landmarks
# ---- HANDOFF ----
# ---- EXPLORE phase: π_E ----
while not explore_done:
    a ~ π_E(·|s); s, r ← env.step(a)
    UpdateArchive(s)
episode ends (life loss by default | explore budget | stall)
```
Every frame of both phases counts toward the frame budget. **Return frames are the price of having no
resets** and must be logged separately (`frames_return`, `frames_explore`).

### 2.2 Landmark archive
Key: `(level, room)` [RAM: `ram[57]`, `ram[3]`]. Each landmark stores:

| field | meaning |
|---|---|
| `obs` | 4×84×84 uint8 frame stack of the representative state (also used as the goal image) |
| `emb` | goal embedding φ(obs), re-computed whenever φ changes (§3) |
| `R` | cumulative extrinsic reward at the representative state, along its own trajectory |
| `U` | RND uncertainty of `obs`, **re-scored with the current RND before every comparison** (GoWU does the same implicitly, because U always comes from the live estimator) |
| `traj_ids` | top-k (default 4) trajectories s0→representative, ranked by (R at arrival ↓, length ↑) |
| `parent` | the landmark that was π_R's goal (or last reached) when this landmark was first found → **landmark tree** |
| `ram_extra` | `(x, y, inventory)` at the representative state [RAM], for logging and sub-landmarks |
| stats | `n_targeted`, `n_reached`, EMA success rate, EMA return length, `t_discovered`, `t_last_improved` |

**Replace rule (GoWU-matched):** a state s visited in room k replaces landmark L_k iff
`(R(s), U(s)) >_lex (R_k, U_k)`, both scored with the current RND. To keep each frame cheap, U(s) is
evaluated on states the RND already scores for training. Candidates are checked at every step
(`archive.check_every = 1`) or every n steps.
**Hysteresis switch** `archive.u_margin`: require `U(s) > (1+m)·U_k` for a tie-break replacement. This
stops the representative from flickering between near-identical frames (default m = 0.1).

**How this differs from GoWU's buffer.** GoWU has no persistent buffer. It keeps a *lineage tree*
pruned down to the currently active lineages. "Landmarks" there are just the particles alive right now,
and winners are picked among them every outer step. In the paper's vocabulary, our landmark is closest
to a **tree node / checkpoint**. Differences: (a) ours persists forever, (b) it is deduplicated by room,
(c) it holds a *trajectory* instead of an emulator snapshot. Because of (b), U only matters *within*
a room. **Across rooms, GoWU's "frontier" pressure must come from the selection rule (§5), not from the
archive itself.**

**Does uncertainty drive exploration in GoWU?** Yes. In GoWU, uncertainty is formally only the
tie-breaker after reward. But in sparse-reward MR almost every comparison is a tie on R, so U ends up
being the main criterion. Their ablation (Fig. 6a) is clear: replacing U with uniform random winner
selection means exploration never clears level 1. We keep U in two places: (i) choosing the
representative within a room, and (ii) choosing *which* landmark to go to (§5).

**Sub-landmarks** (switch `archive.sublandmarks`):
- `none` (default) — exactly one per room.
- `inventory` — key `(level, room, inventory)` [RAM]. Room 1 with a key and room 1 without one become separate landmarks.
- `xy_grid` — key `(level, room, x//gx, y//gy)` [RAM] (Go-Explore-like resolution).
- `emb_cluster` — up to c sub-landmarks per room by online k-centers in φ space (no RAM inside a room).

The room-level landmark is still the max-(R, U) over its sub-landmarks.

### 2.3 Trajectory store
Each completed episode is stored as compressed frames (uint8, zlib) plus actions, rewards, and the
per-step `(level, room)`. It is kept only if it is the top-k trajectory of some landmark or of some
hindsight goal (§4.4). Expected size: one episode ≤ 27k steps × 7 KB/frame ≈ 190 MB raw, about 15 MB
compressed. Capped at `traj_store.max_gb` (default 40 GB on the 128 GB box).

---

## 3. Networks

All networks are feedforward (**no RNN**). Observation = 4×84×84 grayscale frame stack + optional
context channels (§4.2).

| net | options | default |
|---|---|---|
| Torso | `nature` (3 conv, 512 fc) / `impala` (16-32-32 ResNet, 1024 fc, as in GoWU phase II) | `impala` |
| π_R | obs torso ‖ goal encoder → MLP → policy (18) | separate policy and value encoders (like GoWU phase II) |
| V_R | same inputs → value | |
| Goal encoder φ | **E1** RND *target* features (fixed random conv+MLP, 1024-d; stable, no training) / **E2** RND predictor features / **E3** temporal-contrastive embedding trained on stored trajectories (positives within Δt ≤ k, InfoNCE) / **E4** the policy's own obs torso applied to the goal frame stack (end-to-end through PPO+BC) / **E5** [RAM] one-hot(level, room) ⊕ (x, y)/255 — oracle | **E4**, with **E5 as an ablation** |
| Goal-reached test | **G1** [RAM] room match / **G2** [RAM] room match ∧ ‖xy − xy_g‖ < r / **G3** cos(φ(s), φ(g)) > τ in the E3 space / **G4** learned reachability classifier | **G1** (one landmark per room makes this natural); G3 for the RAM-free version |

Why E4 + G1 as the default: E4 lets the goal input be learned for control. G1 makes "success"
unambiguous while we debug everything else. Swapping G1 → G3 is the main step toward a RAM-free
algorithm, so it is its own milestone (§10).

RND (shared with GoWU code): Nature torso + 1024-ReLU-1024 MLP for target and predictor.
Input = last frame, normalised by running mean/std and clipped to [−5, 5]. Adam 3e-4, batch 128.
Trained on frames from **both** phases.

---

## 4. The return (smart) policy π_R

### 4.1 Inputs
`π_R(a | o_t, g, ctx_t)` with `o_t` the frame stack and `g` the goal frame stack (or embedding).
The goal is **fixed for the whole return phase** (s0 → g).

### 4.2 Context — the "temp landmarks list" (replaces the RNN)
In MR, the HUD shows inventory and lives, so the screen is close to Markov. What a feedforward
policy *cannot* see is **progress along its own route**: for example, whether it already went
through room 0 on the way to g. Context switches:

| switch | content | analog |
|---|---|---|
| `ctx.last_landmark_frame` | last frame at which the agent entered a new landmark room this episode (1 extra channel) | GoWU phase II's 5th channel "frame at last reward" |
| `ctx.last_reward_frame` | the frame at the last reward (1 channel) | exactly GoWU phase II |
| `ctx.ancestor_mask` | binary vector over g's ancestor chain in the landmark tree (padded to D_max = 32): which ancestors were reached this episode, in order | DTSIL's trajectory input, but read-only |
| `ctx.visited_rooms` | binary vector over all archive rooms visited this episode (≤ 24 × levels) [RAM-derived] | |
| `ctx.time` | t / T_max scalar | |

Default: `last_reward_frame` + `ancestor_mask`.

### 4.3 Reward for the return phase
```
r_t = c_goal · 1[reached(s_t, g)]                    # terminal success, c_goal = 1
    + c_ext  · clip(r_ext_t, -1, 1)                  # extrinsic, c_ext = 0.1 (switch: 0 / 0.1 / 1)
    + c_anc  · 1[first time reaching the next ancestor of g in order]   # switch, default 0.1, 0 = pure
    − c_step                                          # switch, default 0
```
The return phase ends at handoff. Bootstrapping: goal reached → terminal (V = 0). Timeout → truncation
(bootstrap V). Death → terminal failure. γ_R = 0.99 (switch 0.999). Note `c_anc` only shapes; the
goal stays g, so this is still s0 → g, not S_i → S_{i+1}.

### 4.4 Learning signals for π_R

**(a) Goal-conditioned BC with hindsight relabelling.** Every episode starts at s0, so **any prefix
τ[0:t] of any episode is a valid demo for "reach the state at time t" from s0.** The demos come from
our own exploration. Sticky actions are on, so these are closed-loop demos in the real stochastic env,
which is fine for BC.
- **Landmark demos:** when state s_t replaces landmark L_k, the prefix τ[0:t] joins L_k's top-k
  trajectories with goal = L_k.obs.
- **Hindsight room demos** (switch `bc.hindsight`): for every room k first entered at time t in an episode,
  τ[0:t] is a demo for goal L_k (valid under G1, since any state in room k satisfies the goal). This
  adds demos for free on every episode.
- **Which demos to keep:** top-k per landmark ranked by (R at arrival ↓, length ↑). Shorter is better,
  because it cuts the return cost.
- **Loss:** `L_BC = −E[log π_R(a_t | o_t, g, ctx_t)]`, with options:
  `plain` / `filtered` (only on landmarks whose current return success < τ_f, default 0.8, so BC stops
  pulling once PPO has found a better route) / `awr` (weight exp(A/β), A from V_R) / `sil` (weight max(G_t − V_R, 0)).
  Default: `filtered`.

**(b) PPO on return-phase data.** On-policy, with goal-conditioned value. GoWU phase II PPO
hyperparameters as the starting point (lr 1e-4, clip 0.1, 1 epoch, 8 minibatches, entropy 1e-3, GAE 0.95).
Explore-phase data does **not** train π_R under PPO (it is a different policy). It reaches π_R only
through BC relabelling.

**(c) Entropy boost against derailment** (from policy-based GE, switch): divide the logits by
`1 + (max(0, t̂ − e_T)·0.01)^2` once the agent has gone e_T steps (default: 2× the demo time to g)
without progress.

**Schedule (switch `rl.schedule`):**
- `alternate` (default, per your preference): repeat {1 PPO update on fresh rollouts; n_bc (default 4)
  BC minibatches of 512 sampled across all landmarks, with landmark sampling ∝ (1 − success)}.
  This is safe for on-policy PPO, because each PPO batch recomputes its behaviour log-probs at collection time.
- `joint`: `L = L_PPO + λ_BC·L_BC` in every minibatch (DQfD / SIL style). λ_BC is annealed from 1 → 0.1.
- `pretrain_ft`: when a new landmark appears, run n BC steps on its demos first, then PPO only.

Recommendation: `alternate` with `filtered` BC. Coverage BC also acts as rehearsal against
**catastrophic forgetting** of early landmarks, which is the classic failure of pure-RL agents on MR.

### 4.5 Return-failure detection (handoff trigger 2)
Return fails if any of these happen:
- `t_return > μ·L̄_g + b`, where L̄_g = EMA of successful return lengths to g (initialised from the
  shortest demo). μ = 2, b = 200. These are the GoWU backward-algorithm constants, with b reduced.
- No *new* ancestor or new room for `stall_steps` (default 500).
- Life loss → episode ends by default. Under infinite lives: life loss → count a failure but continue (§7).

On failure, control **still hands off to π_E** from wherever the agent is, and the attempt counts as
a failure for g's stats. This works like GoWU's rollback: we explore from a nearby earlier state
instead of the target.

---

## 5. Landmark selection (GoWU's population logic, made episodic)

GoWU's power is in population management (winner selection, pruning, consolidation). Its ablation
shows the particle policy barely matters. We need an analog without cloning. **Idea: a virtual
population.** There are E parallel envs (default 128), split into M = 4 groups of N = 32, as in GoWU.
Each group holds a **group goal** g_m. Each env in group m targets g_m at episode start (with
probability 1 − p_mix; otherwise it uses the alternatives below).

| GoWU mechanic | LR-GoWU analog |
|---|---|
| particle | one env-episode |
| winner selection (max R, then U, among alive particles) | among landmarks *discovered or improved by group m since its last consolidation*, pick the lex-max (R, U) |
| prune dead → clone winner | episodes that died/failed early re-target to the current group winner at their next reset |
| group consolidation after K outer steps | every K ∈ [K_min, K_max] group-episodes (default [16, 32]), set g_m ← group winner |
| global sync: groups with lower best R reset to global winner | if R(g_m) < R(global winner): g_m ← global winner |
| rollback when all particles die | if the success rate of returning to g_m stays < ρ_min (0.05) for W = 64 attempts: g_m ← a random ancestor of g_m, k ∈ [1, 3] generations back |

Selection switches (`select.rule`):
- `gowu_groups` (default) — the table above.
- `lex_max` — every env targets the global lex-max (R, U) landmark (pure consolidation).
- `u_softmax` — among the max-R landmarks, sample ∝ softmax(U/T).
- `competence` — weight ∝ U · p̂(1 − p̂), where p̂ = return success rate (goals of intermediate difficulty).
- `count` — Go-Explore weights 1/√(n_targeted + 1).
- `mix:<rule>:<p>` — with probability p use `<rule>`, otherwise `gowu_groups` (default p_mix = 0.1 → `count`).

**Reachability gate (all rules):** never select a landmark whose return success is 0 after
`n_gate` (default 32) attempts unless it has demos. A landmark you cannot reach cannot be explored from.

---

## 6. The explore policy π_E (three options + mixed)

| `explore.policy` | description | notes |
|---|---|---|
| `hold_random` | GoWU's ablation policy: sample one action, hold it for T ∈ [5, 15] steps, ε = 0 | reproduces GoWU's particle behaviour; *not RL* |
| `rnd_ppo` | separate PPO with intrinsic RND reward (non-episodic intrinsic return, γ_I = 0.99) + extrinsic (γ_E = 0.999); two value heads (Burda et al.) | pure-RL explorer |
| `goal_explore` | π_R itself with exploration goals: 10% an un-archived *adjacent room* [RAM room graph], 22.5% any adjacent room, 67.5% a landmark sampled by `select.rule`; a new goal every 100 steps or on reaching | policy-based GE found this finds ~4× more cells than random actions |
| `mixed` | choose one of the above per episode, uniformly or with given weights | for head-to-head comparisons on identical budgets |

Explore phase ends on: life loss (default) | `explore.max_steps` (default 1000) | no landmark
created or improved for `explore.stall_steps` (default 300).

---

## 7. Lives

- **Default:** life loss ends the episode (both phases). This is the analog of GoWU's "dead".
- **Infinite lives (baseline B2, switch `lives.mode`):**
  - `ignore` — losing a life does not end the episode. The game still decrements lives and ends at game over.
  - `ram_refill` [RAM] — after each life loss, write `ram[58] := initial lives` via `ale.setRAM`, so the run never reaches game over.
  - Either way: the life loss is logged; during return it counts as a return failure but the episode
    continues; the dying transition gets reward 0 (no death penalty) unless `lives.death_penalty` is set.
  - Caveat: in MR, the respawn point after death is the room's entry point. That is effectively a free
    local reset to the room entrance, which helps exploration. **This must be stated when reporting B2.**

---

## 8. Final policy and evaluation

The deployable policy is **π_R**. Goal schedule at test time (switch `eval.goal_schedule`):
- `best` (default) — goal = landmark with lex-max (R, U), fixed for the whole episode. The `c_ext`
  term in the PPO reward is what makes π_R pick up reward along the way and after arrival.
- `chain_by_R` — after reaching a goal, retarget to the next landmark in descending-R order that has
  not yet been reached in this episode.
- `tree_dfs` — follow the landmark tree from the root through the highest-R branch.

Protocol: sticky 0.25, 18 actions, 100 episodes per checkpoint, greedy vs sampled actions both
reported. The **108k-frame cap is the default, plus a mandatory uncapped run** (see the note at the top).
Like GoWU, keep the top-10 checkpoints by rolling eval and pick the best on 500 fresh episodes.
**Report both the selected-checkpoint score and the final-checkpoint score**, because selecting from
100 checkpoints inflates scores.

---

## 9. Baselines

| id | what | resets? | purpose |
|---|---|---|---|
| **B0a** | GoWU phase I (this repo) | yes | reproduce exploration efficiency (paper: ~98.7k cumulative reward at 400M frames) |
| **B0b** | GoWU phase II backward algorithm (this repo) | yes (to demo states) | reproduce the robustified policy (paper: 182.7k) |
| **B1** | **Oracle goal-RL on GoWU's lineage tree.** Take a finished B0a run, define one landmark per room = the lex-max (R, U) tree node seen in that room, and train π_R s0 → each landmark with *exactly* LR-GoWU's §4 machinery. The BC demos are the root→node paths from GoWU's tree; nodes store their actions, so those paths are executable. No exploration phase. | no (landmarks found with resets; learning is reset-free) | separates "can we learn to return?" from "can we find landmarks?" |
| **B2** | B1 with **infinite lives** (§7) | no | how much of the difficulty is death |
| B2′ (opt) | full LR-GoWU with infinite lives | no | for data only |
| **B3** | pure RL: RND-PPO (CNN, 128 envs), ~2B frames | no | the "pure RL underperforms" reference (paper: RND 8,152) |
| B4 (cite) | policy-based Go-Explore 97,728 @ 12B frames | no | quoted from the paper, not re-run |

Note on B1: GoWU demos are spaced 5–15 steps apart, but every node stores the actions of its rollout,
so a root→node path is a full action sequence. Under sticky actions, an open-loop replay of those
actions will *not* reproduce the path, so these are BC targets, not something to replay.

---

## 10. Milestones

1. **M0** — GoWU phase I reproduction (B0a): 3–5 seeds × 400M frames. Check against Fig. 5.
2. **M1** — B0b: 1 phase II run, 10 demos (10 phase I seeds) or 5 demos, ≤ 5B frames.
3. **M2** — B1 + B2: return learning on fixed GoWU landmarks. Success metric: return success vs. landmark depth (rooms from start).
4. **M3** — LR-GoWU full with `hold_random` explorer, G1, E4, `alternate` BC/PPO.
5. **M4** — explorer comparison (`hold_random` / `rnd_ppo` / `goal_explore` / `mixed`).
6. **M5** — remove RAM: G1 → G3 (temporal-contrastive embedding), room key → embedding clustering.
7. **M6** — uncapped evaluation and the comparison table.

---

## 11. Metrics (log every iteration)

`frames_total`, `frames_return`, `frames_explore` · rooms discovered (per level), max level ·
archive size, max R in archive · return success rate (overall, per landmark, vs. depth) · mean return
length vs. demo length (the "return overhead ratio") · landmark creations/improvements per 1M frames ·
RND loss · π_R entropy, KL, value loss, BC loss · eval score (capped and uncapped) · fps.

---

## 12. Full variable list

| group | variable | default | alternatives |
|---|---|---|---|
| env | `sticky_p` | 0.25 | 0 |
| env | `frame_skip` / max-pool | 4 / last 2 | |
| env | `action_set` | full 18 | minimal |
| env | `max_episode_frames` (train) | none (episodes end via the rules above) | |
| eval | `max_episode_frames` | **108000** | **null (GoWU-style, mandatory second run)** |
| eval | `loiter_steps` | 10000 | |
| eval | `goal_schedule` | best | chain_by_R, tree_dfs |
| archive | `key` | (level, room) [RAM] | + inventory, + xy grid, emb clusters |
| archive | `u_margin` | 0.1 | 0 |
| archive | `topk_traj` | 4 | 1–16 |
| archive | `check_every` | 1 | n |
| rnd | input / lr / batch | last frame / 3e-4 / 128 | stack |
| rnd | `updates_per_env_step` | 1/64 | |
| φ | goal encoder | E4 | E1, E2, E3, E5 |
| reach | test | G1 [RAM] | G2, G3, G4 |
| reach | G2 radius r / G3 τ | 12 px / 0.9 | |
| π_R | torso | impala | nature |
| π_R | ctx | last_reward_frame + ancestor_mask | see §4.2 |
| π_R | `c_goal, c_ext, c_anc, c_step` | 1, 0.1, 0.1, 0 | c_anc = 0 (pure) |
| π_R | γ_R / λ | 0.99 / 0.95 | 0.999 |
| π_R | PPO lr, clip, epochs, minibatches, ent | 1e-4, 0.1, 1, 8, 1e-3 | |
| π_R | entropy boost | on (e_f = 0.01, e_p = 2) | off |
| bc | `schedule` | alternate (n_bc = 4) | joint (λ 1 → 0.1), pretrain_ft |
| bc | `loss` | filtered (τ_f = 0.8) | plain, awr, sil |
| bc | `hindsight` | on | off |
| handoff | timeout μ, b | 2.0, 200 | |
| handoff | stall_steps | 500 | |
| select | rule | gowu_groups (M = 4, N = 32) | lex_max, u_softmax, competence, count, mix |
| select | K range (consolidation) | [16, 32] episodes | |
| select | rollback (ρ_min, W, k) | 0.05, 64, [1, 3] | |
| select | p_mix | 0.1 → count | |
| select | reachability gate n_gate | 32 | |
| explore | policy | hold_random | rnd_ppo, goal_explore, mixed |
| explore | hold T range / ε | [5, 15] / 0 | ε = 0.2 |
| explore | max_steps / stall_steps | 1000 / 300 | |
| lives | mode | episode_end | ignore, ram_refill |
| infra | n_envs | 128 | 64–256 |
| infra | frame budget | 2B frames for M3 | 5B |

---

## 13. Compute plan (2× 3090, 16 cores, 128 GB)

Measured on a laptop core: the ALE runs MR at ~20k raw frames/s per core (~5k agent steps/s before
preprocessing). Estimates for the 16-core box:

- **GoWU phase I:** env-bound. About 14 worker processes → roughly 30–50k agent steps/s → 400M frames
  (100M steps) in ~1–2 h of emulation, plus coordinator overhead. Realistically **2–6 h per seed**.
  10 seeds is about 1–2 days. The GPU is nearly idle with `hold_random` particles. With the
  `ensemble` particle policy it is more GPU-bound but still fine on one 3090.
- **GoWU phase II:** ~10–20k agent steps/s with the IMPALA ResNet and 128 envs → 5B frames ≈ 1–3 days per seed per GPU.
  The paper's 15–20B frames × 50 seed-runs is out of reach; budget 1–3 runs.
- **LR-GoWU:** like phase II PPO, plus return overhead. Policy-based Go-Explore needed 3B frames to
  pass DTSIL and 12B for 97k. Budget **2B frames per run** (~1–2 days), with 2 runs in parallel (one per GPU).
  The CPU becomes the bottleneck with 2 runs (~7 workers each).
- RAM: phase I tree + replay ~10–20 GB. LR-GoWU trajectory store ≤ 40 GB. Fine on 128 GB.

---

## 14. Risks

1. **Direct s0 → g fails for far goals.** This is policy-based GE's own finding. Mitigations: ancestor
   context and shaping (§4.2–4.3), hindsight BC, entropy boost. Kill criterion: if B1 return success
   to depth-5 rooms is < 20% after 500M frames, add `c_anc > 0` and `ctx.ancestor_mask`, then re-test.
2. **Forgetting** of early-room skills → coverage BC rehearsal.
3. **Landmark flicker / RND drift** → hysteresis `u_margin`, re-scoring.
4. **One-landmark-per-room loses inventory distinctions** (room with key vs. without) → `inventory` sub-landmarks.
5. **Exploits** (middle-room key bug, treasure-room loop, key respawn) → same exploit guard as the GoWU code.
6. **Return overhead dominates the budget** once landmarks are deep → log `frames_return / frames_total`.
   This ratio is the honest price of "no resets".
