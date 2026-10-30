# LR-GoWU v2 · infinite lives: architecture at commit `e2d46ef`

Frozen snapshot of what the infinite-lives baseline runs (`runs/lr_v2_inflives_20261030/{a,b}`) executed:
v2 with `lives_mode=ram_refill` (PLAN §7, baseline B2′), warm-started from v2's ~122M-frame checkpoint like v2-resume.
Caveat to state with any result: in MR you respawn at the room's entry point, so each death is a free local reset to
the room entrance. The result is not comparable to GoWU, where a death ends the episode. Later iterations get their
own file; this one is not edited after the fact. Code: `gowu/lr/{trainer,archive,net}.py`. Plan: `docs/PLAN_LR_GOWU.md`.

## One sentence

GoWU's population search over a lex-max (R, U) landmark archive, except that nothing is ever
restored: every episode starts at `env.reset()`, a learned goal-conditioned **return policy π_R**
walks back to the chosen landmark, and only then does random exploration start.

## Episode loop (per env, 128 envs in lockstep, 4 groups × 32)

1. **Goal choice** (GoWU Alg. 1/2 transplanted onto landmarks)
   - Surviving particle (episode ended without death; under infinite lives, almost every episode) → lex-max (R, U) among the landmarks it improved this episode.
   - Dead particle → its group's winner: lex-max (R, U) over the group goal and the landmarks the group improved since its last consolidation.
   - With p = 0.1, a count-based sample instead (weight 1/√(n_targeted+1)).
   - Every K ∈ [16, 32] episodes a group **consolidates**: group goal ← group winner; then **global sync**: groups whose goal has lower R than the global winner jump to it.
   - **Rollback**: a landmark whose last 64 return attempts succeed < 5% sends every group targeting it back 1–3 ancestors.
   - **Reachability gate**: a landmark targeted ≥ 32 times, never reached, *and with no demos* is skipped; a landmark where >90% of ≥16 handoffs die within 8 steps is skipped. An ineligible goal falls back to its nearest eligible ancestor. *(v2 change: v1 gated landmarks even when they had demos.)*
2. **Return phase** (π_R acts)
   - Success = in the goal's cell with R ≥ goal R.
   - Failure = death, timeout (μ·L + b with μ = 2, b = 200, L = EMA of successful return length, else best demo length, else 1000), 500 steps without reaching a new ancestor, or the 5000-step episode cap.
   - **Infinite lives**: a life loss during return counts as a return failure (no death penalty, reward 0), and the agent explores from where it respawns. *(Change vs v2: there, death ended the episode.)*
   - On failure the episode still continues into exploration.
3. **Explore phase** (GoWU particle policy, `hold_random`)
   - A uniformly random action held for 5–15 steps, ε = 0.2 per-step noise.
   - Ends at 1000 steps or 300 steps without a landmark improvement. **A life loss does not end it**: the lives byte (RAM 58) is reset to 5 after each loss, so game over never happens and the agent keeps exploring from the respawn point. *(Change vs v2: there, the first life loss ended the episode.)*
   - The quick-death gate (handoffs that die within 8 steps) never fires, because no death ends an episode.

## Landmark archive (`archive.py`)

- **Cell key** `xy_inv` = (level, room, #keys held, x // 16, y // 16) from RAM.
- One landmark per cell, replaced iff (R, U) beats it lexicographically, with 10% hysteresis on U ties. Never removed.
- U = RND prediction error on the landmark's frame (Nature torso + 1024-1024 head, input-normalised, 31 updates per 1k agent steps from a 512k replay), re-scored fresh at every comparison.
- Parent = the landmark the agent was in when it first created this one → a landmark tree with depth.
- **Demos (hindsight)**: the prefix of *any* episode up to its first arrival in a landmark's cell with that landmark's R is a demo for it. Top-4 per landmark by (R desc, length asc). Shared frame storage, refcounted, 12 GB cap. An R-improvement clears the landmark's demos and statistics.

## Return policy π_R (`net.py`)

- **Input image** (6 × 84 × 84): 4-frame stack, frame at the last reward (GoWU phase II's 5th channel), the goal landmark's frame (early fusion).
- **Context vector**, concatenated after the torso: 32-bit ancestor mask (which of the goal's 32 nearest ancestors this episode has already visited), clip((goal R − R)/1000, 0, 5), t / 5000.
- **Network**: separate IMPALA ResNet encoders (16-32-32, 1024-d) for policy and value; heads 1024+ctx → 512 → 18 / 1.
- **Return reward**: +1 on reaching the goal, +0.1 per new ancestor reached, +0.1·sign(extrinsic reward). Zero during exploration (those steps are not trained on).

## Learning (alternating, every 128-step unroll)

1. **PPO** on return-phase steps only: γ = 0.99, λ = 0.95, lr 1e-4, clip 0.1, 1 epoch, 8 minibatches, entropy 1e-3, value 0.5, grad-norm 0.5, bf16. Entropy boost on stalls: coefficient × (1 + (0.01·steps-without-progress beyond 100)²).
2. **Goal-conditioned BC**: 8 steps × 512 samples of (stacked frames, goal, ctx) → demo action, landmarks sampled ∝ (1 − return success); landmarks with ≥ 8 attempts and success ≥ 0.8 are filtered out.
3. **RND** updates (shared with the archive's U).

## Evaluation (the deployable score)

Unchanged and **not** infinite-lives: π_R from `env.reset()`, sticky actions 0.25, all 5 lives (normal game over), 32 episodes, goal = the archive's best landmark; when reached, retarget to the next best. Reported capped at 108k frames (30 min) and uncapped.

## Fixed settings

128 envs, 14 env workers, 2.5 h wall clock per seed (warm start from `runs/lr_test2_20261027/{a,b}` at ~122M frames, 50-update lr warm-up), one RTX 3090 per seed, middle-room exploit guard on.
