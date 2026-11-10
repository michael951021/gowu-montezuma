# GWRW for Montezuma's Revenge: how it works and how the code is laid out

This is the long explanation that goes with `README.md` (which covers setup and launching). Everything here describes the code in
`gwrw/train_gwrw_comp_cd.py` as shipped, with the live hyper-parameters in `configs/gwrw_cd_v1_live.json`. Where the code and this
text disagree, the code is right. Search for the function names quoted below to find the place.

## 1. The idea in one paragraph

GWRW is "go with the winner" exploration without ever restoring an emulator state. Instead of resetting to a stored game state, a
**route controller** (a neural policy) is trained to walk the game from the start back to places earlier games have reached. When it
gets there, or fails, **explorer agents** take over and play on to game over looking for something new. Anything new they reach
becomes a new place in an **archive**, and so a future destination for the route controller. The route controller learns from the
fastest routes the swarm has found (behaviour cloning) and from its own reward for hitting the route's landmarks (PPO).

Two rules hold throughout: the algorithm sees only pixels and the game score (emulator RAM such as room number, x/y, lives and
inventory is read into `diag_*` log fields and nothing else), and no emulator state is ever restored, so every game starts from the
real game start.

## 2. Life of one game

```
 new_game(i): pick a target (a list of landmarks)  ->  phase 0 (route)  ->  hand-off  ->  phase 1 (explore)  ->  game over  -> bookkeeping
                     |                                  mu follows the         arrival, wrong          explorer plays
                     |                                  target's landmarks     landmark or timeout     (trav net, agent a/b/c)
                     +-- no target (empty archive, p_none): straight to phase 1
```

96 games run in parallel by default (12 worker processes x 8 environments). Each decision is one env step of 4 emulator frames
(`frames` in the logs counts emulator frames). Route games and explorer games share the same batch.

### Landmarks (`gwrw/landmarks.py`, `Scenes`, the loop in `main()`)

A game's **landmark sequence** is the ordered list of events it has produced so far:

* `("r", n)`: the n-th positive game reward in this game.
* `("s", id)`: the first entry, in this game, into a screen background (scene). A pixel scene-change detector decides when the
  background changed; a scene codebook (`st["sigs"]`) maps backgrounds to ids and grows online.
* `("f", uid)`: a **frontier** entry (section 6). It only appears as the last item of a target, never in a game's own sequence.

### Archive and demos

Every prefix of a game's landmark sequence is a **node** (a tuple of landmarks). `archive[node]` holds the visit count and a history of
recent attempts to reach it (last 200 games). The depth of a node is its length. For every node, `demos[node]` keeps the 8 fastest legs
(`DEMOS_PER_NODE`) that went from the previous landmark to this one: frames, actions, and a "kept" index list after **loop erasure**
(`loop_keep`: a state is the last 4 frames; if a later state at least 15 steps on matches it in pixels, the steps in between are dropped, so the kept
action is the one that left the loop; `LE_N`, `LE_GAP`, `le_thr`). A leg is capped at
1000 steps (`LEG_KEEP`).

A node is **reproducible** when its history holds at least 20 attempts and at least 50% of the last (up to) 100 of them reached it
(`MIN_ATTEMPTS`, `REPRO_RATE`, `rate()`, `reproducible()`).

A game ends at game over, or at a step cap: 4000 steps per scene the game has entered (`STEP_CAP_PER_SCENE`), and no more than
`takeover_cap` (8000) steps after the hand-off.

## 3. Choosing the target (`pick_target`, `new_game`)

The live json uses `p_none 0, p_spread 1, p_hard 0, p_consolidate 0, p_stitch 0, p_frontier 0.5`, which gives:

1. No node with demos yet: the game has **no target** and the explorer plays from the first frame. This is the start of a fresh run.
2. If the deepest node's chain is not reproducible and nothing is reproducible: target that weak prefix (`weak`).
3. **Spread wing** (`p_spread`): take the candidate nodes (`hard_cands`: every proper prefix with demos is reproducible; frontier ends
   excluded), group them by how their end landmark *looks* (reference frames kept per landmark, `lmref`), draw a representative from each
   look-alike group by rarity, and pick uniformly over the groups. This spreads games over different places instead of piling onto the
   deepest route.
4. Otherwise `near`: a random reproducible node whose depth is within one of the deepest reproducible depth.
5. Then, with probability `p_frontier`, the target is replaced by a **frontier** entry if the frontier buffer is not empty.

Other wings exist but are off in the shipped json: `hard` (rare/deep reproducible nodes, `p_hard`), `consol` (the deepest
not-yet-reproducible node, `p_consolidate`), `stitch` (splices of two known routes, `p_stitch`), `none` (`p_none`).

## 4. The route controller `mu` (`build_mu`, phase 0)

* Input: the last 4 frames (84x84), the target landmark's id (`lmid`), the position in the target list `k` (depth slot) and the goal
  frame (the picture of where the leg ends, scaled by `GOAL_GAIN`, 0.01). Output: 18 action logits and a value.
* The game follows the target one landmark at a time. A landmark that matches the target's next entry (or one of the next two, which
  are then marked pending) advances `k` and pays +1. A step costs 0.0005. An unrelated landmark ends the route with `fail="other landmark"`;
  more than `LEG_TIMEOUT` (1500) decisions on a leg ends it with `fail="timeout"`. Reaching the end of the list is **arrival**.
* Training, per update, uses only route-phase steps: PPO (`ppo` clip 0.1, `ent_mu` entropy bonus, gamma 0.999) and **behaviour cloning**
  (`bc_step`, `bc_batch`): `bc_steps` steps of `bc_batch` samples per epoch, drawn from the stored demos (kept steps only) of the nodes
  currently being targeted. `bc_hard` up-weights legs that fail more often (`legfail`), `bc_front_frac` mixes in frontier legs.

## 5. Hand-off and the explorers (phase 1)

At arrival, a wrong landmark, or a timeout, the game flips to phase 1 and one of three PPO agents `a`, `b`, `c` is the **owner** (drawn
uniformly per game) and plays until game over.

* **Network**: with `EXPLORER_ARCH=trav` (the shipped setting) the explorer is a second copy of the `mu` architecture (`trav`), initialised
  from `mu`, with per-agent actor heads and per-stream critic heads. It is run with goal `-1` and a fixed depth slot (`EXPL_K`, 20). It
  also gets behaviour-cloning steps on the route controller's legs and on "go-back" legs (re-entering a known scene, `bdemos`), so it
  keeps the ability to travel, while PPO on the owner's rewards pushes it to explore.
* **Action holding**: a sampled action is held for `expl_hold` (10) steps; on each held step with probability `expl_eps` (0.2) a fresh
  sample is taken instead (that step counts as a decision for PPO).
* **Agent `a` is the anchor walker** (`anchor_walker 1`): in games it owns it plays uniformly random actions through the same hold
  process and its policy does not learn (no PPO step for it). Its cell counts still enter the competition below.
* **Rewards for `b` and `c`** (separate critics, "streams"): *task* = RND novelty on pixels (each agent has its own RND
  predictor/target, normalised by a running return scale, times `intrinsic_coef`, times an episodic factor 1/sqrt(count) of the hashed
  feature code within the game); *extrinsic* = game score; *progress* = distance from the current conv feature to the nearest earlier
  one in the same game (features of a frozen network created at start, `build_prog_feat`), normalised by the mean step distance. The *landmark* stream has weight 0. The
  stream advantages are divided by their running scale and combined with weights `w_task` 100, `w_extrinsic` 50, `w_progress` 50.
* **Competition** (`qdef_mode 2`, count-deficit): the last frame is cut into 12x12 blocks of 7x7 pixels, each block's mean is quantised
  in steps of 32 intensity levels (8 levels) and hashed to a cell id (`cell_keys`). Each agent counts the cells it has been in. At a cell where
  another agent (`a` included) has more visits than this one, the agent's advantage is lowered in proportion to the share deficit
  (`qdef_beta` times the advantage's standard deviation), so the agents are pushed apart.
* RND predictors are trained on the pixels of the agent's own takeover steps, with a keep-mask `rnd_keep`.

## 6. Frontier entries (`front_process`, `pick_frontier`, `fck_protos`)

When a game ends having followed a known node (`front_ok`), `front_process` scores the stretch of the game after that node by RND
novelty (mean over the three agents of prediction error over the agent's running error, median-smoothed over `FRONT_W` strided
samples) and offers the best moment to a buffer. An entry is a target `base + (("f", uid),)` with a stored demo: the route controller
walks `base`, then replays towards the novel moment, and the explorers take over from there. This is where the swarm is sent to a spot
that looked promising last time.

* Buffer: at most `FRONT_MAX` (90) entries, `FRONT_K` (6) per landmark set; near-duplicate frames (`FRONT_DUP_D`), over-crowded looks
  (`FRONT_CLUSTER_MAX`) and dark-room frames (`FRONT_DARK_MAX`) are limited; a candidate must beat the weakest aged entry to enter.
* Value and sampling: score times `(1 + depth) ** front_depth_pow`; sampling probability is the value to `front_pow`, mixed with a uniform
  floor `front_floor`.
* Arrival at a frontier target (`p_fprox 1`, `fprox_strict 1`): the game must visually match the checkpoints along the stored demo
  (`fprox_*`); a timer is no longer an arrival and an off-route landmark is a failed leg ("landmark first", "no match"). Each matched
  checkpoint pays +1 to the route controller.

## 7. One update (`while st["frames"] < MAX_FRAMES` loop in `main()`)

1. Re-read `<RUN>_live.json` (the hyper-parameters).
2. Roll out `ROLLOUT` (128) decisions for all games. Each game acts with `mu` (phase 0) or its owner's explorer (phase 1). Landmark events are
   processed per step: archive, demos, target progress, frontier checkpoints, game-over bookkeeping, recorder episodes.
3. `front_process()` for finished games.
4. Route controller: GAE (gamma 0.999), 4 epochs of PPO minibatches on route steps, plus the BC steps.
5. Explorer: RND intrinsic reward and normalisers, count-deficit penalties, per-agent GAE and PPO for `b` and `c`; RND predictor updates;
   one traversal-BC step per epoch.
6. Write one row to `<RUN>_log.jsonl`, print one `update=...` line, save every `SAVE_EVERY` (50) updates, and on SIGINT after the
   current update.

With the defaults an update is `128 x 96 x 4 = 49,152` emulator frames.

## 8. Hyper-parameters in `configs/gwrw_cd_v1_live.json`

Effective values are `LIVE_DEFAULT` (in the trainer) overlaid by `gwrw/out/<RUN>_live.json`, re-read every update. These are the research
run's values at update 90,900, not values tuned for a fresh start.

| Group | Keys | Meaning |
|---|---|---|
| Targets | `p_none` 0, `p_spread` 1, `p_hard` 0, `p_consolidate` 0, `p_stitch` 0, `p_frontier` 0.5 | share of games per wing (section 3); `spread_s_only` 0 allows reward landmarks as spread ends; `hard_*` shape the hard wing |
| Frontier | `front_collect` 1, `front_pow` 2, `front_floor` 0.15, `front_depth_pow` 1 | collection on, sampling sharpness and uniform floor, depth bonus |
| Game length | `takeover_cap` 8000, `rec_x` 40 | steps allowed after the hand-off; threshold for the episode recorder (a diagnostic output, `record_deep`) |
| Frontier arrival | `p_fprox` 1, `fprox_strict` 1, `fprox_thr` 0.006, `fprox_n` 4, `fprox_slack` 1.5, `fprox_seg` 30, `prox_thr` 0.004, `le_thr` 0.0003 | visual-match arrival (section 6) and loop erasure |
| Route controller | `ent_mu` 0.003, `bc_steps` 2, `bc_batch` 512, `bc_hard` 4, `bc_front_frac` 0.6, `bc_front_pow` 1 | entropy bonus and BC schedule (section 4) |
| Explorer action process | `expl_hold` 10, `expl_eps` 0.2, `anchor_walker` 1, `entropy_coef` 1.0 | section 5 |
| Explorer rewards | `intrinsic_coef` 1.5, `w_task` 100, `w_extrinsic` 50, `w_progress` 50, `w_landmark` 0, `gamma_task` 0.99, `gamma_extrinsic` 0.999, `gamma_progress` 0.99, `prog_clip` 3 | streams and discounts |
| Competition | `qdef_mode` 2, `qdef_beta` 2, `qdef_gamma` 0.99 | count-deficit penalty strength and discount |
| PPO | `takeover_lr` 3e-5, `ppo_clip` 0.1, `ppo_epochs` 4, `gae_lambda` 0.98, `value_coef` 1, `minibatch_size` 256, `popart_beta` 0.01, `rnd_keep` 0.25, `episodic_novelty` true, `lm_scale_floor` 0.05 | explorer optimisation |

Environment variables (see `README.md`) set the sizes: `N_PROC`, `ENVS_PER_PROC`, `ROLLOUT`, `LEG_TIMEOUT`, `MAX_FRAMES`, `SAVE_EVERY`,
`GOAL_GAIN`, and so on.

## 9. Where things are

| File | Contents |
|---|---|
| `gwrw/train_gwrw_comp_cd.py` | the whole run: `main()` holds the rollout loop and the closures `pick_target`, `new_game`, `add_demo`, `bc_batch`, `front_process`, `save` |
| same file, top level | `build_mu`, `Prog` (progress reward), `cell_keys` (count-deficit cells), `popart_*`, `gae*`, `seed_from_pool` (not used in a fresh start), `LIVE_DEFAULT` |
| `gwrw/train_gwrw.py` | `Scenes` (scene-change tracker), `Vec` (vector env over worker processes), shared helpers |
| `gwrw/landmarks.py` | scene signature, change detector and thresholds |
| `gwrw/xworker.py`, `gwrw/train_route.py` | environment worker process and the Atari environment construction |
| `algorithms/rnd.py` | RND model (predictor/target, observation normaliser) |
| `models/*.py` | `actor_critic_net.py` and `dueling_qnet.py` are imported by the trainer but only used by the run14 explorer option (needs the earlier checkpoint) |
| `env/*.py` | Atari wrappers and the RAM diagnostic reader (`read_progress`, diagnostics only) |

## 10. Reading the outputs (`gwrw/out/`)

* `<RUN>_stdout.log`: one `update=N frames=... nodes=... max_depth=... scenes=...` line per update plus event lines: `[new] depth D node ...`
  (a new archive node), `[spread] ...`, `frontier uid: N checkpoints ...`, `landmark places ...`, `[stop] ...`.
* `<RUN>_log.jsonl`: the same plus per-agent stats (`ent_`, `vl_`, `rnd_`, `score_`, `qgap_`...) and, when present, `mean_score`,
  `max_score`, `reproducible_nodes`, `reproducible_depth`, `best_route`, `front_n`, `diag_rooms` (rooms seen in recent games, from RAM, diagnostics only).
* `<RUN>_episodes.jsonl`: one row per finished game (wing, target, arrived or failed, `fail`, `seq`, scores, `diag_*`).
* `<RUN>_events.jsonl`: one row per new archive node (update, frames, node, who made it).
* `.pt`, `_demos.pkl`, `_bdemos.pkl`, `_front.pkl`, `_lmref.pkl`: the resumable state (checkpoint plus the stores).

Quick look at progress:

```
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("gwrw/out/gwrw_cd_v1_log.jsonl")]
for r in rows[-3:]:
    print({k: r.get(k) for k in ("update", "frames", "nodes", "max_depth", "scenes", "route_share", "mean_score", "max_score",
                                  "reproducible_depth", "front_n", "diag_rooms")})
PY
```

## 11. What to expect from a fresh start, and what is not known

* At first there are no nodes, so every game is free exploration. A node appears when a game first produces a reward or enters a new
  scene. In a 0.3M-frame check with random-like play there were no nodes yet; with injected fake rewards (`DEBUG_FAKE_R`, tests only) nodes,
  demos, frontier entries and BC all came up and nothing crashed.
* A full-length fresh run has not been done. How long the first reward and the later rooms take is unmeasured; the research run
  reached 3.9e9 frames from a seeded start (an earlier run's agents and an 800-episode pool, which are not part of this package).
* The shipped json is tuned for a mature archive. In particular `p_spread 1`, `bc_hard 4` and `entropy_coef 1.0` assume there are many
  reproducible nodes; with an empty or tiny archive most of those paths are inactive. Changing values is up to you; none were changed here.
* Memory grows with the run: demos are kept in RAM and pickled at each save (about 25 GB of `_demos.pkl` and 55 GB RSS at update 90,900).
* Stop with SIGINT only (`kill -INT $(cat gwrw/out/<RUN>.pid)`); SIGTERM and kill -9 lose everything since the last save.
