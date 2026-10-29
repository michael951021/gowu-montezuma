# Line-by-line walkthrough

Line numbers match the PDF. Each item says what the line does and why, and points to the code
(`gowu/lr/trainer.py` unless noted). Items marked **Clarity?** are places where I think the pseudocode
could still read better. Those are the ones to push back on.

## Algorithm 1′ RunEpisode: one episode of one slot

One call is one episode: reset, walk to the goal landmark g with π_R (return phase), then explore with
random held actions (explore phase). In GoWU terms it is one particle's life between two resets. The
explore phase is a chain of hold segments, and each segment plays the role of one GoWU outer step.
Unlike GoWU, nothing is redistributed between segments, and a reward does not stop the segment.

- **params.** The return time limit is μ·L + b with μ = 2, b = 200. The return stall is 500 steps without
  progress. The explore phase stops after 1000 steps, or after 300 steps without improving the archive.
  Hard cap: 5000 steps per episode (the per-env ring buffer). Hold length 5–15 steps, ε = 0.2, and a death
  within 8 steps of a handoff counts as a "quick death".
- **1.** Reset the emulator to the game's start state s₀. There is never a restore. Zero the episode reward
  R, the step counter t and the improved set I, start the trajectory τ, set *handed* = false, and count one
  more targeting of g (`_begin_episode`).
- **2–3.** Special case: if g is the cell the game starts in and needs no reward (in practice the root),
  there is nothing to walk to. Count it as reached and as a handoff, record
  no return attempt (so W_g is untouched), and jump straight to the explore phase at line 20.
- **4.** Comment: the return phase. It is what replaces GoWU's `Reset(p_i, v)`.
- **5.** L is how long we expect the walk to g to take: the running average of past successful returns to
  g, or, before any success, the length of g's best demo, or else 1000. It sets the time limit on line 15.
- **6.** Loop once per environment step until a `return` or `break`. `Step` increments t.
- **7.** π_R samples an action from the stacked frames, the goal frame and the context κ_t (which ancestors
  of g have been visited, how much reward is still missing, time used). The logits are divided by the
  temperature β_t, which is 1 unless the episode has made no progress for 100+ steps (`_boost`).
- **8.** One environment step: apply the action, add the reward to R, append to τ, put the new frame in the
  RND replay. *dead* = lost a life, game over, or exploit guard fired (`_post_step`).
- **9.** If still alive, try to put the new state into the archive (Algorithm 2′). Any landmark it creates
  or replaces is added to I. A dead state is never archived, mirroring GoWU's "AddChild only for alive
  particles".
- **10.** The PPO reward for this step: +1 if this step reached g, +0.1 for each ancestor of g visited for
  the first time this episode on this step, ±0.1 if the game paid a reward (or penalty). Explore-phase
  steps get no r^R and are not trained on.
- **11–12.** Death during the return: log a failed attempt for g (Algorithm 3′, which may trigger a
  rollback) and end the episode as Dead. There is no explore phase.
- **13–14.** Reached g (right cell, enough reward): log a success, update the average return length
  (the first success sets ℓ_g = t), count a handoff, remember when (t_h) and leave the return loop.
- **15.** Failure without death. The time limit μL + b passed, or 500 steps went by without entering a new
  cell (t_prog is the last step that did), or the 5000-step cap was hit.
- **16.** Log a failed attempt for g. *handed* stays false, so a later death will not count as a
  quick death at g.
- **17.** If the episode is at the 5000-step cap, it ends here as Alive. There are no steps left for
  exploring.
- **18.** Otherwise explore from wherever the agent got to. A failed return still produces exploration
  data, just not from g.
- **19.** Comment: the explore phase. This is GoWU's particle policy.
- **20.** Remember when exploration started (t₀) and treat that moment as the last improvement (t_imp),
  so the 300-step stall clock starts now. h = 0 forces a fresh held action on the first step.
- **21.** Loop once per step until line 29's condition holds.
- **22.** When the current hold runs out, draw a new random action (any of 18) and a new hold length
  h ∈ {5..15}. **This is the held random action.** The particle commits to one action (say RIGHT+JUMP)
  and repeats it for h steps. That gives long, coherent movements: running across a room, jumping off a
  ledge. Fresh i.i.d. uniform actions every step mostly jitter in place, and the paper reports that this
  fails. GoWU v4 (§4.4) uses exactly this: "each particle commits to a single random action held fixed
  for the duration of each rollout segment … a different random action is played with small probability
  at each step (ε = 0.2)". In GoWU a segment is one outer step's T_i inner steps; our GoWU reproduction
  uses T_i ∈ [5, 15], and LR uses the same range for h.
- **23.** Play the held action and count down h. With probability 0.2 this step plays a uniform random
  action instead (the hold continues afterwards). Ours can redraw the held action itself; the paper says
  "a different" action. That is a 1-in-18 difference.
- **24.** One environment step, same bookkeeping as line 8.
- **25–27.** Death ends the episode as Dead. If it happened within 8 steps of a successful handoff, count a
  quick death at g. Many quick deaths make g "doomed", and the eligibility gate stops choosing it.
- **28.** Try to archive the new state. If it created or replaced a landmark (J non-empty), add it to I
  and reset the stall clock.
- **29.** Stop exploring after 1000 explore steps, after 300 steps without an improvement, or at the
  5000-step cap.
- **30.** The episode ends Alive. The slot's next goal is chosen in Algorithm 4′, lines 11–17.

**Clarity?** Line 9 (return) and line 28 (explore) both call UpdateArchive, but only line 28 uses J for
the stall clock. That is faithful to the code (`last_improve` is reset at the handoff, line 20). It may
still read as an inconsistency.

## Algorithm 2′ UpdateArchive: should this state become a landmark?

Called after every alive step of either phase. It is the analogue of GoWU's `AddChild`, but runs every
step instead of at the end of each outer step (`_post_step` + `archive.py`).

- **1.** c is the current cell. c_prev is the cell one step earlier, the candidate parent.
- **2.** If c already has a landmark recorded with exactly this R, the comparison on line 3 will come down
  to uncertainty. Recompute the incumbent's U with today's RND, since its stored value may be stale.
- **3.** Replace (or create) if the cell is new, or this state has more reward, or equal reward and at
  least 10% more uncertainty. This is GoWU's lex-max (R, U) winner rule applied per cell. The 10%
  margin stops two near-identical states from flipping back and forth.
- **4–5.** For a brand-new cell, its parent is the cell the agent just came from, and its depth is the
  parent's plus one. This is how the archive becomes a tree. The parent never changes later.
- **6–7.** For an R-improvement, the old demos and return statistics described routes that reached c with
  less reward. They no longer reach the new landmark, so drop them.
- **8.** Store the new frame, R and U. Report c as improved (J), and add it to the group's improved set
  I_m right away, so other slots of the same group see it before this episode ends.
- **9–10.** Not better: nothing improved.
- **11–13.** Hindsight demo. If this is the first time this episode stands in c with at least R_c, then
  the episode's prefix up to now is a working route from reset to c. Offer it as a demo. The landmark
  keeps its best 4 (most reward, then shortest). These demos are what BC trains π_R on.
- **14.** Return J.

## Algorithm 3′ RecordReturn: log a return attempt, maybe roll back

Called once per return attempt (Alg. 1′ lines 12, 14, 16). It is the analogue of GoWU's rollback
(paper Alg. 1 lines 14–17), with a different trigger: GoWU rolls back when every particle in a group
died, while here it is when π_R keeps failing to reach a landmark.

- **1.** Push the outcome into g's window of the last 64 attempts. A success also counts toward n_reach.
- **2.** With a full window of 64 attempts since the last clear and under 5% successes, g is too hard to
  return to right now.
- **3–4.** Every group currently aiming at g moves its goal 1–3 levels up the landmark tree (k drawn
  separately per group). The ν_j writes are mostly overwritten later; see Remark 1.
- **5.** Clear the window, so g needs 64 fresh attempts before it can trigger another rollback.

## Algorithm 4′ LR-GoWU: the population loop and learning

- **params.** A group consolidates after K ∈ [16, 32] of its episodes have ended. 10% of next goals are
  sampled by count. Learning runs after every unroll of T = 128 steps.
- **1.** The archive starts with only the start cell (R = 0). Create π_R/V_R and the RND.
- **2.** Every group's goal is the root, every group's improved set is empty, each group draws its K.
  Every slot's first episode targets its group's goal.
- **3.** Run forever (until the frame or time budget).
- **4–5.** 128 steps, during which all 128 slots step together (one batched forward pass, one vectorized
  env step).
- **6–8.** A slot whose previous episode just ended starts a new one. The goal is ν_i. If ν_i has become
  ineligible, use the group goal g_m or its nearest eligible ancestor instead.
- **9–10.** Advance this slot's episode by one step (one iteration of Alg. 1′'s loops). If it did not end,
  nothing else happens for this slot this step.
- **11.** The episode ended: pick the goal of this slot's next episode. This is where GoWU's
  redistribution happens.
- **12–13.** With probability 0.1, pick a random eligible landmark, favouring rarely targeted ones
  (weight 1/√(n_tgt + 1)). This keeps every part of the archive in π_R's training distribution.
- **14–15.** Survivor continues: if the episode ended alive and improved an eligible landmark, next time
  go to the best of those (highest R, then highest U). This is GoWU's "alive particles keep their own
  trajectory", except that "keep going" now means "walk back there next episode".
- **16–17.** Otherwise (died, or improved nothing), go to the group winner: the best of the group goal
  and everything the group improved since its last consolidation. This is GoWU's "dead particles are
  reset to the winner".
- **18.** Count one more finished episode for the group.
- **19–21.** Consolidation (GoWU Alg. 1 lines 29–30): after K episodes, the group winner becomes the
  group goal. Start a new improved set, reset the counter, draw a new K. The ν_j writes only stick for
  this slot (Remark 1).
- **22–24.** Global sync (GoWU Alg. 2 lines 8–10, 16): take the best group goal across groups. Any group
  whose goal has strictly less reward adopts it. Groups at equal reward keep searching independently.
- **25.** Learning, after each 128-step unroll (16,384 agent steps).
- **26–27.** PPO on the return-phase steps of this unroll only, with r^R. Reaching g or dying is a true
  end. A timeout or stall is a truncation, so its value is bootstrapped (the trip didn't end, we
  stopped it).
- **28–29.** Behaviour cloning on demos: pick 512 samples by landmark, favouring landmarks π_R still
  fails to reach (weight 1 − success). Skip landmarks it already reaches ≥ 80% of the time (after at
  least 8 tries). Then pick a random demo and a random step j of it, rebuild π_R's input as it would
  have been at that step, and train π_R to output the demo's action. Do 8 such minibatches.
- **30.** Train the RND predictor on random frames from its replay, which holds every frame any slot has
  seen. **This is where U changes.** U is *evaluated* during the episode (Alg. 2′ lines 2–3, and every
  lex-argmax / Winner re-scores U); it is only *trained* here, once per unroll. That mirrors GoWU, where
  a separate learner trains the RND asynchronously from a shared replay.

## Remarks

- **Remark 1.** Slot goals are chosen when that slot's episode ends (lines 11–17). Consolidation, sync
  and rollback write ν_j for slots that are still mid-episode, and those writes are overwritten when the
  slots finish. What really changes is g_m, which is used by dead slots (line 17) and by the eligibility
  fallback (line 7). This differs from GoWU and is a candidate v3 change.
- **Remark 2.** Which counters reset when. n_tgt is counted after the fallback, so it counts the goal
  actually pursued.
