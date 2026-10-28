# GoWU Phase I + room-8 trap fix, at commit `c0ffe06`

Frozen snapshot. Everything is the paper-v4 GoWU Phase I of the baseline (`docs/GOWU_REPRO.md`:
Algorithms 1–2, M = 4 × N = 32, K ∈ [10, 20], T ∈ [5, 15], rollback ∈ [1, 3], held random action
with ε = 0.2, RND at 31 updates per 1k steps, middle-room exploit guard) plus one addition that is
**not in the paper**.

## The problem

- In room 8 (levels 1 and 2) particles can reach a rope post with no exit. They never lose a life there,
  so GoWU never marks them dead. Level 2 pays +1000 twice on the way in.
- Holding that reward, they win every lex-max (R, U) comparison, and global sync resets every group into
  the trap. RND then sees only room-8 frames and U collapses (~1e-4).
- 26 of 32 of our phase I logs end with the global winner confined to room 8; seeds 6/7 of the baseline
  spent 254M and 320M of their 400M frames there.

## The fix: confinement as a dead state (`trap_window = 200`)

- **Trigger**: the global winner keeps the same R and the same (level, room) for 200 consecutive iterations
  (~10M frames). Outside the trap the longest such stretch in any of our runs is ~120 iterations; in the
  trap it is 400–7,000.
- **Dead cells**: every (level, room, R, x // 8, y // 8) that a surviving particle occupied at the winner's
  (level, room, R) during the window becomes a dead state, handled like the exploit guard's flagged
  states: a particle that lands in one is dead and is reset to its group's winner.
- **Relocation**: every particle, and the global winner, sitting in a dead cell moves to its nearest
  lineage ancestor outside the dead cells. The best-ever node (for the demo) is unchanged.
- The cells are keyed by R, so the same spot at a different reward is still allowed.

## Escape test (`runs/trap_escape_20261028`, 20M frames from room 9, trigger 100)

| Start | fix off | fix on | T ∈ [3, 20], no fix |
|---|---|---|---|
| level 2, seed 7 (37,200) | stuck in room 8 at 39,200 | 44,000, 22 level-2 rooms | – |
| level 2, seed 6 (32,200) | stuck in room 8 at 34,200 | left room 8, no new reward in 20M | stuck in room 8 |
| level 1 (two states) | not trapped in 20M | within noise | not trapped |
