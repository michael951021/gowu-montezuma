# GWRW (go-with-the-winner route walker), `gwrw_cd_v1` code package, fresh start

Frozen snapshot. Not our code: the `montezuma_gwrw` package (gwrw_cd_v1 research run), unpacked into
`third_party/montezuma_gwrw/` (tracked in this repo). The algorithm is unchanged. The only addition is an
opt-in `EVAL_GAMES` mode in `gwrw/train_gwrw_comp_cd.py` for the frozen test; with it unset, training is the package's code.

## What it learns from

- Screen pixels and game reward only. Emulator RAM (room, x/y, lives) goes into `diag_*` log fields and nothing else.
- Sticky actions 0.25 (`ALE/MontezumaRevenge-v5`), frame skip 4, 84×84×4 frames, up to 30 no-op starts.
- No emulator restores: every game is played from reset, like LR-GoWU.

## Parts

- **Landmarks**: the n-th positive reward of a game (`r n`) and the first entry into a screen background (`s id`, from a pixel
  scene-change detector whose codebook grows online). Frontier entries (`f uid`) are stored novelty windows used as extra targets.
- **Archive**: every distinct ordered landmark list some game produced is a node; depth = list length. A node is reproducible when
  at least 50% of the last 100 attempts arrive (min. 20 attempts).
- **Route controller μ**: goal-conditioned policy that follows a target's landmark list. Trained by behaviour cloning on the 8
  fastest loop-erased self-generated demos per node, plus PPO on its own landmark-hit reward.
- **Targets** per game: deepest / hard / spread / frontier wings (`pick_target`, shipped live json: `p_spread` 1, `p_frontier` 0.5).
- **Takeover explorer** (`EXPLORER_ARCH=trav`): when μ arrives, hits a wrong landmark or times out (1,500 steps per leg), the
  traversal net plays on until the game ends. Three agents: `a` is a uniform-random walker; `b`/`c` learn by PPO on RND novelty,
  game reward and progress streams, with a count-deficit competition between them. Anything new becomes a node, so a future target.

## Setup of this run (`runs/gwrw_400m_20261103`)

- Fresh start: no `CKPT14` (fresh RND / PopArt / advantage scales) and no 800-episode pool seed. Archive, codebook and demos start empty.
  The research run started from both, so this run is not the research run.
- Shipped `configs/gwrw_cd_v1_live.json` unchanged. Its README says it was tuned at update 90,900 of a mature run, not for a fresh start.
- 12 processes × 8 envs = 96 games per run, rollout 128, one run per RTX 3090. Two replicates, `a` and `b`. The package has no seed
  option: env seeds are the same scheme for both, and network init differs by time-seeded torch.
- 400M frames each (GoWU Phase I budget). Launcher: `scripts/gwrw_pair.sh`.

## Test (`scripts/gwrw_eval.sh`, `EVAL_GAMES=192`)

- Final checkpoint, read through hard-linked copies (prefix `gwrw_<r>_eval`), with no saves, so the training files are never touched.
- No gradient steps (μ, explorers, RND, BC all frozen). Every game targets the deepest reproducible route (the `near` wing; spread,
  hard, frontier and stitch wings off), then the explorers play on to game over.
- Fresh env seeds (+10M). Two complete games per env slot (192 games), so short games are not over-counted.
- Game cap is the trainer's own: 4,000 steps per scene seen, and 8,000 steps after hand-off.

## Comparability

- vs LR-GoWU: same no-restore setting and sticky actions. LR-GoWU's test is π_R alone (32 episodes, 30-min cap). GWRW's test
  includes its explorers after arrival, so it measures the whole system.
- vs GoWU Phase I: GoWU restores emulator states, so its best R is a search result, not a policy score.
- Rooms: GWRW's are raw RAM room ids from its diagnostics (all levels together); LR-GoWU counts level-1 rooms.
