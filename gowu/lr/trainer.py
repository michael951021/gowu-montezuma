"""LR-GoWU trainer (docs/PLAN_LR_GOWU.md). No restoreState: every episode starts from env.reset().

Per env-episode (PLAN §2.1): pick a goal landmark g (GoWU-style group selection, §5), run the return
policy π_R(a | s, g, ctx) until the agent reaches g's cell with g's reward (handoff) or the return fails
(timeout, stall, death), then run the explore policy π_E until death, its step budget or a stall.
Every visited state can create or replace a landmark (lex-max (R, U) per cell, U from a shared RND).

π_R learns from (a) PPO on return-phase steps and (b) goal-conditioned BC on hindsight demos: the
prefix of any episode up to its first arrival at a landmark (with the landmark's reward) is a demo for
that landmark. All envs step in lockstep; return-phase envs act with π_R, explore-phase envs act with
the held-random policy (GoWU's particle policy, eps-noise per step).
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..exploit_guard import ExploitGuard
from ..nets import RND
from ..replay import Replay
from ..vec import EnvPool
from .archive import RAM_GOAL_DIM, Archive, Landmark, cell_key, ram_goal_vec
from .net import GoalActorCritic

RETURN, EXPLORE = 0, 1


@dataclass
class LRConfig:
    game: str = "montezuma_revenge"
    seed: int = 0
    total_frames: int = 2_000_000_000
    max_hours: float | None = None       # wall-clock training budget (evaluation runs after it)
    # warm start from another run's out_dir: network, RND and landmark archive (with per-landmark stats).
    # Not saved there, so rebuilt: demo trajectories (refilled by hindsight), optimizer moments, group goals.
    resume_from: str | None = None
    resume_warmup_updates: int = 50      # linear lr warm-up after a resume (Adam moments start from zero)
    n_envs: int = 128
    n_groups: int = 4                    # M; N = n_envs / M
    n_workers: int = 14
    sticky_p: float = 0.25
    unroll: int = 128
    max_episode_steps: int = 5000        # per-env episode ring buffer
    # archive (§2.2)
    archive_key: str = "room"            # room | room_inv | xy | xy_inv  ([RAM])
    cell_xy: tuple = (16, 16)
    u_margin: float = 0.1
    topk_traj: int = 4
    max_demo_gb: float = 12.0
    # RND (shared with GoWU; paper v4 learner rate and shared 512k replay)
    rnd_lr: float = 3e-4
    rnd_batch: int = 128
    rnd_updates_per_1k_steps: float = 31.0
    rnd_replay: int = 512_000
    # π_R (§3-4)
    goal_encoder: str = "frame"          # frame (E4, early fusion) | ram (E5 oracle) | both
    ctx_ancestors: int = 32              # D_max of the ancestor mask (0 = off)
    dense: int = 1024
    head: int = 512
    c_goal: float = 1.0
    c_ext: float = 0.1
    c_anc: float = 0.1
    c_step: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-4
    clip: float = 0.1
    epochs: int = 1
    minibatches: int = 8
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    amp: bool = True
    entropy_boost: bool = True           # policy-based Go-Explore: flatten logits after a stall
    boost_after: int = 100               # steps without progress before the boost starts
    boost_ef: float = 0.01
    boost_ep: float = 2.0
    # BC (§4.4)
    bc_schedule: str = "alternate"       # alternate | joint | off
    n_bc: int = 8                        # BC minibatches per PPO update (alternate)
    bc_batch: int = 512
    bc_loss: str = "filtered"            # filtered | plain
    bc_tau: float = 0.8
    bc_coef: float = 1.0                 # joint: λ_BC
    # handoff / return failure (§4.5)
    ret_mu: float = 2.0
    ret_b: int = 200
    ret_stall: int = 500
    ret_default_len: int = 1000          # timeout base when the goal has no demo and no success yet
    # selection (§5)
    select_rule: str = "gowu_groups"     # gowu_groups | lex_max | u_softmax | competence | count
    p_mix: float = 0.1
    mix_rule: str = "count"
    u_temp: float = 1.0
    consolidate_k: tuple = (16, 32)      # group-episodes between consolidations
    rollback_rho: float = 0.05
    rollback_w: int = 64
    rollback_k: tuple = (1, 3)
    n_gate: int = 32
    doomed_gate: bool = True             # skip landmarks whose explore phase nearly always dies at once
    doomed_steps: int = 8
    # explore (§6)
    explore_policy: str = "hold_random"  # hold_random | goal_explore
    hold_steps: tuple = (5, 15)
    explore_eps: float = 0.2             # GoWU v4 particle policy
    explore_max_steps: int = 1000
    explore_stall: int = 300
    goal_explore_every: int = 100
    # lives (§7)
    lives_mode: str = "episode_end"      # episode_end | ignore | ram_refill
    exploit_mode: str = "middle_room"
    # evaluation (§8)
    eval_episodes: int = 32
    eval_schedule: str = "best"          # best | chain_by_R
    eval_cap_frames: int = 108_000
    eval_loiter_steps: int = 10_000
    eval_uncapped_limit_steps: int = 250_000
    # infra
    device: str = "auto"
    out_dir: str = "runs/lr_gowu"
    print_every: int = 10
    save_every_sec: float = 900.0
    max_updates: int | None = None


def lexbest(arch: Archive, slots):
    slots = list(slots)
    return max(slots, key=lambda s: (arch.R[s], arch.U[s])) if slots else None


class LRGoWU:
    def __init__(self, cfg: LRConfig):
        self.cfg = cfg
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=1)

        E, M = cfg.n_envs, cfg.n_groups
        assert E % M == 0
        self.E = E
        self.pool = EnvPool(E, cfg.n_workers, cfg.seed, dict(game=cfg.game, sticky_p=cfg.sticky_p),
                            max_rollout_steps=1)
        self.A = self.pool.n_actions
        self.goal_frame = cfg.goal_encoder in ("frame", "both")
        self.goal_ram = cfg.goal_encoder in ("ram", "both")
        self.C = 5 + int(self.goal_frame)
        self.D = cfg.ctx_ancestors
        self.ctx_dim = self.D + 2 + (RAM_GOAL_DIM if self.goal_ram else 0)
        self.net = GoalActorCritic(self.C, self.ctx_dim, self.A, cfg.dense, cfg.head).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.amp = cfg.amp and self.device.type == "cuda"
        self.rnd = RND(1).to(self.device)
        self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=cfg.rnd_lr)
        self.replay = Replay(cfg.rnd_replay, (84, 84), 0, cfg.seed * 31 + 1)
        self.guard = ExploitGuard(cfg.exploit_mode)
        self.arch = Archive(cfg.archive_key, cfg.cell_xy, cfg.topk_traj, cfg.u_margin, cfg.max_demo_gb)

        # per-env episode state
        Tm = cfg.max_episode_steps
        self.ep_frames = np.zeros((E, Tm + 1, 84, 84), np.uint8)
        self.ep_actions = np.zeros((E, Tm), np.uint8)
        self.ep_slots = np.full((E, Tm + 1), -1, np.int32)
        self.ep_R = np.zeros((E, Tm + 1), np.float32)
        self.ep_lrf = np.full((E, Tm + 1), -1, np.int32)
        self.t = np.zeros(E, np.int64)
        self.R = np.zeros(E)
        self.phase = np.zeros(E, np.int8)
        self.goal = np.zeros(E, np.int64)
        self.goal_R = np.zeros(E)
        self.ret_goal = np.zeros(E, np.int64)        # goal of this episode's return phase (stats)
        self.timeout = np.zeros(E, np.int64)
        self.t_phase = np.zeros(E, np.int64)
        self.t_handoff = np.full(E, -1, np.int64)
        self.last_progress = np.zeros(E, np.int64)
        self.last_improve = np.zeros(E, np.int64)
        self.chain = np.full((E, max(self.D, 1)), -1, np.int64)
        self.anc = np.zeros((E, max(self.D, 1)), bool)
        self.goal_ramvec = np.zeros((E, RAM_GOAL_DIM), np.float32)
        self.hold_a = np.zeros(E, np.int64)
        self.hold_left = np.zeros(E, np.int64)
        self.slot = np.zeros(E, np.int64)
        self.prev_ram = [None] * E
        self.ep_best: list[dict] = [dict() for _ in range(E)]
        self.ep_improved: list[set] = [set() for _ in range(E)]
        self.demo_id = np.zeros(E, np.int64)
        self.next_goal = np.full(E, -1, np.int64)
        self.group = np.arange(E) // (E // M)

        # groups
        self.group_goal = [0] * M
        self.group_improved: list[set] = [set() for _ in range(M)]
        self.group_eps = np.zeros(M, np.int64)
        self.group_K = self.rng.integers(cfg.consolidate_k[0], cfg.consolidate_k[1] + 1, M)

        # counters
        self.agent_steps = 0
        self.steps_return = 0
        self.steps_explore = 0
        self.updates = 0
        self.t0 = time.time()
        self._rnd_carry = 0.0
        self._ep_log: list = []
        self._ret_log: list = []
        self._rnd_loss = []
        self.n_rollbacks = 0
        self.n_consolidations = 0
        self.n_exploits = 0
        self.max_episode_R = 0.0
        self._tm = dict(pol=0.0, env=0.0, post=0.0, ppo=0.0, bc=0.0, rnd=0.0)
        self._log = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
        self._last_save = time.time()

        # root landmark and first episodes
        self.pool.reset(range(E))
        rams = self.pool.ram(range(E))
        root = self.arch.slot(rams[0])
        frame0 = self.pool.obs[0, 3].copy()
        self.rnd.norm.update(torch.from_numpy(np.repeat(frame0[None, None], 8, 0)).to(self.device))
        self.arch.update(root, 0.0, float(self._u(frame0[None])[0]), frame0, rams[0], None, 0)
        self.root = root
        self.group_goal = [root] * M
        self._frames0 = 0
        self._updates0 = None
        if cfg.resume_from:
            self._resume(cfg.resume_from)
        for i in range(E):
            self._begin_episode(i, rams[i])

    def _resume(self, d):
        ck = torch.load(os.path.join(d, "ckpt.pt"), map_location=self.device)
        self.net.load_state_dict(ck["net"])
        self.rnd.load_state_dict(ck["rnd"])
        self.agent_steps = self._frames0 = ck["frames"] // 4
        self.updates = self._updates0 = ck["updates"]
        try:   # carry the return/explore frame split so cumulative shares stay comparable
            with open(os.path.join(d, "log.jsonl")) as f:
                last = json.loads(f.readlines()[-1])
            self.steps_return, self.steps_explore = last["frames_return"] // 4, last["frames_explore"] // 4
        except (OSError, ValueError, IndexError, KeyError):
            pass
        with open(os.path.join(d, "landmarks.json")) as f:
            lms = json.load(f)
        z = np.load(os.path.join(d, "landmark_frames.npz"))
        old_frames = dict(zip(z["slots"].tolist(), z["frames"]))
        old_live = sorted(old_frames)   # saved in arch.live() order, i.e. slot order
        arch, c = self.arch, self.cfg.cell_xy
        for k, lm in sorted(zip(old_live, lms), key=lambda t: t[1]["depth"]):
            key = tuple(lm["key"])
            ram = (key[0], key[1], key[-2] * c[0] + c[0] // 2, key[-1] * c[1] + c[1] // 2, 0)
            s = arch.slot_of.get(key)
            if s is None:
                s = arch.slot_of[key] = len(arch.L)
                arch.L.append(None)
                arch.keys.append(key)
                if s >= len(arch.R):
                    n = 2 * len(arch.R)
                    arch.R = np.concatenate([arch.R, np.full(n - len(arch.R), -np.inf)])
                    arch.U = np.concatenate([arch.U, np.zeros(n - len(arch.U))])
                    arch.frames = np.concatenate([arch.frames, np.zeros((n - len(arch.frames), 84, 84), np.uint8)])
            parent = arch.slot_of.get(tuple(lm["parent"])) if lm["parent"] else None
            L = Landmark(key, s, lm["R"], lm["U"], ram, parent, lm["depth"], lm["t_discovered"], lm["t_discovered"])
            L.n_targeted, L.n_reached = lm["n_targeted"], lm["n_reached"]
            L.n_handoff, L.n_quick_death, L.ret_len_ema = lm["n_handoff"], lm["n_quick_death"], lm["ret_len_ema"]
            n_ok = round(lm["success"] * 64) if lm["n_targeted"] else 0
            L.outcomes.extend([1.0] * n_ok + [0.0] * (64 - n_ok) if lm["n_targeted"] >= 64 else [])
            arch.L[s] = L
            arch.R[s], arch.U[s], arch.frames[s] = lm["R"], lm["U"], old_frames[k]
        arch._chain_cache = {}
        live = [s for s in arch.live() if self._eligible(s)]
        best = lexbest(arch, live)
        self.group_goal = [best] * self.cfg.n_groups

    # ------------------------------------------------------------- helpers
    @torch.no_grad()
    def _u(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros(0)
        x = torch.from_numpy(np.ascontiguousarray(frames)).to(self.device)[:, None]
        return self.rnd.uncertainty(x).float().cpu().numpy()

    def _rescore(self, slots):
        slots = list(slots)
        if slots:
            self.arch.U[slots] = self._u(self.arch.frames[slots])

    def _ctx(self, anc, goal_R, R, t, ramvec) -> np.ndarray:
        parts = []
        if self.D:
            parts.append(anc.astype(np.float32))
        parts.append(np.clip((goal_R - R) / 1000.0, 0, 5).astype(np.float32)[:, None])
        parts.append((t / self.cfg.max_episode_steps).astype(np.float32)[:, None])
        if self.goal_ram:
            parts.append(ramvec)
        return np.concatenate(parts, 1)

    def _x(self, ids, goals) -> np.ndarray:
        x = self.pool.obs[ids, :5]
        if self.goal_frame:
            x = np.concatenate([x, self.arch.frames[goals][:, None]], 1)
        return x

    def _forward(self, x, c):
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            logits, v = self.net(x, c)
        return logits.float(), v.float()

    def _boost(self, ids) -> np.ndarray:
        if not self.cfg.entropy_boost:
            return np.ones(len(ids), np.float32)
        stall = np.maximum(0, self.t[ids] - self.last_progress[ids] - self.cfg.boost_after)
        return (1 + (stall * self.cfg.boost_ef) ** self.cfg.boost_ep).astype(np.float32)

    def _eligible(self, s) -> bool:
        lm = self.arch.L[s]
        if lm is None:
            return False
        if s == self.root:
            return True
        # PLAN §5: an unreachable landmark is skipped only if it has no demos to learn the route from
        if lm.n_targeted >= self.cfg.n_gate and lm.n_reached == 0 and not lm.demos:
            return False
        if self.cfg.doomed_gate and lm.n_handoff >= 16 and lm.n_quick_death > 0.9 * lm.n_handoff:
            return False
        return True

    def _eligible_fallback(self, g):
        """g if eligible, else its nearest eligible ancestor (else the root)."""
        if self._eligible(g):
            return g
        for a in self.arch.chain(g, 256):
            if self._eligible(a):
                return a
        return self.root

    # ------------------------------------------------------------ episodes
    def _set_goal(self, i, g):
        lm = self.arch.L[g]
        self.goal[i], self.goal_R[i] = g, lm.R
        ch = self.arch.chain(g, self.D) if self.D else []
        self.chain[i] = -1
        self.chain[i, :len(ch)] = ch
        self.anc[i] = False
        if self.D:
            self.anc[i] = np.isin(self.chain[i], list(self.ep_best[i].keys())) & (self.chain[i] >= 0)
        self.goal_ramvec[i] = ram_goal_vec(lm.ram)

    def _begin_episode(self, i, ram):
        cfg = self.cfg
        g = int(self.next_goal[i]) if self.next_goal[i] >= 0 else self.group_goal[self.group[i]]
        if not self._eligible(g):
            g = self._eligible_fallback(self.group_goal[self.group[i]])
        s = self.arch.slot(ram)
        self.t[i], self.R[i] = 0, 0.0
        self.ep_frames[i, 0] = self.pool.obs[i, 3]
        self.ep_slots[i, 0] = s
        self.ep_R[i, 0] = 0
        self.ep_lrf[i, 0] = -1
        self.slot[i] = s
        self.prev_ram[i] = ram
        self.ep_best[i] = {s: 0.0}
        self.ep_improved[i] = set()
        self.demo_id[i] = self.arch.new_demo()
        self.last_progress[i] = self.last_improve[i] = 0
        self.t_handoff[i] = -1
        self.replay.add(self.pool.obs[i:i + 1, 3])
        self._set_goal(i, g)
        self.ret_goal[i] = g
        lm = self.arch.L[g]
        lm.n_targeted += 1
        if s == g and self.goal_R[i] <= 0:      # goal is the start cell: nothing to return to
            lm.n_reached += 1
            self._handoff(i, success=None)
        else:
            self.phase[i] = RETURN
            self.t_phase[i] = 0
            L = lm.ret_len_ema or (lm.demos[0][1] if lm.demos else cfg.ret_default_len)
            self.timeout[i] = int(cfg.ret_mu * L + cfg.ret_b)

    def _handoff(self, i, success):
        cfg = self.cfg
        g = int(self.ret_goal[i])
        lm = self.arch.L[g]
        if success is not None:
            lm.outcomes.append(1.0 if success else 0.0)
            if success:
                lm.n_reached += 1
                n = int(self.t[i])
                lm.ret_len_ema = n if lm.ret_len_ema is None else 0.9 * lm.ret_len_ema + 0.1 * n
            self._ret_log.append((bool(success), int(self.t[i]), lm.depth))
            self._maybe_rollback(g)
        if success is not False:
            lm.n_handoff += 1
        self.phase[i] = EXPLORE
        self.t_phase[i] = self.t[i]
        self.t_handoff[i] = self.t[i] if success is not False else -1
        self.last_improve[i] = self.t[i]
        self.hold_left[i] = 0
        if cfg.explore_policy == "goal_explore":
            self._set_goal(i, self._sample(cfg.mix_rule))

    def _end_episode(self, i, died: bool):
        cfg = self.cfg
        n = self.t[i]
        if died and self.t_handoff[i] >= 0 and n - self.t_handoff[i] <= cfg.doomed_steps:
            self.arch.L[int(self.ret_goal[i])].n_quick_death += 1
        self.arch.materialize(int(self.demo_id[i]), self.ep_frames[i], self.ep_actions[i], self.ep_R[i],
                              self.ep_lrf[i], self.ep_slots[i])
        self._ep_log.append((float(self.R[i]), int(n), bool(died)))
        self.max_episode_R = max(self.max_episode_R, float(self.R[i]))
        m = self.group[i]
        self.next_goal[i] = self._choose_goal(i, died)
        self.group_eps[m] += 1
        if self.group_eps[m] >= self.group_K[m]:
            self._consolidate(m)

    # ----------------------------------------------------------- selection
    def _sample(self, rule):
        live = [s for s in self.arch.live() if self._eligible(s)]
        if rule == "count":
            w = np.array([1 / math.sqrt(self.arch.L[s].n_targeted + 1) for s in live])
        elif rule == "lex_max":
            self._rescore(live)
            return lexbest(self.arch, live)
        elif rule == "u_softmax":
            self._rescore(live)
            top = self.arch.R[live].max()
            live = [s for s in live if self.arch.R[s] == top]
            u = self.arch.U[live] / self.cfg.u_temp
            w = np.exp((u - u.max()) / max(u.std(), 1e-8))
        elif rule == "competence":
            self._rescore(live)
            p = np.array([self.arch.L[s].success() for s in live])
            w = self.arch.U[live] * (p * (1 - p) + 0.05)
        else:
            raise ValueError(rule)
        w = np.asarray(w, float)
        return int(live[self.rng.choice(len(live), p=w / w.sum())])

    def _group_winner(self, m):
        cands = {s for s in self.group_improved[m] if self._eligible(s)} | {self.group_goal[m]}
        self._rescore(cands)
        return lexbest(self.arch, cands)

    def _choose_goal(self, i, died):
        cfg = self.cfg
        m = self.group[i]
        if cfg.p_mix > 0 and self.rng.random() < cfg.p_mix:
            return self._sample(cfg.mix_rule)
        if cfg.select_rule != "gowu_groups":
            return self._sample(cfg.select_rule)
        alive = [s for s in self.ep_improved[i] if self._eligible(s)]
        if not died and alive:          # a surviving particle continues from its own frontier
            self._rescore(alive)
            return lexbest(self.arch, alive)
        return self._group_winner(m)    # dead particles are reset to the winner

    def _consolidate(self, m):
        cfg = self.cfg
        self.group_goal[m] = self._group_winner(m)
        self.group_improved[m] = set()
        self.group_eps[m] = 0
        self.group_K[m] = self.rng.integers(cfg.consolidate_k[0], cfg.consolidate_k[1] + 1)
        self.next_goal[self.group == m] = self.group_goal[m]
        self.n_consolidations += 1
        if cfg.select_rule == "gowu_groups":     # global sync: groups with lower R reset to the global winner
            self._rescore(set(self.group_goal))
            gw = lexbest(self.arch, self.group_goal)
            for k in range(cfg.n_groups):
                if self.arch.R[self.group_goal[k]] < self.arch.R[gw]:
                    self.group_goal[k] = gw
                    self.next_goal[self.group == k] = gw

    def _maybe_rollback(self, g):
        cfg = self.cfg
        lm = self.arch.L[g]
        if len(lm.outcomes) < cfg.rollback_w or lm.success() >= cfg.rollback_rho:
            return
        for m in range(cfg.n_groups):
            if self.group_goal[m] == g:
                k = int(self.rng.integers(cfg.rollback_k[0], cfg.rollback_k[1] + 1))
                self.group_goal[m] = self.arch.ancestor(g, k)
                self.next_goal[self.group == m] = self.group_goal[m]
                self.n_rollbacks += 1
        lm.outcomes.clear()

    # ------------------------------------------------------------- rollout
    def _explore_actions(self):
        cfg = self.cfg
        a = self.hold_a.copy()
        new = self.hold_left <= 0
        n = int(new.sum())
        if n:
            self.hold_a[new] = self.rng.integers(0, self.A, n)
            self.hold_left[new] = self.rng.integers(cfg.hold_steps[0], cfg.hold_steps[1] + 1, n)
            a[new] = self.hold_a[new]
        self.hold_left -= 1
        noise = self.rng.random(self.E) < cfg.explore_eps
        a[noise] = self.rng.integers(0, self.A, int(noise.sum()))
        return a

    @torch.no_grad()
    def collect(self):
        cfg, E, T = self.cfg, self.E, self.cfg.unroll
        dev = self.device
        buf_x = torch.zeros((T, E, self.C, 84, 84), dtype=torch.uint8, device=dev)
        buf_c = torch.zeros((T, E, self.ctx_dim), device=dev)
        buf_a = torch.zeros((T, E), dtype=torch.long, device=dev)
        buf_logp = torch.zeros((T, E), device=dev)
        buf_scale = torch.ones((T, E), device=dev)
        buf_v = np.zeros((T, E), np.float32)
        rew = np.zeros((T, E), np.float32)
        done = np.zeros((T, E), np.float32)
        valid = np.zeros((T, E), bool)
        all_ids = np.arange(E)
        tm = self._tm
        for t in range(T):
            t0 = time.perf_counter()
            acts = self._explore_actions()
            ret = self.phase == RETURN
            pol = ret | ((self.phase == EXPLORE) & (cfg.explore_policy == "goal_explore"))
            ids = np.where(pol)[0]
            if len(ids):
                x = torch.from_numpy(self._x(ids, self.goal[ids])).to(dev)
                c = torch.from_numpy(self._ctx(self.anc[ids], self.goal_R[ids], self.R[ids], self.t[ids],
                                               self.goal_ramvec[ids])).to(dev)
                logits, v = self._forward(x, c)
                scale = torch.from_numpy(self._boost(ids)).to(dev)
                dist = torch.distributions.Categorical(logits=logits / scale[:, None])
                a = dist.sample()
                acts[ids] = a.cpu().numpy()
                rm = ret[ids]
                rid = torch.from_numpy(ids[rm]).to(dev)
                rmt = torch.from_numpy(rm).to(dev)
                buf_x[t, rid] = x[rmt]
                buf_c[t, rid] = c[rmt]
                buf_a[t, rid] = a[rmt]
                buf_logp[t, rid] = dist.log_prob(a)[rmt]
                buf_scale[t, rid] = scale[rmt]
                buf_v[t, ids[rm]] = v[rmt].cpu().numpy()
                valid[t, ids[rm]] = True
            t1 = time.perf_counter()
            tm["pol"] += t1 - t0
            self.steps_return += int(ret.sum())
            self.steps_explore += int(E - ret.sum())
            r, ll, go, rams = self.pool.step(all_ids, acts)
            self.agent_steps += E
            t2 = time.perf_counter()
            tm["env"] += t2 - t1
            rew[t], done[t], trunc, resets = self._post_step(acts, r, ll, go, rams)
            if len(trunc):   # time-limit truncation of the return phase: bootstrap from s_{t+1}
                _, v_last = self._forward(torch.from_numpy(self._x(trunc, self.goal[trunc])).to(dev),
                                          torch.from_numpy(self._ctx(self.anc[trunc], self.goal_R[trunc],
                                                                     self.R[trunc], self.t[trunc],
                                                                     self.goal_ramvec[trunc])).to(dev))
                rew[t, trunc] += cfg.gamma * v_last.cpu().numpy()
            if resets:
                self.pool.reset(resets)
                rr = self.pool.ram(resets)
                for i in resets:
                    self._begin_episode(i, rr[i])
            if self.updates < 50 or t % 4 == 0:
                self.rnd.norm.update(torch.from_numpy(self.pool.obs[:, 3]).to(dev)[:, None])
            tm["post"] += time.perf_counter() - t2
        # bootstrap the rollout boundary for envs still returning
        v_next = np.zeros(E, np.float32)
        ids = np.where(valid[-1] & (done[-1] == 0))[0]
        if len(ids):
            _, v = self._forward(torch.from_numpy(self._x(ids, self.goal[ids])).to(dev),
                                 torch.from_numpy(self._ctx(self.anc[ids], self.goal_R[ids], self.R[ids], self.t[ids],
                                                            self.goal_ramvec[ids])).to(dev))
            v_next[ids] = v.cpu().numpy()
        return dict(x=buf_x, c=buf_c, a=buf_a, logp=buf_logp, scale=buf_scale, v=buf_v, r=rew, d=done,
                    valid=valid, v_next=v_next)

    def _post_step(self, acts, r, ll, go, rams):
        cfg, arch, E = self.cfg, self.arch, self.E
        tn = self.t + 1
        rows = np.arange(E)
        frames = self.pool.obs[:, 3]
        self.ep_actions[rows, self.t] = acts
        self.ep_frames[rows, tn] = frames
        self.replay.add(frames)

        # rewards, exploits, deaths
        exploit = np.zeros(E, bool)
        for i in np.where(r > 0)[0]:
            if self.guard.mode == "middle_room" and self.guard.middle_room_event(r[i], self.prev_ram[i], rams[i]):
                exploit[i] = True
                self.n_exploits += 1
        r = np.where(exploit, 0.0, r)
        self.R += r
        rewarded = r != 0
        self.ep_lrf[rows, tn] = np.where(rewarded, tn, self.ep_lrf[rows, self.t])
        self.ep_R[rows, tn] = self.R
        if cfg.lives_mode == "episode_end":
            dead = ll | go | exploit
        else:
            dead = go | exploit
            if cfg.lives_mode == "ram_refill" and ll.any():
                self.pool.set_lives([(int(i), 6) for i in np.where(ll)[0]])
        ret_fail_life = ll & ~dead      # infinite lives: a life loss during return is a failure

        slots = np.array([arch.slot(rams[i]) for i in range(E)])
        self.ep_slots[rows, tn] = slots

        # uncertainty: current states and the incumbents they tie with (re-scored with the current RND)
        Rs = arch.R[slots]
        tie_slots = np.unique(slots[(Rs == self.R) & ~dead])
        u_all = self._u(np.concatenate([frames, arch.frames[tie_slots]]))
        U = u_all[:E]
        arch.U[tie_slots] = u_all[E:]

        # archive updates (lex-max (R, U) per cell) and hindsight arrivals; only flagged envs are visited
        cand = ~dead & ((Rs < self.R) | ((Rs == self.R) & (U > (1 + arch.u_margin) * arch.U[slots])))
        moved = ~dead & ((slots != self.slot) | rewarded)
        for i in np.where(cand | moved)[0]:
            s = int(slots[i])
            if cand[i] and arch.is_candidate(s, self.R[i], U[i]):   # re-check: envs can share a cell
                prev = int(self.slot[i])
                parent = prev if prev != s and arch.exists(prev) else None
                kind = arch.update(s, float(self.R[i]), float(U[i]), frames[i], rams[i], parent, self.agent_steps)
                self.ep_improved[i].add(s)
                self.group_improved[self.group[i]].add(s)
                self.last_improve[i] = tn[i]
                if kind == "new" and self.phase[i] == RETURN:
                    self.last_progress[i] = tn[i]
            best = self.ep_best[i].get(s)
            if best is None or self.R[i] > best:
                if best is None:
                    self.last_progress[i] = tn[i]
                self.ep_best[i][s] = self.R[i]
                arch.offer_demo(s, float(self.R[i]), int(tn[i]), int(self.demo_id[i]))

        # ancestor progress (vectorized)
        if self.D:
            hit = (self.chain == slots[:, None]) & (self.chain >= 0) & ~self.anc
            n_anc = hit.sum(1)
            self.anc |= hit
            self.last_progress = np.where(n_anc > 0, tn, self.last_progress)
        else:
            n_anc = np.zeros(E, np.int64)

        self.t = tn
        self.slot = slots
        self.prev_ram = rams

        ret = self.phase == RETURN
        reached = ret & (slots == self.goal) & (self.R >= self.goal_R) & ~dead
        rew = np.where(ret, cfg.c_ext * np.sign(r) + cfg.c_anc * n_anc - cfg.c_step + cfg.c_goal * reached,
                       0.0).astype(np.float32)
        done = np.zeros(E, np.float32)
        ring_full = tn >= cfg.max_episode_steps
        ret_fail = ret & ((tn - self.t_phase >= self.timeout) | (tn - self.last_progress >= cfg.ret_stall)
                          | ring_full | ret_fail_life)
        exp_end = ~ret & ((tn - self.t_phase >= cfg.explore_max_steps) | (tn - self.last_improve >= cfg.explore_stall)
                          | ring_full)
        flagged = dead | reached | ret_fail | exp_end
        if cfg.explore_policy == "goal_explore":
            flagged |= ~ret & (((slots == self.goal) & (self.R >= self.goal_R))
                               | ((tn - self.t_phase) % cfg.goal_explore_every == 0))
        trunc, resets = [], []
        for i in np.where(flagged)[0]:
            if ret[i]:
                if dead[i]:
                    done[i] = 1
                    self._handoff_fail_dead(i)
                    resets.append(i)
                elif reached[i]:
                    done[i] = 1
                    self._handoff(i, success=True)
                elif ret_fail[i]:
                    done[i] = 1
                    trunc.append(i)
                    self._handoff(i, success=False)
                    if ring_full[i]:
                        resets.append(i)
                        self._end_episode(i, died=False)
                continue
            if dead[i]:
                self._end_episode(i, died=True)
                resets.append(i)
            elif exp_end[i]:
                self._end_episode(i, died=False)
                resets.append(i)
            else:
                self._set_goal(i, self._sample(cfg.mix_rule))
        return rew, done, np.array(trunc, np.int64), resets

    def _handoff_fail_dead(self, i):
        lm = self.arch.L[int(self.ret_goal[i])]
        lm.outcomes.append(0.0)
        self._ret_log.append((False, int(self.t[i]), lm.depth))
        self._maybe_rollback(int(self.ret_goal[i]))
        self._end_episode(i, died=True)

    # ------------------------------------------------------------ learning
    def _ppo(self, b):
        cfg = self.cfg
        valid = b["valid"]
        n_valid = int(valid.sum())
        if n_valid < 256:
            return dict(ppo_n=n_valid)
        T, E = valid.shape
        v, r, d = b["v"], b["r"], b["d"]
        adv = np.zeros((T, E), np.float32)
        last = np.zeros(E, np.float32)
        for t in reversed(range(T)):
            nv = b["v_next"] if t == T - 1 else v[t + 1]
            nonterm = 1.0 - d[t]
            delta = r[t] + cfg.gamma * nv * nonterm - v[t]
            last = np.where(valid[t], delta + cfg.gamma * cfg.gae_lambda * nonterm * last, 0.0)
            adv[t] = last
        ret = adv + v
        tt, ee = np.where(valid)
        ti, ei = torch.from_numpy(tt).to(self.device), torch.from_numpy(ee).to(self.device)
        bx, bc, ba = b["x"][ti, ei], b["c"][ti, ei], b["a"][ti, ei]
        blogp, bscale = b["logp"][ti, ei], b["scale"][ti, ei]
        badv = torch.from_numpy(adv[tt, ee]).to(self.device)
        bret = torch.from_numpy(ret[tt, ee]).to(self.device)
        N = len(tt)
        n_mb = max(1, min(cfg.minibatches, N // 64))
        st = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, bc_joint=0.0)
        k = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.device)
            for idx in perm.chunk(n_mb):
                with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
                    logits, val = self.net(bx[idx], bc[idx])
                logits, val = logits.float() / bscale[idx, None], val.float()
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(ba[idx])
                ratio = (logp - blogp[idx]).exp()
                a_mb = badv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_mb).mean()
                vf = F.mse_loss(val, bret[idx])
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                if cfg.bc_schedule == "joint":
                    bcl = self._bc_loss()
                    if bcl is not None:
                        loss = loss + cfg.bc_coef * bcl
                        st["bc_joint"] += bcl.item()
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                st["pg"] += pg.item()
                st["vf"] += vf.item()
                st["ent"] += ent.item()
                st["kl"] += (blogp[idx] - logp).mean().item()
                k += 1
        out = {kk: round(vv / k, 5) for kk, vv in st.items()}
        out["ppo_n"] = N
        return out

    def _bc_sample(self, B):
        """Hindsight goal-conditioned BC batch: (x, ctx, action) from stored s0 -> landmark prefixes."""
        cfg, arch = self.cfg, self.arch
        cands, w = [], []
        for lm in arch.L:
            if lm is None:
                continue
            ds = [(t, d) for R_, t, d in lm.demos if R_ >= lm.R and arch.demos.get(d) is not None
                  and arch.demos[d].frames is not None]
            if not ds:
                continue
            p = lm.success()
            if cfg.bc_loss == "filtered" and len(lm.outcomes) >= 8 and p >= cfg.bc_tau:
                continue
            cands.append((lm, ds))
            w.append(1.0 - p + 1e-3)
        if not cands:
            return None
        w = np.array(w)
        pick = self.rng.choice(len(cands), B, p=w / w.sum())
        x = np.zeros((B, self.C, 84, 84), np.uint8)
        anc = np.zeros((B, max(self.D, 1)), bool)
        goal_R, R, tt = np.zeros(B), np.zeros(B), np.zeros(B)
        ramvec = np.zeros((B, RAM_GOAL_DIM), np.float32)
        a = np.zeros(B, np.int64)
        for k, ci in enumerate(pick):
            lm, ds = cands[ci]
            t_arr, did = ds[self.rng.integers(len(ds))]
            d = arch.demos[did]
            j = int(self.rng.integers(0, t_arr))
            idx = np.clip(np.arange(j - 3, j + 1), 0, None)
            x[k, :4] = d.frames[idx]
            if d.lrf[j] >= 0:
                x[k, 4] = d.frames[d.lrf[j]]
            if self.goal_frame:
                x[k, 5] = arch.frames[lm.slot]
            if self.D:
                ch = arch.chain(lm.slot, self.D)
                anc[k, :len(ch)] = [d.first_visit.get(s, 1 << 30) <= j for s in ch]
            goal_R[k], R[k], tt[k] = lm.R, d.R[j], j
            ramvec[k] = ram_goal_vec(lm.ram)
            a[k] = d.actions[j]
        c = self._ctx(anc, goal_R, R, tt, ramvec)
        return (torch.from_numpy(x).to(self.device), torch.from_numpy(c).to(self.device),
                torch.from_numpy(a).to(self.device))

    def _bc_loss(self, B=None):
        b = self._bc_sample(B or self.cfg.bc_batch)
        if b is None:
            return None
        x, c, a = b
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            logits = self.net.policy(x, c)
        logits = logits.float()
        loss = F.cross_entropy(logits, a)
        self._bc_acc = (logits.argmax(-1) == a).float().mean().item()
        return loss

    def _bc(self):
        cfg = self.cfg
        if cfg.bc_schedule != "alternate":
            return {}
        losses = []
        for _ in range(cfg.n_bc):
            loss = self._bc_loss()
            if loss is None:
                break
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
            self.opt.step()
            losses.append(loss.item())
        return dict(bc_loss=round(float(np.mean(losses)), 4) if losses else None,
                    bc_acc=round(self._bc_acc, 3) if losses else None)

    def _train_rnd(self, new_steps):
        cfg = self.cfg
        self._rnd_carry += new_steps * cfg.rnd_updates_per_1k_steps / 1000
        n = int(self._rnd_carry)
        self._rnd_carry -= n
        if self.replay.size < cfg.rnd_batch:
            return
        for _ in range(n):
            _, x = self.replay.sample(cfg.rnd_batch)
            loss = self.rnd.error(torch.from_numpy(x).to(self.device)[:, None]).mean()
            self.rnd_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.rnd_opt.step()
            self._rnd_loss.append(loss.item())

    # ---------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, n, schedule="best", cap_frames=108_000, loiter_steps=10_000, limit_steps=250_000):
        """π_R alone from env.reset(), all lives, sticky actions (PLAN §8). Goal: the lex-max (R, U) landmark
        (best), or after each arrival the next unreached landmark in descending R (chain_by_R)."""
        arch, cfg = self.arch, self.cfg
        n = min(n, self.E)
        ids = np.arange(n)
        live = arch.live()
        self._rescore(live)
        order = sorted(live, key=lambda s: (-arch.R[s], -arch.U[s]))
        self.pool.reset(ids)
        rams = self.pool.ram(ids)
        goal = np.full(n, order[0])
        R, t, last_r = np.zeros(n), np.zeros(n, np.int64), np.zeros(n, np.int64)
        chain = np.full((n, max(self.D, 1)), -1, np.int64)
        anc = np.zeros((n, max(self.D, 1)), bool)
        prev = [rams[i] for i in ids]
        visited = [set() for _ in ids]
        reached_goals = [set() for _ in ids]
        active = np.ones(n, bool)

        def set_goal(i, g):
            goal[i] = g
            ch = arch.chain(g, self.D) if self.D else []
            chain[i] = -1
            chain[i, :len(ch)] = ch
            anc[i] = np.isin(chain[i], list(visited[i])) & (chain[i] >= 0)

        for i in ids:
            set_goal(i, order[0])
        max_steps = limit_steps if cap_frames is None else cap_frames // 4
        while active.any():
            a_ids = np.where(active)[0]
            gR = arch.R[goal[a_ids]]
            ramvec = np.stack([ram_goal_vec(arch.L[g].ram) for g in goal[a_ids]])
            x = torch.from_numpy(self._x(a_ids, goal[a_ids])).to(self.device)
            c = torch.from_numpy(self._ctx(anc[a_ids], gR, R[a_ids], np.minimum(t[a_ids], cfg.max_episode_steps),
                                           ramvec)).to(self.device)
            logits, _ = self._forward(x, c)
            act = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            r, ll, go, rr = self.pool.step(a_ids, act)
            for k, i in enumerate(a_ids):
                ri = r[k]
                if ri > 0 and self.guard.mode == "middle_room" and self.guard.middle_room_event(ri, prev[i], rr[k]):
                    ri = 0.0
                prev[i] = rr[k]
                R[i] += ri
                t[i] += 1
                if ri != 0:
                    last_r[i] = t[i]
                s = arch.slot_of.get(cell_key(rr[k], arch.key_mode, arch.cell_xy))
                if s is not None:
                    visited[i].add(s)
                    anc[i] |= (chain[i] == s) & (chain[i] >= 0)
                    if s == goal[i] and R[i] >= arch.R[goal[i]]:
                        reached_goals[i].add(int(goal[i]))
                        if schedule == "chain_by_R":
                            nxt = [g for g in order if g not in reached_goals[i] and arch.R[g] > R[i]]
                            if nxt:
                                set_goal(i, nxt[-1])
                if go[k] or t[i] >= max_steps or t[i] - last_r[i] >= loiter_steps:
                    active[i] = False
        return dict(schedule=schedule, cap_frames=cap_frames, n=int(n), mean=float(R.mean()),
                    std=float(R.std()), max=float(R.max()), scores=R.tolist(),
                    goal_reached=float(np.mean([order[0] in g for g in reached_goals])), mean_steps=float(t.mean()))

    # ----------------------------------------------------------------- run
    @property
    def frames(self):
        return self.agent_steps * 4

    def log(self, st):
        arch = self.arch
        live = arch.live()
        dt = time.time() - self.t0
        eps, rets = self._ep_log, self._ret_log
        self._ep_log, self._ret_log = [], []
        rl = [x for x in rets]
        rnd_loss = float(np.mean(self._rnd_loss)) if self._rnd_loss else None
        self._rnd_loss = []
        best = lexbest(arch, live)
        rooms = sorted({(arch.keys[s][0], arch.keys[s][1]) for s in live})
        rec = dict(update=self.updates, frames=self.frames, time=round(dt, 1), fps=round((self.frames - 4 * self._frames0) / dt),
                   frames_return=self.steps_return * 4, frames_explore=self.steps_explore * 4,
                   n_landmarks=len(live), rooms=len(rooms), max_level=max(r[0] for r in rooms),
                   max_R=float(arch.R[live].max()), max_episode_R=self.max_episode_R,
                   best=dict(key=list(map(int, arch.keys[best])), R=float(arch.R[best]),
                             depth=arch.L[best].depth),
                   episodes=len(eps), ep_R=round(float(np.mean([e[0] for e in eps])), 1) if eps else None,
                   ep_len=round(float(np.mean([e[1] for e in eps])), 1) if eps else None,
                   ret_attempts=len(rl), ret_success=round(float(np.mean([x[0] for x in rl])), 3) if rl else None,
                   ret_depth=round(float(np.mean([x[2] for x in rl])), 2) if rl else None,
                   ret_succ_len=round(float(np.mean([x[1] for x in rl if x[0]])), 1) if any(x[0] for x in rl) else None,
                   group_goals=[[*map(int, arch.keys[g]), float(arch.R[g])] for g in self.group_goal],
                   rnd_loss=round(rnd_loss, 6) if rnd_loss is not None else None,
                   demos=len(arch.demos), demo_gb=round(arch.demo_bytes / 2**30, 2),
                   rollbacks=self.n_rollbacks, consolidations=self.n_consolidations, exploits=self.n_exploits,
                   rooms_list=rooms, timing={k: round(v, 2) for k, v in self._tm.items()}, **st)
        self._tm = {k: 0.0 for k in self._tm}
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()
        return rec

    def save(self):
        arch = self.arch
        torch.save(dict(net=self.net.state_dict(), rnd=self.rnd.state_dict(), frames=self.frames,
                        updates=self.updates, C=self.C, ctx_dim=self.ctx_dim, n_actions=self.A),
                   os.path.join(self.cfg.out_dir, "ckpt.pt.tmp"))
        os.replace(os.path.join(self.cfg.out_dir, "ckpt.pt.tmp"), os.path.join(self.cfg.out_dir, "ckpt.pt"))
        lms = [dict(key=list(map(int, lm.key)), R=lm.R, U=float(arch.U[lm.slot]), depth=lm.depth,
                    parent=None if lm.parent is None else list(map(int, arch.keys[lm.parent])),
                    n_targeted=lm.n_targeted, n_reached=lm.n_reached, success=round(lm.success(), 3),
                    ret_len_ema=lm.ret_len_ema, n_demos=len(lm.demos),
                    demo_len=lm.demos[0][1] if lm.demos else None, n_handoff=lm.n_handoff,
                    n_quick_death=lm.n_quick_death, t_discovered=lm.t_discovered)
               for lm in arch.L if lm is not None]
        with open(os.path.join(self.cfg.out_dir, "landmarks.json"), "w") as f:
            json.dump(lms, f, default=float)
        np.savez_compressed(os.path.join(self.cfg.out_dir, "landmark_frames.npz"),
                            frames=arch.frames[arch.live()], slots=np.array(arch.live()))

    def run(self, verbose=True):
        cfg = self.cfg
        deadline = None if cfg.max_hours is None else self.t0 + cfg.max_hours * 3600
        try:
            while self.frames < cfg.total_frames and (cfg.max_updates is None or self.updates < cfg.max_updates):
                if deadline is not None and time.time() > deadline:
                    break
                steps0 = self.agent_steps
                if self._updates0 is not None and cfg.resume_warmup_updates:
                    w = min(1.0, (self.updates - self._updates0 + 1) / cfg.resume_warmup_updates)
                    for g in self.opt.param_groups:
                        g["lr"] = cfg.lr * w
                b = self.collect()
                t0 = time.perf_counter()
                st = self._ppo(b)
                del b
                t1 = time.perf_counter()
                st.update(self._bc())
                t2 = time.perf_counter()
                self._train_rnd(self.agent_steps - steps0)
                tm = self._tm
                tm["ppo"] += t1 - t0
                tm["bc"] += t2 - t1
                tm["rnd"] += time.perf_counter() - t2
                self.updates += 1
                rec = self.log(st)
                if verbose and self.updates % cfg.print_every == 0:
                    print(f"upd {rec['update']:5d} | {rec['frames']/1e6:8.2f}M fr | {rec['fps']:6d} fps | "
                          f"maxR {rec['max_R']:7.0f} | rooms {rec['rooms']:2d} L{rec['max_level']} | "
                          f"lm {rec['n_landmarks']:5d} | ret {rec['ret_success']} d{rec['ret_depth']} | "
                          f"retfr {rec['frames_return']/max(rec['frames'],1):.2f} | bc {rec.get('bc_loss')} "
                          f"{rec.get('bc_acc')} | ent {rec.get('ent')} | rnd {rec['rnd_loss']}", flush=True)
                if time.time() - self._last_save > cfg.save_every_sec:
                    self.save()
                    self._last_save = time.time()
            self.save()
        except KeyboardInterrupt:
            self.save()
        return self

    def final_eval(self):
        cfg = self.cfg
        res = [self.evaluate(cfg.eval_episodes, cfg.eval_schedule, cfg.eval_cap_frames, cfg.eval_loiter_steps),
               self.evaluate(cfg.eval_episodes, cfg.eval_schedule, None, cfg.eval_loiter_steps,
                             cfg.eval_uncapped_limit_steps)]
        with open(os.path.join(cfg.out_dir, "eval.json"), "w") as f:
            json.dump(res, f, indent=1)
        return res
