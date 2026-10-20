"""Backward-algorithm curriculum (Salimans & Chen 2018, as modified in GoWU App. G.1-G.3).

Indices below are *demo node indices*. GoWU demos are lineage-tree paths with nodes spaced 5-15 agent
steps apart, so Δ = 25 means "25 checkpoints" (the paper does not give the unit; see docs).
"""
from __future__ import annotations

import gzip
import pickle
from dataclasses import dataclass, field

import numpy as np


def load_demo(path: str) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


@dataclass
class Demo:
    ckpts: list
    steps: np.ndarray        # agent steps from s0 at each node
    clipped_cum: np.ndarray  # cumulative sign-clipped reward at each node (MR/Venture clipping)
    raw_cum: np.ndarray

    @classmethod
    def from_dict(cls, d: dict, reward_mode: str = "clip"):
        per_node = []
        for evs in d["events"]:
            if reward_mode == "clip":
                per_node.append(sum(float(np.sign(r)) for _, r, _, _ in evs))
            else:  # "scale": Pitfall-style 0.001 scaling
                per_node.append(sum(0.001 * r for _, r, _, _ in evs))
        return cls(d["ckpts"], np.asarray(d["steps"]), np.cumsum(per_node), np.asarray(d["R_cum"], float))

    def __len__(self):
        return len(self.ckpts)


@dataclass
class Segment:
    demo: int
    end: int                 # T_end (node index of the k-th selected reward event)
    t_curr: int
    window: int
    results: list = field(default_factory=list)
    ema_len: float | None = None
    solved: bool = False
    n_episodes: int = 0


@dataclass
class CurriculumConfig:
    k_max: int = 10                       # segments per demo (Atari)
    window: int = 25                      # Δ
    dec_mult: tuple = (0.25, 0.75)        # (α_dec, β_dec)
    inc_mult: tuple = (0.5, 1.0)          # (α_inc, β_inc)
    s_req: float = 0.2
    s_req_begin: float = 0.95
    n_update: int = 32
    eps_tol: float = 0.0
    ema_alpha: float = 0.9
    len_mult: float = 2.0                 # μ
    len_buffer: int = 500                 # b
    extra_log_max: float = 7.0            # after success keep playing ~exp(U(0, 7)) steps
    p_solved: float = 0.1                 # keep rehearsing solved segments (not specified in paper)


class Curriculum:
    def __init__(self, demos: list[Demo], cfg: CurriculumConfig, seed: int = 0):
        self.demos = demos
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.segments: list[Segment] = []
        for di, d in enumerate(demos):
            ev = [i for i in range(1, len(d)) if d.clipped_cum[i] != d.clipped_cum[i - 1]]
            if not ev:
                continue
            if len(ev) > cfg.k_max:  # evenly downsample to K_max boundaries
                ev = [ev[int(round(j))] for j in np.linspace(0, len(ev) - 1, cfg.k_max)]
            for end in ev:
                self.segments.append(Segment(di, end, end - 1, cfg.window))
        assert self.segments, "no reward events in demos"

    # ---------------------------------------------------------- sampling
    def all_solved(self) -> bool:
        return all(s.solved for s in self.segments)

    def sample(self):
        """Returns (segment index, start node, target clipped return, demo length estimate)."""
        uns = [i for i, s in enumerate(self.segments) if not s.solved]
        sol = [i for i, s in enumerate(self.segments) if s.solved]
        if not uns or (sol and self.rng.random() < self.cfg.p_solved):
            si = int(self.rng.choice(sol))
        else:
            si = int(self.rng.choice(uns))
        s = self.segments[si]
        d = self.demos[s.demo]
        lo = max(0, s.t_curr - s.window)
        t0 = int(self.rng.integers(lo, s.t_curr + 1))
        target = d.clipped_cum[s.end] - d.clipped_cum[t0]
        demo_len = int(d.steps[s.end] - d.steps[t0])
        return si, t0, float(target), demo_len

    def max_steps(self, si: int, demo_len: int) -> int:
        s = self.segments[si]
        L = s.ema_len if s.ema_len is not None else demo_len
        return int(self.cfg.len_mult * L + self.cfg.len_buffer)

    def extra_steps(self) -> int:
        return int(np.exp(self.rng.uniform(0, self.cfg.extra_log_max)))

    # ------------------------------------------------------------ update
    def report(self, si: int, success: bool, length: int) -> None:
        cfg = self.cfg
        s = self.segments[si]
        s.n_episodes += 1
        if success:
            s.ema_len = length if s.ema_len is None else cfg.ema_alpha * s.ema_len + (1 - cfg.ema_alpha) * length
        s.results.append(bool(success))
        if len(s.results) < cfg.n_update:
            return
        S = float(np.mean(s.results))
        s.results = []
        if s.t_curr == 0:
            if S >= cfg.s_req_begin:
                s.solved = True
            elif S < cfg.s_req:
                s.t_curr = min(s.end - 1, s.t_curr + self._step(cfg.inc_mult, s.window))
            return
        if S >= cfg.s_req:
            s.t_curr = max(0, s.t_curr - self._step(cfg.dec_mult, s.window))
        else:
            s.t_curr = min(s.end - 1, s.t_curr + self._step(cfg.inc_mult, s.window))

    def _step(self, mult, window) -> int:
        return max(1, int(round(self.rng.uniform(mult[0] * window, mult[1] * window))))

    def summary(self) -> dict:
        segs = self.segments
        return dict(
            n_segments=len(segs), n_solved=sum(s.solved for s in segs),
            progress=round(float(np.mean([1 - s.t_curr / max(s.end, 1) for s in segs])), 4),
            t_curr=[s.t_curr for s in segs])
