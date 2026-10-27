"""Landmark archive and trajectory store (PLAN §2.2-2.3).

A landmark is the lex-max (R, U) state seen in one archive cell. The cell key is (level, room) by default
[RAM]; finer keys (inventory, x/y grid) are the plan's sub-landmark switches. Landmarks are replaced, never
removed. Each landmark keeps its top-k demos: episode prefixes s0 -> arrival at the landmark's cell with
the landmark's reward, ranked by (R at arrival desc, length asc). A demo episode is stored once and shared
by every landmark that references one of its prefixes; it is freed when no landmark references it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

LEVEL_ONEHOT, ROOM_ONEHOT = 4, 24
RAM_GOAL_DIM = LEVEL_ONEHOT + ROOM_ONEHOT + 2


def cell_key(ram, mode: str, cell_xy=(16, 16)):
    level, room, x, y, inv = ram
    if mode == "room":
        return (level, room)
    keys = (int(inv) & 0x1E).bit_count()
    if mode == "room_inv":
        return (level, room, keys)
    if mode == "xy":
        return (level, room, x // cell_xy[0], y // cell_xy[1])
    if mode == "xy_inv":
        return (level, room, keys, x // cell_xy[0], y // cell_xy[1])
    raise ValueError(f"unknown archive key mode: {mode}")


def ram_goal_vec(ram) -> np.ndarray:
    """E5 oracle goal encoding: one-hot(level) + one-hot(room) + (x, y)/255."""
    v = np.zeros(RAM_GOAL_DIM, np.float32)
    v[min(int(ram[0]), LEVEL_ONEHOT - 1)] = 1
    v[LEVEL_ONEHOT + min(int(ram[1]), ROOM_ONEHOT - 1)] = 1
    v[-2:] = (ram[2] / 255.0, ram[3] / 255.0)
    return v


@dataclass
class Demo:
    frames: np.ndarray | None      # (L+1, 84, 84) uint8; None while the episode is still running
    actions: np.ndarray | None     # (L,) uint8
    R: np.ndarray | None           # (L+1,) cumulative reward at each step
    lrf: np.ndarray | None         # (L+1,) index of the last-reward frame, -1 if none yet
    first_visit: dict = field(default_factory=dict)   # slot -> first step index
    refs: int = 0
    nbytes: int = 0
    t_max: int = 0                 # last step any landmark references


@dataclass
class Landmark:
    key: tuple
    slot: int
    R: float
    U: float
    ram: tuple
    parent: int | None
    depth: int
    t_discovered: int
    t_improved: int
    demos: list = field(default_factory=list)       # [(R_arr, t_arr, demo_id)], best first
    n_targeted: int = 0
    n_reached: int = 0
    outcomes: deque = field(default_factory=lambda: deque(maxlen=64))
    ret_len_ema: float | None = None
    n_handoff: int = 0
    n_quick_death: int = 0

    def success(self) -> float:
        return float(np.mean(self.outcomes)) if self.outcomes else 0.0


class Archive:
    def __init__(self, key_mode="room", cell_xy=(16, 16), topk=4, u_margin=0.1, max_demo_gb=12.0, cap=1024):
        self.key_mode, self.cell_xy, self.topk, self.u_margin = key_mode, tuple(cell_xy), topk, u_margin
        self.max_demo_bytes = int(max_demo_gb * 2**30)
        self.slot_of: dict = {}
        self.keys: list = []
        self.L: list[Landmark | None] = []
        self.R = np.full(cap, -np.inf)
        self.U = np.zeros(cap)
        self.frames = np.zeros((cap, 84, 84), np.uint8)
        self.demos: dict[int, Demo] = {}
        self._next_demo = 0
        self.demo_bytes = 0
        self._chain_cache: dict = {}

    # ----------------------------------------------------------------- slots
    def slot(self, ram) -> int:
        k = cell_key(ram, self.key_mode, self.cell_xy)
        s = self.slot_of.get(k)
        if s is None:
            s = self.slot_of[k] = len(self.L)
            self.L.append(None)
            self.keys.append(k)
            if s >= len(self.R):
                n = 2 * len(self.R)
                self.R = np.concatenate([self.R, np.full(n - len(self.R), -np.inf)])
                self.U = np.concatenate([self.U, np.zeros(n - len(self.U))])
                self.frames = np.concatenate([self.frames, np.zeros((n - len(self.frames), 84, 84), np.uint8)])
        return s

    def exists(self, s: int) -> bool:
        return self.L[s] is not None

    def live(self) -> list[int]:
        return [s for s, lm in enumerate(self.L) if lm is not None]

    # ---------------------------------------------------------------- update
    def is_candidate(self, s: int, R: float, U: float) -> bool:
        """Replace rule: (R, U) >_lex (R_k, U_k), with hysteresis on the U tie-break. U_k must be current."""
        if self.L[s] is None or R > self.R[s]:
            return True
        return R == self.R[s] and U > (1 + self.u_margin) * self.U[s]

    def update(self, s: int, R: float, U: float, frame, ram, parent: int | None, t: int) -> str:
        lm = self.L[s]
        if lm is None:
            depth = 0 if parent is None or self.L[parent] is None else self.L[parent].depth + 1
            self.L[s] = Landmark(self.keys[s], s, R, U, tuple(ram), parent, depth, t, t)
            kind = "new"
        else:
            kind = "R" if R > lm.R else "U"
            lm.R, lm.U, lm.ram, lm.t_improved = R, U, tuple(ram), t
            if kind == "R":   # demos at the old reward no longer reach the landmark
                for _, _, d in lm.demos:
                    self._decref(d)
                lm.demos = []
                lm.outcomes.clear()
                lm.ret_len_ema = None
        self.R[s], self.U[s] = R, U
        self.frames[s] = frame
        return kind

    # ----------------------------------------------------------------- demos
    def new_demo(self) -> int:
        d = self._next_demo
        self._next_demo += 1
        self.demos[d] = Demo(None, None, None, None)
        return d

    def offer_demo(self, s: int, R: float, t: int, demo_id: int) -> bool:
        """Arrival at landmark s with reward R at step t of the episode that owns demo_id."""
        lm = self.L[s]
        if lm is None or R < lm.R or t == 0:
            return False
        if any(d == demo_id for _, _, d in lm.demos):
            return False   # an episode counts once per landmark (its first arrival is the shortest)
        cand = (R, t, demo_id)
        rank = lambda e: (-e[0], e[1])  # noqa: E731
        if len(lm.demos) >= self.topk and rank(cand) >= rank(lm.demos[-1]):
            return False
        if self.demo_bytes > self.max_demo_bytes and lm.demos:
            return False   # store full: only first demos for new landmarks
        lm.demos.append(cand)
        lm.demos.sort(key=rank)
        d = self.demos[demo_id]
        d.refs += 1
        d.t_max = max(d.t_max, t)
        while len(lm.demos) > self.topk:
            self._decref(lm.demos.pop()[2])
        return True

    def materialize(self, demo_id, frames, actions, R, lrf, slots):
        """Called at episode end with the episode arrays (copied up to the last referenced step)."""
        d = self.demos.get(demo_id)
        if d is None:
            return
        if d.refs <= 0:
            del self.demos[demo_id]
            return
        t_max = d.t_max
        n = t_max + 1
        d.frames, d.actions = frames[:n].copy(), actions[:t_max].copy()
        d.R, d.lrf = R[:n].copy(), lrf[:n].copy()
        u, idx = np.unique(slots[:n], return_index=True)
        d.first_visit = dict(zip(u.tolist(), idx.tolist()))
        d.nbytes = d.frames.nbytes
        self.demo_bytes += d.nbytes

    def _decref(self, demo_id):
        d = self.demos.get(demo_id)
        if d is None:
            return
        d.refs -= 1
        if d.refs <= 0 and d.frames is not None:
            self.demo_bytes -= d.nbytes
            del self.demos[demo_id]

    # ------------------------------------------------------------- structure
    def chain(self, s: int, D: int) -> list[int]:
        """Up to D ancestors of s in the landmark tree, nearest first."""
        c = self._chain_cache.get(s)
        if c is None:
            c, p = [], self.L[s].parent if self.L[s] is not None else None
            while p is not None and len(c) < 256:
                c.append(p)
                p = self.L[p].parent
            if self.L[s] is not None:
                self._chain_cache[s] = c
        return c[:D]

    def ancestor(self, s: int, k: int) -> int:
        c = self.chain(s, k)
        return c[-1] if c else s
