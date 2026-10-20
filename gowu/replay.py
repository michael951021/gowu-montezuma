"""Per-group FIFO replay (GoWU Sec. 5.1: capacity 128k per group, entries (s, a, m))."""
from __future__ import annotations

import numpy as np


class Replay:
    def __init__(self, capacity: int, obs_shape: tuple, n_members: int = 0, seed: int = 0):
        self.cap = capacity
        self.obs = np.zeros((capacity, *obs_shape), np.uint8)
        self.act = np.zeros(capacity, np.int64)
        self.mask = np.zeros((capacity, n_members), np.float32) if n_members else None
        self.n_members = n_members
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs: np.ndarray, act: np.ndarray | None = None) -> None:
        n = len(obs)
        if n == 0:
            return
        if n > self.cap:
            obs = obs[-self.cap:]
            act = None if act is None else act[-self.cap:]
            n = self.cap
        idx = (self.ptr + np.arange(n)) % self.cap
        self.obs[idx] = obs
        if act is not None:
            self.act[idx] = act
        if self.mask is not None:
            # bootstrap mask drawn independently per member at collection time
            self.mask[idx] = self.rng.random((n, self.n_members)) < 0.5
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)

    def sample(self, b: int):
        idx = self.rng.integers(0, self.size, b)
        return idx, self.obs[idx]


def sample_across(buffers: list[Replay], b: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a minibatch spread over all non-empty buffers (RND trains on every group's data)."""
    live = [r for r in buffers if r.size > 0]
    counts = rng.multinomial(b, np.full(len(live), 1 / len(live)))
    return np.concatenate([r.sample(int(c))[1] for r, c in zip(live, counts) if c > 0])
