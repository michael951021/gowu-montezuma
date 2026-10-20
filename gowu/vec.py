"""Process pool of AtariEnvs with shared-memory observations.

Env i lives in worker i % n_workers (interleaved so every worker gets envs from every group and load
stays balanced when some particles are idle). Two kinds of commands:

- `rollout`: the worker runs a whole GoWU inner-step block for each env on its own (restore, hold an
  action with eps-noise, stop at death / positive reward / T steps). This needs no policy network, so
  there is a single IPC round trip per outer step.
- `step` / `restore` / `reset` / `checkpoint`: lockstep primitives, used by policies that need the
  network every step (ensemble particles, PPO).

Observations are written to shared memory as (stack + 1) channels: the frame stack plus the
last-reward frame (GoWU phase II's 5th channel on MR).
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from dataclasses import dataclass
from multiprocessing import shared_memory

import numpy as np

from .atari import OBS_HW, AtariEnv, Checkpoint


@dataclass
class RolloutItem:
    env: int
    restore: Checkpoint | None
    action: int        # held action (GoWU "fixed random action" particle policy)
    steps: int         # T_i
    eps: float = 0.0   # prob. of a uniform random action at each step
    stop_on_reward: bool = True


@dataclass
class RolloutResult:
    env: int
    n_steps: int
    reward: float
    dead: bool
    ckpt: Checkpoint | None       # endpoint snapshot (only if alive)
    actions: bytes                # uint8 actions actually chosen (pre-sticky)
    ram: tuple                    # (level, room, x, y, inventory) at the endpoint
    reward_events: list           # [(t, r, ram_before, ram_after)]


def _worker(remote, env_ids, env_kwargs, seeds, shm_names, shapes):
    try:
        shms = [shared_memory.SharedMemory(name=n) for n in shm_names]
        obs = np.ndarray(shapes[0], np.uint8, buffer=shms[0].buf)
        traj = np.ndarray(shapes[1], np.uint8, buffer=shms[1].buf)
        envs = {i: AtariEnv(seed=s, **env_kwargs) for i, s in zip(env_ids, seeds)}
        rng = np.random.default_rng(seeds[0] + 7919)
        stack = next(iter(envs.values())).stack

        def write_obs(i):
            e = envs[i]
            obs[i, :stack] = e.frames
            obs[i, stack] = e.last_reward_frame

        while True:
            cmd, data = remote.recv()
            if cmd == "rollout":
                out = []
                for it in data:
                    e = envs[it.env]
                    if it.restore is not None:
                        e.restore(it.restore)
                    acts = np.empty(it.steps, np.uint8)
                    total, dead, events, n = 0.0, False, [], 0
                    for t in range(it.steps):
                        a = it.action if (it.eps <= 0 or rng.random() >= it.eps) else rng.integers(e.n_actions)
                        acts[t] = a
                        ram_before = e.ram_info()
                        frames, r, life_lost, game_over = e.step(int(a))
                        traj[it.env, t] = frames[-1]
                        n = t + 1
                        if r != 0:
                            total += r
                            events.append((t, r, ram_before, e.ram_info()))
                        if life_lost or game_over:
                            dead = True
                            break
                        if r > 0 and it.stop_on_reward:
                            break
                    write_obs(it.env)
                    out.append(RolloutResult(it.env, n, total, dead, None if dead else e.checkpoint(),
                                             acts[:n].tobytes(), e.ram_info(), events))
                remote.send(out)
            elif cmd == "step":
                ids, acts = data
                res = np.zeros((len(ids), 3), np.float32)
                rams = []
                for k, (i, a) in enumerate(zip(ids, acts)):
                    _, r, ll, go = envs[i].step(int(a))
                    res[k] = (r, ll, go)
                    rams.append(envs[i].ram_info())
                    write_obs(i)
                remote.send((res, rams))
            elif cmd == "restore":
                for i, ck in data:
                    envs[i].restore(ck)
                    write_obs(i)
                remote.send(None)
            elif cmd == "reset":
                for i in data:
                    envs[i].reset()
                    write_obs(i)
                remote.send(None)
            elif cmd == "checkpoint":
                remote.send([envs[i].checkpoint() for i in data])
            elif cmd == "set_lives":
                for i, n in data:
                    envs[i].set_lives(n)
                remote.send(None)
            elif cmd == "ram":
                remote.send([envs[i].ram_info() for i in data])
            elif cmd == "close":
                break
    except Exception:  # surface worker crashes in the parent instead of hanging
        remote.send(("__error__", traceback.format_exc()))
    finally:
        remote.close()


class EnvPool:
    def __init__(self, n_envs: int, n_workers: int, seed: int, env_kwargs: dict | None = None,
                 max_rollout_steps: int = 64, stack: int = 4):
        env_kwargs = dict(env_kwargs or {})
        env_kwargs.setdefault("stack", stack)
        self.n_envs = n_envs
        self.n_workers = min(n_workers, n_envs)
        self.stack = env_kwargs["stack"]
        self.max_rollout_steps = max_rollout_steps
        probe = AtariEnv(seed=0, **{k: v for k, v in env_kwargs.items() if k != "max_episode_frames"})
        self.n_actions = probe.n_actions
        del probe
        shapes = [(n_envs, self.stack + 1, OBS_HW, OBS_HW), (n_envs, max_rollout_steps, OBS_HW, OBS_HW)]
        self._shms = [shared_memory.SharedMemory(create=True, size=int(np.prod(s))) for s in shapes]
        self.obs = np.ndarray(shapes[0], np.uint8, buffer=self._shms[0].buf)
        self.traj = np.ndarray(shapes[1], np.uint8, buffer=self._shms[1].buf)
        self.owner = np.arange(n_envs) % self.n_workers
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for w in range(self.n_workers):
            ids = [int(i) for i in np.where(self.owner == w)[0]]
            seeds = [seed * 100_003 + i for i in ids]
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, daemon=True,
                            args=(child, ids, env_kwargs, seeds, [s.name for s in self._shms], shapes))
            p.start()
            child.close()
            self.remotes.append(parent)
            self.procs.append(p)
        self.closed = False

    # -------------------------------------------------------------- plumbing
    def _recv(self, w):
        msg = self.remotes[w].recv()
        if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == "__error__":
            raise RuntimeError(f"env worker {w} crashed:\n{msg[1]}")
        return msg

    def _scatter(self, cmd, per_env: dict):
        """per_env: env -> payload. Returns {worker: reply} for workers that got work."""
        buckets: dict[int, list] = {}
        for i, payload in per_env.items():
            buckets.setdefault(int(self.owner[i]), []).append(payload)
        for w, items in buckets.items():
            self.remotes[w].send((cmd, items))
        return {w: self._recv(w) for w in buckets}

    # ----------------------------------------------------------------- API
    def rollout_send(self, items: list[RolloutItem]) -> None:
        """Dispatch rollouts without waiting, so the caller can train networks meanwhile."""
        assert all(it.steps <= self.max_rollout_steps for it in items)
        buckets: dict[int, list] = {}
        for it in items:
            buckets.setdefault(int(self.owner[it.env]), []).append(it)
        for w, its in buckets.items():
            self.remotes[w].send(("rollout", its))
        self._pending = list(buckets)

    def rollout_recv(self) -> list[RolloutResult]:
        out = [r for w in self._pending for r in self._recv(w)]
        self._pending = []
        out.sort(key=lambda r: r.env)
        return out

    def rollout(self, items: list[RolloutItem]) -> list[RolloutResult]:
        self.rollout_send(items)
        return self.rollout_recv()

    def step(self, ids, actions):
        """Lockstep step for envs `ids`. Returns (reward, life_lost, game_over, rams) aligned to ids."""
        ids = np.asarray(ids)
        actions = np.asarray(actions)
        buckets: dict[int, list] = {}
        for k, i in enumerate(ids):
            buckets.setdefault(int(self.owner[i]), []).append(k)
        for w, ks in buckets.items():
            self.remotes[w].send(("step", (ids[ks].tolist(), actions[ks].tolist())))
        res = np.zeros((len(ids), 3), np.float32)
        rams = [None] * len(ids)
        for w, ks in buckets.items():
            r, rm = self._recv(w)
            res[ks] = r
            for k, x in zip(ks, rm):
                rams[k] = x
        return res[:, 0], res[:, 1].astype(bool), res[:, 2].astype(bool), rams

    def restore(self, pairs: list[tuple[int, Checkpoint]]):
        if pairs:
            self._scatter("restore", {i: (i, ck) for i, ck in pairs})

    def reset(self, ids):
        self._scatter("reset", {int(i): int(i) for i in ids})

    def checkpoint(self, ids) -> dict[int, Checkpoint]:
        ids = [int(i) for i in ids]
        replies = self._scatter("checkpoint", {i: i for i in ids})
        out = {}
        for w, cks in replies.items():
            mine = [i for i in ids if self.owner[i] == w]
            out.update(zip(mine, cks))
        return out

    def set_lives(self, pairs):
        if pairs:
            self._scatter("set_lives", {i: (i, n) for i, n in pairs})

    def ram(self, ids) -> dict[int, tuple]:
        ids = [int(i) for i in ids]
        replies = self._scatter("ram", {i: i for i in ids})
        out = {}
        for w, rams in replies.items():
            mine = [i for i in ids if self.owner[i] == w]
            out.update(zip(mine, rams))
        return out

    def close(self):
        if self.closed:
            return
        self.closed = True
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for s in self._shms:
            s.close()
            s.unlink()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
