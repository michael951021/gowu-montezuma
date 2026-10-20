"""GoWU phase II: backward-algorithm PPO on GoWU demonstrations (App. G).

Synchronous single-learner version of the paper's curriculum-server/actors/learner design. Each env
picks a (demo, segment) from the curriculum when its episode starts, restores the demo checkpoint at
the sampled start index, and plays until it matches the demo's (clipped) return up to the segment's
end, plus a random number of extra steps. The episode also ends on life loss or on the
μ·L + b step limit. Rewards are sign-clipped (MR). The observation is the 4-frame stack plus the
last-reward frame.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from ..nets import ImpalaActorCritic
from ..vec import EnvPool
from .curriculum import Curriculum, CurriculumConfig, Demo, load_demo


@dataclass
class BackwardConfig:
    demos: list = field(default_factory=list)   # paths to demo_best.pkl.gz from phase I runs
    game: str = "montezuma_revenge"
    seed: int = 0
    total_frames: int = 5_000_000_000           # paper: 15-20B; budgeted down for 2x3090
    n_envs: int = 128                           # PPO batch size (trajectories per update)
    unroll: int = 128
    n_workers: int = 14
    sticky_p: float = 0.25
    last_reward_channel: bool = True            # MR: 5th channel = frame at last reward
    reward_mode: str = "clip"                   # clip (MR, Venture) | scale (Pitfall, 0.001)
    lr: float = 1e-4
    gamma: float = 0.999
    gae_lambda: float = 0.95
    epochs: int = 1
    minibatches: int = 8
    ent_coef: float = 1e-3
    vf_coef: float = 0.5
    clip: float = 0.1
    max_grad_norm: float = 0.5
    dense: int = 1024
    head: int = 512
    amp: bool = True                            # bf16 autocast on CUDA
    clean_max_steps: int = 27_000               # episodes from env.reset() once all segments are solved
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    device: str = "auto"
    out_dir: str = "runs/backward"
    log_every: int = 1
    save_every: int = 20                        # updates between ckpt_latest.pt writes
    max_updates: int | None = None


class BackwardTrainer:
    def __init__(self, cfg: BackwardConfig):
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
        assert cfg.demos, "set `demos:` to phase-I demo files"
        self.demos = [Demo.from_dict(load_demo(p), cfg.reward_mode) for p in cfg.demos]
        self.cur = Curriculum(self.demos, cfg.curriculum, cfg.seed)
        self.C = 5 if cfg.last_reward_channel else 4
        self.pool = EnvPool(cfg.n_envs, cfg.n_workers, cfg.seed + 10_000,
                            dict(game=cfg.game, sticky_p=cfg.sticky_p), max_rollout_steps=1)
        self.net = ImpalaActorCritic(self.C, self.pool.n_actions, cfg.dense, cfg.head).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.amp = cfg.amp and self.device.type == "cuda"
        n = cfg.n_envs
        self.ep = dict(seg=np.full(n, -1), target=np.zeros(n), ret=np.zeros(n), raw=np.zeros(n),
                       steps=np.zeros(n, np.int64), max_steps=np.zeros(n, np.int64),
                       success=np.zeros(n, bool), succ_len=np.zeros(n, np.int64), extra=np.zeros(n, np.int64))
        self.frames = 0
        self.updates = 0
        self.t0 = time.time()
        self._log = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
        self._done_stats = []
        for i in range(n):
            self._start_episode(i)
        self._flush_starts()

    # ------------------------------------------------------------ episodes
    def _start_episode(self, i: int):
        e = self.ep
        if not hasattr(self, "_pending_restore"):
            self._pending_restore, self._pending_reset = [], []
        if self.cur.all_solved():
            e["seg"][i], e["target"][i] = -1, np.inf
            e["max_steps"][i] = self.cfg.clean_max_steps
            self._pending_reset.append(i)
        else:
            si, t0, target, demo_len = self.cur.sample()
            e["seg"][i], e["target"][i] = si, target - self.cfg.curriculum.eps_tol
            e["max_steps"][i] = self.cur.max_steps(si, demo_len)
            self._pending_restore.append((i, self.demos[self.cur.segments[si].demo].ckpts[t0]))
        e["ret"][i] = e["raw"][i] = 0
        e["steps"][i] = e["succ_len"][i] = e["extra"][i] = 0
        e["success"][i] = False

    def _flush_starts(self):
        self.pool.restore(self._pending_restore)
        if self._pending_reset:
            self.pool.reset(self._pending_reset)
        self._pending_restore, self._pending_reset = [], []

    def _obs(self) -> torch.Tensor:
        return torch.from_numpy(self.pool.obs[:, :self.C].copy()).to(self.device, non_blocking=True)

    def _forward(self, x):
        with torch.autocast("cuda", torch.bfloat16, enabled=self.amp):
            logits, v = self.net(x)
        return logits.float(), v.float()

    # ------------------------------------------------------------- rollout
    @torch.no_grad()
    def collect(self):
        cfg, e = self.cfg, self.ep
        n, T = cfg.n_envs, cfg.unroll
        buf_obs = torch.zeros((T, n, self.C, 84, 84), dtype=torch.uint8, device=self.device)
        buf_a = torch.zeros((T, n), dtype=torch.long, device=self.device)
        buf_logp = torch.zeros((T, n), device=self.device)
        buf_v = torch.zeros((T, n), device=self.device)
        buf_r = np.zeros((T, n), np.float32)
        buf_d = np.zeros((T, n), np.float32)
        ids = np.arange(n)
        for t in range(T):
            x = self._obs()
            logits, v = self._forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            buf_obs[t], buf_a[t], buf_logp[t], buf_v[t] = x, a, dist.log_prob(a), v
            r, ll, go, _ = self.pool.step(ids, a.cpu().numpy())
            rc = np.sign(r) if cfg.reward_mode == "clip" else 0.001 * r
            e["ret"] += rc
            e["raw"] += r
            e["steps"] += 1
            newly = (~e["success"]) & (e["ret"] >= e["target"])
            e["success"] |= newly
            e["succ_len"][newly] = e["steps"][newly]
            e["extra"][newly] = [self.cur.extra_steps() for _ in range(int(newly.sum()))]
            e["extra"][e["success"] & ~newly] -= 1
            terminal = ll | go
            trunc = ~terminal & ((e["success"] & (e["extra"] <= 0)) | (e["steps"] >= e["max_steps"]))
            rew = rc.astype(np.float32)
            if trunc.any():  # time-limit truncation: bootstrap from the final observation
                _, v_last = self._forward(self._obs()[torch.from_numpy(trunc).to(self.device)])
                rew[trunc] += cfg.gamma * v_last.cpu().numpy()
            buf_r[t] = rew
            done = terminal | trunc
            buf_d[t] = done
            for i in np.where(done)[0]:
                si = int(e["seg"][i])
                if si >= 0:
                    self.cur.report(si, bool(e["success"][i]), int(e["succ_len"][i]))
                self._done_stats.append((si, bool(e["success"][i]), float(e["raw"][i]), int(e["steps"][i])))
                self._start_episode(i)
            self._flush_starts()
        self.frames += n * T * 4
        _, v_next = self._forward(self._obs())
        return buf_obs, buf_a, buf_logp, buf_v, buf_r, buf_d, v_next

    # -------------------------------------------------------------- update
    def update(self, buf_obs, buf_a, buf_logp, buf_v, buf_r, buf_d, v_next):
        cfg = self.cfg
        T, n = buf_a.shape
        r = torch.from_numpy(buf_r).to(self.device)
        d = torch.from_numpy(buf_d).to(self.device)
        adv = torch.zeros_like(r)
        last = torch.zeros(n, device=self.device)
        for t in reversed(range(T)):
            nv = v_next if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - d[t]
            delta = r[t] + cfg.gamma * nv * nonterm - buf_v[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + buf_v
        b_obs, b_a, b_logp = buf_obs.flatten(0, 1), buf_a.flatten(), buf_logp.flatten()
        b_adv, b_ret = adv.flatten(), ret.flatten()
        N = b_a.shape[0]
        mb = N // cfg.minibatches
        stats = dict(pg=0.0, vf=0.0, ent=0.0, kl=0.0, clipfrac=0.0)
        k = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.device)
            for j in range(cfg.minibatches):
                idx = perm[j * mb:(j + 1) * mb]
                logits, v = self._forward(b_obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(b_a[idx])
                ratio = (logp - b_logp[idx]).exp()
                a_mb = b_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = -torch.min(ratio * a_mb, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_mb).mean()
                vf = F.mse_loss(v, b_ret[idx])
                ent = dist.entropy().mean()
                loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item()
                    stats["vf"] += vf.item()
                    stats["ent"] += ent.item()
                    stats["kl"] += (b_logp[idx] - logp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                k += 1
        return {kk: round(vv / k, 5) for kk, vv in stats.items()}

    # ----------------------------------------------------------------- io
    def save(self, name="ckpt_latest.pt"):
        path = os.path.join(self.cfg.out_dir, name)
        torch.save(dict(net=self.net.state_dict(), frames=self.frames, updates=self.updates,
                        C=self.C, n_actions=self.pool.n_actions, dense=self.cfg.dense, head=self.cfg.head,
                        curriculum=[(s.t_curr, s.solved) for s in self.cur.segments]), path + ".tmp")
        os.replace(path + ".tmp", path)

    def log(self, st):
        dt = time.time() - self.t0
        ds = self._done_stats
        self._done_stats = []
        seg_eps = [x for x in ds if x[0] >= 0]
        rec = dict(update=self.updates, frames=self.frames, time=round(dt, 1), fps=round(self.frames / dt),
                   episodes=len(ds), success=round(float(np.mean([x[1] for x in seg_eps])), 3) if seg_eps else None,
                   clean_return=round(float(np.mean([x[2] for x in ds if x[0] < 0])), 1)
                   if any(x[0] < 0 for x in ds) else None,
                   **self.cur.summary(), **st)
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()
        return rec

    def run(self, verbose=True):
        cfg = self.cfg
        try:
            while self.frames < cfg.total_frames and (cfg.max_updates is None or self.updates < cfg.max_updates):
                st = self.update(*self.collect())
                self.updates += 1
                if self.updates % cfg.log_every == 0:
                    rec = self.log(st)
                    if verbose:
                        print(f"upd {rec['update']:6d} | {rec['frames']/1e6:9.2f}M fr | {rec['fps']:6d} fps | "
                              f"solved {rec['n_solved']}/{rec['n_segments']} | prog {rec['progress']:.3f} | "
                              f"succ {rec['success']} | ent {st['ent']:.3f} | kl {st['kl']:.4f}", flush=True)
                if self.updates % cfg.save_every == 0:
                    self.save()
        finally:
            self.save()
            self.pool.close()
