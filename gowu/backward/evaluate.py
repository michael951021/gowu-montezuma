"""Policy evaluation from true env.reset() (GoWU App. G.6).

╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║ EPISODE CAP: the default here is the standard ALE protocol (108,000 frames = 27,000 steps).   ║
║ GoWU's reported scores are NOT capped: they only stop "loitering" episodes (10k steps with no ║
║ reward). Re-run with max_episode_frames=None before comparing to GoWU's 182,671 on MR.        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝

Also implemented, as in the paper:
- loitering prevention: stop after `loiter_steps` steps with no reward
- treasure-room bug: discard an episode with >= 30 consecutive rewards of exactly 1000
"""
from __future__ import annotations

import numpy as np
import torch

from ..nets import ImpalaActorCritic
from ..vec import EnvPool

DEFAULT_MAX_EPISODE_FRAMES = 108_000  # standard ALE cap; GoWU uses None (uncapped). See banner.


def load_policy(path: str, device) -> ImpalaActorCritic:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = ImpalaActorCritic(ck["C"], ck["n_actions"], ck.get("dense", 1024), ck.get("head", 512)).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net


@torch.no_grad()
def evaluate(net, n_episodes: int, seed: int, device, game="montezuma_revenge", sticky_p=0.25,
             greedy: bool = True, max_episode_frames: int | None = DEFAULT_MAX_EPISODE_FRAMES,
             loiter_steps: int = 10_000, n_workers: int = 8, n_envs: int | None = None,
             treasure_run: int = 30) -> dict:
    C = net.pi_enc.conv[0].in_channels
    n_envs = min(n_envs or n_episodes, n_episodes)
    pool = EnvPool(n_envs, n_workers, seed, dict(game=game, sticky_p=sticky_p), max_rollout_steps=1)
    try:
        pool.reset(range(n_envs))
        max_steps = None if max_episode_frames is None else max_episode_frames // 4
        score = np.zeros(n_envs)
        steps = np.zeros(n_envs, np.int64)
        since_r = np.zeros(n_envs, np.int64)
        run1000 = np.zeros(n_envs, np.int64)
        bugged = np.zeros(n_envs, bool)
        active = np.ones(n_envs, bool)
        started = n_envs
        scores, discarded, lengths = [], 0, []
        while active.any():
            ids = np.where(active)[0]
            x = torch.from_numpy(pool.obs[ids, :C].copy()).to(device)
            logits, _ = net(x)
            a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            r, _, go, _ = pool.step(ids, a.cpu().numpy())
            score[ids] += r
            steps[ids] += 1
            since_r[ids] = np.where(r != 0, 0, since_r[ids] + 1)
            run1000[ids] = np.where(r == 1000, run1000[ids] + 1, np.where(r != 0, 0, run1000[ids]))
            bugged[ids] |= run1000[ids] >= treasure_run
            done = go | (since_r[ids] >= loiter_steps) | bugged[ids]
            if max_steps is not None:
                done |= steps[ids] >= max_steps
            restart = []
            for k in np.where(done)[0]:
                i = ids[k]
                if bugged[i]:
                    discarded += 1
                else:
                    scores.append(float(score[i]))
                    lengths.append(int(steps[i]))
                if started < n_episodes:
                    started += 1
                    restart.append(i)
                    score[i] = steps[i] = since_r[i] = run1000[i] = 0
                    bugged[i] = False
                else:
                    active[i] = False
            if restart:
                pool.reset(restart)
    finally:
        pool.close()
    s = np.asarray(scores) if scores else np.zeros(1)
    return dict(mean=float(s.mean()), std=float(s.std()), median=float(np.median(s)), max=float(s.max()),
                n=len(scores), discarded=discarded, mean_len=float(np.mean(lengths)) if lengths else 0.0,
                max_episode_frames=max_episode_frames, greedy=greedy)
