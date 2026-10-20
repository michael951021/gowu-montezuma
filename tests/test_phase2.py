import gzip
import pickle

import numpy as np
import torch

from gowu.atari import AtariEnv
from gowu.backward.curriculum import Curriculum, CurriculumConfig, Demo
from gowu.backward.train import BackwardConfig, BackwardTrainer


def _fake_demo(n=40, events_at=(10, 30)):
    e = AtariEnv(seed=0)
    e.reset()
    ckpts, steps, events, R = [], [], [], []
    tot = 0.0
    for i in range(n):
        ckpts.append(e.checkpoint())
        steps.append(i * 5)
        ev = [(0, 100.0, (0,) * 5, (0,) * 5)] if i in events_at else []
        tot += 100.0 * len(ev)
        events.append(ev)
        R.append(tot)
        for _ in range(5):
            e.step(0)
    return dict(ckpts=ckpts, steps=steps, events=events, R_cum=R)


def test_curriculum_moves_back_and_solves():
    d = Demo.from_dict(_fake_demo())
    cur = Curriculum([d], CurriculumConfig(window=5, n_update=4), seed=0)
    assert [s.end for s in cur.segments] == [10, 30]
    s = cur.segments[0]
    start = s.t_curr
    for _ in range(4):
        cur.report(0, True, 10)
    assert s.t_curr < start
    while s.t_curr > 0:
        for _ in range(4):
            cur.report(0, True, 10)
    for _ in range(4):
        cur.report(0, True, 10)
    assert s.solved
    si, t0, target, _ = cur.sample()
    assert target >= 1  # every start lies strictly before its segment's reward


def test_curriculum_moves_forward_on_failure():
    d = Demo.from_dict(_fake_demo())
    cur = Curriculum([d], CurriculumConfig(window=5, n_update=4), seed=0)
    s = cur.segments[1]
    for _ in range(3):
        for _ in range(4):
            cur.report(1, True, 10)
    back = s.t_curr
    for _ in range(4):
        cur.report(1, False, 0)
    assert s.t_curr > back and s.t_curr <= s.end - 1


def test_ppo_update_prefers_rewarded_action(tmp_path):
    p = tmp_path / "demo.pkl.gz"
    with gzip.open(p, "wb") as f:
        pickle.dump(_fake_demo(), f)
    cfg = BackwardConfig(demos=[str(p)], n_envs=8, unroll=16, n_workers=2, device="cpu", lr=1e-3,
                         minibatches=2, out_dir=str(tmp_path / "run"), dense=64, head=32)
    tr = BackwardTrainer(cfg)
    try:
        T, n = cfg.unroll, cfg.n_envs
        obs = torch.randint(0, 255, (T, n, 5, 84, 84), dtype=torch.uint8)
        a = torch.tensor([3] * 4 + [5] * 4).repeat(T, 1)
        r = np.zeros((T, n), np.float32)
        r[:, :4] = 1.0
        d = np.zeros((T, n), np.float32)
        with torch.no_grad():
            logits, v = tr.net(obs.flatten(0, 1))
        logp = torch.distributions.Categorical(logits=logits).log_prob(a.flatten()).view(T, n)
        p0 = logits.softmax(-1)[:, 3].mean() / logits.softmax(-1)[:, 5].mean()
        for _ in range(5):
            tr.update(obs, a, logp, v.view(T, n), r, d, torch.zeros(n))
        with torch.no_grad():
            logits2, _ = tr.net(obs.flatten(0, 1))
        p1 = logits2.softmax(-1)[:, 3].mean() / logits2.softmax(-1)[:, 5].mean()
        assert p1 > p0 * 1.05
    finally:
        tr.pool.close()


def test_trainer_collect_runs(tmp_path):
    p = tmp_path / "demo.pkl.gz"
    with gzip.open(p, "wb") as f:
        pickle.dump(_fake_demo(), f)
    cfg = BackwardConfig(demos=[str(p)], n_envs=4, unroll=8, n_workers=2, device="cpu", minibatches=2,
                         out_dir=str(tmp_path / "run"), dense=64, head=32, max_updates=2, save_every=1)
    BackwardTrainer(cfg).run(verbose=False)
    assert (tmp_path / "run" / "ckpt_latest.pt").exists()
