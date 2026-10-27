import gzip
import pickle

import numpy as np
import torch
import pytest

from gowu.atari import AtariEnv
from gowu.exploit_guard import ExploitGuard
from gowu.explore import GoWU, GowuConfig
from gowu.tree import LineageTree
from gowu.vec import EnvPool, RolloutItem


def test_rgb_pooling_precedes_grayscale():
    # The paper's RGB pooling retains both channels; grayscale-first loses the red maximum.
    e = AtariEnv.__new__(AtariEnv)
    a = np.zeros((210, 160, 3), np.uint8)
    b = np.zeros_like(a)
    a[..., 0], b[..., 1] = 255, 255
    e._raw = [a, b]
    out = e._grab()
    assert out.shape == (84, 84) and out.dtype == np.uint8
    assert np.all(out == 226)


def test_replay_includes_restored_state_before_first_action():
    pool = EnvPool(1, 1, 16, dict(sticky_p=0.0), max_rollout_steps=5)
    try:
        pool.reset([0])
        start = pool.obs[0, 3].copy()
        ck = pool.checkpoint([0])[0]
        pool.rollout([RolloutItem(0, None, 12, 5)])
        assert np.array_equal(pool.traj[0, 0], start)
        # Restore a different state from the current endpoint: its first replay observation
        # must be the reset target, not a frame produced AFTER leaving it.
        pool.rollout([RolloutItem(0, ck, 11, 5)])
        assert np.array_equal(pool.traj[0, 0], start)
        assert not np.array_equal(pool.obs[0, 3], start)
    finally:
        pool.close()


def test_targeted_guard_rejects_first_bug_reward_but_allows_normal_rewards():
    g = ExploitGuard()
    # (level, room, raw x, raw y, inventory); screen y = 308 - raw y.
    before = (0, 5, 116, 180, 8)
    after = (0, 5, 111, 174, 0)
    assert g.check({}, [(0, 300.0, before, after)])[0]
    # A legitimate opening below the same platform consumes a key too.
    assert not g.check({}, [(0, 300.0, (0, 5, 109, 155, 8), (0, 5, 106, 155, 0))])[0]
    assert not g.check({}, [(0, 300.0, (0, 1, 116, 180, 8), (0, 1, 111, 174, 0))])[0]
    assert not g.check({}, [(0, 300.0, (0, 5, 116, 180, 32), after)])[0]  # sword, not key
    assert not g.check({}, [(0, 300.0, before, (0, 5, 111, 174, 8))])[0]
    counts = {}
    for _ in range(40):
        bad, counts = g.check(counts, [(0, 1000.0, before, before)])
        assert not bad  # no blanket cap on normal treasure/repeated rewards


def test_restore_is_exact_without_sticky():
    e = AtariEnv(seed=0, sticky_p=0.0)
    e.reset()
    rng = np.random.default_rng(0)
    for a in rng.integers(18, size=40):
        e.step(int(a))
    ck = e.checkpoint()
    acts = rng.integers(18, size=60)
    for a in acts:
        e.step(int(a))
    f1 = e.frames.copy()
    e.restore(ck)
    for a in acts:
        e.step(int(a))
    assert np.array_equal(f1, e.frames)


def test_life_loss_detected():
    e = AtariEnv(seed=0, sticky_p=0.0)
    e.reset()
    lost = False
    for _ in range(2000):  # jumping left off the start platform kills Panama Joe
        _, _, ll, go = e.step(12)  # LEFTFIRE
        if ll:
            lost = True
            break
    assert lost


def test_tree_refcount_prunes_dead_branches():
    t = LineageTree()
    root = t.make_root(None, (0,) * 5, None)
    a = t.acquire(t.add_child(root, None, 0, 5, b"", None, [], {}, None))
    b = t.acquire(t.add_child(a, None, 0, 5, b"", None, [], {}, None))
    assert t.n_alive == 3
    t.release(b)              # b freed, a still held
    assert t.n_alive == 2
    t.release(a)              # a freed, root pinned
    assert t.n_alive == 1


def test_exploit_guard_loop():
    g = ExploitGuard("loop", max_repeats=2)
    counts = {}
    ev = [(0, 300.0, (0, 5, 0, 0, 0), (0, 5, 0, 0, 0))]
    for i in range(2):
        bad, counts = g.check(counts, ev)
        assert not bad
    bad, _ = g.check(counts, ev)
    assert bad
    bad, _ = ExploitGuard("loop", rules=[(None, 5, 300.0)]).check({}, ev)
    assert bad


@pytest.mark.parametrize("policy", ["hold_random", "ensemble"])
def test_gowu_runs_and_demo_is_consistent(tmp_path, policy):
    cfg = GowuConfig(n_groups=2, n_particles=4, outer_steps=(2, 3), total_frames=10**9, max_iters=3,
                     n_workers=2, device="cpu", out_dir=str(tmp_path), particle_policy=policy,
                     replay_capacity=5000, save_every_sec=1e9)
    g = GoWU(cfg)
    g.run(verbose=False)
    assert g.frames > 0
    with gzip.open(tmp_path / "demo_best.pkl.gz") as f:
        demo = pickle.load(f)
    assert len(demo["ckpts"]) == len(demo["R_cum"]) >= 1
    assert all(np.diff(demo["R_cum"]) >= 0)
    assert all(np.diff(demo["steps"]) > 0)
    with gzip.open(tmp_path / "landmarks.pkl.gz") as f:
        lm = pickle.load(f)["landmarks"]
    assert lm and all(len(x["path_ckpts"]) == len(x["path_actions"]) for x in lm)
    # every demo checkpoint restores
    e = AtariEnv(seed=1)
    for ck in demo["ckpts"]:
        e.restore(ck)


def test_async_ensemble_learner_trains_and_broadcasts(tmp_path):
    cfg = GowuConfig(n_groups=2, n_particles=4, outer_steps=(2, 3), total_frames=10**9, max_iters=4,
                     n_workers=2, device="cpu", out_dir=str(tmp_path), particle_policy="ensemble",
                     replay_capacity=5000, save_every_sec=1e9, ens_batch=16,
                     ens_updates_per_1k_steps=100)
    g = GoWU(cfg)
    g.run(verbose=False)
    assert g.ens_updates_total > 0          # the learner thread trained while rollouts ran
    g._sync_acting()
    for src, dst in zip(g.ensembles, g.act_ensembles):
        assert all(torch.equal(a, b) for a, b in zip(src.parameters(), dst.parameters()))
