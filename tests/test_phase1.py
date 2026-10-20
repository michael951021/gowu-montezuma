import gzip
import pickle

import numpy as np
import pytest

from gowu.atari import AtariEnv
from gowu.exploit_guard import ExploitGuard
from gowu.explore import GoWU, GowuConfig
from gowu.tree import LineageTree


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
    # every demo checkpoint restores
    e = AtariEnv(seed=1)
    for ck in demo["ckpts"]:
        e.restore(ck)
