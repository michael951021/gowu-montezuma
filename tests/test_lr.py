"""LR-GoWU unit tests (CPU)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gowu.lr.archive import Archive, cell_key  # noqa: E402

F = np.zeros((84, 84), np.uint8)


def test_cell_keys():
    ram = (0, 1, 77, 235, 0b110)
    assert cell_key(ram, "room") == (0, 1)
    assert cell_key(ram, "room_inv") == (0, 1, 2)
    assert cell_key(ram, "xy", (16, 16)) == (0, 1, 4, 14)
    assert cell_key(ram, "xy_inv", (16, 16)) == (0, 1, 2, 4, 14)


def test_replace_rule_lex_with_margin():
    a = Archive("room", u_margin=0.1)
    s = a.slot((0, 1, 0, 0, 0))
    assert a.is_candidate(s, 0.0, 0.0)            # empty cell
    a.update(s, 0.0, 1.0, F, (0, 1, 0, 0, 0), None, 0)
    assert not a.is_candidate(s, 0.0, 1.05)       # within the hysteresis margin
    assert a.is_candidate(s, 0.0, 1.2)            # U tie-break
    assert a.is_candidate(s, 100.0, 0.0)          # higher R wins regardless of U
    assert not a.is_candidate(s, -1.0, 9.0)


def test_demo_topk_refcount_and_R_reset():
    a = Archive("room", topk=2)
    s = a.slot((0, 1, 0, 0, 0))
    a.update(s, 0.0, 1.0, F, (0, 1, 0, 0, 0), None, 0)
    ids = [a.new_demo() for _ in range(3)]
    assert a.offer_demo(s, 0.0, 50, ids[0])
    assert a.offer_demo(s, 0.0, 30, ids[1])
    assert a.offer_demo(s, 0.0, 10, ids[2])       # evicts the longest (50)
    assert [d for _, _, d in a.L[s].demos] == [ids[2], ids[1]]
    assert not a.offer_demo(s, 0.0, 5, ids[2])    # one entry per episode
    n = 60
    fr = np.zeros((n + 1, 84, 84), np.uint8)
    for d in ids:
        a.materialize(d, fr, np.zeros(n, np.uint8), np.zeros(n + 1), np.full(n + 1, -1), np.zeros(n + 1, int))
    assert ids[0] not in a.demos and a.demos[ids[1]].frames.shape[0] == 31
    a.update(s, 100.0, 0.5, F, (0, 1, 0, 0, 0), None, 1)   # higher R: old demos no longer reach it
    assert a.L[s].demos == [] and a.demo_bytes == 0 and not a.demos


def test_chain_and_ancestor():
    a = Archive("room")
    prev = None
    for room in range(5):
        s = a.slot((0, room, 0, 0, 0))
        a.update(s, 0.0, 0.0, F, (0, room, 0, 0, 0), prev, 0)
        prev = s
    assert a.chain(4, 32) == [3, 2, 1, 0]
    assert a.L[4].depth == 4
    assert a.ancestor(4, 2) == 2 and a.ancestor(0, 3) == 0
