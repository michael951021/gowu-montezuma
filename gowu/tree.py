"""State-lineage tree (GoWU Sec. 4.1).

Each node = checkpointed state + cumulative extrinsic reward + parent pointer. We also keep the
actions of the rollout that produced the node, so a root->node path is a complete action sequence
(needed by LR-GoWU baseline B1 for BC). Nodes are freed once no particle occupies them, no child
references them and nothing pins them, so memory scales with the active lineages.
"""
from __future__ import annotations

import itertools

from .atari import Checkpoint


class Node:
    __slots__ = ("id", "parent", "ckpt", "R", "depth", "steps", "actions", "ram", "events",
                 "loop_counts", "refs", "obs_last")

    def __init__(self, id, parent, ckpt, R, depth, steps, actions, ram, events, loop_counts, obs_last):
        self.id = id
        self.parent: Node | None = parent
        self.ckpt: Checkpoint = ckpt
        self.R = R                    # cumulative extrinsic reward
        self.depth = depth            # generations from root
        self.steps = steps            # agent steps from root along this lineage
        self.actions = actions        # bytes: actions of the rollout parent -> self
        self.ram = ram                # (level, room, x, y, inventory) at this state
        self.events = events          # reward events of the rollout parent -> self
        self.loop_counts = loop_counts  # exploit guard bookkeeping, inherited copy-on-write
        self.obs_last = obs_last      # last 84x84 frame (for re-scoring uncertainty)
        self.refs = 0


class LineageTree:
    def __init__(self):
        self._ids = itertools.count()
        self.n_alive = 0
        self.n_created = 0
        self.root: Node | None = None

    def make_root(self, ckpt, ram, obs_last) -> Node:
        self.root = self._new(None, ckpt, 0.0, 0, b"", ram, [], {}, obs_last)
        self.root.refs += 1  # pinned forever
        return self.root

    def _new(self, parent, ckpt, R, steps, actions, ram, events, loop_counts, obs_last) -> Node:
        n = Node(next(self._ids), parent, ckpt, R, 0 if parent is None else parent.depth + 1,
                 steps, actions, ram, events, loop_counts, obs_last)
        if parent is not None:
            parent.refs += 1
        self.n_alive += 1
        self.n_created += 1
        return n

    def add_child(self, parent: Node, ckpt, R, n_steps, actions, ram, events, loop_counts, obs_last) -> Node:
        return self._new(parent, ckpt, R, parent.steps + n_steps, actions, ram, events, loop_counts, obs_last)

    # reference counting --------------------------------------------------
    def acquire(self, n: Node) -> Node:
        n.refs += 1
        return n

    def release(self, n: Node | None) -> None:
        while n is not None:
            n.refs -= 1
            if n.refs > 0:
                return
            parent = n.parent
            n.parent = None
            n.ckpt = None
            n.obs_last = None
            self.n_alive -= 1
            n = parent

    @staticmethod
    def ancestor(n: Node, k: int) -> Node:
        for _ in range(k):
            if n.parent is None:
                break
            n = n.parent
        return n

    @staticmethod
    def path(n: Node) -> list[Node]:
        out = []
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]
