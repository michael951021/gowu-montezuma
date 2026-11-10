"""Landmarks from pixels and reward only (no RAM): scene changes and reward events.

A *scene* is a screen background. A scene change is detected when a large share of playfield pixels changes between two
consecutive frames; the new scene is identified by matching a coarse signature of the new background against a codebook of
signatures. The codebook is built without labels (a signature that matches nothing becomes a new scene), from saved
episodes or online. A *landmark* is an event the agent can observe: ("r", n) = the n-th positive game reward of the episode,
or ("s", scene_id) = entering a scene. A route is an ordered list of landmarks.

Nothing here reads emulator memory. The diagnostic sidecar labels (room ids) are used only by stage0_checks.py to measure how
well this detector agrees with them.
"""
from __future__ import annotations

import numpy as np

TOP = 20            # rows above this are the score / lives / inventory display, which changes without a scene change
CUT_FRAC = 0.10     # share of playfield pixels that must change (by > PIX) between consecutive frames to call a scene change
MATCH_FRAC = 0.06   # a signature matches a codebook entry when fewer than this share of cells differ (by > PIX)
PIX = 24


def playfield(frame84: np.ndarray) -> np.ndarray:
    return frame84[TOP:, :]


def changed_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(playfield(a).astype(np.int16) - playfield(b).astype(np.int16)) > PIX).mean())


def signature(frame84: np.ndarray) -> np.ndarray:
    """Playfield at half resolution (32 x 42), uint8."""
    p = playfield(frame84).astype(np.float32)
    return p.reshape(p.shape[0] // 2, 2, p.shape[1] // 2, 2).mean(axis=(1, 3)).astype(np.uint8)


def sig_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIX).mean())


class Codebook:
    """Scene signatures, in order of discovery. Scene 0 is whatever was seen first (the start screen)."""
    def __init__(self, sigs=None, frozen: bool = False):
        self.sigs = [] if sigs is None else [np.asarray(s) for s in sigs]
        self.frozen = frozen

    def match(self, sig: np.ndarray) -> int:
        """Index of the matching scene; adds a new scene when nothing matches (returns -1 instead if frozen)."""
        if self.sigs:
            d = [sig_dist(sig, s) for s in self.sigs]
            j = int(np.argmin(d))
            if d[j] < MATCH_FRAC:
                return j
        if self.frozen:
            return -1
        self.sigs.append(sig.copy())
        return len(self.sigs) - 1

    def save(self, path):
        np.savez_compressed(path, sigs=np.stack(self.sigs))

    @classmethod
    def load(cls, path, frozen: bool = True):
        return cls(list(np.load(path)["sigs"]), frozen=frozen)


class SceneTracker:
    """Per-environment online tracker. Feed it every decision's newest frame and game reward; it returns the landmarks
    that happened at that step. Call reset() with the first frame of each new game."""
    def __init__(self, codebook: Codebook):
        self.cb = codebook
        self.reset(None)

    def reset(self, frame84):
        self.prev = None
        self.n_rewards = 0
        self.scene = -1
        self.scenes_seen = set()
        if frame84 is not None:
            self.prev = frame84
            self.scene = self.cb.match(signature(frame84))
            self.scenes_seen.add(self.scene)

    def step(self, frame84, reward: float):
        ev = []
        if reward > 0:
            self.n_rewards += 1
            ev.append(("r", self.n_rewards))
        if self.prev is not None and changed_frac(self.prev, frame84) > CUT_FRAC:
            s = self.cb.match(signature(frame84))      # the frame at the cut already shows the new scene
            if s != self.scene and s >= 0:
                self.scene = s
                self.scenes_seen.add(s)
                ev.append(("s", s))
        self.prev = frame84
        return ev


def trace_events(frames: np.ndarray, rewards: np.ndarray, cb: Codebook):
    """Events of one saved episode: list of (t, kind, value). frames: (T, 84, 84) newest frame of each decision."""
    tr = SceneTracker(cb)
    tr.reset(frames[0])
    out = [(0, "s", tr.scene)]
    for t in range(1, len(frames)):
        for k, v in tr.step(frames[t], float(rewards[t - 1])):   # rewards[t-1] is the reward that led to frame t
            out.append((t, k, v))
    return out
