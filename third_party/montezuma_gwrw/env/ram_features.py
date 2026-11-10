"""Symbolic progress features extracted from Montezuma's Revenge RAM.

Byte offsets are the widely-used values from the Montezuma's Revenge RL
literature (e.g. the "Atari Annotated RAM Interface" family of RAM maps
used across Go-Explore / unsupervised-state-representation papers), NOT
independently re-derived byte-by-byte in this project -- with one
exception: `LIVES_BYTE=58` was empirically confirmed here directly
(correlation 0.9999 against the environment's own `info["lives"]` across a
3,000-step rollout; see PROJECT_NOTES.md for the diagnostic script and
output).

`ROOM_BYTE=3` could NOT be independently confirmed the same way: 30
movement-biased episodes and one 3,000-step random rollout never left the
starting room (byte 3 stayed at 1 throughout) -- expected, since escaping
Montezuma's first room requires precise platforming random/biased play
essentially never achieves; this is not evidence the byte is wrong. Trust
level: literature-standard, unconfirmed independently. Self-check: log
`unique room values seen so far` during real training -- if an agent's
score/achievements are clearly progressing but this set never grows beyond
{1}, that is the signal to stop and re-diagnose (e.g. render frames and
visually confirm, or search for the byte among ones that change exactly
when score/x-position jump discontinuously, consistent with a room
transition).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROOM_BYTE = 3
X_BYTE = 42
Y_BYTE = 43
LIVES_BYTE = 58  # empirically confirmed (r=0.9999 vs info["lives"]); note a
# consistent off-by-one at reset (byte reads 5 when info["lives"]==6) --
# likely "lives in reserve" vs "total lives" semantics, not a bug; nothing
# here currently depends on the absolute value, only doesn't matter yet.
INVENTORY_BYTE = 65  # bitmask of held items (keys, sword, torch, etc.)

# Known room graph for the original Montezuma's Revenge has ~24 rooms
# (room numbers roughly 0-23 depending on which map variant/version).
NUM_KNOWN_ROOMS = 24


@dataclass
class ProgressState:
    """One step's worth of symbolic progress signal, derived from RAM."""
    room: int
    x: int
    y: int
    lives: int
    inventory: int


def read_progress(ale) -> ProgressState:
    ram = ale.getRAM()
    return ProgressState(
        room=int(ram[ROOM_BYTE]),
        x=int(ram[X_BYTE]),
        y=int(ram[Y_BYTE]),
        lives=int(ram[LIVES_BYTE]),
        inventory=int(ram[INVENTORY_BYTE]),
    )


@dataclass
class EpisodeProgressTracker:
    """Accumulates achievement/coverage signal over one episode. Call
    `.update(progress_state, raw_score)` every step, `.finalize()` at
    episode end to get the achievement dict logged per evaluation episode
    (matches the plan's section 38 JSON schema)."""
    rooms_visited: set = field(default_factory=set)
    max_room: int = -1
    room_visit_order: list = field(default_factory=list)
    inventory_events: set = field(default_factory=set)  # distinct inventory bitmask values seen
    max_score: float = 0.0
    steps: int = 0
    _last_room: int | None = None
    _last_inventory: int | None = None

    def update(self, p: ProgressState, raw_score: float) -> None:
        self.steps += 1
        if p.room not in self.rooms_visited:
            self.rooms_visited.add(p.room)
            self.room_visit_order.append(p.room)
        self.max_room = max(self.max_room, p.room)
        if p.inventory != self._last_inventory:
            self.inventory_events.add(p.inventory)
            self._last_inventory = p.inventory
        self._last_room = p.room
        self.max_score = max(self.max_score, raw_score)

    def finalize(self) -> dict:
        return {
            "rooms": sorted(self.rooms_visited),
            "num_rooms": len(self.rooms_visited),
            "max_room": self.max_room,
            "room_visit_order": self.room_visit_order,
            "num_inventory_states": len(self.inventory_events),
            "max_score": self.max_score,
            "steps": self.steps,
        }


def progress_vector(tracker: EpisodeProgressTracker) -> np.ndarray:
    """Fixed-length symbolic feature vector phi(tau) for an episode, per
    plan section 9. Room-visitation is one-hot-ish over NUM_KNOWN_ROOMS
    (clipped/ignored beyond that range -- room numbering isn't guaranteed
    contiguous from 0, this is a coarse coverage signal not a precise map)."""
    room_indicator = np.zeros(NUM_KNOWN_ROOMS, dtype=np.float32)
    for r in tracker.rooms_visited:
        if 0 <= r < NUM_KNOWN_ROOMS:
            room_indicator[r] = 1.0
    extra = np.array([
        len(tracker.rooms_visited),
        tracker.max_room,
        len(tracker.inventory_events),
        tracker.max_score,
        tracker.steps,
    ], dtype=np.float32)
    return np.concatenate([room_indicator, extra])
