"""Atari env with checkpoint/restore, following the GoWU paper's pipeline (Sec. 6.1, App. C.1).

- sticky actions (p = 0.25), applied per emulator frame by ALE
- action repeat 4, max-pool over the last 2 frames, grayscale, bilinear resize to 84x84
- 4-frame stack; optionally a 5th channel holding the frame at the last reward (phase II on MR)
- checkpoint = ALE state + frame stack + bookkeeping, so restore reproduces the observation exactly
"""
from __future__ import annotations

import pickle
import zlib
from dataclasses import dataclass

import cv2
import numpy as np
from ale_py import ALEInterface, LoggerMode, roms

ALEInterface.setLoggerMode(LoggerMode.Error)

OBS_HW = 84

# Montezuma's Revenge RAM layout (same addresses as Go-Explore's montezuma_env). Used only for
# logging / exploit detection / LR-GoWU landmarks; GoWU itself never reads RAM for decisions.
MR_RAM = dict(room=3, x=42, y=43, lives=58, level=57, inventory=65)


@dataclass
class Checkpoint:
    ale_state: bytes          # pickled ALEState
    frames: bytes             # zlib-compressed (stack, 84, 84) uint8
    last_reward_frame: bytes  # zlib-compressed (84, 84) uint8
    lives: int
    episode_steps: int

    def nbytes(self) -> int:
        return len(self.ale_state) + len(self.frames) + len(self.last_reward_frame)


class AtariEnv:
    def __init__(self, game: str = "montezuma_revenge", seed: int = 0, sticky_p: float = 0.25,
                 frame_skip: int = 4, stack: int = 4, full_action_space: bool = True,
                 max_episode_frames: int | None = None):
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", int(seed) % (2**31 - 1))
        self.ale.setFloat("repeat_action_probability", float(sticky_p))
        self.ale.setInt("frame_skip", 1)
        self.ale.setInt("max_num_frames_per_episode", int(max_episode_frames or 0))
        self.ale.loadROM(roms.get_rom_path(game))
        self.actions = (self.ale.getLegalActionSet() if full_action_space
                        else self.ale.getMinimalActionSet())
        self.n_actions = len(self.actions)
        self.frame_skip = frame_skip
        self.stack = stack
        self.is_mr = game == "montezuma_revenge"
        h, w = self.ale.getScreenDims()
        self._raw = [np.empty((h, w), np.uint8), np.empty((h, w), np.uint8)]
        self.frames = np.zeros((stack, OBS_HW, OBS_HW), np.uint8)
        self.last_reward_frame = np.zeros((OBS_HW, OBS_HW), np.uint8)
        self.lives = 0
        self.episode_steps = 0

    # ------------------------------------------------------------------ core
    def _grab(self) -> np.ndarray:
        pooled = np.maximum(self._raw[0], self._raw[1])
        return cv2.resize(pooled, (OBS_HW, OBS_HW), interpolation=cv2.INTER_LINEAR)

    def _push(self, frame: np.ndarray) -> None:
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = frame

    def reset(self) -> np.ndarray:
        self.ale.reset_game()
        self.ale.getScreenGrayscale(self._raw[0])
        self._raw[1][:] = self._raw[0]
        f = self._grab()
        self.frames[:] = f
        self.last_reward_frame[:] = 0
        self.lives = self.ale.lives()
        self.episode_steps = 0
        return self.frames

    def step(self, a: int):
        """Returns (frames, reward, life_lost, game_over). `frames` is a view; copy if you keep it."""
        act = self.actions[a]
        reward = 0.0
        for i in range(self.frame_skip):
            reward += self.ale.act(act)
            if i >= self.frame_skip - 2:
                self.ale.getScreenGrayscale(self._raw[i - (self.frame_skip - 2)])
            if self.ale.game_over():
                if i < self.frame_skip - 2:
                    self.ale.getScreenGrayscale(self._raw[0])
                    self._raw[1][:] = self._raw[0]
                break
        f = self._grab()
        self._push(f)
        if reward != 0:
            self.last_reward_frame[:] = f
        lives = self.ale.lives()
        life_lost = lives < self.lives
        self.lives = lives
        self.episode_steps += 1
        return self.frames, reward, life_lost, self.ale.game_over()

    # ------------------------------------------------------------ snapshots
    def checkpoint(self) -> Checkpoint:
        return Checkpoint(
            ale_state=pickle.dumps(self.ale.cloneState()),
            frames=zlib.compress(self.frames.tobytes(), 1),
            last_reward_frame=zlib.compress(self.last_reward_frame.tobytes(), 1),
            lives=self.lives, episode_steps=self.episode_steps)

    def restore(self, ck: Checkpoint) -> np.ndarray:
        # include_rng=False on clone: restored particles keep their own sticky-action RNG stream,
        # so clones of the same node diverge (aleatoric noise is not frozen).
        self.ale.restoreState(pickle.loads(ck.ale_state))
        self.frames[:] = np.frombuffer(zlib.decompress(ck.frames), np.uint8).reshape(self.frames.shape)
        self.last_reward_frame[:] = np.frombuffer(
            zlib.decompress(ck.last_reward_frame), np.uint8).reshape(OBS_HW, OBS_HW)
        self.lives = ck.lives
        self.episode_steps = ck.episode_steps
        return self.frames

    # ------------------------------------------------------------------ RAM
    def ram_info(self) -> tuple[int, int, int, int, int]:
        """(level, room, x, y, inventory) for MR; zeros otherwise. Diagnostics only."""
        if not self.is_mr:
            return (0, 0, 0, 0, 0)
        r = self.ale.getRAM()
        return (int(r[MR_RAM["level"]]), int(r[MR_RAM["room"]]), int(r[MR_RAM["x"]]),
                int(r[MR_RAM["y"]]), int(r[MR_RAM["inventory"]]))

    def set_lives(self, n: int) -> None:
        """Infinite-lives baselines: overwrite the MR lives byte."""
        self.ale.setRAM(MR_RAM["lives"], n)
        self.lives = self.ale.lives()


def unpack_frames(ck: Checkpoint, stack: int = 4) -> np.ndarray:
    return np.frombuffer(zlib.decompress(ck.frames), np.uint8).reshape(stack, OBS_HW, OBS_HW)


def unpack_reward_frame(ck: Checkpoint) -> np.ndarray:
    return np.frombuffer(zlib.decompress(ck.last_reward_frame), np.uint8).reshape(OBS_HW, OBS_HW)
