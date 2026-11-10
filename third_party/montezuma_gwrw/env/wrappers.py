"""Standard Atari preprocessing wrappers (DQN/OpenAI-baselines style),
plus a progress-tracking wrapper built on env/ram_features.py.

Per the plan's section 6 rule -- the exploration mechanism must be
identical across every condition -- these wrappers must be used
identically everywhere too; nothing here should ever be varied per
condition.
"""
from __future__ import annotations

import os
from collections import deque

import cv2
import gymnasium as gym
import numpy as np

from env.ram_features import EpisodeProgressTracker, read_progress

cv2.ocl.setUseOpenCL(False)

# 2026-08-21, per the user's explicit request ("a dynamic max step budget
# per episode of current number of unique rooms * 2000 or something like
# that"): caps episode length at len(rooms_visited) * this many steps,
# checked every step inside ProgressTrackingWrapper (see its own comment
# for why here specifically). Rationale: this project has repeatedly
# produced pathological episodes that survive for tens of thousands of
# steps without dying AND without discovering a new room (e.g. a real
# 26,995-step, zero-new-rooms episode that contributed to an OOM kill --
# a single such episode's obs array alone is ~750MB, and the combined
# training pool caps by EPISODE COUNT, not byte size, so a handful of
# these can blow memory far past what "N average-sized episodes" would
# cost). Scaling the budget BY rooms visited (not a flat cap) means an
# episode that's genuinely progressing keeps its long horizon, while one
# that's stuck oscillating in the same 1-2 rooms gets cut off early --
# the failure mode observed was never "explores slowly", always
# "survives indefinitely without exploring at all". len(rooms_visited)
# is always >= 1 immediately after reset (the start room seeds it), so
# the effective floor is EPISODE_STEP_BUDGET_PER_ROOM itself, never 0.
#
# Raised 2000 -> 4000 (2026-08-21, per the user's explicit follow-up
# request), after checking 2000 against this project's own actual
# historical room-1 -> room-2 transition data (9 saved desktop_v2_run7
# episodes, pre-cap): median 3,063 steps in room 1 before reaching room 2
# (range 1,063-7,565), with the transition landing right near each
# episode's own natural end every time. 2000 would have truncated 7 of
# those 9 episodes before they ever got there -- confirmed empirically as
# the likely cause of zero room-2 discoveries in the first post-cap run.
# 4000 clears the median comfortably while still cutting off genuine
# runaways (the 26,995-step episode this cap exists for) at a small
# fraction of their length.
EPISODE_STEP_BUDGET_PER_ROOM = int(os.environ.get("EPISODE_STEP_BUDGET_PER_ROOM", 4000))


class NoopResetEnv(gym.Wrapper):
    """Sample a random number (up to noop_max) of NOOPs at reset, for
    stochasticity across episodes that start from the same ROM state."""
    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        meanings = env.unwrapped.get_action_meanings()
        assert meanings[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(n):
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat action `skip` times, max-pool the last 2 raw frames (handles
    Atari sprite flicker), sum reward over the skipped frames."""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        shape = env.observation_space.shape
        self._obs_buffer = np.zeros((2, *shape), dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for i in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self.skip - 2:
                self._obs_buffer[0] = obs
            if i == self.skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    """RGB -> grayscale -> resize to 84x84."""
    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class FrameStack(gym.ObservationWrapper):
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=-1)


class ClipRewardEnv(gym.RewardWrapper):
    """Sign-clip reward for training stability. Raw (unclipped) score is
    preserved separately by ProgressTrackingWrapper, which must wrap
    OUTSIDE this one so it sees the true reward."""
    def reward(self, reward):
        return float(np.sign(reward))


class ProgressTrackingWrapper(gym.Wrapper):
    """Reads symbolic RAM progress features every step (env/ram_features.py),
    accumulates them into an EpisodeProgressTracker, and exposes:
      info["raw_score"]        -- cumulative true (unclipped) episode return
      info["progress_state"]   -- this step's ProgressState (room/x/y/lives/inventory)
      info["episode_achievements"] -- only present on the terminal step; the
                                       finalized achievement dict (plan section 38 schema)
    Must wrap OUTSIDE any reward-clipping wrapper so raw_score is the true
    game score, not the clipped training reward."""
    def __init__(self, env):
        super().__init__(env)
        self._tracker = EpisodeProgressTracker()
        self._raw_score = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._tracker = EpisodeProgressTracker()
        self._raw_score = 0.0
        p = read_progress(self.unwrapped.ale)
        self._tracker.update(p, self._raw_score)
        info["raw_score"] = self._raw_score
        info["progress_state"] = p
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._raw_score += float(reward)
        p = read_progress(self.unwrapped.ale)
        self._tracker.update(p, self._raw_score)
        info["raw_score"] = self._raw_score
        info["progress_state"] = p
        # 2026-08-21: dynamic per-episode step budget -- see
        # EPISODE_STEP_BUDGET_PER_ROOM's own module comment for the full
        # rationale. Checked AFTER tracker.update() so this step's own
        # room (if newly discovered) already counts toward the budget
        # before the comparison, and marked `truncated` (not
        # `terminated`) since this is an artificial cutoff, not a real
        # game-over -- matches gymnasium's own terminated/truncated
        # convention, and downstream code (vec_env_mp.py's worker) treats
        # `done = terminated or truncated` identically either way, so the
        # auto-reset and terminal_achievements plumbing both fire exactly
        # as they would for a natural death.
        if self._tracker.steps > len(self._tracker.rooms_visited) * EPISODE_STEP_BUDGET_PER_ROOM:
            truncated = True
        if terminated or truncated:
            info["episode_achievements"] = self._tracker.finalize()
        return obs, reward, terminated, truncated, info
