"""Random Network Distillation intrinsic reward (Burda et al. 2018) --
the plan's recommended default exploration mechanism (section 6, Option A).
Operates on the single most recent frame (not the full 4-frame stack),
matching the original paper. Must be used identically across every
training condition (plan section 6's hard rule).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RunningMeanStd:
    """Welford-style incremental mean/variance, for observation and
    intrinsic-reward normalization."""
    def __init__(self, shape=(), eps: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean, self.var, self.count = new_mean, M2 / tot_count, tot_count


def _rnd_cnn(in_channels: int = 1, out_dim: int = 512):
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(0.2),
        nn.Flatten(),
        nn.LazyLinear(out_dim),
    )


def _rnd_predictor_paper(in_channels: int = 1, out_dim: int = 512, hidden: int = 512):
    """The predictor of Burda et al.'s released code (policies/cnn_policy_param_matched.py, define_self_prediction_rew):
    the same conv encoder as the target, then THREE dense layers (hidden, hidden, out_dim) with ReLU between them, where
    the target has a single linear layer after its convs. hidden = 256 * enlargement(2) = 512 in the default 'normal'
    policy size. Orthogonal init (gain sqrt(2)), zero biases, as in the released code. Fully materialised (no lazy layers):
    the conv stack ends at 64 * 7 * 7 = 3136 for an 84x84 input."""
    net = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.LeakyReLU(0.2),
        nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(0.2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(0.2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
            nn.init.zeros_(m.bias)
    return net


PREDICTOR_ARCHS = ("legacy", "paper")


class RNDModel(nn.Module):
    """Frozen target net + trainable predictor net, same architecture.
    Intrinsic reward = predictor's MSE against the (frozen) target on a
    normalized single-frame observation -- high on states unlike anything
    the predictor has already learned to imitate."""
    def __init__(self, out_dim: int = 512, device: str = "cpu", predictor_arch: str = "legacy"):
        """predictor_arch: "legacy" = a copy of the target (convs + one linear layer; what every run before 2026-10-04
        used, and still the default so older scripts and checkpoints keep working); "paper" = the deeper predictor of
        Burda et al.'s code (see _rnd_predictor_paper). The target is the same in both."""
        super().__init__()
        if predictor_arch not in PREDICTOR_ARCHS:
            raise ValueError(f"predictor_arch must be one of {PREDICTOR_ARCHS}, got {predictor_arch!r}")
        self.predictor_arch = predictor_arch
        self.target = _rnd_cnn(1, out_dim).to(device)
        self.predictor = (_rnd_predictor_paper(1, out_dim) if predictor_arch == "paper" else _rnd_cnn(1, out_dim)).to(device)
        for p in self.target.parameters():
            p.requires_grad = False
        self.device = device
        self.obs_rms = RunningMeanStd(shape=(1, 84, 84))
        self.reward_rms = RunningMeanStd(shape=())

    def _last_frame(self, obs_u8_nhwc: torch.Tensor) -> np.ndarray:
        """obs_u8_nhwc: (N, 84, 84, C) uint8 -> (N, 1, 84, 84) float, the
        single most recent frame (last channel of the stack)."""
        last = obs_u8_nhwc[..., -1].cpu().numpy().astype(np.float64)  # (N, 84, 84)
        return last[:, None, :, :]

    def update_obs_stats(self, obs_u8_nhwc: torch.Tensor):
        frame = self._last_frame(obs_u8_nhwc)
        self.obs_rms.update(frame.reshape(frame.shape[0], -1).reshape(-1, 1, 84, 84))

    def _normalize_obs(self, frame: np.ndarray) -> torch.Tensor:
        mean = self.obs_rms.mean.reshape(1, 84, 84)
        std = np.sqrt(self.obs_rms.var).reshape(1, 84, 84)
        norm = (frame - mean) / (std + 1e-8)
        norm = np.clip(norm, -5.0, 5.0)
        return torch.from_numpy(norm).float().to(self.device)

    def intrinsic_reward(self, obs_u8_nhwc: torch.Tensor) -> torch.Tensor:
        """Returns normalized (by running std, not centered -- RND paper
        convention: intrinsic reward is non-negative) intrinsic reward per
        sample, WITHOUT updating running stats (call update_obs_stats and
        update_reward_stats separately, once per rollout, not per call).
        Pure inference (no_grad internally) -- predictor training happens
        separately via predictor_loss(), which does need gradients. This
        must not depend on being called inside an outer no_grad block."""
        mse, _ = self.intrinsic_reward_with_features(obs_u8_nhwc)
        return mse

    def intrinsic_reward_with_features(self, obs_u8_nhwc: torch.Tensor):
        """Same as intrinsic_reward, but ALSO returns the frozen target
        net's feature vectors ((N, out_dim) tensor) alongside the per-sample
        MSE -- they're already materialized in the same forward pass, so
        exposing them is free. Added for the save-gating redesign's
        episodic-novelty SimHash (train_live_dqn_shared.py): the target
        features are a stable (frozen-net) pixel-derived state fingerprint,
        exactly what within-episode revisit detection needs without touching
        RAM features. A separate method (not a signature change on
        intrinsic_reward) so the 8 legacy call sites across ppo.py/
        train_live.py/train_live_dqn.py/train_live_dqn_shared_mcts.py keep
        working untouched."""
        frame = self._last_frame(obs_u8_nhwc)
        norm_t = self._normalize_obs(frame)
        with torch.no_grad():
            target_out = self.target(norm_t)
            predictor_out = self.predictor(norm_t)
            mse = ((predictor_out - target_out) ** 2).mean(dim=1)
        return mse, target_out

    def update_reward_stats(self, raw_intrinsic: np.ndarray):
        self.reward_rms.update(raw_intrinsic.reshape(-1))

    def normalize_reward(self, raw_intrinsic: np.ndarray) -> np.ndarray:
        return raw_intrinsic / (np.sqrt(self.reward_rms.var) + 1e-8)

    def predictor_loss(self, obs_u8_nhwc: torch.Tensor) -> torch.Tensor:
        """Differentiable version for the actual RND training step."""
        frame = self._last_frame(obs_u8_nhwc)
        norm_t = self._normalize_obs(frame)
        with torch.no_grad():
            target_out = self.target(norm_t)
        predictor_out = self.predictor(norm_t)
        return ((predictor_out - target_out) ** 2).mean()
