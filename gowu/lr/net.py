"""π_R / V_R: goal-conditioned IMPALA actor-critic (PLAN §3-4).

Observation channels: 4-frame stack, last-reward frame (GoWU phase II's 5th channel) and, with
goal_encoder in {frame, both}, the goal landmark's frame (E4 with early fusion: the policy's own torso
reads the goal image). A context vector (ancestor mask, reward still needed, time, optional E5 RAM goal)
is concatenated after the torso. Separate policy and value encoders, as in GoWU phase II.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..nets import ImpalaEncoder


class GoalActorCritic(nn.Module):
    def __init__(self, c_in: int, ctx_dim: int, n_actions: int, dense: int = 1024, head: int = 512):
        super().__init__()
        self.pi_enc = ImpalaEncoder(c_in, dense)
        self.v_enc = ImpalaEncoder(c_in, dense)
        self.pi = nn.Sequential(nn.Linear(dense + ctx_dim, head), nn.ReLU(), nn.Linear(head, n_actions))
        self.v = nn.Sequential(nn.Linear(dense + ctx_dim, head), nn.ReLU(), nn.Linear(head, 1))
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.zeros_(self.pi[-1].bias)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def policy(self, x_uint8, ctx):
        x = x_uint8.float() / 255.0
        return self.pi(torch.cat([self.pi_enc(x), ctx], -1))

    def forward(self, x_uint8, ctx):
        x = x_uint8.float() / 255.0
        logits = self.pi(torch.cat([self.pi_enc(x), ctx], -1))
        v = self.v(torch.cat([self.v_enc(x), ctx], -1)).squeeze(-1)
        return logits, v
