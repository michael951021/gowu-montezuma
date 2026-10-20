"""Networks: Nature-CNN torso, RND (Sec. 5.2), zero-target bootstrapped particle ensemble (Sec. 5.3,
App. E.1), and the IMPALA ResNet actor-critic used by the backward algorithm (App. G.4)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ortho_(m: nn.Module, gain: float = math.sqrt(2)) -> nn.Module:
    for p in m.modules():
        if isinstance(p, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(p.weight, gain)
            nn.init.zeros_(p.bias)
    return m


def nature_torso(c_in: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, 32, 8, 4), nn.ReLU(),
        nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        nn.Flatten())  # 64*7*7 = 3136 for 84x84


# ---------------------------------------------------------------- RND
class RunningNorm(nn.Module):
    """Per-pixel running mean/std (Burda et al. 2018), output clipped to [-5, 5]."""

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.float()
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta.pow(2) * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x.float() - self.mean) / (self.var.sqrt() + 1e-8)).clamp(-5, 5)


class RNDNet(nn.Module):
    def __init__(self, c_in: int, hidden: int = 1024, out: int = 1024):
        super().__init__()
        self.net = ortho_(nn.Sequential(nature_torso(c_in), nn.Linear(3136, hidden), nn.ReLU(),
                                        nn.Linear(hidden, out)))

    def forward(self, x):
        return self.net(x)


class RND(nn.Module):
    """U(s) = || predictor(s) - target(s) ||^2 / d. Input: uint8 (B, c, 84, 84)."""

    def __init__(self, c_in: int = 1):
        super().__init__()
        self.norm = RunningNorm((c_in, 84, 84))
        self.target = RNDNet(c_in)
        self.predictor = RNDNet(c_in)
        for p in self.target.parameters():
            p.requires_grad_(False)

    def error(self, x):
        z = self.norm(x)
        return (self.predictor(z) - self.target(z).detach()).pow(2).mean(-1)

    @torch.no_grad()
    def uncertainty(self, x):
        return self.error(x)


# -------------------------------------------- zero-target particle ensemble
class EnsembleQ(nn.Module):
    """N independent f_psi(s, a) networks (Nature torso + 512 fc + |A| outputs), evaluated in one call
    with grouped convolutions. Trained to output 0 on bootstrap-masked data (App. E.1)."""

    def __init__(self, n_members: int, c_in: int, n_actions: int):
        super().__init__()
        n = self.n = n_members
        self.c_in = c_in
        self.convs = nn.ModuleList([
            nn.Conv2d(c_in * n, 32 * n, 8, 4, groups=n),
            nn.Conv2d(32 * n, 64 * n, 4, 2, groups=n),
            nn.Conv2d(64 * n, 64 * n, 3, 1, groups=n)])
        self.w1 = nn.Parameter(torch.empty(n, 3136, 512))
        self.b1 = nn.Parameter(torch.zeros(n, 512))
        self.w2 = nn.Parameter(torch.empty(n, 512, n_actions))
        self.b2 = nn.Parameter(torch.zeros(n, n_actions))
        for c in self.convs:
            nn.init.orthogonal_(c.weight, math.sqrt(2))
            nn.init.zeros_(c.bias)
        for w in (self.w1, self.w2):
            for k in range(n):
                nn.init.orthogonal_(w.data[k], math.sqrt(2))

    def forward(self, x):
        """x: (B, n, c, 84, 84) float -> (B, n, |A|): member k applied to x[:, k]."""
        B = x.shape[0]
        h = x.reshape(B, self.n * self.c_in, 84, 84)
        for c in self.convs:
            h = F.relu(c(h))
        h = h.reshape(B, self.n, 3136)
        h = F.relu(torch.einsum("bni,nio->bno", h, self.w1) + self.b1)
        return torch.einsum("bni,nio->bno", h, self.w2) + self.b2


# --------------------------------------------------- IMPALA actor-critic
class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c2(F.relu(self.c1(F.relu(x))))


class ImpalaEncoder(nn.Module):
    def __init__(self, c_in: int, dense: int = 1024, chans=(16, 32, 32)):
        super().__init__()
        layers, c = [], c_in
        for co in chans:
            layers += [nn.Conv2d(c, co, 3, padding=1), nn.MaxPool2d(3, 2, padding=1), _Res(co), _Res(co)]
            c = co
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c * 11 * 11, dense)  # 84 -> 42 -> 21 -> 11

    def forward(self, x):
        return F.relu(self.fc(F.relu(self.conv(x)).flatten(1)))


class ImpalaActorCritic(nn.Module):
    """Separate policy and value encoders, dense 1024, heads 512 (GoWU App. G.4)."""

    def __init__(self, c_in: int, n_actions: int, dense: int = 1024, head: int = 512):
        super().__init__()
        self.pi_enc = ImpalaEncoder(c_in, dense)
        self.v_enc = ImpalaEncoder(c_in, dense)
        self.pi = nn.Sequential(nn.Linear(dense, head), nn.ReLU(), nn.Linear(head, n_actions))
        self.v = nn.Sequential(nn.Linear(dense, head), nn.ReLU(), nn.Linear(head, 1))
        nn.init.orthogonal_(self.pi[-1].weight, 0.01)
        nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def forward(self, x_uint8):
        x = x_uint8.float() / 255.0
        return self.pi(self.pi_enc(x)), self.v(self.v_enc(x)).squeeze(-1)
