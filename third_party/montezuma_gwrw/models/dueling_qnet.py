"""Dueling DQN architecture (Wang et al. 2016) -- Q(s,a) = V(s) + A(s,a),
with V(s) exposed as a genuinely separate head. This is exactly the "VAN"
(value-action-network) structure the comparative-value specialization
mechanism (algorithms/competition_van.py) needs: it reads V_i(s) directly
off this net's value head, for both the acting agent and (via a slow
target-network copy) its opponent.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _orthogonal_init(layer, gain=np.sqrt(2)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class DuelingQNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_actions: int = 18):
        super().__init__()
        self.num_actions = num_actions
        self.conv = nn.Sequential(
            _orthogonal_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _orthogonal_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _orthogonal_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.conv(dummy).shape[1]
        self.trunk = nn.Sequential(_orthogonal_init(nn.Linear(flat_dim, 512)), nn.ReLU())
        self.value_head = _orthogonal_init(nn.Linear(512, 1), gain=1.0)
        self.advantage_head = _orthogonal_init(nn.Linear(512, num_actions), gain=0.01)

    def forward(self, obs_u8_nhwc: torch.Tensor):
        """Returns (q, v) -- q: (N, num_actions), v: (N,).
        Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)] (Wang et al.'s
        identifiability-fixing combination -- without the mean subtraction,
        V and A aren't uniquely determined by Q alone)."""
        x = obs_u8_nhwc.permute(0, 3, 1, 2).float() / 255.0
        h = self.trunk(self.conv(x))
        v = self.value_head(h).squeeze(-1)
        a = self.advantage_head(h)
        q = v.unsqueeze(-1) + (a - a.mean(dim=-1, keepdim=True))
        return q, v

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _build_conv_stack(in_channels: int, conv_channels):
    """Nature-DQN-style first three conv layers (kernel/stride 8/4, 4/2,
    3/1, which is what actually shrinks the 84x84 input down), then any
    ADDITIONAL channels get kernel=3/stride=1/padding=1 conv layers that add
    genuine depth without shrinking spatial resolution further -- lets the
    stack go deeper than 3 layers without collapsing to a 1x1 feature map."""
    layers = []
    c_in = in_channels
    for i, c_out in enumerate(conv_channels):
        if i == 0:
            layers.append(_orthogonal_init(nn.Conv2d(c_in, c_out, kernel_size=8, stride=4)))
        elif i == 1:
            layers.append(_orthogonal_init(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2)))
        elif i == 2:
            layers.append(_orthogonal_init(nn.Conv2d(c_in, c_out, kernel_size=3, stride=1)))
        else:
            layers.append(_orthogonal_init(nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)))
        layers.append(nn.ReLU())
        c_in = c_out
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def _make_readout_head(in_dim: int, hidden_dim: int, num_layers: int, out_dim: int, final_gain: float):
    """num_layers total Linear layers: (num_layers-1) x [Linear+ReLU]
    (in_dim -> hidden_dim -> ... -> hidden_dim), then one final
    Linear(hidden_dim, out_dim) with NO activation -- final_gain sets that
    last layer's init gain (1.0 for the value head's unbounded scalar
    estimate, matching the original single-linear value head's init; 0.01
    for the advantage head, matching Wang et al.'s small-init convention so
    advantage starts near-flat across actions rather than dominating Q at
    init). num_layers=1 reduces to the original single-linear-readout head.
    Structurally identical to _make_head_body except the final layer skips
    the trailing ReLU and uses final_gain instead of _orthogonal_init's
    ReLU-tuned default gain=sqrt(2)."""
    layers = []
    d = in_dim
    for _ in range(num_layers - 1):
        layers.append(_orthogonal_init(nn.Linear(d, hidden_dim)))
        layers.append(nn.ReLU())
        d = hidden_dim
    layers.append(_orthogonal_init(nn.Linear(d, out_dim), gain=final_gain))
    return nn.Sequential(*layers)


def load_compat_state_dict(net: nn.Module, state_dict: dict) -> list:
    """load_state_dict, but tolerant of an architecture change to a subset
    of params (e.g. widening value_heads from a single linear layer to a
    multi-layer MLP, per the user's explicit choice): any state_dict key
    whose shape no longer matches the live module is DROPPED (leaving that
    parameter at its freshly-initialized value) instead of raising, while
    every shape-matching key (trunk, head_bodies, advantage_heads, and any
    value_heads key whose shape happens to still match) loads normally --
    so all learned representation/policy progress carries over, and only
    the now-incompatible params retrain from scratch.

    Returns the list of dropped keys (empty if nothing was incompatible) --
    callers MUST check this before loading a checkpointed OPTIMIZER state
    against the same net: torch's Adam.load_state_dict maps saved state to
    live parameters POSITIONALLY (zipped by index across the full
    parameter list, not by name), so a parameter-COUNT change anywhere
    upstream of the end of that list (e.g. value_heads growing from 2
    tensors/key to 8) silently misaligns every later parameter's momentum
    buffers (advantage_heads', here) rather than raising -- there is no
    safe partial-load equivalent for optimizer state the way there is for
    the net's own state_dict, so any non-empty return here means the
    caller should skip loading that optimizer state entirely and use a
    freshly-initialized one instead."""
    own_state = net.state_dict()
    compatible = {}
    dropped = []
    for key, val in state_dict.items():
        if key in own_state and own_state[key].shape == val.shape:
            compatible[key] = val
        else:
            dropped.append(key)
    net.load_state_dict(compatible, strict=False)
    if dropped:
        preview = dropped[:6]
        print(f"load_compat_state_dict: reinitializing {len(dropped)} shape-mismatched "
              f"param(s) (kept everything else): {preview}"
              f"{'...' if len(dropped) > len(preview) else ''}", flush=True)
    return dropped


def _make_head_body(in_dim: int, hidden_dim: int, num_hidden_layers: int):
    """num_hidden_layers stacked Linear+ReLU blocks, in_dim -> hidden_dim ->
    ... -> hidden_dim. This is the per-head specialization depth sitting on
    top of the shared trunk features -- deep enough to let each head express
    a genuinely different nonlinear function of those shared features (a
    single linear readout can only take different linear combinations of
    the same features, which is thin for something like spec's comparative/
    relative target vs. task's blended absolute one), while staying pure
    MLP: the conv layers (actual visual feature extraction) stay in the
    trunk only, per the user's explicit choice not to duplicate them here."""
    layers = []
    d = in_dim
    for _ in range(num_hidden_layers):
        layers.append(_orthogonal_init(nn.Linear(d, hidden_dim)))
        layers.append(nn.ReLU())
        d = hidden_dim
    return nn.Sequential(*layers), d


class MultiHeadDuelingQNet(nn.Module):
    """Shared conv+trunk feature extractor with a separate, deep MLP head
    per key (HEAD_HIDDEN_LAYERS Linear+ReLU blocks, then a (value,
    advantage) output pair) -- per the user's explicit redesign request
    that net_task/net_spec/net_extrinsic, AND agent A/agent B, share ONE
    representation and only split at the final layers. Every head's Double
    DQN loss backpropagates through the SAME trunk parameters (there is
    exactly one trunk in the whole run, not one per net or one per agent as
    the previous per-agent DuelingQNet-per-net design had) -- see
    train_live_dqn_shared.py's module docstring for the full design
    rationale and its consequences for what "competition" means here."""
    def __init__(self, head_keys, in_channels: int = 4, num_actions: int = 18,
                 conv_channels=(64, 128, 128, 128), trunk_dim: int = 1024, trunk_layers: int = 2,
                 head_hidden_dim: int = 256, head_hidden_layers: int = 5, head_value_layers: int = 4,
                 head_advantage_layers: int = 4, use_gru: bool = True, gru_alpha_init: float = 0.0):
        """conv_channels/trunk_dim/trunk_layers default to a WIDER (2x conv
        channels, 2x bottleneck vs. the original DuelingQNet) and DEEPER (4
        conv layers instead of 3, 2 trunk linear layers instead of 1) trunk
        than this class's first version -- ~11.1M total params at these
        defaults, deliberately landing close to parity with the OLD 6-fully-
        separate-DuelingQNet design's ~10.16M total, as a capacity-controlled
        test of whether the shared-trunk redesign's earlier results were
        limited by trunk capacity rather than the shared-representation idea
        itself (the original shared trunk was only 1.68M params -- the same
        size as ONE of the six old trunks, now serving all six heads).

        head_value_layers=4 (was implicitly 1, a bare Linear(head_hidden_dim,
        1)): per the user's explicit request, the value head is now its own
        small MLP instead of a single linear readout -- see
        forward_heads_with_isolated_value's docstring for why this matters
        specifically for the isolated-value training path (BC value
        regression), where v_isolated previously could only ever be a
        LINEAR function of head_bodies[key]'s output; it now has genuine
        nonlinear capacity of its own while staying exactly as isolated
        from head_bodies/trunk as before (still detached at the same
        point, see _head_from_features vs. this method).

        head_advantage_layers=4 (was implicitly 1, a bare
        Linear(head_hidden_dim, num_actions)): per the user's explicit
        follow-up request, the advantage/action head gets the same
        treatment as the value head above, for symmetry -- both final
        readouts off head_bodies[key]'s output are now equally deep. Unlike
        the value head, advantage_heads is never read through the isolated-
        value path, so this doesn't interact with forward_heads_with_
        isolated_value at all -- it only deepens the ordinary (non-isolated)
        q/a computation both forward_heads and forward_heads_with_isolated_
        value already do."""
        super().__init__()
        self.num_actions = num_actions
        self.head_keys = list(head_keys)
        self.conv = _build_conv_stack(in_channels, conv_channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.conv(dummy).shape[1]
        self.trunk, trunk_out_dim = _make_head_body(flat_dim, trunk_dim, trunk_layers)

        # 2026-09-02, per explicit request: a GRU between the shared trunk and
        # the per-head MLPs, matching the RECURRENT variant of Burda et al.'s
        # RND ("The RNN architecture additionally contains GRU cells to capture
        # longer contexts") -- the variant that produced that paper's actual
        # Montezuma's Revenge results. This port had only ever implemented the
        # paper's CNN-only ablation, which the paper itself reports as unstable
        # across seeds and explicitly does not recommend comparing against.
        #
        # Placement (trunk -> GRU -> heads, NOT conv -> GRU -> trunk) is what
        # makes a warm start possible at all: the GRU is trunk_dim -> trunk_dim,
        # so every head's input width is unchanged and all 9 head stacks
        # (task/extrinsic/spec x a/b/c) load from the pre-GRU checkpoint intact.
        #
        # RESIDUAL with a learnable scalar gate (`gru_alpha`) initialized to
        # ZERO, so at load time out == trunk features EXACTLY -- the new net is
        # bit-identical to the old one on its first forward, and ~20 days of
        # learned conv/trunk/head weights keep their exact meaning instead of
        # being fed a randomly-initialized GRU's output. Recurrence then phases
        # itself in as gru_alpha moves off zero (ReZero-style: dL/d_alpha is
        # nonzero immediately even though the GRU's own weights see no gradient
        # until alpha does, so it bootstraps on its own). ONE GRU shared across
        # all agents and all three streams, mirroring this class's existing
        # choice to share conv+trunk and split only at the head MLPs.
        self.use_gru = bool(use_gru)
        self.gru_hidden_dim = trunk_out_dim
        if self.use_gru:
            self.gru = nn.GRUCell(trunk_out_dim, trunk_out_dim)
            self.gru_alpha = nn.Parameter(torch.tensor(float(gru_alpha_init)))

        self.head_bodies = nn.ModuleDict()
        self.value_heads = nn.ModuleDict()
        self.advantage_heads = nn.ModuleDict()
        for k in head_keys:
            body, body_out_dim = _make_head_body(trunk_out_dim, head_hidden_dim, head_hidden_layers)
            self.head_bodies[k] = body
            self.value_heads[k] = _make_readout_head(
                body_out_dim, head_hidden_dim, head_value_layers, out_dim=1, final_gain=1.0)
            self.advantage_heads[k] = _make_readout_head(
                body_out_dim, head_hidden_dim, head_advantage_layers, out_dim=num_actions, final_gain=0.01)

    def features(self, obs_u8_nhwc: torch.Tensor) -> torch.Tensor:
        """PRE-GRU trunk features. Deliberately NOT recurrent: this is the
        stateless conv+trunk representation, which is what trunk-aliasing
        detection (trunk_erase_keep_last) genuinely wants. Every POLICY path
        must go through features_rec/gru_step instead, or it silently drops
        the recurrent state -- see use_gru's comment in __init__."""
        x = obs_u8_nhwc.permute(0, 3, 1, 2).float() / 255.0
        return self.trunk(self.conv(x))

    # alias, for call sites that want to be explicit about which of the two
    # representations they mean now that both exist
    trunk_features = features

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.gru_hidden_dim, device=device)

    def gru_step(self, feat: torch.Tensor, h_prev: torch.Tensor | None):
        """ONE recurrent timestep on already-computed trunk features.

        Split out from features_rec so batched callers can run the (expensive,
        non-recurrent) conv+trunk ONCE over a whole (T*B) block and then loop
        only this (cheap, 1024x1024) cell over T -- otherwise recurrence would
        cost T separate conv passes. Returns (output_features, h_new) where
        output = feat + gru_alpha * h_new (see __init__ on the zero-init gate).
        """
        if not self.use_gru:
            return feat, h_prev
        if h_prev is None:
            h_prev = torch.zeros(feat.shape[0], self.gru_hidden_dim,
                                  device=feat.device, dtype=feat.dtype)
        h_new = self.gru(feat, h_prev)
        return feat + self.gru_alpha * h_new, h_new

    def features_rec(self, obs_u8_nhwc: torch.Tensor, h_prev: torch.Tensor | None = None):
        """Single-timestep recurrent features: (output_features, h_new)."""
        return self.gru_step(self.features(obs_u8_nhwc), h_prev)

    def heads_from_feat(self, feat: torch.Tensor, head_key: str, isolate_value: bool = False):
        """(logits, value) for one head off already-computed (post-GRU)
        features -- the PPO/BC readout, no dueling recombination. Mirrors
        train_live_ppo_shared_competitive_split.stream_logits' body so both
        can share one feature computation across all three heads."""
        hh = self.head_bodies[head_key](feat)
        logits = self.advantage_heads[head_key](hh)
        value = self.value_heads[head_key](hh.detach() if isolate_value else hh).squeeze(-1)
        return logits, value

    def head_q_from_feat(self, feat: torch.Tensor, head_key: str):
        """Full dueling (q, v) off already-computed (post-GRU) features."""
        hh = self.head_bodies[head_key](feat)
        v = self.value_heads[head_key](hh).squeeze(-1)
        a = self.advantage_heads[head_key](hh)
        return v.unsqueeze(-1) + (a - a.mean(dim=-1, keepdim=True)), v

    def forward_head(self, obs_u8_nhwc: torch.Tensor, head_key: str):
        """Same (q, v) contract as DuelingQNet.forward, for one head."""
        h = self.features(obs_u8_nhwc)
        return self._head_from_features(h, head_key)

    def forward_heads(self, obs_u8_nhwc: torch.Tensor, head_keys):
        """Compute the shared TRUNK features ONCE, evaluate several heads off
        it -- avoids redundant conv+trunk forward passes when action
        selection needs more than one head for the same obs (e.g. task_a +
        spec_a every rollout step). Each head's own deep MLP body still runs
        separately per head (that's the whole point -- it's where they stop
        sharing compute)."""
        h = self.features(obs_u8_nhwc)
        return {k: self._head_from_features(h, k) for k in head_keys}

    def _head_from_features(self, h: torch.Tensor, head_key: str):
        hh = self.head_bodies[head_key](h)
        v = self.value_heads[head_key](hh).squeeze(-1)
        a = self.advantage_heads[head_key](hh)
        q = v.unsqueeze(-1) + (a - a.mean(dim=-1, keepdim=True))
        return q, v

    def forward_heads_with_isolated_value(self, obs_u8_nhwc: torch.Tensor, head_keys):
        """Same (q, v) as forward_heads for each key, PLUS a third value
        estimate (v_isolated) computed from a DETACHED copy of that head's
        own body output (hh) -- so a loss on v_isolated backprops ONLY into
        value_heads[key]'s own layers (all of them, now that it's a
        head_value_layers-deep MLP rather than a single linear layer),
        never into head_bodies[key] or the shared trunk. q and v are completely
        unaffected (still the normal, fully-connected computation) -- this
        is purely an ADDITIONAL, isolated readout for auxiliary value-
        regression losses that shouldn't compete with the action
        (advantage/logit) objective for the shared representation. Built
        after direct evidence that a non-isolated value-regression loss,
        even at a loss magnitude fully comparable to the action loss,
        measurably degraded action accuracy over training (see
        training/bc_pretrain_value_regression.py's v1/v2 comparison) --
        confirming genuine gradient competition for shared parameters, not
        just a loss-scale imbalance. Trade-off: v_isolated can only be a
        function of whatever FIXED features the action loss already
        produces at head_bodies[key]'s output -- value_heads[key] itself
        now has real depth (head_value_layers, default 4) so it isn't
        limited to a linear readout of those features the way the original
        single-linear value head was, but it still can't shape
        head_bodies[key]'s output itself to be more useful for value
        prediction, the way an unisolated value loss could (if it didn't
        also hurt the action objective)."""
        h = self.features(obs_u8_nhwc)
        out = {}
        for k in head_keys:
            hh = self.head_bodies[k](h)
            v = self.value_heads[k](hh).squeeze(-1)
            a = self.advantage_heads[k](hh)
            q = v.unsqueeze(-1) + (a - a.mean(dim=-1, keepdim=True))
            v_isolated = self.value_heads[k](hh.detach()).squeeze(-1)
            out[k] = (q, v, v_isolated)
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HeadView:
    """Thin, non-nn.Module view onto ONE head of a MultiHeadDuelingQNet, so
    existing code written against a standalone qnet(obs)->(q,v) with its own
    .parameters() (algorithms/dqn.py's act_boltzmann/double_dqn_update,
    algorithms/competition_van.py's batch_values) works completely
    unmodified -- while every forward/backward call still runs through, and
    every optimizer step still updates, the ONE shared trunk underneath, not
    a private copy of it. .parameters() intentionally returns the FULL
    net's parameters (trunk + every head), not just this head's -- harmless,
    since gradient-norm clipping only touches parameters that actually have
    a grad, and only the trunk + this head will after a forward/backward
    through this view."""
    def __init__(self, net: MultiHeadDuelingQNet, head_key: str):
        self.net = net
        self.head_key = head_key

    def __call__(self, obs_u8_nhwc: torch.Tensor):
        return self.net.forward_head(obs_u8_nhwc, self.head_key)

    def parameters(self):
        return self.net.parameters()
