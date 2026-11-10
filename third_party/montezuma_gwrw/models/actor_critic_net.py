#!/usr/bin/env python3
"""MultiStreamActorCriticNet -- the network for the actor-critic redesign
(2026-09-19), replacing MultiHeadDuelingQNet's "advantage head reinterpreted
as policy logits, then blend the per-stream logits to act" scheme.

WHY THIS EXISTS (the failure mode it removes)
---------------------------------------------
The old design produced the acting policy by blending FOUR streams' raw
logits:

    blended = (1-beta) * (w_p*l_prog/s_prog + w_t*l_task/s_task
                          + w_e*l_ext/s_ext) + beta * (l_spec/s_spec)

Every s_* there is a hand-set, frozen normalizer (logit_sigma_task,
logit_sigma_spec, ...) whose job is to make four unconstrained raw logit
magnitudes comparable. But those raw magnitudes DRIFT during ordinary
training, and nothing ties the frozen constant to the drift: on the live
run, logit_sigma_spec sat at 11,688 while spec's true logit scale had grown
to ~1e13 -- a nine-order-of-magnitude mismatch that caused two separate
catastrophic entropy collapses (blended_std ~1e8, act_entropy 0.0000) before
it was diagnosed and hand-corrected to 7e9. beta_rebalance, the sigma anchor,
and the periodic sigma resync all exist only to chase this drift.

The fix is structural, not another normalizer: there is now ONE actor head
per agent emitting policy logits DIRECTLY. Nothing is blended to act, so
there is no logit-scale comparison to get wrong, and the whole
logit_sigma_*/anchor/resync/beta_rebalance apparatus has nothing left to
normalize. Stream fusion moves to the ONLY place it is well-defined: the
ADVANTAGES in the PPO surrogate, where the scales are set by each stream's
own continuously-updated adv_rms (a measured quantity that cannot go stale
the way a frozen constant can) and are re-normalized per batch anyway.

The old code already blended advantages for the loss AND logits for acting,
with the same weights and a comment warning the two must not desync. This
design keeps exactly one of those two fusion points.

WHAT IS DELIBERATELY UNCHANGED (so a warm start is possible at all)
-------------------------------------------------------------------
conv stack, trunk, and the residual GRU (including gru_alpha) are built by
the SAME helpers, at the same dims, in the same order as
MultiHeadDuelingQNet -- so those parameters load byte-identically from an
existing checkpoint and ~20 days of learned representation carries over.

The critics also carry over EXACTLY. V_k(s) means the same thing in both
designs (that stream's state-value, GAE'd against its own reward and own
gamma), and critic_bodies/critic_heads are built exactly like the old
head_bodies/value_heads, so:

    critic_bodies[f"{stream}_{p}"]  <-  head_bodies[f"{stream}_{p}"]
    critic_heads[f"{stream}_{p}"]   <-  value_heads[f"{stream}_{p}"]

is a shape-exact copy. See warm_start_from_dueling() below.

Only the ACTOR is genuinely new. It is initialized from the task stream's
body+advantage head (shape-compatible, and task is the dominant cooperative
stream at the live weights), which makes it a sane starting point rather
than noise -- but the intended path is to follow that with a short
distillation pass against the old net's blended policy, so the new actor
starts out reproducing the behavior the run already had. See
distill_actor_from_blend() in the training script.
"""
import numpy as np
import torch
import torch.nn as nn

from models.dueling_qnet import (
    _build_conv_stack,
    _make_head_body,
    _make_readout_head,
)


class MultiStreamActorCriticNet(nn.Module):
    """Shared conv+trunk+GRU representation; per-agent actor head; per
    (stream, agent) critic head.

    Parameters mirror MultiHeadDuelingQNet's defaults exactly -- do not
    change them casually, the warm-start path depends on the shapes.

    agent_prefixes: e.g. ["a", "b", "c"]      -> one actor each
    stream_keys:    e.g. ["task", "extrinsic", "spec", "progress"]
                    -> one critic per (stream, agent) pair, keyed
                       f"{stream}_{prefix}" to match the old head keys.
    """

    def __init__(self, agent_prefixes, stream_keys, in_channels: int = 4,
                 num_actions: int = 18, conv_channels=(64, 128, 128, 128),
                 trunk_dim: int = 1024, trunk_layers: int = 2,
                 head_hidden_dim: int = 256, head_hidden_layers: int = 5,
                 critic_readout_layers: int = 4, actor_readout_layers: int = 4,
                 use_gru: bool = True, gru_alpha_init: float = 0.0):
        super().__init__()
        self.num_actions = num_actions
        self.agent_prefixes = list(agent_prefixes)
        self.stream_keys = list(stream_keys)
        self.critic_keys = [f"{s}_{p}" for p in self.agent_prefixes for s in self.stream_keys]

        # --- shared representation: IDENTICAL construction to MultiHeadDuelingQNet ---
        self.conv = _build_conv_stack(in_channels, conv_channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 84, 84)
            flat_dim = self.conv(dummy).shape[1]
        self.trunk, trunk_out_dim = _make_head_body(flat_dim, trunk_dim, trunk_layers)

        self.use_gru = bool(use_gru)
        self.gru_hidden_dim = trunk_out_dim
        if self.use_gru:
            self.gru = nn.GRUCell(trunk_out_dim, trunk_out_dim)
            self.gru_alpha = nn.Parameter(torch.tensor(float(gru_alpha_init)))

        # --- ONE actor per agent: policy logits, directly. No blend, no sigma. ---
        # final_gain=0.01 matches the old advantage head's small-init convention,
        # so the policy starts near-uniform rather than committed at init.
        self.actor_bodies = nn.ModuleDict()
        self.actor_heads = nn.ModuleDict()
        for p in self.agent_prefixes:
            body, body_out_dim = _make_head_body(trunk_out_dim, head_hidden_dim, head_hidden_layers)
            self.actor_bodies[p] = body
            self.actor_heads[p] = _make_readout_head(
                body_out_dim, head_hidden_dim, actor_readout_layers,
                out_dim=num_actions, final_gain=0.01)

        # --- one critic per (stream, agent): V_k(s). Same shapes as the old
        # head_bodies/value_heads, which is what makes the warm start exact. ---
        self.critic_bodies = nn.ModuleDict()
        self.critic_heads = nn.ModuleDict()
        for k in self.critic_keys:
            body, body_out_dim = _make_head_body(trunk_out_dim, head_hidden_dim, head_hidden_layers)
            self.critic_bodies[k] = body
            self.critic_heads[k] = _make_readout_head(
                body_out_dim, head_hidden_dim, critic_readout_layers,
                out_dim=1, final_gain=1.0)

    # ------------------------------------------------------------------
    # representation (verbatim semantics from MultiHeadDuelingQNet)
    # ------------------------------------------------------------------
    def features(self, obs_u8_nhwc: torch.Tensor) -> torch.Tensor:
        """PRE-GRU trunk features -- stateless. Every POLICY path must go
        through features_rec/gru_step instead or it silently drops the
        recurrent state. Kept as its own method because trunk-aliasing
        detection (trunk_erase_keep_last) genuinely wants the stateless
        representation."""
        x = obs_u8_nhwc.permute(0, 3, 1, 2).float() / 255.0
        return self.trunk(self.conv(x))

    trunk_features = features

    def initial_hidden(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.gru_hidden_dim, device=device)

    def gru_step(self, feat: torch.Tensor, h_prev: torch.Tensor | None):
        """ONE recurrent timestep on already-computed trunk features.
        Returns (output_features, h_new) with output = feat + gru_alpha*h_new
        (zero-init gate: at load time output == feat exactly)."""
        if not self.use_gru:
            return feat, h_prev
        if h_prev is None:
            h_prev = torch.zeros(feat.shape[0], self.gru_hidden_dim,
                                  device=feat.device, dtype=feat.dtype)
        h_new = self.gru(feat, h_prev)
        return feat + self.gru_alpha * h_new, h_new

    def features_rec(self, obs_u8_nhwc: torch.Tensor, h_prev: torch.Tensor | None = None):
        return self.gru_step(self.features(obs_u8_nhwc), h_prev)

    # ------------------------------------------------------------------
    # actor / critic readouts
    # ------------------------------------------------------------------
    def actor_logits(self, feat: torch.Tensor, prefix: str) -> torch.Tensor:
        """THE policy, for one agent, off already-computed (post-GRU)
        features. This is the whole of pi: no blending, no per-stream
        sigma, nothing else contributes to action selection."""
        return self.actor_heads[prefix](self.actor_bodies[prefix](feat))

    def critic_value(self, feat: torch.Tensor, critic_key: str,
                      isolate: bool = False) -> torch.Tensor:
        """V_k(s) for one (stream, agent) key.

        isolate=True detaches the INPUT FEATURES, so the critic's gradient
        trains critic_bodies + critic_heads but does not flow back into the
        shared representation (GRU / trunk / conv).

        2026-09-21 BUG FIX. This used to detach between the body and the head
        (`critic_heads[k](hh.detach())`). In the OLD architecture that was
        correct: head_bodies[k] was SHARED between value_heads[k] and
        advantage_heads[k], and the advantage head WAS the policy -- so the
        body still trained via the policy gradient, and the flag only stopped
        v_loss from corrupting a representation the policy depended on.

        In THIS architecture critic_bodies feed nothing but critic_heads (the
        actor has its own actor_bodies), so detaching there left critic_bodies
        with NO gradient source whatsoever. Measured directly: with
        isolate=True the critic-body grad-norm was exactly 0.0, and no other
        loss reaches it either (BC trains only the actor). The 5-layer body --
        most of each critic's capacity -- was frozen at its warm-started
        values for every run, leaving only the 4-layer readout able to adapt.
        That matches the measured critic fit: corr(A_k, own return-to-go) of
        -0.005 / +0.065 / +0.127, with the task critic's advantage varying
        260x more than the return it is supposed to track.

        Detaching `feat` instead preserves the flag's real intent (protect the
        shared representation) while letting the critic actually learn.
        """
        return self.critic_heads[critic_key](
            self.critic_bodies[critic_key](feat.detach() if isolate else feat)).squeeze(-1)

    def critics_for_agent(self, feat: torch.Tensor, prefix: str,
                           isolate: bool = False) -> dict:
        """All of one agent's stream-values in one call: {stream: V_k(s)}."""
        return {s: self.critic_value(feat, f"{s}_{prefix}", isolate)
                for s in self.stream_keys}

    def act_and_value(self, feat: torch.Tensor, prefix: str,
                       isolate_value: bool = False):
        """(logits, {stream: V}) for one agent off shared features -- the
        rollout/PPO readout, one call so the (expensive) feature computation
        is never duplicated."""
        return self.actor_logits(feat, prefix), self.critics_for_agent(feat, prefix, isolate_value)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def warm_start_from_dueling(net: MultiStreamActorCriticNet, state_dict: dict,
                             actor_init_stream: str = "task",
                             actor_readout: str = "reinit",
                             actor_init_scale: float | None = None,
                             verbose: bool = True) -> dict:
    """Load a MultiHeadDuelingQNet checkpoint into this net.

    Three groups, three different stories:

      shared representation (conv/trunk/gru/gru_alpha)
          Shape-identical by construction -> copied verbatim. This is the
          ~20 days of learned visual representation, and the single biggest
          reason to warm start at all.

      critics (critic_bodies/critic_heads <- head_bodies/value_heads)
          Shape-identical AND semantically identical: V_k(s) was already
          that stream's state-value trained by its own GAE returns, which
          is exactly its new job. Copied verbatim, per key.

      actor (actor_bodies/actor_heads <- head_bodies/advantage_heads of
             actor_init_stream)
          Genuinely new. There is no old parameter that means "the blended
          policy", so this seeds from ONE stream's body+action readout
          (shape-compatible; task by default, the dominant cooperative
          stream at the live weights). Treat this as an initialization, not
          a restoration -- follow it with distill_actor_from_blend() to
          actually recover the old blended behavior.

          actor_readout controls what happens to the FINAL readout layer,
          and the default ("reinit") is the only one that trains sanely:

            "reinit" (default) -- body seeded from the source stream, final
                readout freshly initialized at standard scale. The body is
                where the learned, reusable computation lives; the readout
                is one linear map that distillation relearns quickly.

            "scaled" -- copy the readout too, divided by actor_init_scale
                (pass the source stream's logit_sigma_*). This reproduces
                logits_task/sigma_task, which LOOKS right -- healthy ~72%
                entropy on real frames -- but is a trap under Adam: the
                divided weights sit at ~1e-6 while Adam's step size is
                ~lr regardless of parameter magnitude, so at lr=3e-4 a
                single step is ~300x the weights and the seed is destroyed
                instantly. Measured: distillation KL went UP 1.0 -> 8.0 and
                logit_std blew from 1.6 to 62. Only use this with an
                lr matched to the 1e-6 parameter scale, or with SGD.

            "copy" -- copy the readout verbatim. Produces a saturated,
                entropy-0.0000 deterministic actor (raw logits std ~1e6 on
                the live checkpoint). Here for completeness/diagnostics.

          Either way, treat the actor as an initialization, not a
          restoration, and follow with distillation. Verify with
          report_actor_scale() before training on it.

    Returns a report dict {group: (n_loaded, n_expected)} so the caller can
    assert the shared/critic groups came across COMPLETE -- a partial load
    there means the architectures have drifted and the warm start is a lie.
    """
    own = net.state_dict()
    to_load, report = {}, {}

    def _take(dst_prefix: str, src_prefix: str):
        n_ok = n_want = 0
        for dst_key in own:
            if not dst_key.startswith(dst_prefix):
                continue
            n_want += 1
            src_key = src_prefix + dst_key[len(dst_prefix):]
            src = state_dict.get(src_key)
            if src is not None and src.shape == own[dst_key].shape:
                to_load[dst_key] = src
                n_ok += 1
        return n_ok, n_want

    # shared representation -- same names, same shapes
    shared_ok = shared_want = 0
    for pfx in ("conv.", "trunk.", "gru.", "gru_alpha"):
        a, b = _take(pfx, pfx)
        shared_ok += a
        shared_want += b
    report["shared"] = (shared_ok, shared_want)

    # critics: old (head_bodies, value_heads) -> new (critic_bodies, critic_heads)
    c_ok = c_want = 0
    for key in net.critic_keys:
        a, b = _take(f"critic_bodies.{key}.", f"head_bodies.{key}.")
        c_ok += a
        c_want += b
        a, b = _take(f"critic_heads.{key}.", f"value_heads.{key}.")
        c_ok += a
        c_want += b
    report["critics"] = (c_ok, c_want)

    # actor: body seeded from one stream; readout per actor_readout (see docstring)
    if actor_readout not in ("reinit", "scaled", "copy"):
        raise ValueError(f"actor_readout must be reinit/scaled/copy, got {actor_readout!r}")
    a_ok = a_want = 0
    for p in net.agent_prefixes:
        src = f"{actor_init_stream}_{p}"
        a, b = _take(f"actor_bodies.{p}.", f"head_bodies.{src}.")
        a_ok += a
        a_want += b
        if actor_readout != "reinit":
            a, b = _take(f"actor_heads.{p}.", f"advantage_heads.{src}.")
            a_ok += a
            a_want += b
    report["actor_seed"] = (a_ok, a_want)

    net.load_state_dict(to_load, strict=False)

    # actor_readout="scaled": divide the copied final readout so it emits
    # logits_<stream>/sigma_<stream>. See the docstring -- this is an Adam
    # footgun and is not the default.
    if actor_readout == "scaled":
        if not actor_init_scale or actor_init_scale <= 0:
            raise ValueError("actor_readout='scaled' requires a positive actor_init_scale "
                             "(the source stream's logit_sigma_*)")
        for p in net.agent_prefixes:
            last_linear = [m for m in net.actor_heads[p].modules()
                           if isinstance(m, nn.Linear)][-1]
            last_linear.weight.div_(actor_init_scale)
            if last_linear.bias is not None:
                last_linear.bias.div_(actor_init_scale)

    if verbose:
        for group, (ok, want) in report.items():
            flag = "" if ok == want else "   <-- INCOMPLETE"
            print(f"warm_start_from_dueling: {group:12s} {ok:4d}/{want:4d} tensors{flag}",
                  flush=True)
        if report["shared"][0] != report["shared"][1] or report["critics"][0] != report["critics"][1]:
            print("warm_start_from_dueling: WARNING -- shared/critic groups did not load "
                  "completely; architectures have drifted and this warm start is NOT "
                  "carrying over what it claims to.", flush=True)
        if actor_readout == "reinit":
            print(f"warm_start_from_dueling: actor bodies seeded from "
                  f"'{actor_init_stream}'; final readout FRESH at standard scale "
                  f"(distillation relearns it)", flush=True)
        elif actor_readout == "scaled":
            print(f"warm_start_from_dueling: actor readout copied and rescaled by "
                  f"1/{actor_init_scale:g} -- NOTE these weights sit at ~1/{actor_init_scale:g} "
                  f"scale; an Adam lr tuned for standard weights will destroy them in one "
                  f"step.", flush=True)
        else:
            print("warm_start_from_dueling: WARNING -- actor readout copied verbatim. The "
                  "old advantage heads emit raw logits (std ~1e6 on the live checkpoint), "
                  "so this actor is saturated and deterministic (entropy ~0).", flush=True)
    return report


@torch.no_grad()
def report_actor_scale(net: MultiStreamActorCriticNet, obs_u8_nhwc: torch.Tensor,
                        prefix: str, verbose: bool = True) -> dict:
    """Measure the seeded/trained actor's realized policy scale on REAL
    frames -- the check that actually catches a dead actor.

    A healthy policy over `num_actions` actions has entropy somewhere
    meaningfully between 0 and log(num_actions); entropy collapsing to ~0
    means the logits are saturated (the failure the unscaled warm start
    produces), and entropy pinned at exactly log(num_actions) means the
    readout is emitting ~0 for every action (a dead-flat actor, e.g. the
    scale was divided far too aggressively)."""
    feat, _ = net.features_rec(obs_u8_nhwc, None)
    logits = net.actor_logits(feat, prefix)
    probs = torch.softmax(logits, dim=-1)
    ent = float(-(probs * probs.clamp_min(1e-9).log()).sum(-1).mean())
    out = {
        "logit_std": float(logits.std()),
        "entropy": ent,
        "max_entropy": float(np.log(net.num_actions)),
        "entropy_frac": ent / float(np.log(net.num_actions)),
        "max_prob_mean": float(probs.max(dim=-1).values.mean()),
    }
    if verbose:
        verdict = ("SATURATED (dead/deterministic)" if out["entropy_frac"] < 0.05
                   else "FLAT (no preference)" if out["entropy_frac"] > 0.995
                   else "healthy")
        print(f"report_actor_scale[{prefix}]: logit_std={out['logit_std']:.4g} "
              f"entropy={ent:.4f}/{out['max_entropy']:.4f} "
              f"({out['entropy_frac']:.1%}) max_prob={out['max_prob_mean']:.3f} "
              f"-> {verdict}", flush=True)
    return out
