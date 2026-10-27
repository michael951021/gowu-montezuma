"""GoWU phase I: Go-With-Uncertainty exploration (Algorithms 1 and 2 of Mhammedi & Cohan, 2026).

Single-host version of the paper's coordinator/worker/learner design:
- coordinator: this process. It holds the lineage tree and population and runs winner selection,
  pruning, rollback, group consolidation and global sync.
- rollout workers: `EnvPool` processes. Each restores its checkpoint and runs one inner-step block.
- learners: RND (and the optional particle ensembles). They train on the GPU *while* the workers run
  the rollouts, which replaces the paper's asynchronous learning nodes.

All M groups advance in lockstep, one outer step at a time. Each group draws its own K; a group that
has finished its K outer steps sits idle until the iteration ends. Envs are interleaved across
workers, so idle groups do not leave CPU cores idle.
"""
from __future__ import annotations

import gzip
import json
import os
import pickle
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .exploit_guard import ExploitGuard
from .nets import RND, EnsembleQ
from .replay import Replay, sample_across
from .tree import LineageTree, Node
from .vec import EnvPool, RolloutItem, RolloutResult


@dataclass
class GowuConfig:
    game: str = "montezuma_revenge"
    seed: int = 0
    total_frames: int = 400_000_000      # game frames (= agent steps * 4), paper: 400M per seed
    n_groups: int = 4                    # M
    n_particles: int = 32                # N per group
    outer_steps: tuple = (10, 20)        # [K_min, K_max]
    inner_steps: tuple = (5, 15)         # [T_min, T_max]
    rollback: tuple = (1, 3)             # [k_min, k_max]
    sticky_p: float = 0.25
    # particle policy: hold_random (paper ablation, "on par") | ensemble (paper default) | random (fails)
    particle_policy: str = "hold_random"
    particle_eps: float = 0.0            # eps for hold_random; the paper's ensemble uses 0.2
    ensemble_eps: float = 0.2
    # uncertainty (RND input = last frame, as in Burda et al.)
    rnd_lr: float = 3e-4
    rnd_batch: int = 128
    rnd_updates_per_1k_steps: float = 8.0    # learner/actor ratio; ~1 pass over each frame (paper: async, unspecified)
    ens_lr: float = 1e-4
    ens_batch: int = 128
    ens_updates_per_1k_steps: float = 4.0    # used only when ens_async is False
    ens_async: bool = True               # paper App. F.3: learner trains continuously, rollouts use broadcast weights
    replay_capacity: int = 128_000       # per group
    # exploit guard (App. H)
    exploit_mode: str = "middle_room"
    exploit_max_repeats: int = 3
    exploit_rules: list = field(default_factory=list)
    # infra
    n_workers: int = 14
    device: str = "auto"
    out_dir: str = "runs/gowu"
    log_every_iters: int = 1
    save_every_sec: float = 900.0
    max_iters: int | None = None
    track_room_landmarks: bool = True    # export per-room landmarks for LR-GoWU baselines (not in GoWU)


def lexmax(cands):
    """cands: [(R, U, payload)] -> payload with max R, ties broken by max U (lex-argmax)."""
    best = max(cands, key=lambda c: (c[0], c[1]))
    return best[2]


class GoWU:
    def __init__(self, cfg: GowuConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        dev = cfg.device
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(dev)
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=1)

        M, N = cfg.n_groups, cfg.n_particles
        self.P = M * N
        self.pool = EnvPool(self.P, cfg.n_workers, cfg.seed,
                            dict(game=cfg.game, sticky_p=cfg.sticky_p),
                            max_rollout_steps=cfg.inner_steps[1])
        self.n_actions = self.pool.n_actions
        self.guard = ExploitGuard(cfg.exploit_mode, cfg.exploit_max_repeats, cfg.exploit_rules)

        # uncertainty estimator (shared across groups)
        self.rnd = RND(1).to(self.device)
        self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=cfg.rnd_lr)
        self.ensembles = None
        obs_shape = (4, 84, 84) if cfg.particle_policy == "ensemble" else (84, 84)
        n_members = N if cfg.particle_policy == "ensemble" else 0
        self.replays = [Replay(cfg.replay_capacity, obs_shape, n_members, cfg.seed * 31 + m) for m in range(M)]
        if cfg.particle_policy == "ensemble":
            # Paper: Muon, lr 1e-4. Muon needs 2-D weights; our grouped/batched weights are 3-D/4-D,
            # so we use Adam. The paper's ablation shows the particle policy barely matters.
            self.ensembles = [EnsembleQ(N, 4, self.n_actions).to(self.device) for _ in range(M)]
            self.ens_opts = [torch.optim.Adam(e.parameters(), lr=cfg.ens_lr) for e in self.ensembles]
            # Rollouts act with a copy of the weights, refreshed every outer step (the paper's parameter
            # broadcast), so the learner can update its own copy at the same time.
            self.act_ensembles = [EnsembleQ(N, 4, self.n_actions).to(self.device).requires_grad_(False)
                                  for _ in range(M)]
        self._replay_lock = threading.Lock()   # Replay and its numpy RNG are shared with the learner thread
        self._param_lock = threading.Lock()
        self._learner_stop = threading.Event()
        self._learner = None
        self.ens_updates_total = 0
        # Separate CUDA stream: rollout inference must not queue behind whole training updates.
        self._learn_stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        if self.ensembles is not None:
            self._sync_acting()
            if cfg.ens_async:
                self._learner = threading.Thread(target=self._ens_learner_loop, daemon=True)
                self._learner.start()

        # population + tree
        self.pool.reset(range(self.P))
        ck = self.pool.checkpoint([0])[0]
        ram = self.pool.ram([0])[0]
        self.tree = LineageTree()
        root = self.tree.make_root(ck, ram, self.pool.obs[0, 3].copy())
        self.nodes: list[Node] = [self.tree.acquire(root) for _ in range(self.P)]
        self.dirty = np.ones(self.P, bool)  # env state != node state -> restore before next rollout
        self.global_winner = self.tree.acquire(root)
        self.best_node = self.tree.acquire(root)
        self.room_landmarks: dict = {}   # (level, room) -> (node, U at selection, frames)

        # bookkeeping
        self.agent_steps = 0
        self.iteration = 0
        self.t0 = time.time()
        self.rooms_seen: set = {(ram[0], ram[1])}
        self.stats = dict(rollbacks=0, deaths=0, rollouts=0, exploits=0, rnd_loss=0.0, rnd_n=0, rnd_updates_total=0)
        self._upd_carry = {"rnd": 0.0, "ens": 0.0}
        self._log = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")
        self._events = open(os.path.join(cfg.out_dir, "reward_events.jsonl"), "a")
        self._last_save = time.time()
        self._audit_count = 0
        # warm up the RND normaliser with the start state (Burda et al. use random-agent data)
        self.rnd.norm.update(self._rnd_input_nodes([root] * 8))

    # ----------------------------------------------------------- particles
    def _set(self, p: int, node: Node, dirty: bool = True):
        if self.nodes[p] is node:
            return
        self.tree.acquire(node)
        self.tree.release(self.nodes[p])
        self.nodes[p] = node
        self.dirty[p] = dirty

    def group(self, m: int) -> range:
        N = self.cfg.n_particles
        return range(m * N, (m + 1) * N)

    # ---------------------------------------------------------- uncertainty
    def _rnd_input_nodes(self, nodes) -> torch.Tensor:
        return torch.from_numpy(np.stack([n.obs_last for n in nodes])[:, None]).to(self.device)

    def score(self, nodes) -> np.ndarray:
        """Current-estimator uncertainty U(s) for nodes (re-scored each call; U changes online)."""
        if not nodes:
            return np.zeros(0)
        return self.rnd.uncertainty(self._rnd_input_nodes(nodes)).float().cpu().numpy()

    def _train(self, new_steps: int):
        cfg = self.cfg
        self._upd_carry["rnd"] += new_steps * cfg.rnd_updates_per_1k_steps / 1000
        n_rnd = int(self._upd_carry["rnd"])
        self._upd_carry["rnd"] -= n_rnd
        if sum(r.size for r in self.replays) >= cfg.rnd_batch:
            for _ in range(n_rnd):
                with self._replay_lock:
                    x = sample_across(self.replays, cfg.rnd_batch, self.rng)
                x = x[:, -1] if x.ndim == 4 else x  # RND sees the last frame only
                x = torch.from_numpy(x).to(self.device)[:, None]
                loss = self.rnd.error(x).mean()
                self.rnd_opt.zero_grad(set_to_none=True)
                loss.backward()
                self.rnd_opt.step()
                self.stats["rnd_updates_total"] += 1
                self.stats["rnd_loss"] += loss.item()
                self.stats["rnd_n"] += 1
        if self.ensembles is None or cfg.ens_async:
            return
        self._upd_carry["ens"] += new_steps * cfg.ens_updates_per_1k_steps / 1000
        n_ens = int(self._upd_carry["ens"])
        self._upd_carry["ens"] -= n_ens
        for _ in range(n_ens):
            for m in range(self.cfg.n_groups):
                self._ens_update(m)

    def _ens_update(self, m: int) -> bool:
        """One masked zero-target update of group m's ensemble (App. E.1)."""
        cfg = self.cfg
        ens, opt, rep = self.ensembles[m], self.ens_opts[m], self.replays[m]
        with self._replay_lock:
            if rep.size < cfg.ens_batch:
                return False
            idx, obs = rep.sample(cfg.ens_batch)
            act, mask = rep.act[idx].copy(), rep.mask[idx].copy()
        x = torch.from_numpy(obs).to(self.device).float().div_(255.0)
        x = x[:, None].expand(-1, ens.n, -1, -1, -1)
        q = ens(x)                                                # (B, n, A)
        a = torch.from_numpy(act).to(self.device)
        qa = q.gather(2, a[:, None, None].expand(-1, ens.n, 1)).squeeze(-1)
        msk = torch.from_numpy(mask).to(self.device)
        loss = (msk * qa.pow(2)).mean()                           # zero target (App. E.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with self._param_lock:
            opt.step()
        self.ens_updates_total += 1
        return True

    def _ens_learner_loop(self):
        """Asynchronous learning node (Sec. 5.1, App. F.3): train every group's ensemble continuously."""
        if self._learn_stream is not None:
            torch.cuda.set_stream(self._learn_stream)
        while not self._learner_stop.is_set():
            if not any([self._ens_update(m) for m in range(self.cfg.n_groups)]):
                time.sleep(0.05)

    @torch.no_grad()
    def _sync_acting(self):
        """Broadcast the learner's current weights to the copies the rollouts act with."""
        with self._param_lock:
            if self._learn_stream is not None:
                self._learn_stream.synchronize()
            for src, dst in zip(self.ensembles, self.act_ensembles):
                for ps, pd in zip(src.parameters(), dst.parameters()):
                    pd.copy_(ps)

    # -------------------------------------------------------------- rollouts
    def _items(self, particles) -> list[RolloutItem]:
        cfg = self.cfg
        Tlo, Thi = cfg.inner_steps
        items = []
        for p in particles:
            T = int(self.rng.integers(Tlo, Thi + 1))
            if cfg.particle_policy == "hold_random":
                a, eps = int(self.rng.integers(self.n_actions)), cfg.particle_eps
            else:  # "random": i.i.d. uniform actions every step (paper's failing ablation)
                a, eps = 0, 1.0
            items.append(RolloutItem(p, self.nodes[p].ckpt if self.dirty[p] else None, a, T, eps))
        return items

    def _rollout_ensemble(self, particles) -> list[RolloutResult]:
        """Lockstep rollouts: member (p mod N) of group (p div N) picks argmax_a f(s, a), eps = 0.2."""
        cfg = self.cfg
        N = cfg.n_particles
        particles = list(particles)
        self.pool.restore([(p, self.nodes[p].ckpt) for p in particles if self.dirty[p]])
        T = {p: int(self.rng.integers(cfg.inner_steps[0], cfg.inner_steps[1] + 1)) for p in particles}
        acts = {p: bytearray() for p in particles}
        tot = {p: 0.0 for p in particles}
        dead = {p: False for p in particles}
        events = {p: [] for p in particles}
        ram_prev = self.pool.ram(particles)
        running = list(particles)
        t = 0
        while running:
            obs = torch.from_numpy(self.pool.obs[running, :4]).to(self.device).float().div_(255.0)
            grp = np.array(running) // N
            a_gpu = torch.empty(len(running), dtype=torch.long, device=self.device)
            with torch.no_grad():
                for m in np.unique(grp):
                    ks = np.nonzero(grp == m)[0]
                    mem = torch.from_numpy(np.array(running)[ks] % N).to(self.device)
                    ks_t = torch.from_numpy(ks).to(self.device)
                    x = torch.zeros(1, N, 4, 84, 84, device=self.device)
                    x[0, mem] = obs[ks_t]
                    q = self.act_ensembles[m](x)[0][mem]                 # (k, A): member p mod N
                    tie = q == q.max(1, keepdim=True).values
                    # argmax with ties broken uniformly at random
                    a_gpu[ks_t] = torch.where(tie, torch.rand_like(q), -1.0).argmax(1)
            a_sel = a_gpu.cpu().numpy()
            rand = self.rng.random(len(running)) < cfg.ensemble_eps
            a_sel[rand] = self.rng.integers(0, self.n_actions, rand.sum())
            with self._replay_lock:
                for m in np.unique(grp):
                    ks = np.nonzero(grp == m)[0]
                    self.replays[m].add(self.pool.obs[np.array(running)[ks], :4], a_sel[ks])
            r, ll, go, rams = self.pool.step(running, a_sel)
            nxt = []
            for k, p in enumerate(running):
                acts[p].append(int(a_sel[k]))
                if r[k] != 0:
                    tot[p] += float(r[k])
                    events[p].append((t, float(r[k]), ram_prev[p], rams[k]))
                ram_prev[p] = rams[k]
                if ll[k] or go[k]:
                    dead[p] = True
                elif not (r[k] > 0) and len(acts[p]) < T[p]:
                    nxt.append(p)
            running = nxt
            t += 1
        alive = [p for p in particles if not dead[p]]
        cks = self.pool.checkpoint(alive)
        return [RolloutResult(p, len(acts[p]), tot[p], dead[p], cks.get(p), bytes(acts[p]), ram_prev[p], events[p])
                for p in particles]

    # ------------------------------------------------------------ algorithm
    def outer_step(self, active_groups: list[int]):
        """One outer step of Algorithm 1 for every active group (groups advance in lockstep)."""
        cfg = self.cfg
        particles = [p for m in active_groups for p in self.group(m)]
        if cfg.particle_policy == "ensemble":
            self._sync_acting()
            self._train(0)
            results = self._rollout_ensemble(particles)
            steps = sum(r.n_steps for r in results)
            self._train(steps)
            self.rnd.norm.update(torch.from_numpy(self.pool.obs[particles, 3]).to(self.device)[:, None])
        else:
            self.pool.rollout_send(self._items(particles))
            # learners train on earlier data while the workers simulate (asynchronous learner analog)
            self._train(self._last_steps if hasattr(self, "_last_steps") else 0)
            results = self.pool.rollout_recv()
            steps = sum(r.n_steps for r in results)
            self._last_steps = steps
            new = []
            for m in active_groups:
                frames = np.concatenate([self.pool.traj[r.env, :r.n_steps] for r in results
                                         if r.env // cfg.n_particles == m])
                with self._replay_lock:
                    self.replays[m].add(frames)
                new.append(frames)
            new = np.concatenate(new)
            sub = new[self.rng.choice(len(new), min(512, len(new)), replace=False)]
            self.rnd.norm.update(torch.from_numpy(sub).to(self.device)[:, None])
        self.agent_steps += steps
        self.stats["rollouts"] += len(results)

        # expand the tree for survivors
        alive_new: dict[int, Node] = {}
        for res in results:
            p = res.env
            parent = self.nodes[p]
            dead = res.dead
            loop_counts = parent.loop_counts
            if res.reward_events:
                exploit, loop_counts = self.guard.check(parent.loop_counts, res.reward_events)
                for (t, r, rb, ra) in res.reward_events:
                    self._events.write(json.dumps(dict(it=self.iteration, frames=self.frames, p=p, R=parent.R + r,
                                                       r=r, ram_before=rb, ram_after=ra, exploit=exploit)) + "\n")
                    # Preserve every early room-5 key-consumption example, INCLUDING accepted ones,
                    # so false positives/negatives of the RAM surrogate can be reviewed visually.
                    if (self._audit_count < 64 and rb[1] == ra[1] == 5 and r > 0
                            and (int(ra[4]) & 0x1E).bit_count() < (int(rb[4]) & 0x1E).bit_count()):
                        self._audit_count += 1
                        audit_dir = os.path.join(cfg.out_dir, "exploit_audit")
                        os.makedirs(audit_dir, exist_ok=True)
                        base = os.path.join(audit_dir, f"event{self._audit_count:03d}")
                        import cv2
                        frames4 = self.pool.obs[p, :4].transpose(1, 0, 2).reshape(84, 4 * 84)
                        cv2.imwrite(base + ".png", frames4)
                        with gzip.open(base + ".pkl.gz", "wb") as f:
                            pickle.dump(dict(parent=parent.ckpt, endpoint=res.ckpt, actions=res.actions,
                                             ram_before=rb, ram_after=ra, reward=r, exploit=exploit), f)
                if exploit:
                    dead = True
                    self.stats["exploits"] += 1
            if dead:
                self.stats["deaths"] += 1
                self.dirty[p] = True
                continue
            child = self.tree.add_child(parent, res.ckpt, parent.R + res.reward, res.n_steps, res.actions,
                                        res.ram, res.reward_events, loop_counts,
                                        self.pool.obs[p, 3].copy())
            self._set(p, child, dirty=False)
            alive_new[p] = child
            self.rooms_seen.add((res.ram[0], res.ram[1]))
            if child.R > self.best_node.R:
                self.tree.acquire(child)
                self.tree.release(self.best_node)
                self.best_node = child

        U_new = dict(zip(alive_new, self.score(list(alive_new.values()))))
        if cfg.track_room_landmarks:
            self._update_room_landmarks(alive_new, U_new)

        # winner selection + pruning, or rollback
        for m in active_groups:
            g = list(self.group(m))
            alive = [p for p in g if p in alive_new]
            if not alive:
                self.stats["rollbacks"] += 1
                lo, hi = cfg.rollback
                for p in g:
                    k = int(self.rng.integers(lo, hi + 1))
                    self._set(p, self.tree.ancestor(self.nodes[p], k))
                    self.dirty[p] = True
                continue
            winner = lexmax([(self.nodes[p].R, U_new[p], p) for p in alive])
            for p in g:
                if p not in alive_new:
                    self._set(p, self.nodes[winner])
                    self.dirty[p] = True

    def _update_room_landmarks(self, alive_new: dict, U_new: dict):
        """Per-(level, room) lex-max (R, U) node, pinned so its root path survives pruning. Not part of
        GoWU. It exports the landmark set for LR-GoWU baselines B1/B2 (docs/PLAN_LR_GOWU.md §9).
        Ties on R are decided with the incumbent's U re-scored by the current RND."""
        tied = {}
        for p, child in alive_new.items():
            key = (child.ram[0], child.ram[1])
            inc = self.room_landmarks.get(key)
            if inc is None or child.R > inc[0].R:
                self._pin_landmark(key, child, U_new[p])
            elif child.R == inc[0].R:
                tied.setdefault(key, []).append((U_new[p], child))
        if tied:
            U_inc = dict(zip(tied, self.score([self.room_landmarks[k][0] for k in tied])))
            for key, cands in tied.items():
                u, child = max(cands, key=lambda c: c[0])
                if u > U_inc[key]:
                    self._pin_landmark(key, child, u)

    def _pin_landmark(self, key, node: Node, u: float):
        old = self.room_landmarks.get(key)
        self.room_landmarks[key] = (self.tree.acquire(node), float(u), self.frames)
        if old is not None:
            self.tree.release(old[0])

    def consolidate(self, m: int) -> Node:
        """Group consolidation (Alg. 1 l.29-30): reset the whole group to its lex-max particle."""
        g = list(self.group(m))
        U = self.score([self.nodes[p] for p in g])
        wp = lexmax([(self.nodes[p].R, u, p) for p, u in zip(g, U)])
        w = self.nodes[wp]
        for p in g:
            if p != wp:
                self._set(p, w)
        return w

    def iterate(self):
        """One iteration of Algorithm 2."""
        cfg = self.cfg
        M = cfg.n_groups
        # global synchronization
        gw = self.global_winner
        for m in range(M):
            g = list(self.group(m))
            U = self.score([self.nodes[p] for p in g])
            best = lexmax([(self.nodes[p].R, u, p) for p, u in zip(g, U)])
            if self.nodes[best].R < gw.R:
                for p in g:
                    self._set(p, gw)
        # evolve groups (each with its own K)
        K = [int(self.rng.integers(cfg.outer_steps[0], cfg.outer_steps[1] + 1)) for _ in range(M)]
        winners: dict[int, Node] = {}
        for k in range(max(K)):
            active = [m for m in range(M) if k < K[m]]
            self.outer_step(active)
            for m in active:
                if k == K[m] - 1:
                    winners[m] = self.consolidate(m)
        # update global winner
        cands = list(winners.values()) + [gw]
        U = self.score(cands)
        new_gw = lexmax([(n.R, u, i) for i, (n, u) in enumerate(zip(cands, U))])
        new_gw = cands[new_gw]
        if new_gw is not gw:
            self.tree.acquire(new_gw)
            self.tree.release(gw)
            self.global_winner = new_gw
        self.iteration += 1

    # ------------------------------------------------------------ run / io
    @property
    def frames(self) -> int:
        return self.agent_steps * 4

    def log(self):
        dt = time.time() - self.t0
        s = self.stats
        rec = dict(
            iter=self.iteration, frames=self.frames, time=round(dt, 1), fps=round(self.frames / max(dt, 1e-9)),
            best_R=self.best_node.R, global_winner_R=self.global_winner.R,
            group_R=[max(self.nodes[p].R for p in self.group(m)) for m in range(self.cfg.n_groups)],
            rooms=len(self.rooms_seen), max_level=max(l for l, _ in self.rooms_seen),
            landmarks=len(self.room_landmarks),
            tree_nodes=self.tree.n_alive, tree_created=self.tree.n_created,
            best_depth=self.best_node.depth, best_steps=self.best_node.steps,
            rollbacks=s["rollbacks"], deaths=s["deaths"], exploits=s["exploits"],
            death_rate=round(s["deaths"] / max(s["rollouts"], 1), 3),
            rnd_loss=round(s["rnd_loss"] / max(s["rnd_n"], 1), 5),
            rnd_updates_total=s["rnd_updates_total"], ens_updates_total=self.ens_updates_total,
            global_U=float(self.score([self.global_winner])[0]),
            global_winner_in_population=any(n is self.global_winner for n in self.nodes),
            gw_ram=self.global_winner.ram)
        self._log.write(json.dumps(rec) + "\n")
        self._log.flush()
        self._events.flush()
        s["rnd_loss"], s["rnd_n"] = 0.0, 0
        return rec

    def save_demo(self, path: str | None = None):
        """Highest-reward lineage as a demonstration for the backward algorithm (Sec. 5.1)."""
        cands = [self.best_node] + list(self.nodes)
        best = max(cands, key=lambda n: (n.R, -n.steps))
        nodes = self.tree.path(best)
        demo = dict(
            game=self.cfg.game, seed=self.cfg.seed, frames=self.frames, R=best.R,
            ckpts=[n.ckpt for n in nodes], R_cum=[n.R for n in nodes], steps=[n.steps for n in nodes],
            actions=[n.actions for n in nodes], ram=[n.ram for n in nodes], events=[n.events for n in nodes])
        path = path or os.path.join(self.cfg.out_dir, "demo_best.pkl.gz")
        tmp = path + ".tmp"
        with gzip.open(tmp, "wb", compresslevel=3) as f:
            pickle.dump(demo, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        torch.save(dict(rnd=self.rnd.state_dict(), frames=self.frames), os.path.join(self.cfg.out_dir, "rnd.pt"))
        if self.room_landmarks:
            self.save_landmarks()
        return path

    def save_landmarks(self):
        """One landmark per (level, room) with its full root path (checkpoints + actions) for BC."""
        out = []
        for key, (node, u, fr) in sorted(self.room_landmarks.items()):
            nodes = self.tree.path(node)
            out.append(dict(key=key, R=node.R, U=u, found_at_frames=fr, steps=node.steps, ram=node.ram,
                            obs_last=node.obs_last, ckpt=node.ckpt,
                            path_ckpts=[n.ckpt for n in nodes], path_actions=[n.actions for n in nodes],
                            path_R=[n.R for n in nodes]))
        path = os.path.join(self.cfg.out_dir, "landmarks.pkl.gz")
        with gzip.open(path + ".tmp", "wb", compresslevel=3) as f:
            pickle.dump(dict(frames=self.frames, landmarks=out), f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(path + ".tmp", path)

    def run(self, verbose: bool = True):
        cfg = self.cfg
        try:
            while self.frames < cfg.total_frames and (cfg.max_iters is None or self.iteration < cfg.max_iters):
                self.iterate()
                if self.iteration % cfg.log_every_iters == 0:
                    rec = self.log()
                    if verbose:
                        print(f"it {rec['iter']:5d} | {rec['frames']/1e6:8.2f}M fr | {rec['fps']:7d} fps | "
                              f"best R {rec['best_R']:8.0f} | rooms {rec['rooms']:3d} lvl {rec['max_level']} | "
                              f"tree {rec['tree_nodes']:6d} | death {rec['death_rate']:.2f} | "
                              f"rb {rec['rollbacks']} | rnd {rec['rnd_loss']:.4f}", flush=True)
                if time.time() - self._last_save > cfg.save_every_sec:
                    self.save_demo()
                    self._last_save = time.time()
        finally:
            self._learner_stop.set()
            if self._learner is not None:
                self._learner.join(timeout=30)
            self.save_demo()
            self.log()
            self.pool.close()
