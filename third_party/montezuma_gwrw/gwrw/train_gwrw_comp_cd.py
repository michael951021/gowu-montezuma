"""GWRW with run14's competing agents as the takeover policy.

Same route controller / archive / target logic as train_gwrw.py, with two differences:
  * takeover: when the route controller hands off (arrival, wrong landmark, or leg timeout), or from the start in a no-target game, ONE of
    run14's three PPO agents (a, b, c; the shared conv-trunk-GRU net plus per-agent actor and critic heads, RND novelty model, PopArt state
    loaded from run14's final checkpoint) plays to game over. The agent is drawn uniformly per game. No behaviour cloning for them;
    PPO on [novelty (RND, task stream) + game reward (extrinsic stream)] with run14's value-based competition: agents b and c get
    advantage -= QDEF_BETA * gamma * (max_k Vf_k(s') - Vf_self(s')), a is the anchor (beta 0). Frontier temperature, progress, admission and
    spec streams are not used.
  * route controller seeding: the archive, the scene codebook and the imitation demos (8 fastest legs per node) start from run14's
    800-episode pool (pool_legs.py outputs); pool episodes that do not begin at a game start are dropped (first-frame check on pixels).
Emulator memory is read only into diag_* log fields; it never reaches a policy, reward, archive or target.

  RUN_NAME=gwrw_comp_v1 DEVICE=cuda:0 python3 -u gwrw/train_gwrw_comp.py     (one Ctrl-C = finish the update, save, exit)
Checkpoints load only with ~/montezuma_coevo/.venv_gpu/bin/python3.
"""
import collections, json, os, pickle, signal, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "training")); sys.path.insert(0, HERE)
from landmarks import CUT_FRAC, MATCH_FRAC, PIX, TOP, changed_frac, sig_dist, signature
from train_gwrw import Scenes, Vec, lm_id, build_nets as _unused  # noqa: F401  (Vec/Scenes/worker are shared with train_gwrw.py)


def load_opt_grow(opt, sd):
    """Load an optimizer state saved for a prefix of the current parameter list; parameters appended since start with fresh moments."""
    try:
        opt.load_state_dict(sd); return "full"
    except ValueError:
        pass
    assert len(opt.param_groups) == 1 == len(sd["param_groups"]), (len(opt.param_groups), len(sd["param_groups"]))
    n_old, n_new = len(sd["param_groups"][0]["params"]), len(opt.param_groups[0]["params"])
    assert n_old < n_new, (n_old, n_new)
    opt.load_state_dict(dict(sd, param_groups=[dict(sd["param_groups"][0], params=list(range(n_new)))]))
    return f"grown {n_old}->{n_new} params"

OUT = os.path.join(HERE, "out")
RUN = os.environ.get("RUN_NAME", "gwrw_comp_v1")
DEVICE = os.environ.get("DEVICE", "cuda:0")
CKPT14 = os.environ.get("CKPT14", os.path.join(ROOT, "checkpoints", "desktop_v2_run14_ac_resume_final_iter19051_pre_gwrw.pt"))   # optional in this package: absent = fresh RND / PopArt / advantage scales
E = lambda k, d: type(d)(os.environ.get(k, d))
import train_gwrw as G
G.RUN = RUN
from xworker import worker_x


class VecX(Vec):
    """Vec whose workers also report the emulator x coordinate (diag_* logging only)."""
    def __init__(self, seed0):
        ctx = mp.get_context("spawn"); self.conns = []
        for p in range(G.N_PROC):
            a, b = ctx.Pipe()
            ctx.Process(target=worker_x, args=(b, [seed0 + p * 100 + i for i in range(G.ENVS_PER_PROC)]), daemon=True).start()
            self.conns.append(a)
        self.obs = np.stack([o for c in self.conns for o in c.recv()]); self.n = len(self.obs); self.xs = [0] * self.n; self.ys = [0] * self.n

    def step(self, actions, caps):
        for i, c in enumerate(self.conns):
            s = slice(i * G.ENVS_PER_PROC, (i + 1) * G.ENVS_PER_PROC)
            c.send(list(zip(actions[s], caps[s])))
        res = [x for c in self.conns for x in c.recv()]
        self.obs = np.stack([x[0] for x in res]); self.xs = [x[5] for x in res]; self.ys = [x[6] for x in res]
        return (np.array([x[1] for x in res], np.float32), np.array([x[2] for x in res]), [x[3] for x in res], [x[4] for x in res])


def lmid(lm):
    """embedding index of a landmark: rewards 1..15, scenes 16..63 (train_gwrw.lm_id); frontier slots 0..7 use 63..56."""
    return 63 - min(lm[2], 7) if lm[0] == "f" else lm_id(lm)


def room7_xmax(runs):
    """Max x over contiguous room-7 runs of >=10 frames, first/last 3 frames trimmed, 5-frame median (same filter as run14's metric)."""
    best = None
    for run in runs:
        if len(run) < 10:
            continue
        v = np.asarray(run[3:-3], float)
        if len(v) < 1:
            continue
        if len(v) >= 5:
            v = np.array([np.median(v[max(0, i - 2):i + 3]) for i in range(len(v))])
        best = float(v.max()) if best is None else max(best, float(v.max()))
    return best
T = E("ROLLOUT", 128)
LR, LAM, CLIP, EPOCHS, MINIBATCHES, MAX_GRAD = 1e-4, 0.95, 0.1, 4, 4, 1.0
GOAL_GAIN = float(os.environ.get("GOAL_GAIN", 0.01))     # scale of the route controller's goal-frame pathway: keeps Adam's lr-sized first steps on the zero-initialised layer from jolting the policy
G_MU, STEP_COST, BC_W = 0.999, 0.0005, E("BC_W", 0.5)
BC_WARMUP = E("BC_WARMUP", 400)
STEP_CAP_PER_SCENE, LEG_TIMEOUT = 4000, E("LEG_TIMEOUT", 1500)
REPRO_RATE, MIN_ATTEMPTS, WEAK_MIN_ARR = 0.5, 20, 2
DEMOS_PER_NODE, LEG_KEEP = 8, 1000
STITCH_FAIL = 6.0                          # stitched target weight 1 / (1 + failed games / STITCH_FAIL)
STITCH_MIN_VIS = 20                        # a splice's prefix must be a node reached by >= this many games (rare one-off prefixes are not routes the controller can follow)
LE_N, LE_GAP = 4, 15                       # loop erasure: a state = the last LE_N frames; the matching later state must be >= LE_GAP steps on
REC_MAX, REC_STEPS = E("REC_MAX", 400), 2500
REC_RARE_MAX, RARE_FRAC, RARE_MIN_GAMES, RARE_STEPS, DEATH_CAP = E("REC_RARE_MAX", 40), E("RARE_FRAC", 0.01), E("RARE_MIN_GAMES", 1000), 4000, 12
FRONT_K, FRONT_STRIDE, FRONT_W, FRONT_MINLEN, FRONT_MAXDUR, FRONT_RESCORE = 6, 3, 21, 150, 1000, 1
LMREF_K, LMREF_DELAY, LMREF_TIE, LMREF_MAX, LMREF_PLACE, LMREF_OUT = 16, 10, 0.02, 0.15, 0.05, 0.10      # reference frames kept per landmark; steps after the event the frame is taken; tie tolerance and max distance of a match
FRONT_MAX = 90                            # frontier buffer capacity over all landmark sets (FRONT_K per set)
FRONT_DUP_D = 0.01                        # a candidate whose frame differs from an existing entry's frame in fewer than this fraction of playfield pixels is a duplicate and is not admitted
FRONT_CLUSTER_MAX, FRONT_CLUSTER_D = 10, 0.08     # at most this many frontier entries may look alike (fraction of changed playfield pixels below FRONT_CLUSTER_D); pixels only, any number of looks
FRONT_DARK_MAX, FRONT_DARK_LIT = 20, 0.05        # a playfield with fewer than FRONT_DARK_LIT of its pixels above PIX is near-black (a dark room shows only the player): all such frames look alike, share one look with this cap, and in it a deeper entry replaces a shallower one
FRONT_SKIP, FRONT_MINAGE = 100, 30        # ignore the first FRONT_SKIP steps after a landmark (transition frames); entries are protected from eviction for FRONT_MINAGE updates   # buffer size per leaf set; novelty stride and median window (in strided samples)
MAX_FRAMES = E("MAX_FRAMES", 300_000_000)
SAVE_EVERY = E("SAVE_EVERY", 50)
DEBUG_FAKE_R = E("DEBUG_FAKE_R", 0.0)
SEED_FROM_POOL = E("SEED_FROM_POOL", 1)
EVAL_GAMES = E("EVAL_GAMES", 0)   # >0: frozen evaluation of a saved run (no learning, no save): every game targets the deepest reproducible route, then the explorers play on; stops at EVAL_GAMES games (equal per env), writes <RUN>_eval.json
EVAL_WINGS = dict(p_none=0.0, p_spread=0.0, p_hard=0.0, p_consolidate=0.0, p_frontier=0.0, p_stitch=0.0, front_collect=0.0)
ADV_RMS_COUNT_CAP = E("ADV_RMS_COUNT_CAP", 100000.0)  # run14's advantage-scale history (54M samples) is capped so it can adapt; the task stream (scale ~1000x off) starts fresh
RESET_TASK_CRITIC = E("RESET_TASK_CRITIC", 1)  # 1: zero the task critics' output layers + PopArt (inherited values are ~100x too large for takeover novelty)
TRAIN_TRUNK = E("TRAIN_TRUNK", 0)             # 0: conv + trunk frozen (as in run14); actors, critics, GRU train
EXPLORER_ARCH = os.environ.get("EXPLORER_ARCH", "trav")   # "run14": run14's 29M net (GRU); "mu": the route controller's architecture, trunk shared by the agents and warm-started from mu, no GRU
if EXPLORER_ARCH in ("mu", "trav") and "TRAIN_TRUNK" not in os.environ:
    TRAIN_TRUNK = 1
TRUNK_LR_MULT = E("TRUNK_LR_MULT", 0.1)       # explorer trunk learning rate = takeover_lr * this (mu arch)
TRAV_BC, TRAV_X = E("TRAV_BC", 1), E("TRAV_X", 256)   # traversal net: TRAV_BC 1 = one BC step per epoch on the route controller's demo legs; TRAV_X = explorer states per step distilled into goal -1 (0 = off)
TRAV_BACK, TRAV_CONFIRM, TRAV_MIN, BACK_K, TRAV_TRIP = E("TRAV_BACK", 128), E("TRAV_CONFIRM", 8), E("TRAV_MIN", 5), 31, E("TRAV_TRIP", 0.5)   # go-back legs (scene revisits): samples per traversal BC step (0 = off); steps a scene change must hold; shortest leg kept; the k-slot value that marks a go-back goal (route depths are <= 18); share of go-back samples drawn from multi-room trip keys
TRAV_KEYS = ("trav_bc", "trav_acc", "trav_kl", "trav_xent", "trav_back_bc", "trav_back_acc")
EXPL_K, EXPL_WARMUP = E("EXPL_K", 20), E("EXPL_WARMUP", 50)   # EXPLORER_ARCH=trav: explorer = the traversal net with goal -1 and this fixed depth slot (route depths <= 18, go-back 31); critic-only updates after migration
EXPLORE_LM = 0                                # embedding index of goal -1 (never used by a route: rewards 1..15, scenes 16..63, frontier slots 56..63)
PREFIXES = ["a", "b", "c"]; STREAMS = ("task", "extrinsic", "progress", "landmark"); NA, NS = 3, 4
PROG_CAP = 8192
ANCHOR = "a"
ANCHOR_IDX = PREFIXES.index(ANCHOR)
CD_BLOCK, CD_LEVELS, CD_MIN = E("CD_BLOCK", 7), E("CD_LEVELS", 32), E("CD_MIN", 8)   # count-deficit cells: pixel blocks of the last frame, no RAM
CD_W = np.random.default_rng(7).integers(1, 2**63, size=(84 // CD_BLOCK) ** 2, dtype=np.uint64) | np.uint64(1)


def cell_keys(fr):
    k = CD_BLOCK; n = 84 // k; sh = fr.shape[:-2]
    b = fr[..., :n * k, :n * k].reshape(*sh, n, k, n, k).sum(axis=(-3, -1), dtype=np.uint32) // (k * k) // CD_LEVELS
    return (b.reshape(*sh, n * n).astype(np.uint64) * CD_W).sum(-1, dtype=np.uint64)
def hard_probs(nodes, visits, sharp, w_depth, w_rare):
    """Hard-target distribution: softmax(sharp * score) over nodes; score = (w_depth*depth + w_rare*rarity)/(w_depth+w_rare), depth = len/max len, rarity = 1 - log1p(visits)/log1p(max visits), both in [0,1]. sharp 0 = uniform, large = the argmax."""
    d = np.array([len(n) for n in nodes], dtype=np.float64); d /= d.max()
    lv = np.log1p(np.asarray(visits, dtype=np.float64)); r = 1.0 - lv / lv.max() if lv.max() > 0 else np.ones(len(nodes))
    wd, wr = max(float(w_depth), 0.0), max(float(w_rare), 0.0)
    sc = (wd * d + wr * r) / (wd + wr) if wd + wr > 0 else np.zeros(len(nodes))
    z = min(max(float(sharp), 0.0), 1e6) * sc
    p = np.exp(z - z.max()); return p / p.sum()


def lm_sim(refs, dev, pix, out):
    """Landmark x landmark appearance distance: median pixel-change fraction over all pairs of the two landmarks' reference playfields, each landmark's references
    first cut to those within `out` of its medoid. Returns (landmarks, symmetric matrix). Pixels only."""
    import torch
    keep = {}
    for L_, rr_ in refs.items():
        if not rr_:
            continue
        A_ = torch.from_numpy(np.stack(rr_)).to(dev).short()
        D_ = torch.stack([((A_[i_][None] - A_).abs() > pix).float().mean((1, 2)) for i_ in range(len(A_))])
        keep[L_] = A_[D_[int(D_.sum(1).argmin())] < out]
    Ls_ = list(keep); M_ = np.zeros((len(Ls_), len(Ls_)))
    for i_ in range(len(Ls_)):
        for j_ in range(i_ + 1, len(Ls_)):
            a_, b_ = keep[Ls_[i_]], keep[Ls_[j_]]
            M_[i_, j_] = M_[j_, i_] = torch.stack([((a_[k_][None] - b_).abs() > pix).float().mean((1, 2)) for k_ in range(len(a_))]).median().item()
    return Ls_, M_


def spread_list(nodes, visits, similar, sharp, rng):
    """One node per look-alike group of end landmarks. Nodes are taken in a random order drawn without replacement with weight exp(sharp * rarity)
    (Gumbel top-k; rarity = 1 - log1p(visits)/log1p(max visits)); a node is kept only if similar(its last landmark, last landmark of a kept node) is False for every kept node."""
    lv = np.log1p(np.asarray(visits, dtype=np.float64)); r = 1.0 - lv / lv.max() if lv.max() > 0 else np.ones(len(nodes))
    key = max(float(sharp), 0.0) * r + rng.gumbel(size=len(nodes)); kept = []
    for i_ in np.argsort(-key):
        n_ = nodes[i_]
        if all(not similar(n_[-1], k_[-1]) for k_ in kept):
            kept.append(n_)
    return kept


def stitch_pool(nodes, min_len=9, max_len=18, vis=None, min_vis=0):
    """Route targets no visited node realises yet: the prefix of one visited node joined to the suffix of another at a shared landmark. Landmarks are first visits, so
    an edge a->b only holds when the scenes passed between a and b are already seen: a splice is kept when its landmark order is valid (no repeats, r1, r2, ... in
    order), every edge has a visited node realising it whose earlier landmarks all lie inside the splice's own prefix, and at least one (edge, position) pair is not
    realised by any visited node. The prefix joined must itself be a node with >= min_vis visits (vis: node -> visits). Returns (pool, valid splices, supported splices)."""
    ns = set(n for n in nodes if n and all(l[0] in "rs" for l in n))
    cov, ctx = set(), collections.defaultdict(list)
    pre, suf = collections.defaultdict(set), collections.defaultdict(set)
    for n in ns:
        for i in range(len(n) - 1):
            cov.add((n[i], n[i + 1], i + 1))
        if len(n) >= 2:
            ctx[(n[-2], n[-1])].append(frozenset(n[:-1]))
        for i, x in enumerate(n):
            pre[x].add(n[:i + 1])
            if i + 1 < len(n):
                suf[x].add(n[i + 1:])
    seen, pool, n_valid, n_cert = set(), [], 0, 0
    for x, ps in pre.items():
        for p in ps:
            if vis is not None and vis.get(p, 0) < min_vis:
                continue
            for sx in suf.get(x, ()):
                n = p + sx
                if len(n) < min_len or len(n) > max_len or n in ns or n in seen:
                    continue
                seen.add(n)
                if len(set(n)) < len(n):
                    continue
                nr, ok = 0, True
                for l in n:
                    if l[0] == "r":
                        nr += 1
                        if l[1] != nr:
                            ok = False; break
                if not ok:
                    continue
                n_valid += 1
                if all(any(c <= frozenset(n[:j]) for c in ctx.get((n[j - 1], n[j]), ())) for j in range(1, len(n))):
                    n_cert += 1
                    if any((n[i], n[i + 1], i + 1) not in cov for i in range(len(n) - 1)):
                        pool.append(n)
    return sorted(pool), n_valid, n_cert


LIVE_DEFAULT = dict(takeover_lr=3e-5, qdef_mode=2, qdef_beta=1.0, qdef_gamma=0.99, w_task=100.0, w_extrinsic=10.0, w_progress=50.0, prog_clip=1e9, w_landmark=0.0, gamma_landmark=0.999, lm_scale_floor=0.05, prox_thr=0.0, gamma_progress=0.9999, p_none=0.05, p_consolidate=0.4, p_hard=0.05, hard_sharp=1e6, hard_w_depth=1000.0, hard_w_rare=1.0, intrinsic_coef=1.5, expl_hold=5, expl_eps=0.2, anchor_walker=0, gamma_task=0.99,
                    gamma_extrinsic=0.999, gae_lambda=0.95, ppo_clip=0.1, ppo_epochs=4, value_coef=1.0, entropy_coef=0.03, gru_seq_len=16,
                    minibatch_size=256, popart_beta=0.01, episodic_novelty=True, rnd_keep=0.25, rec_x=0.0, p_frontier=0.0, front_collect=0.0, front_pow=1.0, front_floor=0.15, takeover_cap=0, bc_front_frac=0.0, bc_front_pow=0.0, p_fprox=0.0, fprox_thr=0.006, fprox_n=4, fprox_slack=1.5, fprox_seg=200, le_thr=0.0, p_spread=0.0, spread_sim=0.005, spread_sharp=8.0, spread_s_only=1.0, p_stitch=0.0, front_depth_pow=1.0, ent_mu=0.01, bc_steps=1, bc_batch=256, fprox_strict=0, bc_hard=0.0)
POOL_DIR = OUT


def is_dark_pf(pf):
    return float((pf > PIX).mean()) < FRONT_DARK_LIT
EPI_BITS = 64
SEED_NET = 1


# ---------------------------------------------------------------------------------------------------------------- pool seeding
def pool_start_ok(files, log):
    """Pool episodes whose first frame is a game start (pixel distance to the most common first frame). Cached."""
    fp = os.path.join(POOL_DIR, "pool800_start_ok.json")
    if os.path.exists(fp):
        return set(json.load(open(fp)))
    from multiprocessing import Pool
    with Pool(8) as p:
        fr = p.map(_first_frame, files, chunksize=10)
    cnt = collections.Counter(f.tobytes() for f in fr)
    ref = np.frombuffer(cnt.most_common(1)[0][0], np.uint8).reshape(84, 84)
    ok = [f for f, x in zip(files, fr) if changed_frac(x, ref) < MATCH_FRAC]
    json.dump(ok, open(fp, "w"))
    log(f"pool start check: {len(ok)}/{len(files)} episodes start at a game start")
    return set(ok)


def _first_frame(f):
    return np.load(f)["obs"][0][..., -1]


def _leg_demo(args):
    f, legs = args
    obs, act = None, None
    ep = np.load(f); obs, act = ep["obs"], ep["act"]
    out = []
    for node, t0, t1 in legs:
        a0 = max(t0, t1 - LEG_KEEP)
        fr = np.concatenate([np.moveaxis(obs[a0][..., :3], -1, 0), obs[a0:t1][..., -1]])
        out.append((node, (t1 - t0, fr.copy(), act[a0:t1].astype(np.int64))))
    return out


def seed_from_pool(log):
    files = json.load(open(os.path.join(POOL_DIR, "pool800_files.json")))
    ok = pool_start_ok(files, log)
    from landmarks import Codebook
    sigs = list(Codebook.load(os.path.join(POOL_DIR, "pool800_codebook.npz"), frozen=True).sigs)
    byfile = collections.defaultdict(list)
    for l in open(os.path.join(POOL_DIR, "pool800_legs.jsonl")):
        r = json.loads(l)
        if r["file"] in ok:
            byfile[r["file"]].append(r)
    archive, cand = {}, collections.defaultdict(list)
    for f, legs in byfile.items():
        legs.sort(key=lambda r: r["k"]); seq = [tuple(r["to"]) for r in legs]
        for j, r in enumerate(legs):
            n = tuple(seq[:j + 1])
            archive.setdefault(n, dict(visits=0, hist=[]))["visits"] += 1
            if r["t1"] > r["t0"]:
                cand[n].append((r["t1"] - r["t0"], f, r["t0"], r["t1"]))
    jobs = collections.defaultdict(list)
    for n, c in cand.items():
        for dur, f, t0, t1 in sorted(c)[:DEMOS_PER_NODE]:
            jobs[f].append((n, t0, t1))
    from multiprocessing import Pool
    demos = {}
    with Pool(8) as p:
        for res in p.imap_unordered(_leg_demo, list(jobs.items()), chunksize=4):
            for n, d in res:
                demos.setdefault(n, []).append(d)
    for n in demos:
        demos[n].sort(key=lambda x: x[0])
    log(f"seeded from pool: {len(byfile)} episodes, {len(archive)} nodes (max depth {max(map(len, archive))}), {len(demos)} demo nodes, "
        f"{sum(len(v) for v in demos.values())} demo legs, {len(sigs)} scenes")
    return archive, demos, sigs


# ---------------------------------------------------------------------------------------------------------------- networks
def build_mu(heads=False):
    import torch, torch.nn as nn

    def conv():
        return nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())

    def init(m):
        for x in m.modules():
            if isinstance(x, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(x.weight, 2 ** 0.5); nn.init.zeros_(x.bias)

    def head(out, gain):
        body = nn.Sequential(nn.Linear(512, 256), nn.ReLU()); last = nn.Linear(256, out)
        nn.init.orthogonal_(body[0].weight, 2 ** 0.5); nn.init.zeros_(body[0].bias); nn.init.orthogonal_(last.weight, gain); nn.init.zeros_(last.bias)
        return body, nn.Sequential(last)

    class Mu(nn.Module):
        use_gru = False
        gru_hidden_dim = 512

        def __init__(self):
            super().__init__()
            self.conv, self.e_lm, self.e_k = conv(), nn.Embedding(64, 64), nn.Embedding(32, 32)
            self.fc = nn.Sequential(nn.Linear(3136 + 96, 512), nn.ReLU()); self.pi, self.v = nn.Linear(512, 18), nn.Linear(512, 1)
            init(self); nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.orthogonal_(self.v.weight, 1.0)
            if heads:
                self.actor_bodies, self.actor_heads, self.critic_bodies, self.critic_heads = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
                for p in PREFIXES:
                    self.actor_bodies[p], self.actor_heads[p] = head(18, 0.01)
                    for s_ in STREAMS:
                        self.critic_bodies[f"{s_}_{p}"], self.critic_heads[f"{s_}_{p}"] = head(1, 1.0)
            self.g_lin = nn.Linear(3136, 512); nn.init.zeros_(self.g_lin.weight); nn.init.zeros_(self.g_lin.bias)   # goal-frame pathway, zero at init = the goal-free network

        def features(self, obs):
            B = obs.shape[0]
            lm = torch.full((B,), EXPLORE_LM, dtype=torch.long, device=obs.device); kk = torch.full((B,), EXPL_K, dtype=torch.long, device=obs.device)
            return self.fc(torch.cat([self.conv(obs.permute(0, 3, 1, 2).float() / 255.0), self.e_lm(lm), self.e_k(kk)], 1))

        def gru_step(self, feat, h_prev):
            return feat, h_prev

        def actor_logits(self, feat, p):
            return self.actor_heads[p](self.actor_bodies[p](feat))

        def critic_value(self, feat, key, isolate=False):
            f = feat.detach() if isolate else feat
            return self.critic_heads[key](self.critic_bodies[key](f)).squeeze(-1)

        def forward(self, obs, lm, k, goal=None):
            z = self.fc[0](torch.cat([self.conv(obs.permute(0, 3, 1, 2).float() / 255.0), self.e_lm(lm), self.e_k(k.clamp(max=31))], 1))
            if goal is not None:                                          # goal frame (B, 84, 84) uint8: the target's appearance, through the shared conv
                z = z + GOAL_GAIN * self.g_lin(self.conv((goal.float() / 255.0)[:, None].repeat(1, 4, 1, 1)))
            h = self.fc[1](z)
            return self.pi(h), self.v(h).squeeze(-1)
    return Mu()


def build_mu_explorer():
    """Explorer with the route controller's architecture: mu's conv + 512 FC as one trunk shared by all agents (no GRU, no goal input);
    per agent one actor and one critic per stream on top. Same call surface as MultiStreamActorCriticNet where the trainer uses it."""
    import torch, torch.nn as nn

    def head(out, gain):
        body = nn.Sequential(nn.Linear(512, 256), nn.ReLU()); last = nn.Linear(256, out)
        nn.init.orthogonal_(body[0].weight, 2 ** 0.5); nn.init.zeros_(body[0].bias); nn.init.orthogonal_(last.weight, gain); nn.init.zeros_(last.bias)
        return body, nn.Sequential(last)

    class MuExplorer(nn.Module):
        use_gru = False

        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            self.trunk = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
            for m in list(self.conv) + list(self.trunk):
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, 2 ** 0.5); nn.init.zeros_(m.bias)
            self.gru_hidden_dim = 512
            self.actor_bodies, self.actor_heads, self.critic_bodies, self.critic_heads = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
            for p in PREFIXES:
                self.actor_bodies[p], self.actor_heads[p] = head(18, 0.01)
                for s in STREAMS:
                    self.critic_bodies[f"{s}_{p}"], self.critic_heads[f"{s}_{p}"] = head(1, 1.0)

        def features(self, obs):
            return self.trunk(self.conv(obs.permute(0, 3, 1, 2).float() / 255.0))

        def gru_step(self, feat, h_prev):
            return feat, h_prev

        def actor_logits(self, feat, p):
            return self.actor_heads[p](self.actor_bodies[p](feat))

        def critic_value(self, feat, key, isolate=False):
            f = feat.detach() if isolate else feat
            return self.critic_heads[key](self.critic_bodies[key](f)).squeeze(-1)

        def warm_start_from_mu(self, sd):
            """conv weights verbatim; the FC's columns over the conv features (the landmark / k embedding columns are dropped = zero goal input)."""
            with torch.no_grad():
                self.conv.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("conv.")})
                self.trunk[0].weight.copy_(sd["fc.0.weight"][:, :3136]); self.trunk[0].bias.copy_(sd["fc.0.bias"])
    return MuExplorer()


def build_prog_feat():
    import torch, torch.nn as nn

    class PF(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            self.trunk = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())

        def features(self, obs):
            return self.trunk(self.conv(obs.permute(0, 3, 1, 2).float() / 255.0))
    return PF()


# PopArt (copied from run14's training script, unchanged)
POPART_SIGMA_MIN, POPART_SIGMA_MAX = 1e-4, 1e6


def popart_mu_sigma(state):
    if not state or not state.get("inited"):
        return 0.0, 1.0
    mu = float(state["mu"]); var = max(float(state["nu"]) - mu * mu, 1e-8)
    return mu, float(np.clip(np.sqrt(var), POPART_SIGMA_MIN, POPART_SIGMA_MAX))


def popart_rescale_head(head, mu_old, sigma_old, mu_new, sigma_new):
    import torch
    last = head[-1]
    with torch.no_grad():
        last.weight.mul_(sigma_old / sigma_new); last.bias.mul_(sigma_old / sigma_new); last.bias.add_((mu_old - mu_new) / sigma_new)


def popart_update(ag, stream, ret_raw, beta):
    st = ag.popart[stream]; inited = bool(st.get("inited"))
    mu_old, sigma_old = popart_mu_sigma(st)
    bm, bq = float(np.mean(ret_raw)), float(np.mean(np.asarray(ret_raw, np.float64) ** 2))
    if not inited:
        st["mu"], st["nu"], st["inited"] = bm, bq, True
    else:
        st["mu"] = (1 - beta) * st["mu"] + beta * bm; st["nu"] = (1 - beta) * st["nu"] + beta * bq
    mu_new, sigma_new = popart_mu_sigma(st)
    if (mu_new, sigma_new) != (mu_old, sigma_old):
        popart_rescale_head(ag.net.critic_heads[f"{stream}_{ag.p}"], mu_old, sigma_old, mu_new, sigma_new)
    return (ret_raw - mu_new) / sigma_new


def gru_scan(net, feat_seq, h0, done_seq):
    outs, h = [], h0
    for t in range(feat_seq.shape[0]):
        o, h = net.gru_step(feat_seq[t], h); outs.append(o)
        h = h * (1.0 - done_seq[t]).unsqueeze(-1)
    return torch_stack(outs), h


def torch_stack(x):
    import torch
    return torch.stack(x, 0)


def qdef_penalty(vall, last_vall, done, coef, iA, gamma):
    Vf = np.einsum("tnks,s->tnk", vall.astype(np.float64), coef); Vl = np.einsum("nks,s->nk", last_vall.astype(np.float64), coef)
    Vn = np.concatenate([Vf[1:], Vl[None]], 0)
    return gamma * (1.0 - done.astype(np.float64)) * (Vn.max(2) - Vn[:, :, iA])


def qavoid_penalty(vall, last_vall, done, coef, iA, gamma):
    Vf = np.einsum("tnks,s->tnk", vall.astype(np.float64), coef); Vl = np.einsum("nks,s->nk", last_vall.astype(np.float64), coef)
    Vn = np.concatenate([Vf[1:], Vl[None]], 0)
    return gamma * (1.0 - done.astype(np.float64)) * Vn[:, :, iA]


def gae(R, V, Vlast, done, gamma, lam, nonep):
    Vx = np.vstack([V, Vlast[None]]); adv = np.zeros_like(R, dtype=np.float64); last = np.zeros(R.shape[1])
    for t in reversed(range(len(R))):
        bm = 1.0 if nonep else 1.0 - done[t]
        delta = R[t] + gamma * Vx[t + 1] * bm - V[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last; adv[t] = last
    return adv


def gae_mu(R, V, done, gamma):
    adv, last = np.zeros_like(R), 0.0
    for t in reversed(range(len(R))):
        nd = 1.0 - done[t]
        delta = R[t] + gamma * V[t + 1] * nd - V[t]
        last = delta + gamma * LAM * nd * last; adv[t] = last
    return adv


def rms_to(r):
    return dict(mean=r.mean, var=r.var, count=r.count)


def rms_from(r, d):
    r.mean, r.var, r.count = d["mean"], d["var"], d["count"]


class Prog:
    """Run14's conv/L2 progress reward: min L2 distance of the pre-GRU conv feature to every earlier one in the same game, over the game's mean consecutive-step distance."""
    def __init__(self, N, D, dev):
        import torch
        from algorithms.rnd import RunningMeanStd
        self.buf = torch.zeros(N, PROG_CAP, D, device=dev); self.len = np.zeros(N, np.int64); self.ssum = np.zeros(N); self.scnt = np.zeros(N)
        self.bn = torch.zeros(N, PROG_CAP, device=dev); self.prev = torch.zeros(N, D, device=dev); self.rms = RunningMeanStd(shape=()); self.dev = dev

    def step(self, cf):
        import torch
        mx = int(self.len.max())
        if mx > 0:
            d = (self.bn[:, :mx] + (cf * cf).sum(-1, keepdim=True) - 2.0 * torch.bmm(self.buf[:, :mx], cf.unsqueeze(2)).squeeze(2)).clamp_min(0).sqrt()   # same mm formula as cdist, without its mx-sized copy of buf
            d = torch.where(torch.arange(mx, device=self.dev).unsqueeze(0) < torch.from_numpy(self.len).to(self.dev).unsqueeze(1), d, torch.full_like(d, float("inf")))
            raw = np.where(self.len > 0, d.min(1).values.cpu().numpy(), 0.0)
        else:
            raw = np.zeros(len(self.len))
        has = self.len > 0
        if has.any():
            sd = (cf - self.prev).norm(dim=-1).cpu().numpy(); self.ssum[has] += sd[has]; self.scnt[has] += 1
        self.prev = cf.clone(); self.rms.update(raw.astype(np.float64))
        sc = np.maximum(np.where(self.scnt >= 5, self.ssum / np.maximum(self.scnt, 1), max(float(np.sqrt(self.rms.var)), 1e-6)), 1e-6)
        ei = np.flatnonzero(self.len < PROG_CAP)
        if len(ei):
            et = torch.from_numpy(ei).to(self.dev)
            et_pos = torch.from_numpy(self.len[ei]).to(self.dev)
            self.buf[et, et_pos] = cf[et]; self.bn[et, et_pos] = (cf[et] * cf[et]).sum(-1)
        self.len = np.minimum(self.len + 1, PROG_CAP)
        return (raw / sc).astype(np.float32)

    def reset(self, done):
        self.len = np.where(done, 0, self.len); self.ssum[done] = 0.0; self.scnt[done] = 0.0


class Ag:
    """One of run14's agents: its RND novelty model, PopArt stats and advantage scales; its heads live in the shared net."""
    def __init__(self, p, net, N, ck):
        from algorithms.rnd import RNDModel, RunningMeanStd
        self.p, self.net = p, net
        self.rnd = RNDModel(device=DEVICE, predictor_arch="paper")
        self.ret_rms = RunningMeanStd(shape=())
        if ck is not None:
            self.rnd.predictor.load_state_dict(ck[f"{p}_rnd_predictor"]); self.rnd.target.load_state_dict(ck[f"{p}_rnd_target"])
            rms_from(self.rnd.obs_rms, ck[f"{p}_rnd_obs_rms"]); rms_from(self.ret_rms, ck[f"{p}_rnd_return_rms"])
        self.popart = {s: (dict(ck[f"{p}_popart"][s]) if ck is not None and s in ck[f"{p}_popart"] else dict(mu=0.0, nu=1.0, inited=False)) for s in STREAMS}; self._popart0 = {s: dict(v) for s, v in self.popart.items()}
        if RESET_TASK_CRITIC:
            import torch as _t
            with _t.no_grad():
                last = net.critic_heads[f"task_{p}"][-1]; last.weight.zero_(); last.bias.zero_()
            self.popart["task"] = dict(mu=0.0, nu=1.0, inited=False)
        self.adv_rms = {s: RunningMeanStd(shape=()) for s in STREAMS}
        for s_ in (("extrinsic", "progress") if ck is not None else ()):
            rms_from(self.adv_rms[s_], ck[f"{p}_adv_rms"][s_]); self.adv_rms[s_].count = min(self.adv_rms[s_].count, ADV_RMS_COUNT_CAP)
        if not RESET_TASK_CRITIC and ck is not None:
            rms_from(self.adv_rms["task"], ck[f"{p}_adv_rms"]["task"]); self.adv_rms["task"].count = min(self.adv_rms["task"].count, ADV_RMS_COUNT_CAP)
        import torch
        self.rnd_opt = torch.optim.Adam(self.rnd.predictor.parameters(), lr=1e-4)
        if ck is not None and f"{p}_rnd_opt" in ck:
            try: self.rnd_opt.load_state_dict(ck[f"{p}_rnd_opt"])
            except Exception: pass
        self.filt = np.zeros(N)

    def state(self):
        return dict(pred=self.rnd.predictor.state_dict(), target=self.rnd.target.state_dict(), obs_rms=rms_to(self.rnd.obs_rms), ret_rms=rms_to(self.ret_rms),
                    popart=self.popart, adv_rms={s: rms_to(self.adv_rms[s]) for s in STREAMS}, rnd_opt=self.rnd_opt.state_dict(), filt=self.filt)

    def load(self, d):
        self.rnd.predictor.load_state_dict(d["pred"]); self.rnd.target.load_state_dict(d["target"]); rms_from(self.rnd.obs_rms, d["obs_rms"])
        rms_from(self.ret_rms, d["ret_rms"]); self.popart = d["popart"]; self.rnd_opt.load_state_dict(d["rnd_opt"])
        f_ = np.asarray(d["filt"]); self.filt = np.zeros(len(self.filt)); n_ = min(len(f_), len(self.filt)); self.filt[:n_] = f_[:n_]   # env count may differ from the checkpoint
        for s in STREAMS:
            if s in d["adv_rms"]: rms_from(self.adv_rms[s], d["adv_rms"][s])
        for s in STREAMS:
            self.popart.setdefault(s, dict(self._popart0[s]))


def main():
    import torch, torch.nn.functional as F
    from algorithms.rnd import RNDModel
    from models.actor_critic_net import MultiStreamActorCriticNet
    from models.dueling_qnet import load_compat_state_dict
    torch.set_num_threads(2)
    dev = torch.device(DEVICE)
    P = lambda s: os.path.join(OUT, f"{RUN}{s}")
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    vec = None
    mu = build_mu().to(dev); trav = build_mu(heads=(EXPLORER_ARCH == "trav")).to(dev)
    os.makedirs(OUT, exist_ok=True)
    ck14 = torch.load(CKPT14, mmap=True, weights_only=False, map_location="cpu") if os.path.exists(CKPT14) else None
    log(f"run14 checkpoint {'loaded: ' + CKPT14 if ck14 is not None else 'not found -> fresh RND models, PopArt and advantage scales for agents a/b/c'}")
    if ck14 is None and EXPLORER_ARCH != "trav":
        raise SystemExit(f"EXPLORER_ARCH={EXPLORER_ARCH} needs the run14 checkpoint (CKPT14={CKPT14}); use EXPLORER_ARCH=trav for a fresh start")
    if EXPLORER_ARCH == "trav":
        net = trav
    elif EXPLORER_ARCH == "mu":
        net = build_mu_explorer().to(dev)
    else:
        net = MultiStreamActorCriticNet(PREFIXES, ("task", "spec", "extrinsic", "progress", "admission", "landmark"), 4, 18).to(dev)
        dropped = load_compat_state_dict(net, ck14["shared_net"])
        assert not dropped, f"checkpoint keys dropped: {dropped[:5]}"
    N = G.N_PROC * G.ENVS_PER_PROC
    agents = [Ag(p, net, N, ck14) for p in PREFIXES]
    prog = Prog(N, net.gru_hidden_dim, dev)
    pnet, pf, migrated = None, None, False
    if EXPLORER_ARCH == "trav":
        pnet = build_prog_feat().to(dev)
        for q in pnet.parameters():
            q.requires_grad = False
        pf = pnet.features
    del ck14
    live_path = P("_live.json")
    if not os.path.exists(live_path):
        json.dump(LIVE_DEFAULT, open(live_path, "w"), indent=1)

    def read_live():
        try:
            d = dict(LIVE_DEFAULT); d.update(json.load(open(live_path)))
        except Exception:
            d = dict(LIVE_DEFAULT)
        if EVAL_GAMES:
            d.update(EVAL_WINGS)
        return d
    hp = read_live()

    def loop_keep(fr, thr):
        """Loop erasure of a demo (frames [S+3,84,84]; step s = stack fr[s:s+4]). Step i and a later step j are the same state when each of the
        LE_N aligned frame pairs differs in < thr of the playfield pixels (rows TOP:, abs diff > 24). Scanning forward, a step jumps to the LAST
        later matching step (>= LE_GAP on); the steps in between are dropped, so the action at the state is the one that left the loop.
        Returns the kept step indices. Pixels only."""
        D = len(fr); S = D - 3
        if thr <= 0 or S < LE_N + LE_GAP + 1:
            return np.arange(S)
        X = torch.from_numpy(np.ascontiguousarray(fr[:, TOP:, :])).to(dev).short().reshape(D, -1); rows = {}
        def row(r):
            if r not in rows:
                rows[r] = ((X[r][None] - X).abs() > 24).float().mean(1)
            return rows[r]
        drop = np.zeros(S, bool); i = 0
        while i < D:
            j_ = None
            if i >= LE_N - 1 and i + LE_GAP < D:
                W = row(i).clone()
                for k in range(1, LE_N):
                    Wk = torch.full_like(W, 9.0); Wk[k:] = row(i - k)[:D - k]; W = torch.maximum(W, Wk)
                W[:i + LE_GAP] = 9.0; W[:LE_N - 1] = 9.0
                ok = torch.nonzero(W < thr).flatten()
                if len(ok):
                    j_ = int(ok.max())
            if j_ is None:
                i += 1
            else:
                drop[i - 3:j_ - 3] = True; i = j_
            if len(rows) > 64:
                for r in sorted(rows)[:-16]:
                    del rows[r]
        return np.nonzero(~drop)[0]

    def mkdemo(dur, fr, a):
        return (dur, fr, a, loop_keep(fr, hp["le_thr"]))

    if EXPLORER_ARCH == "trav":
        trunk = [q for m in (net.conv, net.fc, net.e_lm, net.e_k) for q in m.parameters()]
    else:
        trunk = list(net.conv.parameters()) + list(net.trunk.parameters())
        for p_ in trunk:
            p_.requires_grad = bool(TRAIN_TRUNK)
    actor_p = [q for p in PREFIXES for m in (net.actor_bodies[p], net.actor_heads[p]) for q in m.parameters()]
    crit_p = [q for p in PREFIXES for s in STREAMS if s != "landmark" for m in (net.critic_bodies[f"{s}_{p}"], net.critic_heads[f"{s}_{p}"]) for q in m.parameters()]
    gru_p = (list(net.gru.parameters()) + [net.gru_alpha]) if getattr(net, "use_gru", True) else []
    lm_p = [q for p in PREFIXES for m in (net.critic_bodies[f"landmark_{p}"], net.critic_heads[f"landmark_{p}"]) for q in m.parameters()]
    groups = [dict(params=actor_p, name="actor"), dict(params=crit_p, name="critic")] + ([dict(params=gru_p, name="gru")] if gru_p else [])
    if TRAIN_TRUNK:
        groups.append(dict(params=trunk, name="trunk"))
    groups.append(dict(params=lm_p, name="lm_critic"))
    opt_tk = torch.optim.Adam(groups, lr=hp["takeover_lr"], eps=1e-5)
    opt_mu = torch.optim.Adam(mu.parameters(), lr=LR, eps=1e-5); tr_params = [q for n_, q in trav.named_parameters() if not n_.startswith(("actor_", "critic_"))]; opt_tr = torch.optim.Adam(tr_params, lr=LR, eps=1e-5)
    st = dict(update=0, frames=0, sigs=[], archive={})
    demos, bdemos = {}, {}
    if os.path.exists(P(".pt")):
        ck = torch.load(P(".pt"), map_location=dev, weights_only=False)
        miss_mu = mu.load_state_dict(ck["mu"], strict=False); assert not miss_mu.unexpected_keys and all(k_.startswith("g_lin.") for k_ in miss_mu.missing_keys), (miss_mu.missing_keys[:4], miss_mu.unexpected_keys[:4])
        log(f"route controller: {len(miss_mu.missing_keys)} goal-pathway tensors fresh (zero), opt_mu {load_opt_grow(opt_mu, ck['opt_mu'])}")
        if EXPLORER_ARCH == "trav":
            if ck.get("net_arch") == "trav":
                miss_n = net.load_state_dict(ck["net"], strict=False); assert not miss_n.unexpected_keys and all(k_.startswith("g_lin.") for k_ in miss_n.missing_keys), (miss_n.missing_keys[:4], miss_n.unexpected_keys[:4])
                pnet.load_state_dict(ck["progfeat"]); log(f"traversal net: {len(miss_n.missing_keys)} goal-pathway tensors fresh, opt_tr {load_opt_grow(opt_tr, ck['opt_tr2'])}")
                try: opt_tk.load_state_dict(ck["opt_tk"])
                except ValueError: log("opt_tk state not restored; Adam moments restart for the takeover agents")
            else:
                miss = trav.load_state_dict(ck["trav2"], strict=False)
                assert not miss.unexpected_keys and all(k_.startswith(("actor_", "critic_")) for k_ in miss.missing_keys), (miss.missing_keys[:4], miss.unexpected_keys[:4])
                opt_tr.load_state_dict(ck["opt_tr2"])
                pnet.load_state_dict({k_: v_ for k_, v_ in ck["net"].items() if k_.startswith(("conv.", "trunk."))})
                with torch.no_grad():
                    for k_ in net.critic_heads.keys():
                        last = net.critic_heads[k_][-1]; last.weight.zero_(); last.bias.zero_()
                migrated = True
                log(f"explorer arch trav: the traversal net is the explorer; fresh actor/critic heads on the traversal trunk (critic outputs zeroed), progress-reward features frozen from the old explorer trunk, {EXPL_WARMUP} critic-only updates, goal -1 depth slot {EXPL_K}")
        elif ck.get("net_arch", "run14") == EXPLORER_ARCH:
            miss = net.load_state_dict(ck["net"], strict=False)
            assert not miss.unexpected_keys and all("landmark_" in k_ for k_ in miss.missing_keys), (miss.missing_keys[:4], miss.unexpected_keys[:4])
            if miss.missing_keys:
                with torch.no_grad():
                    for p_ in PREFIXES:
                        last = net.critic_heads[f"landmark_{p_}"][-1]; last.weight.zero_(); last.bias.zero_()
                log(f"net checkpoint lacks {len(miss.missing_keys)} landmark-critic tensors; fresh, output layer zeroed")
            osd = ck["opt_tk"]
            if len(osd["param_groups"]) == len(opt_tk.param_groups) - 1:
                n_old = sum(len(g_["params"]) for g_ in osd["param_groups"]); g_new = dict(osd["param_groups"][-1]); g_new["params"] = list(range(n_old, n_old + len(lm_p))); g_new["name"] = "lm_critic"; osd["param_groups"].append(g_new)
            try: opt_tk.load_state_dict(osd)
            except ValueError: log("opt_tk state not restored (checkpoint predates the progress critic); Adam moments restart for the takeover agents")
        elif EXPLORER_ARCH == "mu":
            net.warm_start_from_mu(ck["mu"]); log(f"explorer arch {EXPLORER_ARCH}: checkpoint net is {ck.get('net_arch', 'run14')}; fresh heads, trunk warm-started from the route controller, Adam fresh")
        else:
            raise SystemExit(f"checkpoint explorer arch {ck.get('net_arch', 'run14')} != EXPLORER_ARCH {EXPLORER_ARCH}")
        if EXPLORER_ARCH == "trav":
            pass
        elif "trav2" in ck:
            trav.load_state_dict(ck["trav2"]); opt_tr.load_state_dict(ck["opt_tr2"])
        else:
            trav.load_state_dict(ck["mu"]); log("traversal net initialised from the route controller")
        for ag, d in zip(agents, ck["agents"]): ag.load(d)
        st = ck["st"]
        if migrated:
            st["expl_warm"] = EXPL_WARMUP
        if int(os.environ.get("REINIT_EXPLORER", 0)) and EXPLORER_ARCH == "trav":
            torch.manual_seed(int(time.time()))
            with torch.no_grad():
                trav.load_state_dict(build_mu(heads=True).state_dict())
                for k_ in net.critic_heads.keys():
                    last = net.critic_heads[k_][-1]; last.weight.zero_(); last.bias.zero_()
            for q in trav.parameters():
                opt_tk.state.pop(q, None); opt_tr.state.pop(q, None)
            st["expl_warm"] = EXPL_WARMUP
            log(f"REINIT_EXPLORER: the explorer / traversal net (conv, fc, goal embeddings, pi/v, all actor and critic heads) re-initialised at random; critic outputs zeroed; Adam moments cleared; {EXPL_WARMUP} critic-only updates; mu, agents' RND/PopArt, archive, frontier and progress features kept")
        if int(os.environ.get("REINIT_HEADS", 0)) and EXPLORER_ARCH not in ("mu", "trav"):
            torch.manual_seed(int(time.time()))
            fresh = MultiStreamActorCriticNet(PREFIXES, ("task", "spec", "extrinsic", "progress", "admission", "landmark"), 4, 18).to(dev)
            for p in PREFIXES:
                net.actor_bodies[p].load_state_dict(fresh.actor_bodies[p].state_dict()); net.actor_heads[p].load_state_dict(fresh.actor_heads[p].state_dict())
            for k_ in net.critic_bodies.keys():
                net.critic_bodies[k_].load_state_dict(fresh.critic_bodies[k_].state_dict()); net.critic_heads[k_].load_state_dict(fresh.critic_heads[k_].state_dict())
            for q in actor_p + crit_p:
                opt_tk.state.pop(q, None)
            del fresh
            log("REINIT_HEADS: actor and critic bodies+heads of all agents re-initialised; their Adam moments cleared")
        if os.path.exists(P("_demos.pkl")):
            demos = pickle.load(open(P("_demos.pkl"), "rb"))
        if os.path.exists(P("_bdemos.pkl")):
            bdemos = pickle.load(open(P("_bdemos.pkl"), "rb"))
        log(f"resumed at update {st['update']}, frames {st['frames']}, nodes {len(st['archive'])}, scenes {len(st['sigs'])}, demo nodes {len(demos)}, go-back keys {len(bdemos)} ({sum(len(v) for v in bdemos.values())} legs)")
    else:
        if SEED_FROM_POOL and os.path.exists(os.path.join(POOL_DIR, "pool800_files.json")):
            st["archive"], demos, st["sigs"] = seed_from_pool(log)
        else:
            log("fresh start: empty archive, no demos, no scene codebook (everything is discovered from the first game on)")
        miss_ = trav.load_state_dict(mu.state_dict(), strict=False)     # trav = mu's trunk + the explorer's actor/critic heads (those stay at their own init)
        assert not miss_.unexpected_keys and all(k_.startswith(("actor_", "critic_")) for k_ in miss_.missing_keys), (miss_.missing_keys[:4], miss_.unexpected_keys[:4])
    t_le = time.time(); n_old = sum(1 for v in demos.values() for d in v if len(d) == 3)
    demos = {n_: [d if len(d) == 4 else mkdemo(*d) for d in v] for n_, v in demos.items()}
    log(f"loop erasure (le_thr {hp['le_thr']}): {n_old} demos newly processed in {time.time() - t_le:.0f}s; all demos steps {sum(len(d[2]) for v in demos.values() for d in v)} -> kept {sum(len(d[3]) for v in demos.values() for d in v)}")
    archive = st["archive"]
    vec = VecX(1000 + st["update"] + (10_000_000 if EVAL_GAMES else 0)); assert vec.n == N
    ev_per = -(-EVAL_GAMES // N); ev_games = [[] for _ in range(N)]; ev_paths = [[] for _ in range(N)]; ev_frames0 = st["frames"]
    rng = np.random.default_rng(st["update"]); hrng = np.random.default_rng()
    scenes = Scenes(N, st["sigs"])
    ev_f, log_f, ep_f = open(P("_events.jsonl"), "a"), open(P("_log.jsonl"), "a"), open(P("_episodes.jsonl"), "a")

    def node(n):
        if n not in archive:
            archive[n] = dict(visits=0, hist=[])
        return archive[n]

    def rate(n):
        h = archive[n]["hist"][-100:]
        return (sum(1 for x in h if x[0]) / len(h)) if len(h) >= MIN_ATTEMPTS else None

    def reproducible(n):
        r_ = rate(n)
        return r_ is not None and r_ >= REPRO_RATE

    def hard_cands(nodes):
        return [n for n in nodes if all(reproducible(n[:k]) for k in range(1, len(n)) if n[:k] in demos and n[:k] in archive)]

    def hard_dist(nodes):
        hc = hard_cands(nodes)
        return (hc, hard_probs(hc, [archive[n]["visits"] for n in hc], hp["hard_sharp"], hp["hard_w_depth"], hp["hard_w_rare"])) if hc else ([], None)

    SPR = dict(upd=None, lst=[], key=None, Ls=[], M=None, ix={})
    PW = [""]

    def spread_target(nodes):
        """New wing: one target per look-alike group of end landmarks (by reference-frame appearance), representative drawn by rarity; the target is uniform over the groups. The list is rebuilt once per update."""
        if SPR["upd"] != st["update"]:
            SPR["upd"] = st["update"]; k_ = (st["update"] // 20, len(lmref["refs"]))
            if SPR["key"] != k_:
                SPR["key"] = k_; SPR["Ls"], SPR["M"] = lm_sim(lmref["refs"], dev, PIX, LMREF_OUT); SPR["ix"] = {L_: i_ for i_, L_ in enumerate(SPR["Ls"])}
            hc_ = [n for n in hard_cands(nodes) if n[-1][0] != "f" and (n[-1][0] == "s" or not hp["spread_s_only"])]; ix_, M_, thr_ = SPR["ix"], SPR["M"], hp["spread_sim"]
            similar = lambda a, b: a == b or (a in ix_ and b in ix_ and M_[ix_[a], ix_[b]] < thr_)
            SPR["lst"] = spread_list(hc_, [archive[n]["visits"] for n in hc_], similar, hp["spread_sharp"], rng) if hc_ else []
            if st["update"] % 100 == 0:
                log(f"[spread] update {st['update']}: {len(SPR['lst'])} targets from {len(hc_)} candidates: " + ", ".join(f"{n[-1][0]}{n[-1][1]}d{len(n)}v{archive[n]['visits']}" for n in SPR["lst"]))
        return SPR["lst"][int(rng.integers(len(SPR["lst"])))] if SPR["lst"] else None

    def pick_target():
        nodes = [n for n in archive if n and n in demos]
        u = rng.random()
        if not nodes or u < hp["p_none"]:
            PW[0] = "none"; return ()
        maxd = max(len(n) for n in nodes)
        deepest = min((n for n in nodes if len(n) == maxd), key=lambda n: archive[n]["visits"])
        weak = next((deepest[:k] for k in range(1, maxd + 1) if deepest[:k] in demos and not reproducible(deepest[:k])), None)
        rep = [n for n in nodes if reproducible(n)]
        if weak is not None and not rep:
            PW[0] = "weak"; return weak
        if hp["p_spread"] > 0 and rng.random() < hp["p_spread"]:
            sp_ = spread_target(nodes)
            if sp_ is not None:
                PW[0] = "spread"; return sp_
        if u < hp["p_none"] + hp["p_hard"]:
            hc, ph = hard_dist(nodes)
            if hc:
                PW[0] = "hard"; return hc[rng.choice(len(hc), p=ph)]
        if u < hp["p_none"] + hp["p_hard"] + hp["p_consolidate"]:
            cand = [n for n in nodes if not reproducible(n) and (rate(n) is None or sum(1 for x in archive[n]["hist"][-100:] if x[0]) >= WEAK_MIN_ARR)]
            if cand:
                md_ = max(len(n) for n in cand)
                PW[0] = "consol"; return min((n for n in cand if len(n) == md_), key=lambda n: archive[n]["visits"])
        if not rep:
            PW[0] = "none"; return ()
        md = max(len(n) for n in rep)
        near = [n for n in rep if len(n) >= md - 1]
        PW[0] = "near"; return near[rng.integers(len(near))]

    STC = dict(upd=-10**9, pool=[], att={}, arr={})

    def stitch_target():
        """Stitched wing: a target from stitch_pool (rebuilt every 50 updates from the visited nodes), weighted down by its failed games. The node joins the archive with demos once a game arrives."""
        if st["update"] - STC["upd"] >= 50:
            t_ = time.time(); pool_, nv_, nc_ = stitch_pool([n for n in archive if n and n in demos], vis={n: archive[n].get("visits", 0) for n in archive if n}, min_vis=STITCH_MIN_VIS); STC.update(upd=st["update"], pool=pool_)
            log(f"[stitch] update {st['update']}: pool {len(pool_)} ({nv_} valid splices, {nc_} supported, {time.time() - t_:.1f}s); games so far {sum(STC['att'].values())}, arrivals {sum(STC['arr'].values())}")
        pool_ = STC["pool"]
        if not pool_:
            return None
        w_ = np.array([1.0 / (1.0 + (STC["att"].get(n, 0) - STC["arr"].get(n, 0)) / STITCH_FAIL) for n in pool_]); w_ /= w_.sum()
        return pool_[int(rng.choice(len(pool_), p=w_))]

    PX = dict(Ls=[], sl=[], P=None, upd=-10**9)
    def build_prox():
        by = {}
        for n_, legs_ in demos.items():
            if n_ and n_[-1][0] != "f":
                for _, fr_, *_r in legs_:
                    by.setdefault(n_[-1], []).append(fr_[-8:-1, 20:, :])
        Ls, sl, chunks, o = [], [], [], 0
        for L_, fl in by.items():
            f_ = np.concatenate(fl)
            if len(f_) > 96:
                f_ = f_[np.random.default_rng(0).choice(len(f_), 96, replace=False)]
            Ls.append(L_); sl.append((o, o + len(f_))); chunks.append(f_); o += len(f_)
        PX.update(Ls=Ls, sl=sl, P=torch.from_numpy(np.concatenate(chunks)).to(dev).short() if chunks else None, upd=st["update"])
        log(f"proximity prototypes: {len(Ls)} landmarks, {o} frames")

    def prox_hits(frame_last, done):
        thr = hp["prox_thr"]; hits = {}
        if thr <= 0 or PX["P"] is None:
            return hits
        x = None
        for L_, (a_, b_) in zip(PX["Ls"], PX["sl"]):
            elig = [i for i, gi in enumerate(g) if gi["phase"] == 1 and not done[i] and L_ not in gi["prox"]
                    and ((L_[0] == "s" and L_[1] not in gi["seen"]) or (L_[0] == "r" and L_[1] == gi["nrew"] + 1))]
            if not elig:
                continue
            if x is None:
                x = torch.from_numpy(np.ascontiguousarray(frame_last[:, 20:, :])).to(dev).short()
            xe = x[elig]
            dm_ = ((xe[:, None] - PX["P"][a_:b_][None]).abs() > 24).float().mean((2, 3)).min(1).values.cpu().numpy()
            for j, i in enumerate(elig):
                if dm_[j] < thr:
                    hits.setdefault(i, []).append(L_)
        return hits

    FPX = {}
    def fck_protos(uid):
        """Checkpoint prototypes along a frontier demo: one every ~fprox_seg steps (each shifted within +-40 steps to a frame that is
        locally stable and distinct from the rest of the demo), plus the demo's end. Each is the last fprox_n frames before the checkpoint."""
        n_, seg_, thr = int(hp["fprox_n"]), int(hp["fprox_seg"]), hp["fprox_thr"]; key = (uid, n_, seg_)
        if key in FPX:
            return FPX[key]
        dm_ = demos[fkey(front[uid])][0]; fr_ = dm_[1]; kf = dm_[3] + 3; D = len(kf) + 3             # the erased demo: its kept steps' last frames, in order
        x = torch.from_numpy(np.ascontiguousarray(fr_[:, 20:, :])).to(dev).short()
        dist = lambda a, b: ((a[:, None] - b[None]).abs() > 24).float().mean((2, 3))
        st6 = np.zeros(len(fr_), bool)              # st6[j]: frames j..j+5 static up to the 2-step laser flicker (a death or stuck animation): no checkpoint right before one
        if len(fr_) >= 6:
            pd_ = lambda k_: ((x[:-k_] - x[k_:]).abs() > 24).float().mean((1, 2)).cpu().numpy()
            d2, d4 = pd_(2), pd_(4)
            for j in range(len(fr_) - 5):
                st6[j] = max(d2[j], d2[j + 1], d2[j + 2], d2[j + 3], d4[j], d4[j + 1]) < 0.002
        cks = []; last_c = -seg_
        if seg_ > 0:
            for c0 in range(seg_, D - 1 - seg_ // 2, seg_):
                best = None
                for c in range(max(n_ - 1, c0 - 40, last_c + seg_ // 2), min(D - 22, c0 + 40) + 1):
                    ix = int(kf[c - 3])
                    if ix < n_ - 1 or c - 18 <= 0:
                        continue
                    pr = x[ix - n_ + 1:ix + 1]
                    if dist(pr, pr).max().item() > 0.1 or st6[max(0, ix - 3):ix + 9].any():
                        continue
                    marg = dist(pr, x[torch.from_numpy(kf[:c - 18]).to(dev)]).min().item()     # distinct from everything earlier on the erased path (hits are taken in order)
                    if marg >= thr and (best is None or marg > best[0]):
                        best = (marg, c)
                if best is not None:
                    cks.append(best[1]); last_c = best[1]
        cks.append(D - 1)
        def proto(c):
            ix = int(kf[c - 3]); sl = x[max(0, ix - n_ + 1):ix + 1]
            return torch.cat([sl[:1].expand(n_ - len(sl), -1, -1), sl]) if len(sl) < n_ else sl
        FPX[key] = torch.stack([proto(c) for c in cks])
        log(f"frontier {uid}: {len(cks)} checkpoints at demo steps {cks} (erased demo {D} frames of {len(fr_)})")
        return FPX[key]

    def fprox_live(gi):
        return gi["fprox"] and gi["front"] in front and fkey(front[gi["front"]]) in demos

    def trail_live(gi):
        return gi["trail"] is not None and gi["trail"] in front and fkey(front[gi["trail"]]) in demos

    def trail_pick(base):
        """The frontier whose checkpoint trail a takeover game follows: the highest-scoring live entry grown from the game's landmark sequence."""
        c_ = [e for e in front.values() if set(e["base"]) == set(base) and len(e["base"]) == len(base) and fkey(e) in demos]      # order-free, as frontier groups are
        return max(c_, key=lambda e: e["score"])["uid"] if c_ else None

    def fprox_hits(frame_last, done):
        """Checkpoint matches {game: checkpoint index}: a game replaying a frontier under proximity arrival (phase 0, from checkpoint fck) or following a
        frontier trail (phase 1, from tck) hits the furthest remaining checkpoint whose prototype is within fprox_thr of its last frame; skipped ones are forfeited."""
        thr = hp["fprox_thr"]; hits = {}; by = {}
        for i, gi in enumerate(g):
            if done[i]:
                continue
            if gi["phase"] == 0 and gi["k"] == len(gi["target"]) - 1 and fprox_live(gi):
                uid, c0 = gi["front"], gi["fck"]
            elif gi["phase"] == 1 and trail_live(gi):
                uid, c0 = gi["trail"], gi["tck"]
            else:
                continue
            if c0 < len(fck_protos(uid)):
                by.setdefault(uid, []).append((i, c0))
        if not by:
            return hits
        x = torch.from_numpy(np.ascontiguousarray(frame_last[:, 20:, :])).to(dev).short()
        for uid, lst in by.items():
            P_ = fck_protos(uid)
            d_ = ((x[[i for i, _ in lst]][:, None, None] - P_[None]).abs() > 24).float().mean((3, 4)).min(2).values.cpu().numpy()     # (games, checkpoints)
            for (i, c0), row in zip(lst, d_):
                q = np.flatnonzero(row[c0:] < thr)
                if len(q):
                    hits[i] = c0 + int(q[-1])
        return hits

    front, fnext = {}, [0]
    if os.path.exists(P("_front.pkl")):
        fd = pickle.load(open(P("_front.pkl"), "rb")); front, fnext[0] = fd["front"], fd["next"]
    anc_of = lambda e: e.get("anc", len(e["base"]))                              # length of the reproducible landmark prefix the entry's demo starts after
    fkey = lambda e: e["base"][:anc_of(e)] + (("f", e["uid"], e["slot"]),)
    for e in list(front.values()):
        if fkey(e) not in demos:
            front.pop(e["uid"])
    for k_ in [k_ for k_ in demos if k_ and k_[-1][0] == "f" and k_[-1][1] not in front]:
        demos.pop(k_)
    for e in front.values():                                                    # a frontier's duration (timer arrival) is the length of its erased demo
        d_ = demos[fkey(e)][0]
        if len(d_[3]) < len(d_[2]):
            e["dur_raw"] = e.get("dur_raw", e["dur"]); e["dur"] = int(len(d_[3]))
    log(f"frontier buffer: {len(front)} entries; durations {[(e['uid'], e['dur']) for e in front.values()]}")
    log(f"frontier admission: any known landmark set may offer a frontier (anchored at its longest reproducible prefix), near-duplicate threshold {FRONT_DUP_D}, per-look cap {FRONT_CLUSTER_MAX} (near-black look {FRONT_DARK_MAX}, deeper replaces shallower), buffer cap {FRONT_MAX}, depth bonus (1+len(base))**{hp['front_depth_pow']} on sampling and admission")
    front_f = open(P("_front.jsonl"), "a")
    fin = []; frej = [0, 0]; gseq = [()]; fdup = [0]

    def is_leaf_set(base):
        sb = set(base)
        return not any(len(n) > len(base) and sb < set(n) for n in archive)

    def front_ok(base):
        """Any game whose landmark sequence is a known node may offer a frontier, leaf or not."""
        return base in archive

    def anchor_len(base):
        """Length of the longest prefix of base that is a node the route controller reproduces (the frontier demo starts after it); None if there is none."""
        for k_ in range(len(base), 0, -1):
            if base[:k_] in archive and reproducible(base[:k_]):
                return k_
        return None

    def front_novelty(frames):
        """Mean over the three agents of RND prediction error / that agent's running mean error. Pixels only."""
        x = torch.from_numpy(np.stack(frames))[..., None]
        ne = st.setdefault("nov_ema", [None] * NA); out = np.zeros(len(frames))
        for k, ag in enumerate(agents):
            raw = np.concatenate([ag.rnd.intrinsic_reward_with_features(x[j:j + 2048])[0].cpu().numpy() for j in range(0, len(x), 2048)])
            if ne[k] is None:
                ne[k] = float(raw.mean())
            out += raw / max(ne[k], 1e-12)
        return out / NA

    def fw(e):
        return max(e["score"], 1e-6) * (0.5 + min(1.0, (e["imp"] + 2 * e["adv"]) / (e["tries"] + 2))) * (0.5 if e["tries"] >= 30 and e["imp"] + e["adv"] == 0 else 1.0)

    def fwd(e):
        """Frontier value: fw times a depth bonus (1 + landmark-sequence length) ** front_depth_pow; used for sampling and for every admission / eviction comparison."""
        return fw(e) * (1 + len(e["base"])) ** hp["front_depth_pow"]

    def front_cap(pf):
        return FRONT_DARK_MAX if is_dark_pf(pf) else FRONT_CLUSTER_MAX

    def front_key(dark):
        return (lambda x: (len(x["base"]), fwd(x))) if dark else fwd

    def front_victim(pool, dark):
        """Weakest aged entry of the pool (in the dark look: the shallowest, ties by fwd); None if none is aged."""
        o_ = [x for x in pool if st["update"] - x["born"] >= FRONT_MINAGE]
        return min(o_, key=front_key(dark)) if o_ else None

    def front_beats(base, score, victim, dark):
        """Whether a candidate (landmark sequence base, window novelty score) takes the place of the victim."""
        if dark and len(base) != len(victim["base"]):
            return len(base) > len(victim["base"])
        return score * 0.5 * (1 + len(base)) ** hp["front_depth_pow"] > fwd(victim)

    def front_drop(e, why):
        front.pop(e["uid"], None); demos.pop(fkey(e), None)
        front_f.write(json.dumps(dict(update=st["update"], ev="drop", why=why, uid=e["uid"], base=e["base"], tries=e["tries"], ho=e["ho"], imp=e["imp"], adv=e["adv"], score=e["score"], diag=e["diag"])) + "\n"); front_f.flush()

    def pick_frontier():
        cand = [e for e in front.values() if fkey(e) in demos and anc_of(e) >= 1 and reproducible(e["base"][:anc_of(e)])]
        if not cand:
            return None
        w = np.array([fwd(e) for e in cand]) ** hp["front_pow"]; w = hp["front_floor"] / len(cand) + (1 - hp["front_floor"]) * w / w.sum()
        return fkey(cand[rng.choice(len(cand), p=w / w.sum())])

    lmref = pickle.load(open(P("_lmref.pkl"), "rb")) if os.path.exists(P("_lmref.pkl")) else dict(refs={}, cnt={})
    if not lmref["refs"]:                                                           # backfill: frame LMREF_DELAY+4 steps into the legs that follow each landmark (post-event, past the transition)
        for n_, legs_ in demos.items():
            if len(n_) >= 2 and n_[-1][0] != "f" and n_[-2][0] != "f":
                for _, fr_, *_r in legs_:
                    if len(fr_) > 3 + LMREF_DELAY + 4:
                        lmref["refs"].setdefault(n_[-2], []).append(fr_[3 + LMREF_DELAY + 4][TOP:].copy())
        for L_ in list(lmref["refs"]):
            rr_ = lmref["refs"][L_]; lmref["cnt"][L_] = len(rr_)
            if len(rr_) > LMREF_K:
                lmref["refs"][L_] = [rr_[i_] for i_ in np.random.default_rng(0).choice(len(rr_), LMREF_K, replace=False)]
    log(f"landmark reference frames: {{ {', '.join(f'{L_}:{len(v_)}' for L_, v_ in lmref['refs'].items())} }}")

    def lmref_add(L_, frame):
        """Automatic: the playfield of the frame LMREF_DELAY steps after a landmark event is stored (reservoir of LMREF_K) under that landmark."""
        c_ = lmref["cnt"][L_] = lmref["cnt"].get(L_, 0) + 1; rr_ = lmref["refs"].setdefault(L_, [])
        if len(rr_) < LMREF_K:
            rr_.append(frame[TOP:].copy())
        elif rng.random() < LMREF_K / c_:
            rr_[int(rng.integers(LMREF_K))] = frame[TOP:].copy()

    def lmref_match(frames):
        """Per frame: the landmarks whose reference frames it matches best (pixel change fraction; ties within LMREF_TIE; none if the best is over LMREF_MAX).
        Landmarks whose reference frames look alike (change fraction < LMREF_TIE) form one place; a place's rank is the earliest route position of its landmarks.
        Returns a bool array: False when every matched place ranks earlier in the route than the place of the last landmark."""
        refs_ = {}
        for L_, rr_ in lmref["refs"].items():                                         # drop outlier references (e.g. captured mid-transition): keep those near the medoid
            if not rr_:
                continue
            A_ = torch.from_numpy(np.stack(rr_)).to(dev).short()
            D_ = torch.stack([((A_[i_][None] - A_).abs() > PIX).float().mean((1, 2)) for i_ in range(len(A_))])
            keep_ = (D_[int(D_.sum(1).argmin())] < LMREF_OUT).cpu().numpy()
            refs_[L_] = [r_ for r_, k_ in zip(rr_, keep_) if k_]
        Ls_ = list(refs_)
        if not Ls_:
            return np.ones(len(frames), bool)
        R_ = torch.from_numpy(np.stack([f_ for L_ in Ls_ for f_ in refs_[L_]])).to(dev).short()
        own = torch.tensor([i_ for i_, L_ in enumerate(Ls_) for _ in refs_[L_]], device=dev)
        def dist(X_):
            d_ = torch.cat([((X_[j_:j_ + 16, None] - R_[None]).abs() > PIX).float().mean((2, 3)) for j_ in range(0, len(X_), 16)])
            return torch.stack([d_[:, own == i_].min(1).values for i_ in range(len(Ls_))], 1).cpu().numpy()
        dl_ = dist(torch.from_numpy(np.stack([f_[TOP:] for f_ in frames])).to(dev).short())
        match = (dl_ <= dl_.min(1, keepdims=True) + LMREF_TIE) & (dl_.min(1, keepdims=True) < LMREF_MAX)
        dd_ = np.stack([dist(torch.from_numpy(np.stack(refs_[L_])).to(dev).short()).max(0) for L_ in Ls_])    # landmark x landmark: largest change between a landmark's references and another's nearest reference
        place, groups = [0] * len(Ls_), []
        for a_ in range(len(Ls_)):                                                    # a landmark joins the first place all of whose members it resembles
            for gk_, mem_ in enumerate(groups):
                if all(max(dd_[a_, b_], dd_[b_, a_]) < LMREF_PLACE for b_ in mem_):
                    mem_.append(a_); place[a_] = gk_; break
            else:
                groups.append([a_]); place[a_] = len(groups) - 1
        sig_ = str([[Ls_[i_] for i_ in m_] for m_ in groups]) + str({L_: len(v_) for L_, v_ in refs_.items()})
        if sig_ != lmref.get("_sig"):
            lmref["_sig"] = sig_; log(f"landmark places: {[[f'{Ls_[i_][0]}{Ls_[i_][1]}' for i_ in m_] for m_ in groups]}  refs kept: { {f'{L_[0]}{L_[1]}': len(v_) for L_, v_ in refs_.items()} }")
        pos = {L_: i_ for i_, L_ in enumerate(gseq[0])}
        prank = {}
        for i_, L_ in enumerate(Ls_):
            prank[place[i_]] = min(prank.get(place[i_], 10 ** 9), pos.get(L_, 10 ** 9))
        last_ = gseq[0][-1]
        lr_ = prank[place[Ls_.index(last_)]] if last_ in Ls_ else 0
        ok = np.ones(len(frames), bool)
        for j_ in range(len(frames)):
            m_ = np.flatnonzero(match[j_])
            if len(m_) and all(prank[place[i_]] < lr_ for i_ in m_):
                ok[j_] = False
        return ok

    def same_look(a, b):
        return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > PIX).mean()) < LMREF_MAX

    def ent_pf(e):
        return e["clip"][len(e["clip"]) // 2][TOP:]

    def front_near(pf, pool):
        """Entries of the pool whose frame looks like pf (pixels only)."""
        return [x for x in pool if float((np.abs(ent_pf(x).astype(np.int16) - pf.astype(np.int16)) > PIX).mean()) < FRONT_CLUSTER_D]

    def front_trim():
        """Cap per look: while some entry has more than FRONT_CLUSTER_MAX look-alike entries (itself included), drop the weakest of the largest such look."""
        while True:
            worst = None
            for x in list(front.values()):
                nb = front_near(ent_pf(x), front.values())
                if len(nb) > front_cap(ent_pf(x)) and (worst is None or len(nb) > len(worst)):
                    worst = nb
            if worst is None:
                return
            front_drop(min(worst, key=front_key(is_dark_pf(ent_pf(worst[0])))), "cluster")

    if front:
        n0_ = len(front); front_trim()
        log(f"frontier look cap {FRONT_CLUSTER_MAX}: {n0_} -> {len(front)} entries")

    def front_rej(why, gi, tf, base, score):
        """Diagnostic (RAM room, never read by the algorithm): why a candidate with a near-black frame was not admitted."""
        if is_dark_pf(gi["frames"][tf][TOP:]):
            front_f.write(json.dumps(dict(update=st["update"], ev="rej", why=why, depth=len(base), score=float(score), room=int(gi["dpos"][min(int(tf), len(gi["dpos"]) - 1)][2]))) + "\n")

    def front_process():
        """Games finished since the last update: update the stats of the frontier entry they targeted, offer their most promising moment to the buffer."""
        W = FRONT_W; half = (W // 2) * FRONT_STRIDE
        for gi in fin:
            base = tuple(gi["seq"]); L0 = gi["leg0"]; leg = gi["frames"][L0:]
            e = front.get(gi["front"]) if gi["front"] is not None else None
            v = None
            if base and len(leg) >= FRONT_MINLEN:
                v = np.median(np.lib.stride_tricks.sliding_window_view(front_novelty(leg[::FRONT_STRIDE]), W), axis=1)    # v[j]: window centred at leg step (j + W//2) * stride
            if e is not None:
                e["tries"] += 1
                if gi["t_arrive"] is not None:
                    e["ho"] += 1
                    if gi.get("fend"):
                        e["fh"] = e.get("fh", 0) + 1
                if len(base) > len(e["base"]) and is_leaf_set(base) and (gi["t_arrive"] is not None or not hp["fprox_strict"]):
                    e["adv"] += 1
                elif v is not None and gi["t_arrive"] is not None:
                    jm_ = max(0, -(-FRONT_SKIP // FRONT_STRIDE) - W // 2)
                    if jm_ < len(v) and v[jm_:].max() > 1.15 * e["score"]:
                        e["imp"] += 1
            if v is None or not front_ok(base):
                continue
            anc = anchor_len(base)
            if anc is None:
                continue
            Lanc = L0 if anc == len(base) else gi["legs0"][anc - 1]
            jmin = max(0, -(-FRONT_SKIP // FRONT_STRIDE) - W // 2)
            if jmin >= len(v):
                continue
            vm = v.copy(); vm[:jmin] = -np.inf
            if e is not None and gi["t_arrive"] is not None:          # the route controller's own part of a frontier leg competes only where it beat the entry it was replaying
                ja_ = min(len(v), max(jmin, (gi["t_arrive"] - L0) // FRONT_STRIDE))
                vm[jmin:ja_][v[jmin:ja_] <= 1.15 * e["score"]] = -np.inf
            ctr = L0 + (np.arange(len(v)) + W // 2) * FRONT_STRIDE
            gseq[0] = base; okr = lmref_match([gi["frames"][min(c_, len(gi["frames"]) - 1)] for c_ in ctr])
            jb_ = int(np.argmax(vm)); frej[0] += 1; frej[1] += int(not okr[jb_])
            front_f.write(json.dumps(dict(update=st["update"], ev="filt", l0_room=int(gi["dpos"][max(0, L0 - 1)][2]), cand_room=int(gi["dpos"][min(int(ctr[jb_]), len(gi["dpos"]) - 1)][2]), rej=int(not okr[jb_]), n_rej=int((~okr).sum()), n=len(okr), last=base[-1])) + "\n")
            vm[~okr] = -np.inf
            if not np.isfinite(vm.max()):
                continue
            j = int(np.flatnonzero(vm >= 0.97 * vm.max())[0]); tf = L0 + (j + W // 2) * FRONT_STRIDE
            if tf + 1 - Lanc > FRONT_MAXDUR:
                continue
            cpf_ = gi["frames"][tf][TOP:].astype(np.int16)
            if any(float((np.abs(ent_pf(x).astype(np.int16) - cpf_) > PIX).mean()) < FRONT_DUP_D for x in front.values()):
                fdup[0] += 1
                front_rej("dup", gi, tf, base, v[j])
                continue
            group = [x for x in front.values() if set(x["base"]) == set(base)]
            slot = None; pl = None
            cpf2 = gi["frames"][tf][TOP:]; dk_ = is_dark_pf(cpf2)
            cmem = front_near(cpf2, front.values()); cvict = None
            if len(cmem) >= front_cap(cpf2):                            # this look is full: the candidate must beat the weakest aged entry of the look, which it then replaces
                cvict = front_victim(cmem, dk_)
                if cvict is None or not front_beats(base, v[j], cvict, dk_):
                    front_rej("cluster", gi, tf, base, v[j])
                    continue
            if len(group) < FRONT_K:
                if len(front) >= FRONT_MAX and cvict is None:                             # buffer full: the new entry takes the place of the weakest aged entry of any set
                    w_ = front_victim(front.values(), False)
                    if w_ is None or not front_beats(base, v[j], w_, False):
                        front_rej("capacity", gi, tf, base, v[j])
                        continue
                    front_drop(w_, "capacity")
                slot = min(set(range(8)) - {x["slot"] for x in group})
            else:                                                       # full: entries are grouped by what their frame looks like; scores are only compared inside a group
                cpf = gi["frames"][tf][TOP:]; cl = []
                for x in group:
                    for c_ in cl:
                        if same_look(ent_pf(c_[0]), ent_pf(x)):
                            c_.append(x); break
                    else:
                        cl.append([x])
                mine = next((c_ for c_ in cl if same_look(ent_pf(c_[0]), cpf)), []); big = max(cl, key=len); pl = [len(mine), len(big)]
                aged = lambda c_: [x for x in c_ if st["update"] - x["born"] >= FRONT_MINAGE]
                victim = None
                if len(mine) < len(big) - 1:                            # under-represented look: take a slot from the largest group
                    o_ = aged(big); victim = min(o_, key=fw) if o_ else None
                elif mine:                                              # otherwise compete with the weakest entry of the same look
                    o_ = aged(mine); w_ = min(o_, key=fw) if o_ else None
                    if w_ is not None and v[j] * 0.5 > fw(w_):
                        victim = w_
                if victim is not None:
                    slot = victim["slot"]; front_drop(victim, "replaced")
            if slot is None:
                front_rej("group", gi, tf, base, v[j])
                continue
            if cvict is not None and cvict["uid"] in front and len(front_near(cpf2, front.values())) >= front_cap(cpf2):
                front_drop(cvict, "cluster")
            a0 = max(Lanc, tf - LEG_KEEP + 1); fr = gi["frames"][max(0, a0 - 3):tf + 1]; fr = [fr[0]] * (3 - (a0 - max(0, a0 - 3))) + fr
            pos = gi["dpos"][max(0, tf - 1)]
            ne = dict(uid=fnext[0], slot=slot, base=base, score=float(v[j]), score0=float(v[j]), tries=0, ho=0, imp=0, adv=0, born=st["update"], dur=tf + 1 - Lanc, anc=anc,
                      clip=np.stack(gi["frames"][tf - half:tf + half + 1:FRONT_STRIDE]), diag=dict(x=int(pos[0]), y=int(pos[1]), room=int(pos[2])))
            fnext[0] += 1
            front[ne["uid"]] = ne; demos[fkey(ne)] = [mkdemo(ne["dur"], np.stack(fr), np.array(gi["acts"][a0:tf + 1], np.int64))]
            d_ = demos[fkey(ne)][0]
            if len(d_[3]) < len(d_[2]):
                ne["dur_raw"] = ne["dur"]; ne["dur"] = int(len(d_[3]))
            front_f.write(json.dumps(dict(update=st["update"], ev="add", uid=ne["uid"], slot=slot, base=base, anc=anc, score=ne["score"], dur=ne["dur"], dur_raw=ne.get("dur_raw"), diag=ne["diag"], src_front=gi["front"], group=len(group), pl=pl, l0_room=int(gi["dpos"][max(0, L0 - 1)][2]))) + "\n"); front_f.flush()
        fin.clear()
        if front and st["update"] % FRONT_RESCORE == 0:                  # same normaliser for every entry at every comparison
            for e in front.values():
                e["score"] = float(np.median(front_novelty(list(e["clip"]))))

    deep_dir = P("_deep"); deep_n = [len(os.listdir(deep_dir)) if os.path.isdir(deep_dir) else 0]
    def record_deep(i, gi, rec):
        """Diagnostic recorder (rec_x > 0): frames, actions and RAM x/y/room (diag only) of games whose room-7 x reached rec_x. Never read by the algorithm."""
        xm = rec["diag_r7_xmax"]
        if hp["rec_x"] <= 0 or xm is None or xm < hp["rec_x"] or deep_n[0] >= REC_MAX:
            return
        os.makedirs(deep_dir, exist_ok=True)
        pos = np.array(gi["dpos"], np.int32); fr = np.stack(gi["frames"]); ac = np.array(gi["acts"], np.int64)
        i7 = np.flatnonzero(pos[:, 2] == 7); a0 = max(0, int(i7[0]) - 100) if len(i7) else 0; a0 = max(a0, len(fr) - REC_STEPS)
        np.savez_compressed(os.path.join(deep_dir, f"ep_u{st['update']}_e{i}_x{int(xm)}.npz"), frames=fr[a0:], acts=ac[a0:], pos=pos[a0:], start=a0, target=np.array(str(rec["target"])), seq=np.array(str(rec["seq"])))
        deep_n[0] += 1

    rare_dir = P("_deep_rare"); rare_n = [len(os.listdir(rare_dir)) if os.path.isdir(rare_dir) else 0]
    room_games, n_games = collections.Counter(), [0]
    if os.path.exists(P("_episodes.jsonl")):
        for epl_ in open(P("_episodes.jsonl")):
            try: rare_seen_ = json.loads(epl_)["diag_rooms"]
            except Exception: continue
            n_games[0] += 1; room_games.update(rare_seen_)
    log(f"rare-room recorder: {n_games[0]} past games; rooms visited by < {RARE_FRAC:.2%} of games count as rare (now {sorted(r_ for r_, c_ in room_games.items() if c_ < RARE_FRAC * n_games[0])}); {rare_n[0]}/{REC_RARE_MAX} files")
    def rare_rooms_of(rooms_seen):
        """Diagnostic only: rooms of this game that were visited by < RARE_FRAC of all earlier games (data-driven 'outside the usual set')."""
        if n_games[0] < RARE_MIN_GAMES:
            return []
        return [r_ for r_ in rooms_seen if room_games[r_] < RARE_FRAC * n_games[0]]
    def record_rare(i, gi, rec, rr):
        """Diagnostic recorder: whole-game frames/acts/RAM positions of games that visited a rare room, own cap, no rec_x condition. Never read by the algorithm."""
        if REC_RARE_MAX <= 0 or rare_n[0] >= REC_RARE_MAX:
            return False
        try:
            os.makedirs(rare_dir, exist_ok=True)
            pos = np.array(gi["dpos"], np.int32); fr = np.stack(gi["frames"]); ac = np.array(gi["acts"], np.int64)
            ir = np.flatnonzero(np.isin(pos[:, 2], rr)); a0 = max(0, int(ir[0]) - 300) if len(ir) else 0; a0 = max(a0, len(fr) - RARE_STEPS)
            np.savez_compressed(os.path.join(rare_dir, f"ep_u{st['update']}_e{i}_rare{'-'.join(map(str, rr))}.npz"), frames=fr[a0:], acts=ac[a0:], pos=pos[a0:], start=a0,
                                deaths=np.array(gi["diag_deaths"], np.int32).reshape(-1, 4), rare=np.array(rr), meta=np.array(json.dumps(rec)))
            rare_n[0] += 1; log(f"rare-room game recorded: update {st['update']} env {i} rooms {rr} ({rare_n[0]}/{REC_RARE_MAX})")
            return True
        except Exception as ex_:
            log(f"rare-room recorder failed: {ex_!r}"); return False

    lmroom = json.load(open(P("_lmroom.json"))) if os.path.exists(P("_lmroom.json")) else {}
    g = [None] * N
    epi = [dict() for _ in range(N)]
    def new_game(i):
        scenes.reset(i, vec.obs[i][..., -1])
        fid = None; fdur = 0; fx = []; wing = "stitch"
        tgt = stitch_target() if hp["p_stitch"] > 0 and rng.random() < hp["p_stitch"] else None
        if tgt is None:
            tgt = pick_target(); wing = PW[0]
            if hp["p_frontier"] > 0 and front and rng.random() < hp["p_frontier"]:
                fk = pick_frontier()
                if fk is not None:
                    tgt, fid, wing = fk, fk[-1][1], "front"; fdur = front[fid]["dur"]; fx = list(front[fid]["base"][anc_of(front[fid]):])
        g[i] = dict(wing=wing, front=fid, fdur=fdur, fprox=bool(fid is not None and rng.random() < hp["p_fprox"]), fhit=None, fck=0, fckm=None, fhn=0, fend=False, trail=None, trail_seq=None, tck=0, tckn=0, target=tgt, k=0, pend=set(), phase=0 if tgt else 1, t=0, seq=[], seen={int(scenes.scene[i])}, bscene=int(scenes.scene[i]), bprev=None, bleg0=0, bpend=None, bvis={int(scenes.scene[i])}, brun=None, nrew=0, leg0=0, legs0=[], fexp=fx, frames=[], acts=[], score=0.0,
                    t_arrive=(0 if not tgt else None), diag_lives_arrive=None, diag_rooms=set(), diag_r7runs=[], fail=None, owner=int(rng.integers(NA)), prox=set(),
                    score_ho=(0.0 if not tgt else None), dpos=[], capq=[], diag_lives=None, diag_deaths=[], diag_r13=0)
    for i in range(N):
        new_game(i)

    def add_demo(i, n, store=None, leg0=None, te=None):
        gi = g[i]; store = demos if store is None else store; leg0 = gi["leg0"] if leg0 is None else leg0; te = gi["t"] if te is None else te
        a0 = max(leg0, te - LEG_KEEP + 1)
        fr = gi["frames"][max(0, a0 - 3):te + 1]
        fr = [fr[0]] * (3 - (a0 - max(0, a0 - 3))) + fr
        d = store.setdefault(n, [])
        dur = te + 1 - leg0
        if len(d) < DEMOS_PER_NODE or dur < d[-1][0]:
            d.append(mkdemo(dur, np.stack(fr), np.array(gi["acts"][a0:te + 1], np.int64)))
            d.sort(key=lambda x: x[0]); del d[DEMOS_PER_NODE:]
            return True
        return False

    BK = dict(new=0, tnew=0)

    def back_commit(i):
        """Go-back leg: the route legs' logic applied to scene revisits. When a game re-enters a scene it has already been in, the leg from the step the scene it
        is leaving began to the transition step is stored like a route leg (add_demo, fastest DEMOS_PER_NODE per key, either phase) in bdemos, keyed by landmark
        tuples (scene before, scene left, scene re-entered), or (scene left, scene re-entered) when the scene left is the start scene. The key fixes where the
        leg starts (the entrance used) and ends (the exit), so the fastest legs are comparable. Trip: a run of consecutive revisits that never repeats a scene
        (a real way back through >= 2 rooms) is also stored as one leg from the entrance of the first scene of the run (the first-visit landmark when the run
        follows one) to every scene it reaches after the second hop, under a (("t", 0), scene before, first scene, ..., last scene) key. A first visit, or a
        revisit of a scene already in the run (a bounce or loop), ends the run. Pixels only."""
        gi = g[i]; s_to, te = gi["bpend"]; s_from = gi["bscene"]; dur = te + 1 - gi["bleg0"]
        if s_to in gi["bvis"]:
            if dur >= TRAV_MIN:
                n_ = tuple(("s", int(x_)) for x_ in ((gi["bprev"],) if gi["bprev"] is not None else ()) + (s_from, s_to))
                BK["new"] += int(add_demo(i, n_, bdemos, gi["bleg0"], te))
            run = gi["brun"]
            if run is None:
                run = gi["brun"] = dict(t0=gi["bleg0"], path=([gi["bprev"]] if gi["bprev"] is not None else []) + [s_from], r=0)
            if s_to in run["path"]:
                gi["brun"] = None
            else:
                run["path"].append(s_to); run["r"] += 1
                if run["r"] >= 2:
                    BK["tnew"] += int(add_demo(i, (("t", 0),) + tuple(("s", int(x_)) for x_ in run["path"]), bdemos, run["t0"], te))
        else:
            gi["brun"] = None
        gi["bprev"], gi["bscene"], gi["bleg0"], gi["bpend"] = s_from, s_to, te + 1, None
        gi["bvis"].add(s_to)

    def back_track(i, s_now):
        """A scene change counts once the new scene has held for TRAV_CONFIRM steps (room-transition frames can flicker between known scenes for a step or two);
        a flip back to the scene being left cancels it, a further change re-points it."""
        gi = g[i]
        if s_now != gi["bscene"]:
            if gi["bpend"] is None or gi["bpend"][0] != s_now:
                gi["bpend"] = (s_now, gi["t"])
            elif gi["t"] - gi["bpend"][1] >= TRAV_CONFIRM:
                back_commit(i)
        else:
            gi["bpend"] = None

    def back_batch(n):
        """Same sampler as bc_batch (a node, one of its demos, one of the demo's kept steps), over all go-back keys (a TRAV_TRIP share of the samples from the multi-room trip keys, the rest from the single-revisit keys); goal = the re-entered scene's landmark, k slot = BACK_K."""
        keys = list(bdemos)
        if not keys:
            return None
        tk_ = [k_ for k_ in keys if k_[0][0] == "t"]; hk_ = [k_ for k_ in keys if k_[0][0] != "t"]
        ob, lm, kk, ac = [], [], [], []
        for _ in range(n):
            pool = tk_ if tk_ and (not hk_ or rng.random() < TRAV_TRIP) else hk_
            nd_ = pool[rng.integers(len(pool))]
            dur, fr, a, kp = bdemos[nd_][rng.integers(len(bdemos[nd_]))]
            j = kp[rng.integers(len(kp))]
            ob.append(np.moveaxis(fr[j:j + 4], 0, -1)); lm.append(lmid(nd_[-1])); kk.append(BACK_K); ac.append(a[j])
        return np.stack(ob), np.array(lm), np.array(kk), np.array(ac)

    def trav_step(n, xb=None):
        """One BC step of the traversal net. Route part: the route controller's own sampler (bc_batch: same legs, same landmark and depth inputs). Go-back part:
        back_batch over bdemos (same sampler, k slot = BACK_K). Goal -1 part (xb = obs and owner of explorer states from this rollout): the takeover explorer's
        policy on those states, KL-distilled, with a random depth input."""
        b = bc_batch(n) if n else None
        bb = back_batch(TRAV_BACK) if TRAV_BACK else None
        parts = [(torch.from_numpy(x_[0]), torch.from_numpy(x_[1]), torch.from_numpy(x_[2])) for x_ in (b, bb) if x_ is not None]
        nb_ = len(b[3]) if b is not None else 0; nk_ = len(bb[3]) if bb is not None else 0
        lg_x = None
        if xb is not None:
            ox = torch.from_numpy(xb[0]).to(dev)
            with torch.no_grad():
                lg_x = torch.stack([net.actor_logits(net.features(ox), p) for p in PREFIXES])[torch.from_numpy(xb[1]).to(dev), torch.arange(len(ox), device=dev)]
            parts.append((torch.from_numpy(xb[0]), torch.full((len(ox),), EXPLORE_LM, dtype=torch.int64), torch.from_numpy(rng.integers(0, 24, len(ox)))))
        if not parts:
            return None
        lg, _ = trav(*[torch.cat([p_[j] for p_ in parts]).to(dev) for j in range(3)])
        loss, out = 0.0, [None] * len(TRAV_KEYS)
        if b is not None:
            y = torch.from_numpy(b[3]).to(dev); ce = F.cross_entropy(lg[:nb_], y); loss = loss + ce
            out[0], out[1] = ce.item(), (lg[:nb_].argmax(1) == y).float().mean().item()
        if bb is not None:
            y = torch.from_numpy(bb[3]).to(dev); ce = F.cross_entropy(lg[nb_:nb_ + nk_], y); loss = loss + ce
            out[4], out[5] = ce.item(), (lg[nb_:nb_ + nk_].argmax(1) == y).float().mean().item()
        if lg_x is not None:
            lx = lg[nb_ + nk_:]; kl = F.kl_div(F.log_softmax(lx, 1), F.softmax(lg_x, 1), reduction="batchmean"); loss = loss + kl
            ls_ = F.log_softmax(lx, 1); out[2], out[3] = kl.item(), -(ls_.exp() * ls_).sum(1).mean().item()
        opt_tr.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(tr_params, MAX_GRAD); opt_tr.step()
        return out

    Z84 = np.zeros((84, 84), np.uint8)

    def goal_frame(n):
        """The target's appearance: last frame of the shortest demo leg into node n (a frontier node: its frontier frame); black if unknown."""
        d_ = demos.get(n)
        return d_[0][1][-1] if d_ else Z84

    def goal_of(gi):
        return goal_frame(tuple(gi["target"][:gi["k"] + 1])) if gi["phase"] == 0 else Z84

    def bc_batch(n):
        active = set()
        for gi in g:
            for k in range(1, len(gi["target"]) + 1):
                if gi["target"][:k] in demos: active.add(gi["target"][:k])
        if not active:
            return None
        active = list(active); ob, lm, kk, ac, gl = [], [], [], [], []
        fa_ = [a_ for a_ in active if a_[-1][0] == "f"]; ra_ = [a_ for a_ in active if a_[-1][0] != "f"]; ff_ = float(hp["bc_front_frac"]) if fa_ and ra_ else 0.0
        fw_ = np.array([fw(front[a_[-1][1]]) if a_[-1][1] in front else 1e-6 for a_ in fa_]) ** float(hp["bc_front_pow"]) if fa_ else None
        fp_ = fw_ / fw_.sum() if fa_ and fw_.sum() > 0 else None
        bh_ = float(hp["bc_hard"]); rw_ = None
        if bh_ > 0 and ra_:
            def legfail(n_):                                            # share of games that reached the leg's start landmark but not its end
                r_ = rate(n_) if n_ in archive else None
                if r_ is None:
                    return 0.0
                r0_ = rate(n_[:-1]) if len(n_) > 1 and n_[:-1] in archive else None
                return max(0.0, 1.0 - r_ / r0_) if r0_ else max(0.0, 1.0 - r_)
            rw_ = np.array([1.0 + bh_ * legfail(n_) for n_ in ra_]); rw_ /= rw_.sum()
        for _ in range(n):
            if ff_ > 0 and rng.random() < ff_:
                nd_ = fa_[rng.choice(len(fa_), p=fp_)] if fp_ is not None else fa_[rng.integers(len(fa_))]
            else:
                pool_ = ra_ if (ff_ > 0 or not fa_) else active
                nd_ = pool_[rng.choice(len(pool_), p=rw_)] if (rw_ is not None and pool_ is ra_) else pool_[rng.integers(len(pool_))]
            dur, fr, a, kp = demos[nd_][rng.integers(len(demos[nd_]))]
            j = kp[rng.integers(len(kp))]
            ob.append(np.moveaxis(fr[j:j + 4], 0, -1)); lm.append(lmid(nd_[-1])); kk.append(len(nd_) - 1); ac.append(a[j]); gl.append(fr[-1])
        return np.stack(ob), np.array(lm), np.array(kk), np.array(ac), np.stack(gl)

    def bc_step(n):
        b = bc_batch(n)
        if b is None:
            return None
        lg, _ = mu(torch.from_numpy(b[0]).to(dev), torch.from_numpy(b[1]).to(dev), torch.from_numpy(b[2]).to(dev), torch.from_numpy(b[4]).to(dev))
        bl = F.cross_entropy(lg, torch.from_numpy(b[3]).to(dev))
        opt_mu.zero_grad(); (BC_W * bl).backward(); torch.nn.utils.clip_grad_norm_(mu.parameters(), MAX_GRAD); opt_mu.step()
        return bl.item()

    if st["update"] == 0 and BC_WARMUP:
        for gi in g:                                  # warm-up targets are the games' own targets; all prefixes' demos are active
            pass
        t0_ = time.time(); last = None
        for s_ in range(BC_WARMUP):
            last = bc_step(256)
            if last is None: break
        log(f"BC warm-up of the route controller: {BC_WARMUP} steps, last loss {last}, {time.time() - t0_:.0f}s")

    stop = {"now": False}
    def on_int(*_):
        if not stop["now"]:
            print("[stop] signal received -- finishing this update, then saving and exiting.", flush=True)
        stop["now"] = True
    signal.signal(signal.SIGINT, on_int)

    def save():
        st["sigs"] = scenes.sigs
        torch.save(dict(mu=mu.state_dict(), net=net.state_dict(), net_arch=EXPLORER_ARCH, opt_mu=opt_mu.state_dict(), opt_tk=opt_tk.state_dict(), agents=[a.state() for a in agents], st=st,
                        trav2=trav.state_dict(), opt_tr2=opt_tr.state_dict(), progfeat=(pnet.state_dict() if pnet is not None else None)), P(".pt.tmp"))
        os.replace(P(".pt.tmp"), P(".pt"))
        pickle.dump(demos, open(P("_demos.pkl.tmp"), "wb"), protocol=4); os.replace(P("_demos.pkl.tmp"), P("_demos.pkl"))
        pickle.dump(bdemos, open(P("_bdemos.pkl.tmp"), "wb"), protocol=4); os.replace(P("_bdemos.pkl.tmp"), P("_bdemos.pkl"))
        pickle.dump(lmref, open(P("_lmref.pkl.tmp"), "wb"), protocol=4); os.replace(P("_lmref.pkl.tmp"), P("_lmref.pkl"))
        pickle.dump(dict(front=front, next=fnext[0]), open(P("_front.pkl.tmp"), "wb"), protocol=4); os.replace(P("_front.pkl.tmp"), P("_front.pkl"))

    H = net.gru_hidden_dim
    h = torch.zeros(N, H, device=dev)
    popart_dn = lambda ag, s, x: x * popart_mu_sigma(ag.popart[s])[1] + popart_mu_sigma(ag.popart[s])[0]
    epi_reset = np.zeros(N, bool)
    ar = torch.arange(N, device=dev)
    recent, t_last = [], time.time()
    while st["frames"] < MAX_FRAMES and not stop["now"]:
        hp = read_live()
        wu = st.get("expl_warm", 0) > 0
        for gr in opt_tk.param_groups:
            gr["lr"] = hp["takeover_lr"] * (TRUNK_LR_MULT if gr.get("name") == "trunk" and EXPLORER_ARCH in ("mu", "trav") else 1.0)
        O = np.zeros((T + 1, N, 84, 84, 4), np.uint8); A = np.zeros((T, N), np.int64); LP = np.zeros((T, N), np.float32)
        LMI = np.zeros((T, N), np.int64); KK = np.zeros((T, N), np.int64); OWN = np.zeros((T, N), np.int64); GL = np.zeros((T, N, 84, 84), np.uint8)
        MR = np.zeros((T, N), bool); Rm = np.zeros((T, N), np.float32); Dm = np.ones((T, N), np.float32); Vm = np.zeros((T + 1, N), np.float32)
        Re = np.zeros((T, N), np.float32); Rp = np.zeros((T, N), np.float32); Rl = np.zeros((T, N), np.float32); Dg = np.zeros((T, N), np.float32); VALL = np.zeros((T + 1, N, NA, NS), np.float64)
        Hs = torch.zeros(T, N, H, device=dev); ent_ = np.zeros((T, N), np.float32); DEC = np.ones((T, N), bool)
        for t in range(T):
            O[t] = vec.obs
            ph = np.array([gi["phase"] for gi in g]); MR[t] = ph == 0
            LMI[t] = [lmid(gi["target"][gi["k"]]) if gi["phase"] == 0 else 0 for gi in g]; KK[t] = [gi["k"] for gi in g]; GL[t] = [goal_of(gi) for gi in g]
            OWN[t] = [gi["owner"] for gi in g]
            ot = torch.from_numpy(O[t]).to(dev)
            with torch.no_grad():
                lg_m, v_m = mu(ot, torch.from_numpy(LMI[t]).to(dev), torch.from_numpy(KK[t]).to(dev), torch.from_numpy(GL[t]).to(dev))
                cf0 = net.features(ot)
                Rp[t] = np.minimum(prog.step(pf(ot) if pf is not None else cf0), hp["prog_clip"])
                feat, h_new = net.gru_step(cf0, h)
                Hs[t] = h
                lg_all = torch.stack([net.actor_logits(feat, p) for p in PREFIXES])            # (NA, N, 18)
                own_t = torch.from_numpy(OWN[t]).to(dev)
                lg_e = lg_all[own_t, ar]
                if int(hp["anchor_walker"]):                                                    # anchor-owned takeover games: uniform actions through the same hold/eps process (the U10 walker)
                    lg_e = torch.where((own_t == ANCHOR_IDX).unsqueeze(1), torch.zeros_like(lg_e), lg_e)
                for k, ag in enumerate(agents):
                    for s_i, s in enumerate(STREAMS):
                        VALL[t, :, k, s_i] = popart_dn(ag, s, net.critic_value(feat, f"{s}_{ag.p}", True)).cpu().numpy()
                dm, de = torch.distributions.Categorical(logits=lg_m), torch.distributions.Categorical(logits=lg_e)
                am, ae = dm.sample(), de.sample()
            mr = torch.from_numpy(MR[t]).to(dev)
            R_hold = max(1, int(hp["expl_hold"])); eps_dev = float(hp["expl_eps"])
            hl = np.array([gi.get("hold_left", 0) for gi in g]); ha = np.array([gi.get("hold_a", 0) for gi in g], dtype=np.int64); ae_np = ae.cpu().numpy()
            in_seg = (~MR[t]) & (hl > 0) & (R_hold > 1)
            devi = in_seg & (hrng.random(N) < eps_dev)
            held = in_seg & ~devi
            new_seg = (~MR[t]) & ~in_seg
            DEC[t] = ~held
            for i, gi in enumerate(g):
                if MR[t, i]: gi["hold_left"] = 0
                elif new_seg[i]: gi["hold_a"], gi["hold_left"] = int(ae_np[i]), R_hold - 1
                else: gi["hold_left"] = int(hl[i]) - 1
            ae = torch.from_numpy(np.where(held, ha, ae_np).astype(np.int64)).to(dev)
            a = torch.where(mr, am, ae)
            A[t] = a.cpu().numpy(); LP[t] = torch.where(mr, dm.log_prob(am), de.log_prob(ae)).cpu().numpy()
            ent_[t] = de.entropy().cpu().numpy(); Vm[t] = v_m.cpu().numpy()
            for i, gi in enumerate(g):
                gi["frames"].append(O[t, i, :, :, -1].copy()); gi["acts"].append(int(A[t, i]))
                while gi["capq"] and gi["capq"][0][1] < len(gi["frames"]):
                    L_, k_, n_try = gi["capq"].pop(0)
                    if changed_frac(gi["frames"][k_], gi["frames"][k_ - 3]) < 0.01:                  # the screen has settled (no room-change wipe or other animation in progress)
                        lmref_add(L_, gi["frames"][k_])
                    elif n_try < 3:
                        gi["capq"].append((L_, k_ + 4, n_try + 1))
            caps = []
            for gi in g:
                c_ = STEP_CAP_PER_SCENE * len(gi["seen"])
                if gi["phase"] == 1:
                    gi.setdefault("t_ho", gi["t"])
                    if hp["takeover_cap"] > 0: c_ = min(c_, gi["t_ho"] + int(hp["takeover_cap"]))
                caps.append(c_)
            r, done, lives, rooms = vec.step(A[t], caps)
            if DEBUG_FAKE_R:
                r = np.where(rng.random(N) < DEBUG_FAKE_R, 1.0, r).astype(np.float32)
            h = h_new * torch.from_numpy(1.0 - done.astype(np.float32)).to(dev).unsqueeze(-1)
            prog.reset(done.astype(bool))
            new_scene = scenes.step(vec.obs[..., -1], done)
            if hp["prox_thr"] > 0 and st["update"] - PX["upd"] >= 50:
                build_prox()
            pxh = prox_hits(vec.obs[..., -1], done)
            fph = fprox_hits(vec.obs[..., -1], done)
            for i, gi in enumerate(g):
                gi["dpos"].append((vec.xs[i], vec.ys[i], rooms[i]))
                lv_ = int(lives[i])
                if gi["diag_lives"] is not None and lv_ < gi["diag_lives"] and len(gi["diag_deaths"]) < DEATH_CAP:
                    gi["diag_deaths"].append((len(gi["dpos"]) - 1, int(rooms[i]), int(vec.xs[i]), int(vec.ys[i])))
                gi["diag_lives"] = lv_
                gi["diag_r13"] += int(rooms[i] == 13)
                gi["score"] += r[i]; gi["diag_rooms"].add(rooms[i] if not done[i] else -1)
                if not done[i] and rooms[i] == 7:
                    if gi.get("_r7open"): gi["diag_r7runs"][-1].append(vec.xs[i])
                    else: gi["diag_r7runs"].append([vec.xs[i]]); gi["_r7open"] = True
                else:
                    gi["_r7open"] = False
                lms = []
                if r[i] > 0:
                    gi["nrew"] += 1; lms.append(("r", gi["nrew"]))
                if not done[i]:
                    back_track(i, int(scenes.scene[i]))
                if i in new_scene and new_scene[i] not in gi["seen"]:
                    gi["seen"].add(new_scene[i]); lms.append(("s", int(new_scene[i])))
                route_end = False
                if gi["phase"] == 1:
                    Rl[t, i] += sum(1 for lm in lms if lm not in gi["prox"])
                    for L_ in pxh.get(i, ()):
                        gi["prox"].add(L_); Rl[t, i] += 1
                    if i in fph:                                                  # frontier trail checkpoint, paid like a landmark
                        gi["tck"] = fph[i] + 1; gi["tckn"] += 1; Rl[t, i] += 1
                if MR[t, i]:
                    Rm[t, i] = -STEP_COST
                for lm in lms:
                    k_ = f"{lm[0]}{lm[1]}@{rooms[i]}"; lmroom[k_] = lmroom.get(k_, 0) + 1                  # diagnostic: true room at each landmark event
                    gi["seq"].append(lm); n_ = tuple(gi["seq"]); gi["capq"].append((lm, gi["t"] + 1 + LMREF_DELAY, 0))
                    if n_ not in archive:
                        ev_f.write(json.dumps(dict(update=st["update"], frames=st["frames"], node=n_, t=gi["t"], by="route" if gi["phase"] == 0 else PREFIXES[gi["owner"]], diag_room=rooms[i])) + "\n"); ev_f.flush()
                        log(f"[new] depth {len(n_)} node {n_} at decision {gi['t']} ({'route' if gi['phase'] == 0 else PREFIXES[gi['owner']]})")
                    node(n_)["visits"] += 1
                    add_demo(i, n_); gi["leg0"] = gi["t"] + 1; gi["legs0"].append(gi["leg0"])
                    if gi["phase"] == 0:
                        tg = gi["target"]; j_ = None
                        if lm == tg[gi["k"]]:
                            gi["k"] += 1; Rm[t, i] += 1.0
                            while gi["k"] in gi["pend"]:
                                gi["k"] += 1
                            if gi["k"] == len(tg):
                                gi["phase"], gi["t_arrive"], gi["diag_lives_arrive"], route_end = 1, gi["t"] + 1, lives[i], True
                                gi["score_ho"] = gi["score"]
                        else:
                            j_ = next((j for j in range(gi["k"] + 1, min(gi["k"] + 3, len(tg))) if tg[j] == lm and j not in gi["pend"]), None)
                            if j_ is not None:
                                gi["pend"].add(j_); Rm[t, i] += 1.0
                            elif tg[gi["k"]][0] == "f" and gi["fexp"] and lm == gi["fexp"][0]:      # a landmark the anchored frontier's own demo passes through
                                gi["fexp"].pop(0); Rm[t, i] += 1.0
                            elif tg[gi["k"]][0] == "f":                       # the real next landmark came first
                                m_ = len(fck_protos(gi["front"])) if fprox_live(gi) else 0
                                if hp["fprox_strict"] and not (m_ > 0 and gi["fck"] >= max(1, m_ - 1)):   # strict: an off-route landmark is not an arrival (unless every checkpoint but the last was matched = on-route overshoot); the game keeps the landmark, the explorers take over
                                    gi["phase"], gi["fail"], route_end = 1, "landmark first", True
                                else:
                                    gi["phase"], gi["t_arrive"], gi["diag_lives_arrive"], route_end = 1, gi["t"] + 1, lives[i], True; Rm[t, i] += 1.0
                                gi["score_ho"] = gi["score"]; gi["fhit"], gi["fckm"], gi["fend"] = gi["fhn"], m_, bool(m_ > 0 and gi["fck"] >= m_)
                            else:
                                gi["phase"], gi["fail"], route_end = 1, "other landmark", True; gi["score_ho"] = gi["score"]
                if gi["phase"] == 0 and gi["front"] is not None and gi["k"] == len(gi["target"]) - 1:
                    el_ = gi["t"] + 1 - gi["leg0"]
                    if gi["fprox"]:                                            # arrival by visual match to checkpoints along the demo, +1 each; the timer (x slack) is an unpaid fallback
                        m_ = len(fck_protos(gi["front"])) if fprox_live(gi) else 0
                        if i in fph:
                            gi["fck"] = fph[i] + 1; gi["fhn"] += 1; Rm[t, i] += 1.0
                        done_ = m_ > 0 and gi["fck"] >= m_
                        if done_ or el_ >= gi["fdur"] * hp["fprox_slack"]:
                            if done_ or not hp["fprox_strict"]:
                                gi["phase"], gi["t_arrive"], gi["diag_lives_arrive"], route_end = 1, gi["t"] + 1, lives[i], True
                            else:                                          # strict: the timer no longer counts as arrival; the leg fails and the explorers take over where the game is
                                gi["phase"], gi["fail"], route_end = 1, "no match", True
                            gi["score_ho"] = gi["score"]; gi["fhit"], gi["fckm"], gi["fend"] = gi["fhn"], m_, bool(done_)
                    elif el_ >= gi["fdur"]:                                     # frontier arrival is by elapsed time
                        gi["phase"], gi["t_arrive"], gi["diag_lives_arrive"], route_end = 1, gi["t"] + 1, lives[i], True
                        gi["score_ho"] = gi["score"]; Rm[t, i] += 1.0
                if gi["phase"] == 0 and gi["t"] + 1 - gi["leg0"] > LEG_TIMEOUT:
                    gi["phase"], gi["fail"], route_end = 1, "timeout", True; gi["score_ho"] = gi["score"]
                if gi["phase"] == 1 and (gi["trail_seq"] != tuple(gi["seq"]) or (gi["trail"] is not None and not trail_live(gi))):
                    gi["trail_seq"] = tuple(gi["seq"]); u_ = trail_pick(gi["trail_seq"])
                    if u_ != gi["trail"]:
                        gi["trail"] = u_; gi["tck"] = gi["fck"] if (u_ is not None and u_ == gi["front"]) else 0
                Re[t, i] = r[i]; Dg[t, i] = float(done[i])
                Dm[t, i] = 1.0 if (route_end or done[i] or not MR[t, i]) else 0.0
                gi["t"] += 1
                if done[i]:
                    tg_ = gi["target"][:-1] if gi["front"] is not None else gi["target"]
                    for j in range(1, len(tg_) + 1):
                        if gi["k"] >= j - 1 and (gi["wing"] != "stitch" or tg_[:j] in archive):
                            hh = node(tg_[:j])["hist"]; full = j == len(tg_) and gi["t_arrive"] is not None and gi["front"] is None
                            hh.append((gi["k"] >= j, gi["t_arrive"] if full else None, gi["diag_lives_arrive"] if full else None)); del hh[:-200]
                    rec = dict(update=st["update"], owner=PREFIXES[gi["owner"]], wing=gi["wing"], target=gi["target"], arrived=gi["t_arrive"] is not None, t_arrive=gi["t_arrive"], fail=gi["fail"],
                               seq=gi["seq"], length=gi["t"], handoff=gi["score_ho"] is not None, score=float(gi["score"]), takeover_score=float(gi["score"] - (gi["score_ho"] or 0.0)),
                               diag_lives_arrive=gi["diag_lives_arrive"], diag_rooms=sorted(x for x in gi["diag_rooms"] if x >= 0), diag_r7_xmax=room7_xmax(gi["diag_r7runs"]), prox=len(gi["prox"]), front=gi["front"], fprox=gi["fhit"], fprox_m=gi["fckm"], fprox_end=gi["fend"], trail=gi["trail"], tck=gi["tckn"], tck_pos=gi["tck"], tck_m=(len(fck_protos(gi["trail"])) if trail_live(gi) else None), diag_ho=([int(x_) for x_ in gi["dpos"][gi["t_arrive"] - 1]] if gi["t_arrive"] else None),
                               diag_end=[int(gi["dpos"][-1][2]), int(gi["dpos"][-1][0]), int(gi["dpos"][-1][1]), gi["diag_lives"]], diag_deaths=[list(d_) for d_ in gi["diag_deaths"]], diag_r13=int(gi["diag_r13"]))
                    rare_hit = rare_rooms_of(rec["diag_rooms"]); n_games[0] += 1; room_games.update(rec["diag_rooms"])
                    recent.append(rec); ep_f.write(json.dumps(rec) + "\n")
                    if EVAL_GAMES and len(ev_games[i]) < ev_per:
                        ev_games[i].append(rec); ev_paths[i].append(np.array(gi["dpos"], np.int16))   # RAM x, y, room per step (diag only, for the path viewer)
                    if gi["wing"] == "stitch":
                        tk_ = tuple(gi["target"]); STC["att"][tk_] = STC["att"].get(tk_, 0) + 1; STC["arr"][tk_] = STC["arr"].get(tk_, 0) + (1 if gi["t_arrive"] is not None else 0)
                    if not (rare_hit and record_rare(i, gi, rec, rare_hit)):
                        record_deep(i, gi, rec)
                    if (hp["front_collect"] > 0 or hp["p_frontier"] > 0) and gi["seq"] and front_ok(tuple(gi["seq"])):
                        fin.append(gi)
                    new_game(i)
        O[T] = vec.obs
        if EVAL_GAMES:
            st["frames"] += T * N * 4; st["update"] += 1; recent = recent[-300:]; n_ev = sum(map(len, ev_games))
            if st["update"] % 20 == 0:
                log(f"[eval] {n_ev}/{ev_per * N} games, {st['frames'] - ev_frames0} frames")
            if n_ev >= ev_per * N:
                break
            continue
        front_process()
        TK = ~MR                                              # takeover steps
        ph_end = np.array([gi["phase"] for gi in g])
        with torch.no_grad():
            oT = torch.from_numpy(O[T]).to(dev)
            featT, _ = net.gru_step(net.features(oT), h)
            for k, ag in enumerate(agents):
                for s_i, s in enumerate(STREAMS):
                    VALL[T, :, k, s_i] = popart_dn(ag, s, net.critic_value(featT, f"{s}_{ag.p}", True)).cpu().numpy()
            lmT = torch.from_numpy(np.array([lmid(gi["target"][gi["k"]]) if gi["phase"] == 0 else 0 for gi in g])).to(dev)
            gT = torch.from_numpy(np.stack([goal_of(gi) for gi in g])).to(dev)
            Vm[T] = np.where(ph_end == 0, mu(oT, lmT, torch.from_numpy(np.array([gi["k"] for gi in g])).to(dev), gT)[1].cpu().numpy(), 0.0)
        adv_m = gae_mu(Rm, Vm, Dm, G_MU); ret_m = adv_m + Vm[:T]
        fo = O[:T].reshape(-1, 84, 84, 4); fg = GL.reshape(-1, 84, 84); flat = lambda x: x.reshape(-1)
        S = collections.defaultdict(float); nb_m = 0
        idx_tk = np.flatnonzero(flat(TK)); idx_m = np.flatnonzero(flat(MR)); tt = lambda x, mb: torch.from_numpy(flat(x)[mb]).to(dev)
        # ---- route controller update
        for _ in range(EPOCHS):
            if len(idx_m) >= 64:
                rng.shuffle(idx_m)
                for mb in np.array_split(idx_m, MINIBATCHES):
                    mb = np.sort(mb)
                    lg, v = mu(torch.from_numpy(fo[mb]).to(dev), tt(LMI, mb), tt(KK, mb), torch.from_numpy(fg[mb]).to(dev))
                    dist = torch.distributions.Categorical(logits=lg); lp = dist.log_prob(tt(A, mb))
                    a_ = tt(adv_m, mb); a_ = (a_ - a_.mean()) / (a_.std() + 1e-8); ratio = (lp - tt(LP, mb)).exp()
                    pg = -torch.min(ratio * a_, ratio.clamp(1 - CLIP, 1 + CLIP) * a_).mean()
                    vl = F.mse_loss(v, tt(ret_m, mb)); ent = dist.entropy().mean()
                    loss = pg + 0.5 * vl - float(hp["ent_mu"]) * ent
                    opt_mu.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mu.parameters(), MAX_GRAD); opt_mu.step()
                    S["mu_vl"] += vl.item(); S["mu_ent"] += ent.item(); nb_m += 1
            for _b in range(int(hp["bc_steps"])):
                bl = bc_step(int(hp["bc_batch"]))
                if bl is not None:
                    S["bc"] += bl; S["bc_n"] += 1
            if TRAV_BC and EXPLORER_ARCH != "trav":
                xb = None
                if TRAV_X and EXPLORER_ARCH == "mu" and len(idx_tk) >= 64:
                    mbx = np.sort(rng.choice(idx_tk, min(TRAV_X, len(idx_tk)), replace=False)); xb = (fo[mbx], flat(OWN)[mbx])
                tb = trav_step(256, xb)
                if tb is not None:
                    for key_, v_ in zip(TRAV_KEYS, tb):
                        if v_ is not None: S[key_] += v_; S[key_ + "_n"] += 1
        # ---- takeover agents: novelty reward with each owner's own RND, then GAE, competition, PPO
        Rint_raw = np.zeros((T, N), np.float32); epi_mult = np.ones((T, N), np.float32); Rt = np.zeros((T, N), np.float32)
        M = [TK & (OWN == k) for k in range(NA)]
        keys_flat = {}
        for k, ag in enumerate(agents):
            idx = np.flatnonzero(flat(M[k]))
            if len(idx) == 0:
                continue
            ag.rnd.update_obs_stats(torch.from_numpy(fo[idx[::4]]))
            raws, bits = [], []
            if ag.p and hp["episodic_novelty"]:
                proj = np.random.default_rng(SEED_NET).standard_normal((512, EPI_BITS)).astype(np.float32)
            for j in range(0, len(idx), 2048):
                m_, f_ = ag.rnd.intrinsic_reward_with_features(torch.from_numpy(fo[idx[j:j + 2048]]))
                raws.append(m_.cpu().numpy())
                if hp["episodic_novelty"]:
                    keys_flat.update(zip(idx[j:j + 2048].tolist(), np.packbits((f_.cpu().numpy() @ proj) > 0, axis=1)))
            Rint_raw.reshape(-1)[idx] = np.concatenate(raws)
        for t in range(T):                                    # forward-filtered novelty return -> normaliser, per owner (non-episodic filter)
            for k, ag in enumerate(agents):
                m = M[k][t]
                if m.any():
                    ag.filt[m] = ag.filt[m] * hp["gamma_task"] + Rint_raw[t, m]
                    S[f"filt_{k}"] = 1.0; ag.ret_rms.update(ag.filt[m].astype(np.float64))
        for t in range(T):                                    # episodic count multiplier (counts reset at each game start)
            for i in np.flatnonzero(epi_reset):
                epi[i] = {}
            epi_reset = Dg[t] > 0
            if hp["episodic_novelty"]:
                for i in np.flatnonzero(TK[t]):
                    kb = keys_flat[t * N + i].tobytes(); c = epi[i].get(kb, 0) + 1; epi[i][kb] = c; epi_mult[t, i] = 1.0 / np.sqrt(c)
        ne_ = st.setdefault("nov_ema", [None] * NA)
        for k in range(NA):
            if M[k].any():
                mk = float(Rint_raw[M[k]].mean()); ne_[k] = mk if ne_[k] is None else 0.98 * ne_[k] + 0.02 * mk
        stats = {}
        for k, ag in enumerate(agents):
            sd = max(float(ag.ret_rms.var), 1e-12) ** 0.5 + 1e-8
            Rt += np.where(M[k], hp["intrinsic_coef"] * Rint_raw / sd * epi_mult, 0.0).astype(np.float32)
        # per-agent GAE on that agent's own critic values (all steps of the (T,N) grid), then keep the agent's own takeover steps
        ADV, RET = {}, {}
        for k, ag in enumerate(agents):
            for s_i, s in enumerate(STREAMS):
                rew = {"task": Rt, "extrinsic": Re, "progress": Rp, "landmark": Rl}[s]
                gam = hp[f"gamma_{s}"]
                adv = gae(rew, VALL[:T, :, k, s_i], VALL[T, :, k, s_i], Dg, gam, hp["gae_lambda"], nonep=(s == "task"))
                ADV[k, s] = adv; RET[k, s] = adv + VALL[:T, :, k, s_i]
        scale = {}
        for k, ag in enumerate(agents):
            if M[k].sum() >= 64:
                for s in STREAMS:
                    ag.adv_rms[s].update(ADV[k, s][M[k]].astype(np.float64))
            scale[k] = {s: max(float(np.sqrt(ag.adv_rms[s].var)), 1e-6) for s in STREAMS}
            scale[k]["landmark"] = max(scale[k]["landmark"], hp["lm_scale_floor"])
        CDd = {}
        if int(hp["qdef_mode"]) == 2:
            cd = st.setdefault("cd", [dict() for _ in range(NA)])
            ck_ = cell_keys(O[..., -1])                                   # (T+1, N) pixel-cell id of every frame
            uk, inv = np.unique(ck_, return_inverse=True); inv = inv.reshape(ck_.shape)
            cnt = np.array([[c.get(int(u), 0) for u in uk] for c in cd], np.float64)    # counts before this batch
            nmax = cnt.max(0)
            for k in range(NA):
                dd = (nmax - cnt[k]) / np.maximum(nmax, 1.0) * np.minimum(1.0, nmax / CD_MIN)     # share deficit in [0,1]
                CDd[k] = hp["qdef_gamma"] * (1.0 - Dg) * dd[inv[1:]]
            for k in range(NA):
                u_, c_ = np.unique(ck_[:T][M[k]], return_counts=True)
                for a_, b_ in zip(u_.tolist(), c_.tolist()):
                    cd[k][a_] = cd[k].get(a_, 0) + b_
            stats["cd_cells"] = len(set().union(*[set(c) for c in cd]))
        Ot = torch.from_numpy(O).to(dev); Dgt = torch.from_numpy(Dg).to(dev)
        nbt = 0
        for k, ag in enumerate(agents):
            Mk = M[k]; n_k = int(Mk.sum()); stats[f"n_{ag.p}"] = n_k; Md = Mk & DEC
            if int(hp["anchor_walker"]) and k == ANCHOR_IDX:                                    # the walker does not learn; log its true entropy
                S[f"ent_{ag.p}"], S[f"nb_{ag.p}"] = float(np.log(18)), 1
                continue
            if n_k < 64:
                continue
            coef = np.array([hp[f"w_{s}"] / scale[k][s] for s in STREAMS])
            adv_tot = sum(coef[i_] * ADV[k, s] for i_, s in enumerate(STREAMS))
            qb = 0.0 if (ag.p == ANCHOR and int(hp["qdef_mode"]) != 2) else hp["qdef_beta"]
            if int(hp["qdef_mode"]) == 2:
                pen = CDd[k] * (float(adv_tot[Mk].std()) + 1e-12)
                stats[f"cdd_{ag.p}"] = float(CDd[k][Mk].mean() / max(hp["qdef_gamma"], 1e-9)); stats[f"cdfree_{ag.p}"] = float((CDd[k][Mk] == 0).mean())
            elif int(hp["qdef_mode"]) == 1:
                pen = qavoid_penalty(VALL[:T], VALL[T], Dg, coef, ANCHOR_IDX, hp["qdef_gamma"])
            else:
                pen = qdef_penalty(VALL[:T], VALL[T], Dg, coef, k, hp["qdef_gamma"])
            coop_std = float(adv_tot[Mk].std()) + 1e-12
            adv_tot = adv_tot - qb * (pen - pen[Mk].mean())
            stats[f"qgap_{ag.p}"] = float(pen[Mk].mean()); stats[f"qratio_{ag.p}"] = float(qb * pen[Mk].std() / coop_std)
            Ms_ = Md if int(Md.sum()) >= 8 else Mk; am_, as_ = adv_tot[Ms_].mean(), adv_tot[Ms_].std() + 1e-8
            advn = torch.from_numpy(((adv_tot - am_) / as_).astype(np.float32)).to(dev)
            tgt = {}
            for s in STREAMS:
                rn = np.zeros((T, N), np.float32); rn[Mk] = popart_update(ag, s, RET[k, s][Mk], hp["popart_beta"]); tgt[s] = torch.from_numpy(rn).to(dev)
            stats[f"popsig_task_{ag.p}"] = popart_mu_sigma(ag.popart["task"])[1]; stats[f"s_task_{ag.p}"] = scale[k]["task"]; stats[f"s_ext_{ag.p}"] = scale[k]["extrinsic"]
            stats[f"s_prog_{ag.p}"] = scale[k]["progress"]; stats[f"popsig_prog_{ag.p}"] = popart_mu_sigma(ag.popart["progress"])[1]; stats[f"rp_{ag.p}"] = float(Rp[Mk].mean())
            stats[f"rl_{ag.p}"] = float(Rl[Mk].mean()); stats[f"s_lm_{ag.p}"] = scale[k]["landmark"]
            At, LPt, Mt = torch.from_numpy(A).to(dev), torch.from_numpy(LP).to(dev), torch.from_numpy(Mk.astype(np.float32)).to(dev); Mdt = torch.from_numpy(Md.astype(np.float32)).to(dev)
            L = min(int(hp["gru_seq_len"]), T); nch = T // L
            anyM = Mk[:nch * L].reshape(nch, L, N).any(1)
            pairs = np.argwhere(anyM); ppm = max(1, int(hp["minibatch_size"]) // L)
            for _ in range(int(hp["ppo_epochs"])):
                perm = np.random.permutation(len(pairs))
                for s0 in range(0, len(pairs), ppm):
                    sel = pairs[perm[s0:s0 + ppm]]
                    cs = torch.from_numpy(sel[:, 0]).long().to(dev); ns = torch.from_numpy(sel[:, 1]).long().to(dev)
                    t0 = cs * L; tidx = t0.unsqueeze(0) + torch.arange(L, device=dev).unsqueeze(1); nidx = ns.unsqueeze(0).expand(L, -1)
                    Pn = len(sel)
                    with torch.set_grad_enabled(bool(TRAIN_TRUNK)):
                        cf = net.features(Ot[tidx, nidx].reshape(L * Pn, 84, 84, 4))
                    fs = cf if not getattr(net, "use_gru", True) else gru_scan(net, cf.view(L, Pn, -1), Hs[t0, ns], Dgt[tidx, nidx])[0]
                    fs = fs.reshape(L * Pn, -1)
                    dist = torch.distributions.Categorical(logits=net.actor_logits(fs, ag.p))
                    m = Mt[tidx, nidx].reshape(-1); den = m.sum().clamp(min=1.0); md = Mdt[tidx, nidx].reshape(-1); den_d = md.sum().clamp(min=1.0)
                    lp = dist.log_prob(At[tidx, nidx].reshape(-1)); ratio = (lp - LPt[tidx, nidx].reshape(-1)).exp()
                    a_ = advn[tidx, nidx].reshape(-1)
                    pg = -((torch.min(ratio * a_, ratio.clamp(1 - hp["ppo_clip"], 1 + hp["ppo_clip"]) * a_)) * md).sum() / den_d
                    vt = net.critic_value(fs, f"task_{ag.p}", True); ve = net.critic_value(fs, f"extrinsic_{ag.p}", True)
                    vl_t = (((vt - tgt["task"][tidx, nidx].reshape(-1)) ** 2) * m).sum() / den
                    vl_e = (((ve - tgt["extrinsic"][tidx, nidx].reshape(-1)) ** 2) * m).sum() / den
                    vp = net.critic_value(fs, f"progress_{ag.p}", True)
                    vl_p = (((vp - tgt["progress"][tidx, nidx].reshape(-1)) ** 2) * m).sum() / den
                    vlm = net.critic_value(fs, f"landmark_{ag.p}", True)
                    vl_l = (((vlm - tgt["landmark"][tidx, nidx].reshape(-1)) ** 2) * m).sum() / den
                    ent = (dist.entropy() * md).sum() / den_d
                    loss = (0.0 if wu else pg - hp["entropy_coef"] * ent) + hp["value_coef"] * (vl_t + vl_e + vl_p + vl_l)
                    opt_tk.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_([q for gr in opt_tk.param_groups for q in gr["params"]], MAX_GRAD); opt_tk.step()
                    for nm, v_ in (("vl_t", vl_t), ("vl_e", vl_e), ("vl_p", vl_p), ("vl_l", vl_l), ("ent", ent)):
                        S[f"{nm}_{ag.p}"] += v_.item()
                    S[f"nb_{ag.p}"] += 1
            idx = np.flatnonzero(flat(Mk))                                        # RND predictor: masked loss on the agent's own takeover frames
            for _ in range(int(hp["ppo_epochs"])):
                np.random.shuffle(idx)
                for mb in np.array_split(idx, MINIBATCHES):
                    from_ = torch.from_numpy(fo[np.sort(mb)])
                    norm = ag.rnd._normalize_obs(ag.rnd._last_frame(from_))
                    with torch.no_grad():
                        tg = ag.rnd.target(norm)
                    per = ((ag.rnd.predictor(norm) - tg) ** 2).mean(1); keep = (torch.rand_like(per) < hp["rnd_keep"]).float()
                    rl = (keep * per).sum() / keep.sum().clamp(min=1.0)
                    ag.rnd_opt.zero_grad(); rl.backward(); torch.nn.utils.clip_grad_norm_(ag.rnd.predictor.parameters(), MAX_GRAD); ag.rnd_opt.step()
                    S[f"rnd_{ag.p}"] += rl.item(); S[f"rndn_{ag.p}"] += 1
        if TRAV_BC and EXPLORER_ARCH == "trav":
            for _ in range(EPOCHS):
                tb = trav_step(256, None)
                if tb is not None:
                    for key_, v_ in zip(TRAV_KEYS, tb):
                        if v_ is not None: S[key_] += v_; S[key_ + "_n"] += 1
        if st.get("expl_warm", 0) > 0:
            st["expl_warm"] -= 1
        st["update"] += 1; st["frames"] += T * N * 4
        recent = recent[-300:]
        nodes = [n for n in archive if n]
        maxd = max((len(n) for n in nodes), default=0)
        row = dict(update=st["update"], frames=st["frames"], fps=round(T * N * 4 / (time.time() - t_last)), nodes=len(nodes), max_depth=maxd, scenes=len(scenes.sigs),
                   route_share=round(float(MR.mean()), 3), mu_ent=round(S["mu_ent"] / max(1, nb_m), 3), mu_vl=round(S["mu_vl"] / max(1, nb_m), 4), bc=round(S["bc"] / max(1, S["bc_n"]), 3))
        if "cd_cells" in stats: row["cd_cells"] = stats["cd_cells"]
        row.update(explorer=EXPLORER_ARCH, expl_warm=st.get("expl_warm", 0), **{k_: round(S[k_] / S[k_ + "_n"], 3) for k_ in TRAV_KEYS if S[k_ + "_n"]}, back_keys=sum(1 for k_ in bdemos if k_[0][0] != "t"), back_legs=sum(len(v) for k_, v in bdemos.items() if k_[0][0] != "t"), back_new=BK["new"], trip_keys=sum(1 for k_ in bdemos if k_[0][0] == "t"), trip_legs=sum(len(v) for k_, v in bdemos.items() if k_[0][0] == "t"), trip_new=BK["tnew"]); BK["new"] = BK["tnew"] = 0
        for k, ag in enumerate(agents):
            p = ag.p; nb = max(1, S[f"nb_{p}"])
            row.update({f"n_{p}": stats.get(f"n_{p}", 0), f"ent_{p}": round(S[f"ent_{p}"] / nb, 3), f"vl_t_{p}": round(S[f"vl_t_{p}"] / nb, 4), f"vl_e_{p}": round(S[f"vl_e_{p}"] / nb, 5), f"vl_p_{p}": round(S[f"vl_p_{p}"] / nb, 4), f"vl_l_{p}": round(S[f"vl_l_{p}"] / nb, 4), f"rl_{p}": round(stats.get(f"rl_{p}", 0.0), 5), f"s_lm_{p}": round(stats.get(f"s_lm_{p}", 0.0), 4),
                        f"s_prog_{p}": round(stats.get(f"s_prog_{p}", 0.0), 3), f"popsig_prog_{p}": round(stats.get(f"popsig_prog_{p}", 0.0), 3), f"rp_{p}": round(stats.get(f"rp_{p}", 0.0), 3),
                        f"rnd_{p}": round(S[f"rnd_{p}"] / max(1, S[f"rndn_{p}"]), 5), f"qgap_{p}": round(stats.get(f"qgap_{p}", 0.0), 3), f"qratio_{p}": round(stats.get(f"qratio_{p}", 0.0), 3),
                        f"s_task_{p}": round(stats.get(f"s_task_{p}", 0.0), 3), f"s_ext_{p}": round(stats.get(f"s_ext_{p}", 0.0), 4), f"popsig_task_{p}": round(stats.get(f"popsig_task_{p}", 0.0), 3)})
            if f"cdd_{p}" in stats: row.update({f"cdd_{p}": round(stats[f"cdd_{p}"], 3), f"cdfree_{p}": round(stats[f"cdfree_{p}"], 3)})
            rk = [e for e in recent if e["owner"] == p]
            if rk:
                row.update({f"games_{p}": len(rk), f"score_{p}": round(float(np.mean([e["score"] for e in rk])), 1), f"len_{p}": round(float(np.mean([e["length"] for e in rk]))),
                            f"takeover_score_{p}": round(float(np.mean([e["takeover_score"] for e in rk if e["handoff"]] or [0.0])), 2), f"landmarks_{p}": round(float(np.mean([len(e["seq"]) for e in rk])), 2)})
        if recent:
            row.update(dec_frac=round(float(DEC[TK].mean()) if TK.any() else 1.0, 3), handoff_frac=round(float(np.mean([e["handoff"] for e in recent])), 3), mean_score=round(float(np.mean([e["score"] for e in recent])), 1), max_score=float(max(e["score"] for e in recent)),
                       mean_len=round(float(np.mean([e["length"] for e in recent]))), mean_landmarks=round(float(np.mean([len(e["seq"]) for e in recent])), 2),
                       diag_rooms="".join(str(x) + " " for x in sorted({x for e in recent for x in e["diag_rooms"]})).strip())
        if nodes:
            top = min((n for n in nodes if len(n) == maxd), key=lambda n: archive[n]["visits"])
            rep = [n for n in nodes if reproducible(n)]
            row.update(top_target=" ".join(f"{k}{v}" for k, v in top), top_arrive_rate=rate(top), reproducible_nodes=len(rep), reproducible_depth=max((len(n) for n in rep), default=0))
            hc, ph = hard_dist(nodes)
            if hc:
                row.update(hard_top=" | ".join("%s:%.2f" % (".".join(f"{k}{v}" for k, v in hc[i][-3:]), ph[i]) for i in np.argsort(-ph)[:4]))
            if rep:
                best = max(rep, key=lambda n: (len(n), rate(n))); arr = [x for x in archive[best]["hist"][-100:] if x[1] is not None]
                row.update(best_route=" ".join(f"{k}{v}" for k, v in best), best_route_rate=round(rate(best), 3))
                if arr:
                    row.update(best_route_t_median=float(np.median([x[1] for x in arr])), diag_best_route_lives_mean=round(float(np.mean([x[2] for x in arr])), 2))
        if front or hp["p_frontier"] > 0:
            fg = [e for e in recent if e.get("front") is not None]
            row.update(front_n=len(front), front_games=len(fg), front_adv=sum(1 for e in fg if len(e["seq"]) > len(e["target"]) - 1),
                       front_score=round(float(np.mean([e["score"] for e in fg])), 1) if fg else None, front_best=round(max((e["score"] for e in front.values()), default=0.0), 2), front_cand=frej[0], front_rej=frej[1], front_dup=fdup[0], front_anch=sum(1 for e in front.values() if anc_of(e) < len(e["base"])))
            tg_ = [e for e in recent if e.get("trail") is not None]
            row.update(trail_games=len(tg_), trail_hits=round(float(np.mean([e["tck"] for e in tg_])), 2) if tg_ else None, trail_end=sum(1 for e in tg_ if e.get("tck_m") and e["tck_pos"] >= e["tck_m"]))
        if hp["p_stitch"] > 0 or STC["att"]:
            sg_ = [e for e in recent if e["wing"] == "stitch"]
            row.update(stitch_pool=len(STC["pool"]), stitch_games=len(sg_), stitch_arr=sum(1 for e in sg_ if e["arrived"]), stitch_tot=sum(STC["att"].values()), stitch_tot_arr=sum(STC["arr"].values()))
        if st["update"] % 10 == 0:
            json.dump(lmroom, open(P("_lmroom.json"), "w"))
        t_last = time.time()
        log_f.write(json.dumps(row) + "\n"); log_f.flush(); ep_f.flush()
        if st["update"] % 5 == 0 or st["update"] == 1:
            log(" ".join(f"{k}={v}" for k, v in row.items()))
        if st["update"] % SAVE_EVERY == 0:
            save()
    if EVAL_GAMES:
        eps = [e for v in ev_games for e in v]; sc = np.array([e["score"] for e in eps], float)
        summ = dict(games=len(eps), per_env=ev_per, mean=float(sc.mean()) if len(sc) else None, std=float(sc.std()) if len(sc) else None,
                    median=float(np.median(sc)) if len(sc) else None, max=float(sc.max()) if len(sc) else None, min=float(sc.min()) if len(sc) else None,
                    arrived=float(np.mean([e["arrived"] for e in eps])) if eps else None, mean_landmarks=float(np.mean([len(e["seq"]) for e in eps])) if eps else None,
                    rooms=sorted({x for e in eps for x in e["diag_rooms"]}), mean_rooms=float(np.mean([len(e["diag_rooms"]) for e in eps])) if eps else None,
                    targets={" ".join(f"{k}{v}" for k, v in t_): c_ for t_, c_ in collections.Counter(tuple(map(tuple, e["target"])) for e in eps).most_common(5)},
                    eval_frames=st["frames"] - ev_frames0, train_frames=ev_frames0, train_update=st["update"] - (st["frames"] - ev_frames0) // (T * N * 4), scores=sc.tolist())
        json.dump(summ, open(P("_eval.json"), "w"), indent=1)
        pickle.dump(dict(paths=[p_ for v in ev_paths for p_ in v], scores=sc.tolist(), deaths=[e["diag_deaths"] for e in eps]), open(P("_eval_paths.pkl"), "wb"), protocol=4)
        log(f"[eval] done: {len(eps)} games, mean {summ['mean']}, median {summ['median']}, max {summ['max']}, arrived {summ['arrived']}, rooms {summ['rooms']}")
        vec.close(); return
    save()
    log(f"[stop] checkpoint saved at update {st['update']}, frames {st['frames']}; exiting cleanly.")
    vec.close()


if __name__ == "__main__":
    main()
