"""GWRW v2: automatic frontier, online archive, route controller + explorer. Starts from nothing (no saved episodes).

Everything the algorithm uses comes from the screen and the game reward:
  * landmarks: the n-th positive game reward of a game, and the first entry into each scene (pixel scene-change detector
    with a codebook that grows online; see landmarks.py for the thresholds);
  * archive: every distinct ordered list of landmarks some game has produced is a node; its depth is the list's length;
  * target of a game (chosen at the game's start, automatically): usually the deepest node (ties: the least visited),
    sometimes a node near that depth, sometimes none (explore from the start);
  * route controller mu: follows the target's landmarks in order; reward +1 per landmark, -STEP_COST per decision; it also
    imitates the fastest self-generated instances of each leg (kept per node, replaced when a faster one appears);
  * explorer: takes over at arrival (or when the route attempt fails: wrong landmark or LEG_TIMEOUT decisions without
    progress) and plays on to game over with PPO on game reward + RND novelty. Anything new it reaches becomes a node,
    hence a future target. Games end only at game over or the step cap (4000 decisions per scene seen).
Emulator memory is read only into diag_* log fields (rooms, lives) and never reaches a policy, a reward, or the archive.

  RUN_NAME=gwrw_v2 DEVICE=cuda:0 python3 -u gwrw/train_gwrw.py        (one Ctrl-C = finish the update, save, exit)
"""
import collections, json, os, pickle, signal, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from landmarks import CUT_FRAC, MATCH_FRAC, PIX, TOP, sig_dist, signature

OUT = os.path.join(HERE, "out")
RUN = os.environ.get("RUN_NAME", "gwrw_v2")
DEVICE = os.environ.get("DEVICE", "cuda:0")
E = lambda k, d: type(d)(os.environ.get(k, d))
N_PROC, ENVS_PER_PROC, T = E("N_PROC", 12), E("ENVS_PER_PROC", 4), E("ROLLOUT", 128)
LR, LAM, CLIP, EPOCHS, MINIBATCHES, MAX_GRAD = 1e-4, 0.95, 0.1, 4, 4, 1.0
G_MU, ENT_MU, STEP_COST, BC_W = 0.999, 0.01, 0.0005, E("BC_W", 0.5)
G_EXT, G_INT, ENT_EX, W_EXT, W_INT, RND_KEEP = 0.999, 0.99, 0.001, 2.0, 1.0, 0.25
STEP_CAP_PER_SCENE, LEG_TIMEOUT = 4000, E("LEG_TIMEOUT", 1500)
P_NONE, P_CONSOLIDATE = 0.2, 0.4                   # the rest: branch from a reproducible frontier
REPRO_RATE, MIN_ATTEMPTS = 0.5, 20                 # a node is reproducible when its recent arrival rate is at least this
DEMOS_PER_NODE, LEG_KEEP = 8, 1000
MAX_FRAMES = E("MAX_FRAMES", 300_000_000)
SAVE_EVERY = 50
DEBUG_FAKE_R = E("DEBUG_FAKE_R", 0.0)
WARMUP_STEPS = E("WARMUP_STEPS", 200)


def lm_id(lm):                                     # landmark -> embedding index: rewards 1..15, scenes 16..63
    return min(lm[1], 15) if lm[0] == "r" else 16 + min(lm[1], 47)


def worker(conn, seeds):
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C reaches the whole process group; only the trainer handles it
    from train_route import make_env
    from env.ram_features import read_progress     # diagnostics only
    envs = [make_env(s) for s in seeds]
    steps = [0] * len(envs)
    conn.send([e.reset()[0] for e in envs])
    try:
        while True:
            msg = conn.recv()
            if msg is None:
                break
            out = []
            for j, (e, (a, cap)) in enumerate(zip(envs, msg)):
                obs, r, term, trunc, _ = e.step(int(a)); steps[j] += 1
                p = read_progress(e.unwrapped.ale)
                done = term or trunc or steps[j] > cap
                if done:
                    obs, _ = e.reset(); steps[j] = 0
                out.append((obs, float(r), done, int(p.lives), int(p.room)))
            conn.send(out)
    except (EOFError, KeyboardInterrupt):
        pass


class Vec:
    def __init__(self, seed0):
        ctx = mp.get_context("spawn")
        self.conns = []
        for p in range(N_PROC):
            a, b = ctx.Pipe()
            ctx.Process(target=worker, args=(b, [seed0 + p * 100 + i for i in range(ENVS_PER_PROC)]), daemon=True).start()
            self.conns.append(a)
        self.obs = np.stack([o for c in self.conns for o in c.recv()])
        self.n = len(self.obs)

    def step(self, actions, caps):
        for i, c in enumerate(self.conns):
            s = slice(i * ENVS_PER_PROC, (i + 1) * ENVS_PER_PROC)
            c.send(list(zip(actions[s], caps[s])))
        res = [x for c in self.conns for x in c.recv()]
        self.obs = np.stack([x[0] for x in res])
        return (np.array([x[1] for x in res], np.float32), np.array([x[2] for x in res]), [x[3] for x in res], [x[4] for x in res])

    def close(self):
        for c in self.conns:
            try: c.send(None)
            except Exception: pass


class Scenes:
    """Online pixel scene tracker for N games. A background that matches no stored signature becomes a new scene once
    the following frame confirms it (guards against a blended transition frame)."""
    def __init__(self, n, sigs=None):
        self.sigs = [] if sigs is None else list(sigs)
        self.prev = np.zeros((n, 84, 84), np.uint8); self.scene = np.zeros(n, int); self.pending = [None] * n

    def _match(self, sig):
        if self.sigs:
            d = [sig_dist(sig, s) for s in self.sigs]
            j = int(np.argmin(d))
            if d[j] < MATCH_FRAC:
                return j
        return -1

    def reset(self, i, frame):
        s = self._match(signature(frame))
        if s < 0:
            self.sigs.append(signature(frame)); s = len(self.sigs) - 1
        self.prev[i], self.scene[i], self.pending[i] = frame, s, None

    def step(self, frames, skip):
        """Returns {env: scene id} for envs whose scene changed at this step. skip: envs that were just reset."""
        cut = (np.abs(self.prev[:, TOP:].astype(np.int16) - frames[:, TOP:].astype(np.int16)) > PIX).mean(axis=(1, 2)) > CUT_FRAC
        out = {}
        for i in range(len(frames)):
            if skip[i]:
                continue
            if cut[i]:
                sig = signature(frames[i]); s = self._match(sig)
                self.pending[i] = sig if s < 0 else None
                if s >= 0 and s != self.scene[i]:
                    self.scene[i] = s; out[i] = s
            elif self.pending[i] is not None:
                sig = signature(frames[i])
                if sig_dist(sig, self.pending[i]) < MATCH_FRAC:
                    s = self._match(sig)
                    if s < 0:
                        self.sigs.append(sig); s = len(self.sigs) - 1
                    if s != self.scene[i]:
                        self.scene[i] = s; out[i] = s
                self.pending[i] = None
        self.prev[:] = frames
        return out


def build_nets():
    import torch, torch.nn as nn

    def conv():
        return nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())

    def init(m):
        for x in m.modules():
            if isinstance(x, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(x.weight, 2 ** 0.5); nn.init.zeros_(x.bias)

    class Mu(nn.Module):                           # route controller: frames + which landmark is next + how many are done
        def __init__(self):
            super().__init__()
            self.conv, self.e_lm, self.e_k = conv(), nn.Embedding(64, 64), nn.Embedding(32, 32)
            self.fc = nn.Sequential(nn.Linear(3136 + 96, 512), nn.ReLU()); self.pi, self.v = nn.Linear(512, 18), nn.Linear(512, 1)
            init(self); nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.orthogonal_(self.v.weight, 1.0)

        def forward(self, obs, lm, k):
            h = self.fc(torch.cat([self.conv(obs.permute(0, 3, 1, 2).float() / 255.0), self.e_lm(lm), self.e_k(k.clamp(max=31))], 1))
            return self.pi(h), self.v(h).squeeze(-1)

    class Ex(nn.Module):                           # explorer: frames only; game-reward and novelty value heads
        def __init__(self):
            super().__init__()
            self.conv = conv(); self.fc = nn.Sequential(nn.Linear(3136, 512), nn.ReLU())
            self.pi, self.ve, self.vi = nn.Linear(512, 18), nn.Linear(512, 1), nn.Linear(512, 1)
            init(self); nn.init.orthogonal_(self.pi.weight, 0.01)

        def forward(self, obs):
            h = self.fc(self.conv(obs.permute(0, 3, 1, 2).float() / 255.0))
            return self.pi(h), self.ve(h).squeeze(-1), self.vi(h).squeeze(-1)
    return Mu(), Ex()


def gae(R, V, done, gamma):
    adv, last = np.zeros_like(R), 0.0
    for t in reversed(range(len(R))):
        nd = 1.0 - done[t]
        delta = R[t] + gamma * V[t + 1] * nd - V[t]
        last = delta + gamma * LAM * nd * last; adv[t] = last
    return adv


def main():
    import torch, torch.nn.functional as F
    from algorithms.rnd import RNDModel
    torch.set_num_threads(2)
    dev = torch.device(DEVICE)
    P = lambda s: os.path.join(OUT, f"{RUN}{s}")
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    mu, ex = build_nets(); mu, ex = mu.to(dev), ex.to(dev)
    rnd = RNDModel(device=DEVICE, predictor_arch="paper")
    opt_mu = torch.optim.Adam(mu.parameters(), lr=LR, eps=1e-5)
    opt_ex = torch.optim.Adam(list(ex.parameters()) + list(rnd.predictor.parameters()), lr=LR, eps=1e-5)
    st = dict(update=0, frames=0, sigs=[], archive={}, int_var=1.0, int_count=1e-4, int_mean=0.0)
    demos = {}                                     # node -> list of (duration, frames (n+3, 84, 84), actions (n,))
    if os.path.exists(P(".pt")):
        ck = torch.load(P(".pt"), map_location=dev, weights_only=False)
        mu.load_state_dict(ck["mu"]); ex.load_state_dict(ck["ex"]); rnd.load_state_dict(ck["rnd"]); opt_mu.load_state_dict(ck["opt_mu"]); opt_ex.load_state_dict(ck["opt_ex"])
        rnd.obs_rms.mean, rnd.obs_rms.var, rnd.obs_rms.count = ck["obs_rms"]; st = ck["st"]
        if os.path.exists(P("_demos.pkl")):
            demos = pickle.load(open(P("_demos.pkl"), "rb"))
        log(f"resumed at update {st['update']}, frames {st['frames']}, nodes {len(st['archive'])}, scenes {len(st['sigs'])}, demo nodes {len(demos)}")
    archive = st["archive"]                        # node (tuple of landmarks) -> dict(visits, hist)
    vec = Vec(1000 + st["update"]); N = vec.n
    rng = np.random.default_rng(st["update"])
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

    def pick_target():
        """No target (explore from the start) with probability P_NONE. Otherwise either consolidate: the earliest prefix of
        the deepest node that is not yet reproducible (forward curriculum), or branch: a reproducible node among the deepest."""
        nodes = [n for n in archive if n and n in demos]
        u = rng.random()
        if not nodes or u < P_NONE:
            return ()
        maxd = max(len(n) for n in nodes)
        deepest = min((n for n in nodes if len(n) == maxd), key=lambda n: archive[n]["visits"])
        weak = next((deepest[:k] for k in range(1, maxd + 1) if deepest[:k] in demos and not reproducible(deepest[:k])), None)
        rep = [n for n in nodes if reproducible(n)]
        if weak is not None and (u < P_NONE + P_CONSOLIDATE or not rep):
            return weak
        if not rep:
            return ()
        md = max(len(n) for n in rep)
        near = [n for n in rep if len(n) >= md - 1]
        return near[rng.integers(len(near))]

    # per-game state
    g = [None] * N
    def new_game(i):
        scenes.reset(i, vec.obs[i][..., -1])
        tgt = pick_target()
        g[i] = dict(target=tgt, k=0, phase=0 if tgt else 1, t=0, seq=[], seen={int(scenes.scene[i])}, nrew=0, leg0=0, frames=[], acts=[], score=0.0,
                    t_arrive=(0 if not tgt else None), diag_lives_arrive=None, diag_rooms=set(), fail=None)
    if st["update"] == 0:                          # observation statistics for the novelty model, from random play
        for _ in range(WARMUP_STEPS):
            vec.step(rng.integers(0, 18, N), [10 ** 9] * N)
            rnd.update_obs_stats(torch.from_numpy(vec.obs))
    for i in range(N):
        new_game(i)
    rff = np.zeros(N)                              # discounted novelty return filter (for reward normalisation)

    def add_demo(i, n):
        gi = g[i]
        a0 = max(gi["leg0"], gi["t"] - LEG_KEEP + 1)
        fr = gi["frames"][max(0, a0 - 3):gi["t"] + 1]
        fr = [fr[0]] * (3 - (a0 - max(0, a0 - 3))) + fr
        d = demos.setdefault(n, [])
        dur = gi["t"] + 1 - gi["leg0"]
        if len(d) < DEMOS_PER_NODE or dur < d[-1][0]:
            d.append((dur, np.stack(fr), np.array(gi["acts"][a0:gi["t"] + 1], np.int64)))
            d.sort(key=lambda x: x[0]); del d[DEMOS_PER_NODE:]

    def bc_batch(n):
        active = set()
        for gi in g:
            for k in range(1, len(gi["target"]) + 1):
                if gi["target"][:k] in demos: active.add(gi["target"][:k])
        if not active:
            return None
        active = list(active); ob, lm, kk, ac = [], [], [], []
        for _ in range(n):
            nd_ = active[rng.integers(len(active))]; dur, fr, a = demos[nd_][rng.integers(len(demos[nd_]))]
            j = rng.integers(len(a))
            ob.append(np.moveaxis(fr[j:j + 4], 0, -1)); lm.append(lm_id(nd_[-1])); kk.append(len(nd_) - 1); ac.append(a[j])
        return np.stack(ob), np.array(lm), np.array(kk), np.array(ac)

    stop = {"now": False}
    def on_int(*_):
        if not stop["now"]:
            print("[stop] signal received -- finishing this update, then saving and exiting.", flush=True)
        stop["now"] = True
    signal.signal(signal.SIGINT, on_int)

    def save():
        st["sigs"] = scenes.sigs
        torch.save(dict(mu=mu.state_dict(), ex=ex.state_dict(), rnd=rnd.state_dict(), opt_mu=opt_mu.state_dict(), opt_ex=opt_ex.state_dict(),
                        obs_rms=(rnd.obs_rms.mean, rnd.obs_rms.var, rnd.obs_rms.count), st=st), P(".pt.tmp"))
        os.replace(P(".pt.tmp"), P(".pt"))
        pickle.dump(demos, open(P("_demos.pkl.tmp"), "wb"), protocol=4); os.replace(P("_demos.pkl.tmp"), P("_demos.pkl"))

    recent, t_last = [], time.time()
    while st["frames"] < MAX_FRAMES and not stop["now"]:
        O = np.zeros((T + 1, N, 84, 84, 4), np.uint8); A = np.zeros((T, N), np.int64); LP = np.zeros((T, N), np.float32)
        LMI = np.zeros((T, N), np.int64); KK = np.zeros((T, N), np.int64)
        MR = np.zeros((T, N), bool); Rm = np.zeros((T, N), np.float32); Dm = np.ones((T, N), np.float32); Vm = np.zeros((T + 1, N), np.float32)
        Re = np.zeros((T, N), np.float32); Dg = np.zeros((T, N), np.float32); Ve = np.zeros((T + 1, N), np.float32); Vi = np.zeros((T + 1, N), np.float32)
        for t in range(T):
            O[t] = vec.obs
            ph = np.array([gi["phase"] for gi in g]); MR[t] = ph == 0
            LMI[t] = [lm_id(gi["target"][gi["k"]]) if gi["phase"] == 0 else 0 for gi in g]; KK[t] = [gi["k"] for gi in g]
            ot = torch.from_numpy(O[t]).to(dev)
            with torch.no_grad():
                lg_m, v_m = mu(ot, torch.from_numpy(LMI[t]).to(dev), torch.from_numpy(KK[t]).to(dev))
                lg_e, v_e, v_i = ex(ot)
                dm, de = torch.distributions.Categorical(logits=lg_m), torch.distributions.Categorical(logits=lg_e)
                am, ae = dm.sample(), de.sample()
            mr = torch.from_numpy(MR[t]).to(dev)
            a = torch.where(mr, am, ae)
            A[t] = a.cpu().numpy(); LP[t] = torch.where(mr, dm.log_prob(am), de.log_prob(ae)).cpu().numpy()
            Vm[t], Ve[t], Vi[t] = v_m.cpu().numpy(), v_e.cpu().numpy(), v_i.cpu().numpy()
            for i, gi in enumerate(g):
                gi["frames"].append(O[t, i, :, :, -1].copy()); gi["acts"].append(int(A[t, i]))
            caps = [STEP_CAP_PER_SCENE * len(gi["seen"]) for gi in g]
            r, done, lives, rooms = vec.step(A[t], caps)
            if DEBUG_FAKE_R:                                # smoke test only: pretend rewards happen, to exercise the route code
                r = np.where(rng.random(N) < DEBUG_FAKE_R, 1.0, r).astype(np.float32)
            new_scene = scenes.step(vec.obs[..., -1], done)
            for i, gi in enumerate(g):
                gi["score"] += r[i]; gi["diag_rooms"].add(rooms[i] if not done[i] else -1)
                lms = []
                if r[i] > 0:
                    gi["nrew"] += 1; lms.append(("r", gi["nrew"]))
                if i in new_scene and new_scene[i] not in gi["seen"]:
                    gi["seen"].add(new_scene[i]); lms.append(("s", int(new_scene[i])))
                route_end = False
                if MR[t, i]:
                    Rm[t, i] = -STEP_COST
                for lm in lms:
                    gi["seq"].append(lm); n_ = tuple(gi["seq"])
                    if n_ not in archive:
                        ev_f.write(json.dumps(dict(update=st["update"], frames=st["frames"], node=n_, t=gi["t"], by="route" if gi["phase"] == 0 else "explorer", diag_room=rooms[i])) + "\n"); ev_f.flush()
                        log(f"[new] depth {len(n_)} node {n_} at decision {gi['t']} ({'route' if gi['phase'] == 0 else 'explorer'})")
                    node(n_)["visits"] += 1
                    add_demo(i, n_); gi["leg0"] = gi["t"] + 1
                    if gi["phase"] == 0:
                        if lm == gi["target"][gi["k"]]:
                            gi["k"] += 1; Rm[t, i] += 1.0
                            if gi["k"] == len(gi["target"]):
                                gi["phase"], gi["t_arrive"], gi["diag_lives_arrive"], route_end = 1, gi["t"] + 1, lives[i], True
                        else:
                            gi["phase"], gi["fail"], route_end = 1, "other landmark", True
                if gi["phase"] == 0 and gi["t"] + 1 - gi["leg0"] > LEG_TIMEOUT:
                    gi["phase"], gi["fail"], route_end = 1, "timeout", True
                Re[t, i] = np.sign(r[i]); Dg[t, i] = float(done[i])
                Dm[t, i] = 1.0 if (route_end or done[i] or not MR[t, i]) else 0.0
                gi["t"] += 1
                if done[i]:
                    for j in range(1, len(gi["target"]) + 1):           # every prefix the game had the chance to try
                        if gi["k"] >= j - 1:
                            h = node(gi["target"][:j])["hist"]; full = j == len(gi["target"]) and gi["t_arrive"] is not None
                            h.append((gi["k"] >= j, gi["t_arrive"] if full else None, gi["diag_lives_arrive"] if full else None)); del h[:-200]
                    rec = dict(update=st["update"], target=gi["target"], arrived=gi["t_arrive"] is not None, t_arrive=gi["t_arrive"], fail=gi["fail"], seq=gi["seq"], length=gi["t"],
                               score=float(gi["score"]), diag_lives_arrive=gi["diag_lives_arrive"], diag_rooms=sorted(x for x in gi["diag_rooms"] if x >= 0))
                    recent.append(rec); ep_f.write(json.dumps(rec) + "\n")
                    new_game(i)
        O[T] = vec.obs
        ME = ~MR
        # novelty reward on the frame each explorer step led to; none on a game's last step (the next frame is a new game)
        rnd.update_obs_stats(torch.from_numpy(O[1:][ME][::4]) if ME.any() else torch.from_numpy(O[T]))
        Ri = np.zeros((T, N), np.float32)
        if ME.any():
            nxt = O[1:][ME]
            raw = np.concatenate([rnd.intrinsic_reward(torch.from_numpy(nxt[j:j + 2048])).cpu().numpy() for j in range(0, len(nxt), 2048)])
            Ri[ME] = raw; Ri *= (1.0 - Dg)
            vals = []
            for t in range(T):
                rff = rff * G_INT + Ri[t]; vals.append(rff[ME[t]])
            vals = np.concatenate(vals)
            if st["update"] < 5:                            # the first updates' errors are huge; do not let them set the scale
                st["int_count"], st["int_mean"], st["int_var"] = 1e-4, 0.0, 1.0
            if len(vals) > 1:
                c, m, v = st["int_count"], st["int_mean"], st["int_var"]
                bm, bv, bc_ = vals.mean(), vals.var(), len(vals); tot = c + bc_; d = bm - m
                st["int_mean"], st["int_var"], st["int_count"] = m + d * bc_ / tot, (v * c + bv * bc_ + d * d * c * bc_ / tot) / tot, tot
            Ri /= (st["int_var"] ** 0.5 + 1e-8)
        ph_end = np.array([gi["phase"] for gi in g])
        with torch.no_grad():
            oT = torch.from_numpy(O[T]).to(dev)
            lmT = torch.from_numpy(np.array([lm_id(gi["target"][gi["k"]]) if gi["phase"] == 0 else 0 for gi in g])).to(dev)
            Vm[T] = np.where(ph_end == 0, mu(oT, lmT, torch.from_numpy(np.array([gi["k"] for gi in g])).to(dev))[1].cpu().numpy(), 0.0)
            _, ve_T, vi_T = ex(oT); Ve[T], Vi[T] = np.where(ph_end == 1, ve_T.cpu().numpy(), 0.0), np.where(ph_end == 1, vi_T.cpu().numpy(), 0.0)
        De = np.maximum(Dg, MR.astype(np.float32))          # the explorer's stretch of a game ends at game over; route steps carry nothing
        adv_m = gae(Rm, Vm, Dm, G_MU); ret_m = adv_m + Vm[:T]
        adv_e, adv_i = gae(Re * ME, Ve, De, G_EXT), gae(Ri, Vi, De, G_INT)
        ret_e, ret_i = adv_e + Ve[:T], adv_i + Vi[:T]
        fo = O[:T].reshape(-1, 84, 84, 4); flat = lambda x: x.reshape(-1)
        S = collections.defaultdict(float); nb_m = nb_e = 0
        idx_m, idx_e = np.flatnonzero(flat(MR)), np.flatnonzero(flat(ME))
        tt = lambda x, mb: torch.from_numpy(flat(x)[mb]).to(dev)
        for _ in range(EPOCHS):
            if len(idx_m) >= 64:
                rng.shuffle(idx_m)
                for mb in np.array_split(idx_m, MINIBATCHES):
                    mb = np.sort(mb)
                    lg, v = mu(torch.from_numpy(fo[mb]).to(dev), tt(LMI, mb), tt(KK, mb))
                    dist = torch.distributions.Categorical(logits=lg); lp = dist.log_prob(tt(A, mb))
                    a_ = tt(adv_m, mb); a_ = (a_ - a_.mean()) / (a_.std() + 1e-8); ratio = (lp - tt(LP, mb)).exp()
                    pg = -torch.min(ratio * a_, ratio.clamp(1 - CLIP, 1 + CLIP) * a_).mean()
                    vl = F.mse_loss(v, tt(ret_m, mb)); ent = dist.entropy().mean()
                    loss = pg + 0.5 * vl - ENT_MU * ent
                    opt_mu.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mu.parameters(), MAX_GRAD); opt_mu.step()
                    S["mu_vl"] += vl.item(); S["mu_ent"] += ent.item(); nb_m += 1
            b = bc_batch(256)                               # imitation of the fastest self-generated legs
            if b is not None:
                lg, _ = mu(torch.from_numpy(b[0]).to(dev), torch.from_numpy(b[1]).to(dev), torch.from_numpy(b[2]).to(dev))
                bl = F.cross_entropy(lg, torch.from_numpy(b[3]).to(dev))
                opt_mu.zero_grad(); (BC_W * bl).backward(); torch.nn.utils.clip_grad_norm_(mu.parameters(), MAX_GRAD); opt_mu.step()
                S["bc"] += bl.item(); S["bc_n"] += 1
            if len(idx_e) >= 64:
                rng.shuffle(idx_e)
                for mb in np.array_split(idx_e, MINIBATCHES):
                    mb = np.sort(mb)
                    lg, ve, vi = ex(torch.from_numpy(fo[mb]).to(dev))
                    dist = torch.distributions.Categorical(logits=lg); lp = dist.log_prob(tt(A, mb))
                    a_ = W_EXT * tt(adv_e, mb) + W_INT * tt(adv_i, mb); a_ = (a_ - a_.mean()) / (a_.std() + 1e-8); ratio = (lp - tt(LP, mb)).exp()
                    pg = -torch.min(ratio * a_, ratio.clamp(1 - CLIP, 1 + CLIP) * a_).mean()
                    vl = F.mse_loss(ve, tt(ret_e, mb)) + F.mse_loss(vi, tt(ret_i, mb)); ent = dist.entropy().mean()
                    nxt = O[1:].reshape(-1, 84, 84, 4)[mb]
                    norm = rnd._normalize_obs(rnd._last_frame(torch.from_numpy(nxt)))
                    with torch.no_grad():
                        tgt = rnd.target(norm)
                    per = ((rnd.predictor(norm) - tgt) ** 2).mean(1); keep = (torch.rand_like(per) < RND_KEEP).float()
                    rl = (keep * per).sum() / keep.sum().clamp(min=1.0)
                    loss = pg + 0.5 * vl - ENT_EX * ent + rl
                    opt_ex.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(ex.parameters()) + list(rnd.predictor.parameters()), MAX_GRAD); opt_ex.step()
                    S["ex_ent"] += ent.item(); S["rnd_loss"] += rl.item(); nb_e += 1
        st["update"] += 1; st["frames"] += T * N * 4
        recent = recent[-300:]
        nodes = [n for n in archive if n]
        maxd = max((len(n) for n in nodes), default=0)
        row = dict(update=st["update"], frames=st["frames"], fps=round(T * N * 4 / (time.time() - t_last)), nodes=len(nodes), max_depth=maxd, scenes=len(scenes.sigs),
                   route_share=round(float(MR.mean()), 3), mu_ent=round(S["mu_ent"] / max(1, nb_m), 3), mu_vl=round(S["mu_vl"] / max(1, nb_m), 4), bc=round(S["bc"] / max(1, S["bc_n"]), 3),
                   ex_ent=round(S["ex_ent"] / max(1, nb_e), 3), rnd_loss=round(S["rnd_loss"] / max(1, nb_e), 5), novelty_mean=round(float(Ri[ME].mean()) if ME.any() else 0.0, 4))
        if recent:
            row.update(mean_score=round(float(np.mean([e["score"] for e in recent])), 1), max_score=float(max(e["score"] for e in recent)), mean_len=round(float(np.mean([e["length"] for e in recent]))),
                       mean_landmarks=round(float(np.mean([len(e["seq"]) for e in recent])), 2), diag_rooms="".join(str(x) + " " for x in sorted({x for e in recent for x in e["diag_rooms"]})).strip())
        if nodes:                                           # the frontier, and the deepest node that is reached reliably
            top = min((n for n in nodes if len(n) == maxd), key=lambda n: archive[n]["visits"])
            rep = [n for n in nodes if reproducible(n)]
            row.update(top_target=" ".join(f"{k}{v}" for k, v in top), top_arrive_rate=rate(top), reproducible_nodes=len(rep), reproducible_depth=max((len(n) for n in rep), default=0))
            if rep:
                best = max(rep, key=lambda n: (len(n), rate(n))); arr = [x for x in archive[best]["hist"][-100:] if x[1] is not None]
                row.update(best_route=" ".join(f"{k}{v}" for k, v in best), best_route_rate=round(rate(best), 3))
                if arr:
                    row.update(best_route_t_median=float(np.median([x[1] for x in arr])), diag_best_route_lives_mean=round(float(np.mean([x[2] for x in arr])), 2))
        t_last = time.time()
        log_f.write(json.dumps(row) + "\n"); log_f.flush(); ep_f.flush()
        if st["update"] % 5 == 0 or st["update"] == 1:
            log(" ".join(f"{k}={v}" for k, v in row.items()))
        if st["update"] % SAVE_EVERY == 0:
            save()
    save()
    log(f"[stop] checkpoint saved at update {st['update']}, frames {st['frames']}; exiting cleanly.")
    vec.close()


if __name__ == "__main__":
    main()
