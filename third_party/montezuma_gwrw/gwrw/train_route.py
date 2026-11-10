"""Stage 1 of tex/gwrw_plan.tex: a route controller that goes from the ordinary game start to the frontier scene.

Separate network, own optimiser; shares nothing with the run14 scripts. Trained on pixels and game reward only:
  - landmarks (n-th game reward, scene entries) come from gwrw/landmarks.py with a frozen pixel codebook;
  - reward = +1 for each route landmark reached in order, minus STEP_COST per decision; no novelty reward;
  - the controller never reads lives, room ids or positions. Those are read from emulator memory ONLY into the per-episode
    diagnostic record (keys starting with diag_) and never reach the policy, the reward or the episode cap.
After arrival the rest of the game is played with uniformly random actions (the slot the Stage-2 explorers will take); those
steps are not trained on but are counted in the frame total. Games end only at game over or at the step cap
(STEP_CAP_PER_SCENE decisions per scene seen, scenes counted by the pixel tracker).

  train:  RUN_NAME=gwrw_route_v1 DEVICE=cuda:0 python3 -u gwrw/train_route.py
  check:  RUN_NAME=gwrw_route_v1 DEVICE=cuda:0 python3 -u gwrw/train_route.py --eval 200
One Ctrl-C = finish the update, save, exit.
"""
import json, os, signal, sys, time
import multiprocessing as mp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "out")
RUN = os.environ.get("RUN_NAME", "gwrw_route_v1")
DEVICE = os.environ.get("DEVICE", "cuda:0")
E = lambda k, d: type(d)(os.environ.get(k, d))
N_PROC, ENVS_PER_PROC, T = E("N_PROC", 12), E("ENVS_PER_PROC", 4), E("ROLLOUT", 128)
LR, GAMMA, LAM, CLIP, ENT, VF = E("LR", 1e-4), 0.999, 0.95, 0.1, E("ENT", 0.01), 0.5
EPOCHS, MINIBATCHES, MAX_GRAD = 4, 4, 0.5
STEP_COST = E("STEP_COST", 0.0005)
STEP_CAP_PER_SCENE = 4000
BC_EPOCHS, BC_LR, BC_AUX0, BC_AUX_UPDATES = E("BC_EPOCHS", 6), 2.5e-4, 1.0, E("BC_AUX_UPDATES", 300)
MAX_FRAMES = E("MAX_FRAMES", 100_000_000)
SAVE_EVERY = 25


def make_env(seed):
    import ale_py, gymnasium as gym
    from env.wrappers import FrameStack, MaxAndSkipEnv, NoopResetEnv, WarpFrame
    if os.path.isdir(os.environ.get("ALE_ROMS_DIR", "")):
        os.environ.setdefault("ALE_ROMS_PATH", os.environ["ALE_ROMS_DIR"])    # only needed for a ROM directory outside ale-py; ale-py 0.12 bundles Montezuma
    gym.register_envs(ale_py)
    env = gym.make("ALE/MontezumaRevenge-v5", frameskip=1)      # same settings as the saved routes (sticky 0.25, noop starts)
    env = FrameStack(WarpFrame(MaxAndSkipEnv(NoopResetEnv(env, noop_max=30), skip=4)), k=4)
    env.reset(seed=seed)
    return env


class RouteEnv:
    """One game with the pixel landmark tracker and the route bookkeeping."""
    def __init__(self, seed, route, codebook_path):
        from landmarks import Codebook, SceneTracker
        from env.ram_features import read_progress          # diagnostics only
        self._diag = read_progress
        self.env = make_env(seed)
        self.route = [tuple(x) for x in route]
        self.tr = SceneTracker(Codebook.load(codebook_path, frozen=True))
        self.obs = self._new_game()

    def _new_game(self):
        obs, _ = self.env.reset()
        self.tr.reset(obs[..., -1])
        self.k, self.phase, self.steps, self.score = 0, 0, 0, 0.0
        self.t_arrive, self.diag_lives_arrive, self.t_landmarks = None, None, []
        self.diag_rooms = set()
        return obs

    def step(self, action):
        obs, r, term, trunc, _ = self.env.step(int(action))
        self.steps += 1; self.score += r
        rr, route_end = (-STEP_COST if self.phase == 0 else 0.0), False
        for ev in self.tr.step(obs[..., -1], r):
            if self.phase == 0 and ev == self.route[self.k]:
                self.k += 1; rr += 1.0; self.t_landmarks.append(self.steps)
                if self.k == len(self.route):
                    self.phase, route_end, self.t_arrive = 1, True, self.steps
                    self.diag_lives_arrive = int(self._diag(self.env.unwrapped.ale).lives)
        if self.steps % 16 == 0:
            self.diag_rooms.add(int(self._diag(self.env.unwrapped.ale).room))
        if self.steps > STEP_CAP_PER_SCENE * max(1, len(self.tr.scenes_seen)):
            trunc = True
        done, info = term or trunc, None
        if done:
            if self.phase == 0:
                route_end = True
            info = dict(arrived=self.t_arrive is not None, t_arrive=self.t_arrive, k=self.k, length=self.steps, score=self.score,
                        t_landmarks=self.t_landmarks, scenes=len(self.tr.scenes_seen), capped=bool(trunc and not term),
                        diag_lives_arrive=self.diag_lives_arrive, diag_rooms=sorted(self.diag_rooms))
            obs = self._new_game()
        self.obs = obs
        return obs, rr, route_end, done, self.k, self.phase, info


def worker(conn, seeds, route, cb):
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl-C reaches the whole process group; only the trainer handles it
    envs = [RouteEnv(s, route, cb) for s in seeds]
    conn.send(([e.obs for e in envs], [e.k for e in envs], [e.phase for e in envs]))
    try:
        while True:
            acts = conn.recv()
            if acts is None:
                break
            conn.send([e.step(a) for e, a in zip(envs, acts)])
    except (EOFError, KeyboardInterrupt):
        pass


class Vec:
    def __init__(self, route, cb, seed0):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for p in range(N_PROC):
            a, b = ctx.Pipe()
            pr = ctx.Process(target=worker, args=(b, [seed0 + p * 100 + i for i in range(ENVS_PER_PROC)], route, cb), daemon=True)
            pr.start(); self.conns.append(a); self.procs.append(pr)
        first = [c.recv() for c in self.conns]
        self.obs = np.stack([o for f in first for o in f[0]])
        self.k = np.array([k for f in first for k in f[1]]); self.phase = np.array([p for f in first for p in f[2]])
        self.n = len(self.obs)

    def step(self, actions):
        for i, c in enumerate(self.conns):
            c.send(actions[i * ENVS_PER_PROC:(i + 1) * ENVS_PER_PROC])
        res = [x for c in self.conns for x in c.recv()]
        self.obs = np.stack([x[0] for x in res]); self.k = np.array([x[4] for x in res]); self.phase = np.array([x[5] for x in res])
        return (np.array([x[1] for x in res], np.float32), np.array([x[2] for x in res]), [x[6] for x in res])

    def close(self):
        for c in self.conns:
            try: c.send(None)
            except Exception: pass


def build_model(n_landmarks):
    import torch, torch.nn as nn

    class RouteNet(nn.Module):
        """Feed-forward on the 4-frame stack, conditioned on which landmark is next (the route position is the memory)."""
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(), nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
            self.emb = nn.Embedding(n_landmarks + 1, 64)
            self.fc = nn.Sequential(nn.Linear(3136 + 64, 512), nn.ReLU())
            self.pi, self.v = nn.Linear(512, 18), nn.Linear(512, 1)
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, 2 ** 0.5); nn.init.zeros_(m.bias)
            nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.orthogonal_(self.v.weight, 1.0)

        def forward(self, obs_u8, k):
            x = self.conv(obs_u8.permute(0, 3, 1, 2).float() / 255.0)
            h = self.fc(torch.cat([x, self.emb(k)], 1))
            return self.pi(h), self.v(h).squeeze(-1)
    return RouteNet()


def bc_pretrain(net, opt_lr, dev, log):
    import torch, torch.nn.functional as F
    d = np.load(os.path.join(OUT, "bc_data.npz"))
    obs, act, k = d["obs"], d["act"], d["k"]
    rng = np.random.default_rng(0); perm = rng.permutation(len(obs)); n_val = len(obs) // 20
    val, tr = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=opt_lr)
    for ep in range(BC_EPOCHS):
        rng.shuffle(tr); tot = 0.0
        for i in range(0, len(tr) - 255, 256):
            b = np.sort(tr[i:i + 256])
            logits, _ = net(torch.from_numpy(obs[b]).to(dev), torch.from_numpy(k[b]).to(dev))
            loss = F.cross_entropy(logits, torch.from_numpy(act[b]).to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        with torch.no_grad():
            acc = np.mean([(net(torch.from_numpy(obs[np.sort(val[i:i + 1024])]).to(dev), torch.from_numpy(k[np.sort(val[i:i + 1024])]).to(dev))[0].argmax(1).cpu().numpy() == act[np.sort(val[i:i + 1024])]).mean() for i in range(0, len(val), 1024)])
        log(f"[bc] epoch {ep + 1}/{BC_EPOCHS} loss {tot / (len(tr) // 256):.3f} held-out action accuracy {acc:.3f}")
    return obs, act, k


def summarize(eps):
    arr = [e for e in eps if e["arrived"]]
    n, s = len(eps), len(arr)
    out = dict(games=n, arrive_rate=s / max(1, n), mean_k=float(np.mean([e["k"] for e in eps])) if eps else 0.0,
               mean_len=float(np.mean([e["length"] for e in eps])) if eps else 0.0, mean_score=float(np.mean([e["score"] for e in eps])) if eps else 0.0)
    if arr:
        out.update(t_arrive_median=float(np.median([e["t_arrive"] for e in arr])), diag_lives_arrive_median=float(np.median([e["diag_lives_arrive"] for e in arr])),
                   diag_lives_arrive_mean=float(np.mean([e["diag_lives_arrive"] for e in arr])))
    return out


def main():
    import torch, torch.nn.functional as F
    torch.set_num_threads(2)
    route_cfg = json.load(open(os.path.join(OUT, "route.json")))
    route, L = route_cfg["route"], len(route_cfg["route"])
    cb = os.path.join(OUT, "codebook.npz")
    dev = torch.device(DEVICE)
    ck_path = os.path.join(OUT, f"{RUN}.pt"); log_path = os.path.join(OUT, f"{RUN}_log.jsonl"); ep_path = os.path.join(OUT, f"{RUN}_episodes.jsonl")
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    net = build_model(L).to(dev)
    evaln = int(sys.argv[sys.argv.index("--eval") + 1]) if "--eval" in sys.argv else 0
    state = dict(update=0, frames=0)
    bc = None
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); state = ck["state"]
        log(f"loaded {ck_path} at update {state['update']}, frames {state['frames']}")
    elif evaln:
        sys.exit("no checkpoint to evaluate")
    else:
        log(f"route {route}; imitation pretraining on the fastest saved legs")
        bc = bc_pretrain(net, BC_LR, dev, log)
        torch.save(dict(net=net.state_dict(), state=state), os.path.join(OUT, f"{RUN}_bc_only.pt"))
    opt = torch.optim.Adam(net.parameters(), lr=LR, eps=1e-5)
    if os.path.exists(ck_path) and "opt" in ck and not evaln:
        opt.load_state_dict(ck["opt"])
    if bc is None and not evaln and state["update"] < BC_AUX_UPDATES:
        d = np.load(os.path.join(OUT, "bc_data.npz")); bc = (d["obs"], d["act"], d["k"])

    vec = Vec(route, cb, seed0=(10_000_000 if evaln else 1000 + state["update"]))
    N = vec.n
    rng = np.random.default_rng(state["update"])

    if evaln:                                            # frozen controller, fresh games, no learning
        eps = []
        while len(eps) < evaln:
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(vec.obs).to(dev), torch.from_numpy(np.minimum(vec.k, L)).to(dev))
                a = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
            a = np.where(vec.phase == 0, a, rng.integers(0, 18, N))
            _, _, infos = vec.step(a)
            eps += [i for i in infos if i]
        eps = eps[:evaln]; s = summarize(eps); n = len(eps)
        s["arrive_rate_lower_95"] = max(0.0, s["arrive_rate"] - (np.log(1 / 0.05) / (2 * n)) ** 0.5)
        s["reached_landmark_k_or_more"] = {k: float(np.mean([e["k"] >= k for e in eps])) for k in range(1, L + 1)}
        s["diag_lives_arrive_hist"] = {str(v): int(sum(1 for e in eps if e["arrived"] and e["diag_lives_arrive"] == v)) for v in range(6)}
        print(json.dumps(s, indent=1)); json.dump(dict(summary=s, episodes=eps), open(os.path.join(OUT, f"{RUN}_eval_u{state['update']}.json"), "w"))
        vec.close(); return

    stop = {"now": False}
    def on_int(*_):
        if not stop["now"]:
            print("[stop] signal received -- finishing this update, then saving and exiting.", flush=True)
        stop["now"] = True
    signal.signal(signal.SIGINT, on_int)
    recent, t_last = [], time.time()
    logf, epf = open(log_path, "a"), open(ep_path, "a")
    while state["frames"] < MAX_FRAMES and not stop["now"]:
        O = np.zeros((T, N, 84, 84, 4), np.uint8); K = np.zeros((T, N), np.int64); A = np.zeros((T, N), np.int64)
        LP = np.zeros((T, N), np.float32); V = np.zeros((T + 1, N), np.float32); R = np.zeros((T, N), np.float32)
        D = np.zeros((T, N), bool); M = np.zeros((T, N), bool)
        for t in range(T):
            O[t], K[t], M[t] = vec.obs, np.minimum(vec.k, L - 1), vec.phase == 0
            with torch.no_grad():
                logits, v = net(torch.from_numpy(O[t]).to(dev), torch.from_numpy(K[t]).to(dev))
                dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
            a_np = np.where(M[t], a.cpu().numpy(), rng.integers(0, 18, N))
            A[t], LP[t], V[t] = a_np, dist.log_prob(a).cpu().numpy(), v.cpu().numpy()
            r, route_end, infos = vec.step(a_np)
            R[t], D[t] = r, route_end | ~M[t]          # nothing bootstraps across the end of a route phase or through random play
            for i in infos:
                if i:
                    recent.append(i); epf.write(json.dumps(dict(update=state["update"], **i)) + "\n")
        with torch.no_grad():
            V[T] = net(torch.from_numpy(vec.obs).to(dev), torch.from_numpy(np.minimum(vec.k, L - 1)).to(dev))[1].cpu().numpy()
        V[T] = np.where(vec.phase == 0, V[T], 0.0)
        adv, last = np.zeros((T, N), np.float32), 0.0
        for t in reversed(range(T)):
            nd = 1.0 - D[t]
            delta = R[t] + GAMMA * V[t + 1] * nd - V[t]
            last = delta + GAMMA * LAM * nd * last; adv[t] = last
        ret = adv + V[:T]
        idx = np.flatnonzero(M.reshape(-1))
        fo, fk, fa = O.reshape(-1, 84, 84, 4), K.reshape(-1), A.reshape(-1)
        flp, fadv, fret = LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        aux_w = BC_AUX0 * max(0.0, 1.0 - state["update"] / BC_AUX_UPDATES) if bc is not None else 0.0
        stats = dict(pg=0.0, vl=0.0, ent=0.0, bc=0.0, kl=0.0); nb = 0
        for _ in range(EPOCHS):
            rng.shuffle(idx)
            for mb in np.array_split(idx, MINIBATCHES):
                mb = np.sort(mb)
                tt = lambda x: torch.from_numpy(x[mb]).to(dev)
                logits, v = net(tt(fo), tt(fk))
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(tt(fa)); a_ = tt(fadv); a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                ratio = (lp - tt(flp)).exp()
                pg = -torch.min(ratio * a_, ratio.clamp(1 - CLIP, 1 + CLIP) * a_).mean()
                vl = F.mse_loss(v, tt(fret)); ent = dist.entropy().mean()
                loss = pg + VF * vl - ENT * ent
                bcl = torch.zeros((), device=dev)
                if aux_w > 0:
                    b = np.sort(rng.choice(len(bc[0]), 512, replace=False))
                    bl, _ = net(torch.from_numpy(bc[0][b]).to(dev), torch.from_numpy(bc[2][b]).to(dev))
                    bcl = F.cross_entropy(bl, torch.from_numpy(bc[1][b]).to(dev)); loss = loss + aux_w * bcl
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD); opt.step()
                stats["pg"] += pg.item(); stats["vl"] += vl.item(); stats["ent"] += ent.item(); stats["bc"] += bcl.item()
                stats["kl"] += (tt(flp) - lp).mean().item(); nb += 1
        state["update"] += 1; state["frames"] += T * N * 4
        recent = recent[-300:]
        row = dict(update=state["update"], frames=state["frames"], route_share=float(M.mean()), fps=round(T * N * 4 / (time.time() - t_last)),
                   bc_aux_w=round(aux_w, 3), **{k: round(v / nb, 4) for k, v in stats.items()}, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarize(recent).items()})
        t_last = time.time()
        logf.write(json.dumps(row) + "\n"); logf.flush(); epf.flush()
        if state["update"] % 5 == 0 or state["update"] == 1:
            log(" ".join(f"{k}={v}" for k, v in row.items()))
        if state["update"] % SAVE_EVERY == 0 or stop["now"] or state["frames"] >= MAX_FRAMES:
            torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state), ck_path + ".tmp"); os.replace(ck_path + ".tmp", ck_path)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), state=state), ck_path + ".tmp"); os.replace(ck_path + ".tmp", ck_path)
    log(f"[stop] checkpoint saved at update {state['update']}, frames {state['frames']}; exiting cleanly.")
    vec.close()


if __name__ == "__main__":
    main()
