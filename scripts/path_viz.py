"""Montezuma's Revenge Level-1 path map for the hub: where agents walk, die and stop, drawn on the real rooms.

    .venv/bin/python scripts/path_viz.py [--gwrw runs/gwrw_400m_20261103:gwrw_a ...] [--lr runs/lr_v2_resume_20261028/a ...]
                                         [--out ~/Documents/coders/loop/reports/montezuma-paths] [--late 0.25]

How it works
  1. Rooms: each of the 24 Level-1 rooms is rendered by the emulator itself. Joe is placed at the edge of a neighbouring
     room and walked through the doorway, so the game loads the room's own objects (lasers, skulls, keys). Cached in OUT/rooms/.
  2. Coordinates: Joe's RAM position (x = byte 42, y = byte 43, room = byte 3) maps to screen pixels as
     px = x + 3, py = 325 - y (feet). Fitted on 8.5k frames where Joe's sprite was found by colour (median error < 0.5 px).
     Both GWRW (diag_* fields) and LR-GoWU (archive cells) log these same bytes; neither algorithm uses them to act.
  3. Layers, each a heat image over the whole pyramid (2-px bins, Gaussian blur, log scale):
     GWRW visits  = every step of the recorded games (deep / rare recorders, and the frozen test's paths when present),
     GWRW deaths  = every life lost in the last LATE share of training games (episodes.jsonl diag_deaths),
     GWRW ends    = where those games ended,
     LR landmarks = LR-GoWU archive cells (level, room, x//16, y//16), weighted by how often they were reached,
     GoWU route   = the Level-1 part of GoWU Phase I's best route (demo_best.pkl.gz: one RAM position per tree node).
  4. Walkthrough: a few recorded GWRW games replayed frame by frame next to a moving marker on the map, plus GoWU's best
     route replayed from its own emulator snapshots (each node restored, its actions stepped, so it re-syncs every node).
Writes OUT/index.html, OUT/map.png, OUT/layer_*.png, OUT/walk_*.png (frame sheets) and OUT/data.json.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
HUD, W, H = 47, 160, 210 - 47                     # playfield = screen rows 47..209
LAYOUT = [[None] * 3 + [0, 1, 2] + [None] * 3, [None] * 2 + [3, 4, 5, 6, 7] + [None] * 2,
          [None] + list(range(8, 15)) + [None], list(range(15, 24))]
CELL = {room: (r, c) for r, row in enumerate(LAYOUT) for c, room in enumerate(row) if room is not None}
MAP_H, MAP_W = len(LAYOUT) * H, 9 * W


def to_map(room, x, y):
    """RAM (room, x, y) -> map pixel (mx, my), or None outside Level 1's grid."""
    if room not in CELL:
        return None
    r, c = CELL[room]
    return c * W + int(x) + 3, r * H + (325 - int(y)) - HUD


# ------------------------------------------------------------------ rooms
def render_rooms(cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    if all((cache / f"room_{k:02d}.png").exists() for k in range(24)):
        return {k: cv2.imread(str(cache / f"room_{k:02d}.png"))[:, :, ::-1] for k in range(24)}
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    env = gym.make("ALE/MontezumaRevenge-v5", frameskip=1, repeat_action_probability=0.0)
    env.reset(seed=0); ale = env.unwrapped.ale
    for _ in range(60):
        env.step(0)
    st = ale.cloneState(); out = {}
    for R in range(24):
        for src, x0, act in ((R - 1, 148, 3), (R + 1, 4, 4)):          # walk in from the left / right neighbour
            if R in out or not 0 <= src < 24:
                continue
            for y in (148, 235, 192, 165, 210, 252):
                ale.restoreState(st); ale.setRAM(3, src); ale.setRAM(42, x0); ale.setRAM(43, y)
                lives = ale.getRAM()[58]
                for _ in range(40):
                    env.step(act)
                    if ale.getRAM()[3] == R:
                        break
                for _ in range(20):
                    env.step(0)
                ram = ale.getRAM()
                if ram[3] == R and ram[58] == lives:
                    f = ale.getScreenRGB().copy()
                    px, py = int(ram[42]) + 3, 325 - int(ram[43])              # paint Joe out (he stands at the doorway)
                    f[max(0, py - 22):py + 2, max(0, px - 6):px + 6][((f[max(0, py - 22):py + 2, max(0, px - 6):px + 6] == (200, 72, 72)).all(-1)
                                                                     | (f[max(0, py - 22):py + 2, max(0, px - 6):px + 6] == (210, 182, 86)).all(-1))] = 0
                    out[R] = f; cv2.imwrite(str(cache / f"room_{R:02d}.png"), f[:, :, ::-1])
                    break
    env.close()
    return out


def montage(rooms):
    m = np.zeros((MAP_H, MAP_W, 3), np.uint8)
    for room, (r, c) in CELL.items():
        if room in rooms:
            m[r * H:(r + 1) * H, c * W:(c + 1) * W] = rooms[room][HUD:]
    return m


# ------------------------------------------------------------------ data
def fix_transitions(P):
    """P: (n, 3) room, x, y. On the frame of a room change the room byte is already new while x/y are still the old room's
    (e.g. room 6 x=146 -> room 7 x=149 -> room 7 x=1), which would plot as a jump to the far edge. That frame takes the
    next frame's x/y."""
    P = P.astype(np.int64)
    ch = np.flatnonzero(P[1:-1, 0] != P[:-2, 0]) + 1
    for i in ch:
        if P[i + 1, 0] == P[i, 0]:
            P[i, 1:] = P[i + 1, 1:]
    # one-frame glitches at a doorway: a single frame in the other room (6, 6, 7@x149, 6) or a single wrapped x (7@5, 7@152, 7@4)
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        if (a[0] == c[0] != b[0]) or (a[0] == b[0] == c[0] and abs(b[1] - a[1]) > 60 and abs(b[1] - c[1]) > 60):
            P[i] = a
    return P


def gwrw_data(run_dir: Path, prefix: str, late: float):
    """Visits (list of (room, x, y) arrays), deaths and ends (lists of (room, x, y)) of one GWRW run."""
    paths = []
    for f in sorted(glob.glob(str(run_dir / f"{prefix}_deep" / "*.npz"))) + sorted(glob.glob(str(run_dir / f"{prefix}_deep_rare" / "*.npz"))):
        try:
            p = np.load(f)["pos"]; paths.append(fix_transitions(np.stack([p[:, 2], p[:, 0], p[:, 1]], 1)))
        except Exception:
            pass
    ev = run_dir / f"{prefix}_eval_eval_paths.pkl"
    if ev.exists():
        for p in pickle.load(ev.open("rb"))["paths"]:
            paths.append(fix_transitions(np.stack([p[:, 2], p[:, 0], p[:, 1]], 1)))
    rows = [json.loads(l) for l in (run_dir / f"{prefix}_episodes.jsonl").open()]
    rows = rows[int(len(rows) * (1 - late)):]
    deaths = [(d[1], d[2], d[3]) for e in rows for d in e.get("diag_deaths", [])]
    ends = [tuple(e["diag_end"][:3]) for e in rows if e.get("diag_end")]
    x7 = [e["diag_r7_xmax"] for e in rows if e.get("diag_r7_xmax") is not None]
    from collections import Counter
    ho = Counter(e["diag_ho"][2] for e in rows if e.get("diag_ho"))
    h7 = [e for e in rows if e.get("diag_ho") and e["diag_ho"][2] == 7 and e.get("t_arrive") is not None]
    after = [[d for d in e["diag_deaths"] if d[0] >= e["t_arrive"]] for e in h7]
    tg = [e for e in rows if e.get("target")]
    diag = dict(handoff_rooms=[[int(k), v / max(1, sum(ho.values()))] for k, v in ho.most_common(6)],
                h7_games=len(h7), h7_deaths7=float(np.mean([sum(d[1] == 7 for d in a_) for a_ in after])) if h7 else None,
                h7_first_death=float(np.median([(min(d[0] for d in a_) if a_ else e["length"]) - e["t_arrive"] for e, a_ in zip(h7, after)])) if h7 else None,
                arrived=float(np.mean([e["arrived"] for e in tg])) if tg else None,
                wings={k: v / len(rows) for k, v in Counter(e.get("wing") for e in rows).items()})
    return dict(paths=paths, deaths=deaths, ends=ends, games=len(rows), x7=x7, ev=ev.exists(), diag=diag)


def lr_data(run_dir: Path):
    pts = []
    for lm in json.load((run_dir / "landmarks.json").open()):
        level, room, _keys, cx, cy = lm["key"]
        if level == 0:
            pts.append((room, cx * 16 + 8, cy * 16 + 8, max(1, lm.get("n_reached", 1)), lm.get("R", 0)))
    return pts


def gowu_load(run_dir: Path):
    import gzip, sys
    sys.path.insert(0, str(REPO))
    d = pickle.load(gzip.open(run_dir / "demo_best.pkl.gz"))
    n = next((i for i, r in enumerate(d["ram"]) if r[0] >= 1), len(d["ram"]))      # level-1 part of the best route
    return d, n


def gowu_data(run_dir: Path):
    """GoWU's best route (restores allowed) through Level 1: one point per tree node (5-15 steps each)."""
    d, n = gowu_load(run_dir)
    P = fix_transitions(np.array([(r[1], r[2], r[3]) for r in d["ram"][:n]]))
    st = d["steps"]
    first = {}
    for i, (room, _, _) in enumerate(P):
        first.setdefault(int(room), int(st[i]))
    i7 = [i for i in range(n) if P[i, 0] == 7]
    return dict(points=[tuple(p) for p in P], steps=int(st[n - 1]), R=float(d["R_cum"][n - 1]), rooms=sorted(first),
                first_step=first, nodes=n, found_frames=int(d["frames"]),
                r7_in=int(st[i7[0]]) if i7 else None,
                r7_cross=next((int(st[i]) for i in range(1, n) if P[i - 1, 0] == 7 and P[i, 0] == 13), None))


def gowu_walk(run_dir: Path, out: Path, tag: str, lo_room=None, hi_room=None, max_frames=1100, cols=40):
    """Replays GoWU's best route step by step: restore each node's snapshot, step its actions without sticky noise (re-synced
    at every node, so drift stays within one node). Either the whole Level-1 route (subsampled) or the first stretch from the
    first entry into lo_room to the first entry into hi_room."""
    import sys
    sys.path.insert(0, str(REPO))
    from gowu.atari import AtariEnv
    d, n = gowu_load(run_dir)
    rooms = [r[1] for r in d["ram"][:n]]
    a_, b_ = 1, n
    if lo_room is not None:
        a_ = max(1, rooms.index(lo_room) - 3); b_ = next(i for i in range(a_, n) if rooms[i] == hi_room) + 12
    env = AtariEnv(sticky_p=0.0)
    fr, pos, ev, lives_prev, R_prev = [], [], [], d["ckpts"][a_ - 1].lives, d["R_cum"][a_ - 1]
    total = sum(len(d["actions"][i]) for i in range(a_, b_))
    stride = max(1, -(-total // max_frames)); k = 0
    for i in range(a_, b_):
        env.restore(d["ckpts"][i - 1])
        for act in d["actions"][i]:
            obs, _, _, _ = env.step(act)
            if k % stride == 0:
                r = env.ram_info(); fr.append(obs[-1].copy()); pos.append((r[1], r[2], r[3]))
            k += 1
        if d["ckpts"][i].lives < lives_prev:
            ev.append((len(fr) - 1, f"loses a life in room {rooms[i]} (GoWU restores an earlier snapshot only when a branch dies; this one kept going)"))
        if d["R_cum"][i] > R_prev:
            ev.append((len(fr) - 1, f"+{int(d['R_cum'][i] - R_prev)} (score {int(d['R_cum'][i])})"))
        lives_prev, R_prev = d["ckpts"][i].lives, d["R_cum"][i]
    P = fix_transitions(np.array(pos))
    w = write_walk(np.array(fr), P, out, tag, stride, cols, extra_events=ev)
    w.update(file=f"{run_dir.parent.name}/{run_dir.name} best route", target="best route GoWU found (restores)", seq="",
             score=int(d["R_cum"][b_ - 1]), step0=int(d["steps"][a_ - 1]))
    return w


def heat(points, weights=None, sigma=3.0):
    """Points (room, x, y) -> RGBA heat image over the map (transparent where empty)."""
    acc = np.zeros((MAP_H, MAP_W), np.float32)
    for k, (room, x, y) in enumerate(points):
        m = to_map(room, x, y)
        if m and 0 <= m[0] < MAP_W and 0 <= m[1] < MAP_H:
            acc[m[1], m[0]] += 1.0 if weights is None else weights[k]
    if acc.max() == 0:
        return np.zeros((MAP_H, MAP_W, 4), np.uint8), 0
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    v = np.log1p(acc / acc[acc > 0].min()); v /= v.max()
    rgb = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
    alpha = (np.clip(v * 2.2, 0, 1) * 235).astype(np.uint8)
    return np.dstack([rgb, alpha]), int(len(points))


# ------------------------------------------------------------------ walkthrough
def pick_samples(run_dir: Path, prefix: str, k=2):
    """The deep-recorder game that got furthest right in room 7, plus the first rare-room games."""
    deep = sorted(glob.glob(str(run_dir / f"{prefix}_deep" / "*.npz")), key=lambda f: -int(f.rsplit("_x", 1)[1][:-4]))
    rare = sorted(glob.glob(str(run_dir / f"{prefix}_deep_rare" / "*.npz")))
    return (deep[:1] + rare[:k - 1]) if deep else rare[:k]


def write_walk(fr, P, out: Path, tag: str, stride, cols=40, extra_events=()):
    """fr: (n, 84, 84) frames, P: (n, 3) room, x, y (already subsampled). Writes the frame sheet; returns the walk record."""
    n = len(fr); rows = -(-n // cols)
    sheet = np.zeros((rows * 84, cols * 84), np.uint8)
    for i, im in enumerate(fr):
        sheet[(i // cols) * 84:(i // cols + 1) * 84, (i % cols) * 84:(i % cols + 1) * 84] = im
    cv2.imwrite(str(out / f"walk_{tag}.png"), sheet)
    mp = [to_map(int(p[0]), p[1], p[2]) or (None, None) for p in P]
    events, last = [], None
    for i, p in enumerate(P):
        if int(p[0]) != last:
            events.append(dict(i=i, text=f"enters room {int(p[0])}")); last = int(p[0])
    events += [dict(i=int(i), text=t) for i, t in extra_events if 0 <= i < n]
    return dict(tag=tag, n=n, cols=cols, stride=stride, map=mp, rooms=[int(p[0]) for p in P],
                events=sorted(events, key=lambda e: e["i"]), step0=0)


def export_walk(f, out: Path, tag: str, stride=2, max_frames=1100, cols=40):
    d = np.load(f)
    P = fix_transitions(np.stack([d["pos"][:, 2], d["pos"][:, 0], d["pos"][:, 1]], 1))[::stride][:max_frames]
    w = write_walk(d["frames"][::stride][:max_frames], P, out, tag, stride, cols)
    n, events = w["n"], w["events"]
    start = int(d["start"]) if "start" in d.files else 0
    if "deaths" in d.files:
        for dd in d["deaths"]:
            i = (int(dd[0]) - start) // stride
            if 0 <= i < n:
                events.append(dict(i=i, text=f"loses a life in room {int(dd[1])} at x={int(dd[2])}"))
    meta = {}
    if "meta" in d.files:
        meta = json.loads(str(d["meta"]))
    target = str(d["target"]) if "target" in d.files else str(meta.get("target", ""))
    seq = str(d["seq"]) if "seq" in d.files else str(meta.get("seq", ""))
    w.update(file=os.path.basename(f), start=start, events=sorted(events, key=lambda e: e["i"]),
             target=target, seq=seq, score=meta.get("score"), fail=meta.get("fail"))
    return w


# ------------------------------------------------------------------ page
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwrw", nargs="*", default=["runs/gwrw_400m_20261103:gwrw_a", "runs/gwrw_400m_20261103:gwrw_b"])
    ap.add_argument("--lr", nargs="*", default=["runs/lr_v2_resume_20261028/a", "runs/lr_v2_resume_20261028/b"])
    ap.add_argument("--gowu", nargs="*", default=["runs/trapfix_400m_20261028/seed6", "runs/trapfix_400m_20261028/seed7"],
                    help="GoWU Phase I run dirs with demo_best.pkl.gz (best route; the level-1 part is drawn)")
    ap.add_argument("--out", type=Path, default=Path.home() / "Documents/coders/loop/reports/montezuma-paths")
    ap.add_argument("--late", type=float, default=0.25, help="share of the latest training games used for deaths / ends")
    a = ap.parse_args()
    out = a.out.expanduser(); out.mkdir(parents=True, exist_ok=True)
    rooms = render_rooms(out / "rooms")
    cv2.imwrite(str(out / "map.png"), montage(rooms)[:, :, ::-1])

    layers, stats, walks = [], {}, []
    G = {}
    for spec in a.gwrw:
        d_, p_ = spec.split(":"); G[p_] = gwrw_data(REPO / d_, p_, a.late)
    vis = [(int(r), x, y) for g in G.values() for P in g["paths"] for r, x, y in P]
    for name, pts, label in (("visits", vis, "GWRW · where recorded games walked"),
                             ("deaths", [p for g in G.values() for p in g["deaths"]], f"GWRW · lives lost (last {int(a.late * 100)}% of training)"),
                             ("ends", [p for g in G.values() for p in g["ends"]], "GWRW · where games ended")):
        img, n = heat(pts)
        cv2.imwrite(str(out / f"layer_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)); layers.append(dict(id=name, label=label, n=n))
    lrp = [p for d_ in a.lr if (REPO / d_ / "landmarks.json").exists() for p in lr_data(REPO / d_)]
    img, n = heat([p[:3] for p in lrp], [float(np.log1p(p[3])) for p in lrp], sigma=5)
    cv2.imwrite(str(out / "layer_lr.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)); layers.append(dict(id="lr", label="LR-GoWU · archive landmarks (weighted by visits)", n=n))

    for p_, g in G.items():
        from collections import Counter
        dc = Counter((r, x // 16 * 16, y // 16 * 16) for r, x, y in g["deaths"])
        x7 = np.array(g["x7"]) if g["x7"] else np.zeros(0)
        stats[p_] = dict(games=g["games"], deaths=len(g["deaths"]), top_deaths=[[*k, v] for k, v in dc.most_common(6)],
                         rooms=sorted({int(r) for P in g["paths"] for r in P[:, 0]}),
                         x7_hist=np.histogram(x7, bins=16, range=(0, 160))[0].tolist() if len(x7) else [],
                         x7_median=float(np.median(x7)) if len(x7) else None, x7_past100=float((x7 >= 100).mean()) if len(x7) else None,
                         recorded_games=len(g["paths"]), eval_paths=g["ev"], diag=g["diag"])
        for k, f in enumerate(pick_samples(REPO / a.gwrw[list(G).index(p_)].split(":")[0], p_)):
            walks.append(export_walk(f, out, f"{p_}_{k}"))
    lr_rooms = sorted({p[0] for p in lrp}); lr_x7 = max((p[1] for p in lrp if p[0] == 7), default=None)

    gw = {}
    for d_ in a.gowu:
        gw[d_] = gowu_data(REPO / d_)
    if gw:
        img, n = heat([p for g in gw.values() for p in g["points"]])
        cv2.imwrite(str(out / "layer_gowu.png"), cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        layers.append(dict(id="gowu", label="GoWU · best route through Level 1 (restores)", n=n))
        d0 = a.gowu[0]
        walks.insert(0, gowu_walk(REPO / d0, out, "gowu_r7", lo_room=6, hi_room=13))
        walks.insert(1, gowu_walk(REPO / a.gowu[-1], out, "gowu_lvl1"))
    gowu_stats = {f"{Path(d_).parent.name}/{Path(d_).name}": {k: v for k, v in g.items() if k != "points"} for d_, g in gw.items()}
    data = dict(layers=layers, stats=stats, walks=walks, lr=dict(rooms=lr_rooms, landmarks=len(lrp), room7_max_x=lr_x7), gowu=gowu_stats,
                layout=LAYOUT, W=W, H=H)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    print(out / "index.html")


PAGE = open(Path(__file__).with_name("path_viz_page.html")).read() if Path(__file__).with_name("path_viz_page.html").exists() else "__DATA__"

if __name__ == "__main__":
    main()
