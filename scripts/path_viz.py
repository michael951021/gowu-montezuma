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
     LR landmarks = LR-GoWU archive cells (level, room, x//16, y//16), weighted by how often they were reached.
  4. Walkthrough: a few recorded GWRW games replayed frame by frame next to a moving marker on the map.
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
def gwrw_data(run_dir: Path, prefix: str, late: float):
    """Visits (list of (room, x, y) arrays), deaths and ends (lists of (room, x, y)) of one GWRW run."""
    paths = []
    for f in sorted(glob.glob(str(run_dir / f"{prefix}_deep" / "*.npz"))) + sorted(glob.glob(str(run_dir / f"{prefix}_deep_rare" / "*.npz"))):
        try:
            p = np.load(f)["pos"]; paths.append(np.stack([p[:, 2], p[:, 0], p[:, 1]], 1))
        except Exception:
            pass
    ev = run_dir / f"{prefix}_eval_eval_paths.pkl"
    if ev.exists():
        for p in pickle.load(ev.open("rb"))["paths"]:
            paths.append(np.stack([p[:, 2], p[:, 0], p[:, 1]], 1))
    rows = [json.loads(l) for l in (run_dir / f"{prefix}_episodes.jsonl").open()]
    rows = rows[int(len(rows) * (1 - late)):]
    deaths = [(d[1], d[2], d[3]) for e in rows for d in e.get("diag_deaths", [])]
    ends = [tuple(e["diag_end"][:3]) for e in rows if e.get("diag_end")]
    x7 = [e["diag_r7_xmax"] for e in rows if e.get("diag_r7_xmax") is not None]
    return dict(paths=paths, deaths=deaths, ends=ends, games=len(rows), x7=x7, ev=ev.exists())


def lr_data(run_dir: Path):
    pts = []
    for lm in json.load((run_dir / "landmarks.json").open()):
        level, room, _keys, cx, cy = lm["key"]
        if level == 0:
            pts.append((room, cx * 16 + 8, cy * 16 + 8, max(1, lm.get("n_reached", 1)), lm.get("R", 0)))
    return pts


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


def export_walk(f, out: Path, tag: str, stride=2, max_frames=1100, cols=40):
    d = np.load(f)
    fr, pos = d["frames"][::stride][:max_frames], d["pos"][::stride][:max_frames]
    n = len(fr); rows = -(-n // cols)
    sheet = np.zeros((rows * 84, cols * 84), np.uint8)
    for i, im in enumerate(fr):
        sheet[(i // cols) * 84:(i // cols + 1) * 84, (i % cols) * 84:(i % cols + 1) * 84] = im
    cv2.imwrite(str(out / f"walk_{tag}.png"), sheet)
    mp = [to_map(int(p[2]), p[0], p[1]) or (None, None) for p in pos]
    events, last = [], None
    for i, p in enumerate(pos):
        if int(p[2]) != last:
            events.append(dict(i=i, text=f"enters room {int(p[2])}")); last = int(p[2])
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
    return dict(tag=tag, file=os.path.basename(f), n=n, cols=cols, stride=stride, start=start, map=mp,
                rooms=[int(p[2]) for p in pos], events=sorted(events, key=lambda e: e["i"]),
                target=target, seq=seq, score=meta.get("score"), fail=meta.get("fail"))


# ------------------------------------------------------------------ page
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gwrw", nargs="*", default=["runs/gwrw_400m_20261103:gwrw_a", "runs/gwrw_400m_20261103:gwrw_b"])
    ap.add_argument("--lr", nargs="*", default=["runs/lr_v2_resume_20261028/a", "runs/lr_v2_resume_20261028/b"])
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
                         recorded_games=len(g["paths"]), eval_paths=g["ev"])
        for k, f in enumerate(pick_samples(REPO / a.gwrw[list(G).index(p_)].split(":")[0], p_)):
            walks.append(export_walk(f, out, f"{p_}_{k}"))
    lr_rooms = sorted({p[0] for p in lrp}); lr_x7 = max((p[1] for p in lrp if p[0] == 7), default=None)
    data = dict(layers=layers, stats=stats, walks=walks, lr=dict(rooms=lr_rooms, landmarks=len(lrp), room7_max_x=lr_x7),
                layout=LAYOUT, W=W, H=H)
    (out / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    (out / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/")))
    print(out / "index.html")


PAGE = open(Path(__file__).with_name("path_viz_page.html")).read() if Path(__file__).with_name("path_viz_page.html").exists() else "__DATA__"

if __name__ == "__main__":
    main()
