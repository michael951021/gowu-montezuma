"""Build the LR-GoWU research page on the hub from research/lr_gowu/registry.json.

    python3 scripts/build_research_page.py [--out ~/Documents/coders/loop/reports/lr-gowu]

Writes index.html (curves of every registered iteration against the GoWU baseline, an iteration table,
data locations) and arch/<id>.html (the frozen architecture snapshot, the exact run config, and what
changed in config and code since the previous iteration). Stdlib only; re-run after every new run.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "research" / "lr_gowu"
ALGO = RESEARCH / "algorithm"
DEFAULT_OUT = Path.home() / "Documents/coders/loop/reports/lr-gowu"
HUB = ("<script>(s=>{s.src='http://'+location.hostname+':8765/hub.js';document.head.append(s)})"
       "(document.createElement('script'))</script>")
MAX_POINTS = 400
E = html.escape


# ------------------------------------------------------------------ data
def read_log(path: Path):
    recs = []
    for line in path.open():
        try:
            recs.append(json.loads(line))
        except ValueError:
            pass
    return recs


def thin(rows, key_cols):
    """At most MAX_POINTS rows, always keeping the rows where a step-series column changes."""
    if len(rows) <= MAX_POINTS:
        return rows
    keep = {0, len(rows) - 1}
    for c in key_cols:
        keep |= {i for i in range(1, len(rows)) if rows[i][c] != rows[i - 1][c]}
    stride = max(1, len(rows) // MAX_POINTS)
    keep |= set(range(0, len(rows), stride))
    return [rows[i] for i in sorted(keep)]


def gowu_series(run_dir: Path):
    recs = read_log(run_dir / "log.jsonl")
    rows = [[r["frames"], r["best_R"], r["rooms"]] for r in recs]
    last = recs[-1] if recs else {}
    return {"rows": thin(rows, [1, 2]), "cols": ["frames", "best_R", "rooms"],
            "final": {"frames": last.get("frames"), "best_R": last.get("best_R"), "rooms": last.get("rooms"),
                      "fps": last.get("fps")}}


def lr_series(run_dir: Path, continued=()):
    """One seed's curve. `continued`: later run dirs that resumed this one; their logs are appended (records at or
    before the last frame count already seen are dropped) and the last one's eval is the run's eval."""
    recs, hours = read_log(run_dir / "log.jsonl"), 0.0
    for c in continued:
        hours += recs[-1].get("time", 0) if recs else 0
        upto = recs[-1]["frames"] if recs else -1
        recs += [r for r in read_log(c / "log.jsonl") if r["frames"] > upto]
        run_dir = c
    rows, win = [], []
    for r in recs:
        if r.get("ret_attempts"):
            win.append((r["ret_success"] * r["ret_attempts"], r["ret_attempts"]))
        win = win[-20:]
        succ = sum(a for a, _ in win) / max(1, sum(n for _, n in win)) if win else None
        share = r["frames_return"] / r["frames"] if r.get("frames") else None
        rows.append([r["frames"], r["max_R"], r["rooms"], r["n_landmarks"],
                     None if succ is None else round(succ, 3), None if share is None else round(share, 3),
                     r.get("ep_R")])
    last = recs[-1] if recs else {}
    ev = None
    if (run_dir / "eval.json").exists():
        ev = [{k: e.get(k) for k in ("schedule", "cap_frames", "n", "mean", "std", "max")}
              for e in json.load((run_dir / "eval.json").open())]
    return {"rows": thin(rows, [1, 2, 3]),
            "cols": ["frames", "best_R", "rooms", "landmarks", "ret_success", "return_share", "ep_R"],
            "final": {"frames": last.get("frames"), "best_R": last.get("max_R"), "rooms": last.get("rooms"),
                      "landmarks": last.get("n_landmarks"), "fps": last.get("fps"), "hours": (hours + last.get("time", 0)) / 3600,
                      "timing": last.get("timing")},
            "eval": ev}


def gwrw_series(run_dir: Path, prefix: str):
    """One GWRW run (third_party/montezuma_gwrw; several runs share a dir, files prefixed `prefix`). best_R / rooms are
    running max score / distinct rooms (RAM room id, diagnostics only) over finished games; landmarks = archive nodes;
    ep_R = mean score of the last 300 games. The frozen test is `<prefix>_eval_eval.json` (scripts/gwrw_eval.sh)."""
    recs = read_log(run_dir / f"{prefix}_log.jsonl")
    by_upd = {}
    for e in read_log(run_dir / f"{prefix}_episodes.jsonl"):
        b = by_upd.setdefault(e["update"], [0.0, set(), 0, 0])
        b[0] = max(b[0], e["score"]); b[1].update(e.get("diag_rooms", []))
        if e.get("target"):   # a game with a route to walk: did μ arrive (all its landmarks, or the frontier clip)?
            b[2] += 1; b[3] += bool(e.get("arrived"))
    best, rooms, upds, rows = 0.0, set(), sorted(by_upd), []
    j, win, share_sum = 0, [], 0.0
    for n, r in enumerate(recs, 1):
        while j < len(upds) and upds[j] <= r["update"]:
            u = by_upd[upds[j]]
            best = max(best, u[0]); rooms |= u[1]; win.append((u[3], u[2])); j += 1
        win = [w for w in win[-400:]]   # ~400 updates = ~20M frames of finished games
        tries = sum(t for _, t in win)
        succ = round(sum(a for a, _ in win) / tries, 3) if tries >= 20 else None
        share_sum += r.get("route_share") or 0.0   # share of this update's frames spent walking the route (μ)
        rows.append([r["frames"], best, len(rooms), r.get("nodes"), succ, round(share_sum / n, 3), r.get("mean_score")])
    last = recs[-1] if recs else {}
    ev = None
    if (run_dir / f"{prefix}_eval_eval.json").exists():
        e = json.load((run_dir / f"{prefix}_eval_eval.json").open())
        ev = [{"schedule": "frozen", "cap_frames": "game cap", "n": e["games"], "mean": e["mean"], "std": e["std"], "max": e["max"],
               "arrived": e.get("arrived"), "rooms": e.get("rooms")}]
    return {"rows": thin(rows, [1, 2, 3]),
            "cols": ["frames", "best_R", "rooms", "landmarks", "ret_success", "return_share", "ep_R"],
            "final": {"frames": last.get("frames"), "best_R": best, "rooms": len(rooms), "landmarks": last.get("nodes"),
                      "fps": last.get("fps"), "best_route": last.get("best_route"), "reproducible_depth": last.get("reproducible_depth")},
            "eval": ev}


def reach_lr(run_dir: Path):
    """Every LR landmark the return policy was ever sent to: [depth, arrivals / tries, arrivals, tries, label]."""
    p = run_dir / "landmarks.json"
    if not p.exists():
        return None
    pts = []
    for l in json.loads(p.read_text()):
        if l["n_targeted"]:
            k = l["key"]   # (level, room), (level, room, x//16, y//16) or (level, room, keys, x//16, y//16)
            where = f"L{k[0] + 1} room {k[1]}"
            if len(k) >= 4:
                cx, cy = k[-2:]
                where += f" · x {cx * 16}–{cx * 16 + 15}, y {cy * 16}–{cy * 16 + 15}" + (f" · {k[2]} key" if len(k) == 5 and k[2] else "")
            pts.append([l["depth"], round(l["n_reached"] / l["n_targeted"], 4), l["n_reached"], l["n_targeted"],
                        where + f" · R {l['R']:,.0f}"])
    return pts


def reach_gwrw(run_dir: Path, prefix: str):
    """Every GWRW archive node μ was sent to (logged finished games): [depth, arrivals / tries, arrivals, tries, label].
    Frontier-clip targets are pooled into one extra entry per depth, labelled as such."""
    p = run_dir / f"{prefix}_episodes.jsonl"
    if not p.exists():
        return None
    lmroom = {}
    if (run_dir / f"{prefix}_lmroom.json").exists():
        for k, v in json.loads((run_dir / f"{prefix}_lmroom.json").read_text()).items():
            lm, room = k.split("@")
            if v > lmroom.get(lm, (None, -1))[1]:
                lmroom[lm] = (room, v)
    C = {}
    for e in read_log(p):
        t = e.get("target")
        if not t:
            continue
        key = ("front", len(t)) if t[-1][0] == "f" else tuple(f"{a}{b}" for a, b, *_ in t)
        c = C.setdefault(key, [0, 0]); c[0] += 1; c[1] += bool(e.get("arrived"))
    pts = []
    for key, (n, a) in C.items():
        if key[0] == "front":
            pts.append([key[1], round(a / n, 4), a, n, f"frontier clips at depth {key[1]} (pooled)"])
        else:
            last = key[-1]
            room = lmroom.get(last, ("?",))[0]
            kind = "reward" if last[0] == "r" else "screen"
            pts.append([len(key), round(a / n, 4), a, n, f"route ending in {kind} {last[1:]} (room {room}) · " + " ".join(key)])
    return pts


def load_config(run_dir: Path):
    p = run_dir / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def series(kind, d: Path, r):
    if kind == "gowu":
        return gowu_series(d) if (d / "log.jsonl").exists() else None
    if kind == "gwrw":
        return gwrw_series(d, r["prefix"]) if (d / f"{r['prefix']}_log.jsonl").exists() else None
    return lr_series(d, [REPO / c for c in r.get("continued", [])]) if (d / "log.jsonl").exists() else None


def git(*args):
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------ markdown
def inline(s):
    s = E(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<i>\1</i>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", s)
    return s


def md(text):
    """The subset the arch snapshots use: headings, paragraphs, nested - lists, 1. lists."""
    out, stack, para = [], [], []

    def flush_para():
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def close_to(depth):
        while len(stack) > depth:
            out.append(f"</li></{stack.pop()}>")

    for line in text.splitlines():
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush_para()
            depth = len(m.group(1)) // 2 + 1
            tag = "ol" if m.group(2)[0].isdigit() else "ul"
            if depth > len(stack):
                while depth > len(stack):
                    out.append(f"<{tag}><li>")
                    stack.append(tag)
            else:
                close_to(depth)
                out.append("</li><li>")
            out.append(inline(m.group(3)))
            continue
        if stack and line.startswith("  ") and line.strip():
            out.append(" " + inline(line.strip()))
            continue
        close_to(0)
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        if h:
            flush_para()
            n = len(h.group(1))
            out.append(f"<h{n + 1}>{inline(h.group(2))}</h{n + 1}>")
        elif not line.strip():
            flush_para()
        else:
            para.append(line.strip())
    flush_para()
    close_to(0)
    return "\n".join(out).replace("<li>\n</li><li>", "<li>")


# ------------------------------------------------------------------ page
CSS = """
:root{color-scheme:light;--bg:#f6f6f4;--surface:#fcfcfb;--line:#e4e3de;--grid:#ecebe7;--ink:#0b0b0b;--ink2:#52514e;--muted:#7a7974;
--accent:#2a78d6;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948;--ref:#8a8984;--code:#efeee9}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme=light])){color-scheme:dark;--bg:#121211;--surface:#1a1a19;--line:#2e2e2b;
--grid:#262624;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--accent:#3987e5;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#008300;--s7:#9085e9;--s8:#e66767;--ref:#8f8e86;--code:#242422}}
:root[data-theme=dark]{color-scheme:dark;--bg:#121211;--surface:#1a1a19;--line:#2e2e2b;--grid:#262624;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;
--accent:#3987e5;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767;--ref:#8f8e86;--code:#242422}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
h3{font-size:15px;margin:22px 0 6px}.sub{color:var(--ink2)}.muted{color:var(--muted);font-size:13px}a{color:var(--accent)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,520px),1fr));gap:16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:16px}
.tile .v{font-size:24px;font-weight:650;font-variant-numeric:tabular-nums}.tile .k{color:var(--ink2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13.5px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--ink2);font-weight:600;white-space:nowrap}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tw{overflow-x:auto}.sw{display:inline-block;width:12px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin:0 0 10px}
.legend button{all:unset;cursor:pointer;display:inline-flex;align-items:center}.legend button[aria-pressed=false]{opacity:.35}
.ctrl{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0 14px;font-size:13px;color:var(--ink2)}
.ctrl button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
.ctrl button[aria-pressed=true]{border-color:var(--accent);color:var(--accent)}
.chart{position:relative}.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart .ax{fill:var(--muted);font-size:11px}.chart .gl{stroke:var(--grid)}.chart .bl{stroke:var(--line)}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:8px 10px;
font-size:12px;box-shadow:0 4px 16px #0002;white-space:nowrap;display:none;z-index:5}.tip b{font-variant-numeric:tabular-nums}
.chip{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:99px;padding:1px 8px;color:var(--ink2)}
pre{background:var(--code);padding:12px;border-radius:8px;overflow:auto;font-size:12.5px}
.doc li{margin:3px 0}.doc h2{border-top:1px solid var(--line);padding-top:18px}.doc h2:first-child{border:0;padding:0;margin-top:0}
details summary{cursor:pointer;color:var(--ink2)}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const fmtF = v => v >= 1e9 ? (v/1e9).toFixed(1)+'B' : (v/1e6).toFixed(v < 1e7 ? 1 : 0)+'M';
const fmtN = v => v == null ? '–' : Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : (+v.toFixed(3)).toString();
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const hidden = new Set();
let xMax = D.lrMaxFrames * 1.05;
const charts = [];

function stepAt(rows, xi, yi, x){ let v = null; for (const r of rows){ if (r[xi] > x) break; if (r[yi] != null) v = r[yi]; } return v; }

function chart(el, {col, logY, lrOnly, yFmt, paper}){
  const W = 560, H = 300, m = {l: 54, r: 14, t: 10, b: 30};
  const tip = document.createElement('div'); tip.className = 'tip'; el.appendChild(tip);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); el.appendChild(svg);
  const fams = D.families.filter(f => !(lrOnly && f.kind === 'gowu'));
  function draw(){
    svg.innerHTML = '';
    const vis = fams.filter(f => !hidden.has(f.id));
    let ys = [];
    for (const f of vis) for (const r of f.runs) for (const row of r.rows) {
      const ci = r.cols.indexOf(col); if (ci < 0 || row[0] > xMax) continue;
      const y = row[ci]; if (y != null && (!logY || y > 0)) ys.push(y); }
    if (paper && !hidden.has('paper')) for (const p of D.paper.points) if (p[0] <= xMax) ys.push(p[1]);
    if (!ys.length) ys = [1];
    let y0 = logY ? Math.pow(10, Math.floor(Math.log10(Math.min(...ys))) - 1) : 0;
    let y1 = Math.max(...ys); y1 = logY ? Math.pow(10, Math.ceil(Math.log10(y1 * 1.05))) : (y1 <= 1 ? 1 : y1 * 1.08);
    const X = x => m.l + (x / xMax) * (W - m.l - m.r);
    const Y = y => logY ? H - m.b - (Math.log10(Math.max(y, y0)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0)) * (H - m.t - m.b)
                        : H - m.b - (y - y0) / (y1 - y0) * (H - m.t - m.b);
    const add = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.appendChild(e); return e; };
    const ticks = [];
    if (logY) for (let p = Math.log10(y0); p <= Math.log10(y1) + 1e-9; p++) ticks.push(Math.pow(10, p));
    else { const st = niceStep(y1 / 5); for (let v = 0; v <= y1 + 1e-9; v += st) ticks.push(+v.toFixed(6)); }
    for (const t of ticks){ add('line', {x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: 'gl'}); add('text', {x: m.l - 6, y: Y(t) + 4, 'text-anchor': 'end', class: 'ax'}, yFmt(t)); }
    const xs = niceStep(xMax / 6);
    for (let v = 0; v <= xMax; v += xs){ add('text', {x: X(v), y: H - m.b + 16, 'text-anchor': 'middle', class: 'ax'}, fmtF(v)); }
    add('line', {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, class: 'bl'});
    add('text', {x: W - m.r, y: H - 2, 'text-anchor': 'end', class: 'ax'}, 'environment frames');
    for (const f of vis) for (const r of f.runs) {
      const ci = r.cols.indexOf(col); if (ci < 0) continue;
      let d = '', prev = null;
      for (const row of r.rows) { if (row[0] > xMax) break; const y = row[ci]; if (y == null || (logY && y <= 0)) continue;
        if (f.step && prev != null) d += `L${X(row[0]).toFixed(1)},${Y(prev).toFixed(1)}`;
        d += (d ? 'L' : 'M') + X(row[0]).toFixed(1) + ',' + Y(y).toFixed(1); prev = y; }
      add('path', {d, fill: 'none', stroke: css(f.color), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-opacity': f.runs.length > 1 ? .85 : 1});
    }
    if (paper && !hidden.has('paper')) for (const p of D.paper.points) if (p[0] <= xMax)
      add('circle', {cx: X(p[0]), cy: Y(p[1]), r: 4.5, fill: css('--surface'), stroke: css('--ref'), 'stroke-width': 2});
    const cross = add('line', {y1: m.t, y2: H - m.b, stroke: css('--muted'), 'stroke-dasharray': '3 3', visibility: 'hidden'});
    const hit = add('rect', {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent'});
    hit.addEventListener('mousemove', ev => {
      const b = svg.getBoundingClientRect(); const sx = (ev.clientX - b.left) / b.width * W;
      const x = Math.max(0, Math.min(xMax, (sx - m.l) / (W - m.l - m.r) * xMax));
      cross.setAttribute('x1', X(x)); cross.setAttribute('x2', X(x)); cross.setAttribute('visibility', 'visible');
      let h = `<div class=muted>${fmtF(x)} frames</div>`;
      for (const f of vis) for (const r of f.runs) { const ci = r.cols.indexOf(col); if (ci < 0) continue;
        const last = r.rows[r.rows.length - 1][0]; if (x > last) continue;
        h += `<div><span class=sw style="background:${css(f.color)}"></span>${f.label} · s${r.seed}: <b>${yFmt(stepAt(r.rows, 0, ci, x))}</b></div>`; }
      tip.innerHTML = h; tip.style.display = 'block';
      const px = (ev.clientX - el.getBoundingClientRect().left);
      tip.style.left = Math.min(px + 14, el.clientWidth - tip.offsetWidth - 4) + 'px'; tip.style.top = '8px';
    });
    hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); });
  }
  charts.push(draw); draw();
}
function niceStep(raw){ const p = Math.pow(10, Math.floor(Math.log10(raw))); const n = raw / p; return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * p; }
const redraw = () => charts.forEach(d => d());

const lg = document.getElementById('legend');
for (const f of D.families) { const b = document.createElement('button'); b.setAttribute('aria-pressed', 'true');
  b.innerHTML = `<span class=sw style="background:var(${f.color})"></span>${f.label}`;
  b.onclick = () => { hidden.has(f.id) ? hidden.delete(f.id) : hidden.add(f.id); b.setAttribute('aria-pressed', !hidden.has(f.id)); redraw(); };
  lg.appendChild(b); }
{ const b = document.createElement('button'); b.setAttribute('aria-pressed', 'true');
  b.innerHTML = `<span class=sw style="background:transparent;border:2px solid var(--ref);width:9px;height:9px;border-radius:50%"></span>${D.paper.label}`;
  b.onclick = () => { hidden.has('paper') ? hidden.delete('paper') : hidden.add('paper'); b.setAttribute('aria-pressed', !hidden.has('paper')); redraw(); };
  lg.appendChild(b); }
for (const b of document.querySelectorAll('[data-x]')) b.onclick = () => {
  xMax = b.dataset.x === 'lr' ? D.lrMaxFrames * 1.05 : D.allMaxFrames * 1.02;
  document.querySelectorAll('[data-x]').forEach(o => o.setAttribute('aria-pressed', o === b)); redraw(); };

const pct = v => v == null ? '–' : Math.round(v * 100) + '%';
chart(document.getElementById('c-best'), {col: 'best_R', logY: true, yFmt: fmtN, paper: true});
chart(document.getElementById('c-rooms'), {col: 'rooms', yFmt: v => v == null ? '–' : String(Math.round(v))});
chart(document.getElementById('c-ret'), {col: 'ret_success', lrOnly: true, yFmt: pct});
chart(document.getElementById('c-share'), {col: 'return_share', lrOnly: true, yFmt: pct});

function reachChart(fam){
  const card = document.createElement('div'); card.className = 'card';
  const all = fam.runs.flatMap(r => r.pts);
  const md = Math.max(...all.map(p => p[0])), deep = all.filter(p => p[0] >= md - 2);
  const zero = all.filter(p => p[2] === 0).length, ok = all.filter(p => p[1] >= 0.5);
  const okd = ok.length ? Math.max(...ok.map(p => p[0])) : 0;
  card.innerHTML = `<h3 style=margin-top:0><span class=sw style="background:var(${fam.color})"></span>${fam.label}</h3>
    <div class=muted>Deepest landmark reached in ≥50% of tries: depth ${okd} of ${md}. Never reached once: ${zero} of ${all.length}.
    Deepest three depths: ${deep.reduce((a, p) => a + p[2], 0).toLocaleString()} arrivals in ${deep.reduce((a, p) => a + p[3], 0).toLocaleString()} tries.</div>`;
  const el = document.createElement('div'); el.className = 'chart'; card.appendChild(el);
  document.getElementById('reach-grid').appendChild(card);
  const W = 560, H = 260, m = {l: 44, r: 14, t: 10, b: 34}, ns = 'http://www.w3.org/2000/svg';
  const tip = document.createElement('div'); tip.className = 'tip'; el.appendChild(tip);
  const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); el.appendChild(svg);
  const X = d => m.l + (d / (md + 1)) * (W - m.l - m.r), Y = v => m.t + (1 - v) * (H - m.t - m.b);
  const add = (tag, at) => { const e = document.createElementNS(ns, tag); for (const k in at) e.setAttribute(k, at[k]); svg.appendChild(e); return e; };
  function draw(){
    svg.innerHTML = '';
    for (const v of [0, .25, .5, .75, 1]) { add('line', {x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: css('--grid')});
      add('text', {x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', class: 'ax', fill: css('--muted'), 'font-size': 11}).textContent = Math.round(v * 100) + '%'; }
    const st = niceStep(md / 6);
    for (let d = 0; d <= md; d += st) add('text', {x: X(d), y: H - m.b + 16, 'text-anchor': 'middle', fill: css('--muted'), 'font-size': 11}).textContent = d;
    add('text', {x: (W + m.l) / 2, y: H - 4, 'text-anchor': 'middle', fill: css('--muted'), 'font-size': 11}).textContent = 'landmark depth (→ newer, further from the start)';
    const col = css(fam.color), bg = css('--surface');
    fam.runs.forEach((r, si) => r.pts.forEach(p => {
      const jit = (si ? 0.18 : -0.18) * (W - m.l - m.r) / (md + 1);
      const c = add('circle', {cx: X(p[0]) + jit, cy: Y(p[1]), r: 4.5, fill: si ? bg : col, stroke: si ? col : bg, 'stroke-width': 2});
      c.style.cursor = 'default';
      c.addEventListener('mousemove', ev => { const b = el.getBoundingClientRect();
        tip.innerHTML = `seed ${r.seed} · depth ${p[0]}<br>${p[4]}<br><b>${(p[1] * 100).toFixed(1)}%</b> · ${p[2].toLocaleString()} of ${p[3].toLocaleString()} tries`;
        tip.style.display = 'block'; tip.style.left = Math.min(ev.clientX - b.left + 12, b.width - 260) + 'px'; tip.style.top = (ev.clientY - b.top - 60) + 'px'; });
      c.addEventListener('mouseleave', () => tip.style.display = 'none');
    }));
  }
  draw(); charts.push(draw);
}
D.reach.forEach(reachChart);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);
"""


def shell(title, body, script=""):
    return (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{E(title)}</title><style>{CSS}</style>{HUB}</head><body><main>{body}</main>{script}</body></html>")


def fmt(v, nd=0):
    if v is None:
        return "–"
    return f"{v:,.{nd}f}"


def eval_cell(fam):
    if fam["kind"] == "gowu":
        return "<span class=muted>Phase II not run</span>"
    if fam["kind"] == "gwrw":
        evs = [e for r in fam["runs"] for e in (r.get("eval") or [])]
        if not evs:
            return "<span class=muted>test pending</span>"
        return " / ".join(fmt(e["mean"]) for e in evs) + f"<div class=muted>max {fmt(max(e['max'] for e in evs))} · frozen, {evs[0]['n']} games</div>"
    caps = [e for r in fam["runs"] for e in (r.get("eval") or []) if e["cap_frames"]]
    if not caps:
        return "–"
    return " / ".join(fmt(e["mean"]) for e in caps) + f"<div class=muted>max {fmt(max(e['max'] for e in caps))}</div>"


def build(out: Path):
    reg = json.loads((RESEARCH / "registry.json").read_text())
    fams = []
    for k, f in enumerate(reg["baselines"] + reg["iterations"]):
        kind = f.get("kind", "lr")
        runs = []
        for r in f["runs"]:
            d = REPO / r["dir"]
            s = series(kind, d, r)
            if s is None:
                continue
            cfg = load_config(d) if kind != "gwrw" else json.loads((d / f"{r['prefix']}_live.json").read_text())
            s.update(seed=r["seed"], dir=str(d), config=cfg, continued=r.get("continued", []))
            if kind != "gowu":
                last = REPO / r["continued"][-1] if r.get("continued") else d
                s["reach"] = reach_gwrw(d, r["prefix"]) if kind == "gwrw" else reach_lr(last)
            runs.append(s)
        fams.append({**f, "kind": kind, "color": f"--s{((f.get('slot') or k + 1) - 1) % 8 + 1}", "step": True, "runs": runs})

    lr_max = max((r["rows"][-1][0] for f in fams if f["kind"] == "lr" for r in f["runs"]), default=1e8)
    all_max = max(r["rows"][-1][0] for f in fams for r in f["runs"])
    data = {"families": [{k: f[k] for k in ("id", "label", "kind", "color", "step")} |
                         {"runs": [{"seed": r["seed"], "cols": r["cols"], "rows": r["rows"]} for r in f["runs"]]} for f in fams],
            "paper": reg["paper"], "lrMaxFrames": lr_max, "allMaxFrames": all_max,
            "reach": [{"id": f["id"], "label": f["label"], "kind": f["kind"], "color": f["color"],
                       "runs": [{"seed": r["seed"], "pts": r["reach"]} for r in f["runs"] if r.get("reach")]}
                      for f in fams if f["kind"] != "gowu" and any(r.get("reach") for r in f["runs"])]}

    # ---- iteration table
    rows = []
    for f in fams:
        fin = [r["final"] for r in f["runs"]]
        best = max((x["best_R"] or 0) for x in fin) if fin else None
        rooms = max((x["rooms"] or 0) for x in fin) if fin else None
        frames = max((x["frames"] or 0) for x in fin) if fin else None
        arch = (f"<a href='/{out.name}/arch/{E(f['id'])}.html'>architecture →</a>" if f.get("arch") else
                "<span class=muted><code>docs/GOWU_REPRO.md</code></span>")
        rows.append(f"<tr><td><span class=sw style='background:var({f['color']})'></span><b>{E(f['label'])}</b>"
                    f"<div class=muted>{E(f['date'])} · <code>{E(f['commit'])}</code> · {len(f['runs'])} seed(s)</div></td>"
                    f"<td>{E(f.get('change') or f.get('about', ''))}</td>"
                    f"<td class=n>{fmt((frames or 0) / 1e6)}M</td><td class=n>{fmt(best)}</td><td class=n>{fmt(rooms)}</td>"
                    f"<td class=n>{eval_cell(f)}</td><td>{E(f.get('verdict', ''))}</td><td>{arch}</td></tr>")

    latest = next((f for f in reversed(fams) if f["kind"] == "lr" and f["runs"]), None)
    tiles = ""
    if latest:
        fin = [r["final"] for r in latest["runs"]]
        evs = [e for r in latest["runs"] for e in (r.get("eval") or []) if e["cap_frames"]]
        base = next(f for f in fams if f["kind"] == "gowu")
        bR = [stepAt(r["rows"], latest_frames := max(x["frames"] for x in fin)) for r in base["runs"]]
        bR = [v for v in bR if v is not None]
        tiles = "".join(f"<div class='card tile'><div class=k>{k}</div><div class=v>{v}</div><div class=muted>{s}</div></div>" for k, v, s in [
            ("Latest iteration", E(latest["label"]), f"<code>{E(latest['commit'])}</code> · <a href='/{out.name}/arch/{E(latest['id'])}.html'>architecture</a>"),
            ("Deployable score (eval mean)", " / ".join(fmt(e["mean"]) for e in evs) or "–", "π_R from reset, sticky actions, per seed"),
            ("Best reward in training", fmt(max(x["best_R"] for x in fin)), f"{max(x['rooms'] for x in fin)} rooms · {fmt(latest_frames / 1e6)}M frames"),
            ("GoWU Phase I at same frames", (f"{fmt(min(bR))}–{fmt(max(bR))}" if bR else "–"), f"best trajectory, restores states, {len(bR)} seeds"),
        ])

    # ---- head-to-head at equal frames (best R and rooms; range over seeds; – = no seed got that far)
    marks = [25e6, 50e6, 100e6, 250e6, 400e6]

    def at(rows, x, c):
        if not rows or rows[-1][0] < x * 0.98:
            return None
        v = None
        for r in rows:
            if r[0] > x:
                break
            v = r[c] if r[c] is not None else v
        return v

    def rng_cell(vals, rooms):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "<td class=n>–</td>"
        rr = [v for v in rooms if v is not None]
        top = fmt(min(vals)) if min(vals) == max(vals) else f"{fmt(min(vals))}–{fmt(max(vals))}"
        return f"<td class=n>{top}<div class=muted>{max(rr) if rr else '–'} rooms</div></td>"
    h2h = []
    for f in fams:
        if not f["runs"]:
            continue
        cells = "".join(rng_cell([at(r["rows"], x, 1) for r in f["runs"]], [at(r["rows"], x, 2) for r in f["runs"]]) for x in marks)
        h2h.append(f"<tr><td><span class=sw style='background:var({f['color']})'></span>{E(f['label'])}</td>{cells}<td class=n>{eval_cell(f)}</td></tr>")

    locs = []
    for f in fams:
        for r in f["runs"]:
            locs.append(f"<tr><td>{E(f['label'])} · s{r['seed']}</td><td><code>{E(r['dir'])}</code></td></tr>")

    body = f"""
<h1>LR-GoWU research</h1>
<div class=sub>Learned-Return GoWU on Montezuma's Revenge: GoWU's search, but no emulator restores. A goal-conditioned
policy has to walk back to every landmark, so the result is a deployable policy rather than a lucky trajectory.
Baseline: our GoWU reproduction (Mhammedi &amp; Cohan 2026). Plan: <code>docs/PLAN_LR_GOWU.md</code>.
<a href='/{out.name}/algorithm/'>Algorithm (LaTeX / PDF) →</a> · <a href='/montezuma-paths/'>Path map: where agents walk, die and stop →</a> · <a href='/lr-gowu/models/'>Networks + exploration compared (LR-GoWU · GWRW · GoWU) →</a></div>
<div class=tiles>{tiles}</div>

<h2>Training curves</h2>
<div class=ctrl>Frames shown: <button data-x=lr aria-pressed=true>LR-GoWU range</button><button data-x=all aria-pressed=false>All (to 400M)</button>
<span class=muted>Click a legend entry to hide it. One line per seed.</span></div>
<div class=legend id=legend></div>
<div class=grid>
<div class=card><h3 style=margin-top:0>Best reward found during training <span class=muted>(log scale)</span></h3>
<div class=muted>GoWU: best restored trajectory. LR-GoWU: best reward any episode reached from reset.</div><div class=chart id=c-best></div></div>
<div class=card><h3 style=margin-top:0>Rooms discovered</h3><div class=muted>Level-1 rooms visited by any particle / episode.</div><div class=chart id=c-rooms></div></div>
<div class=card><h3 style=margin-top:0>Return success <span class=muted>(LR-GoWU and GWRW)</span></h3>
<div class=muted>Share of trips back to the target that arrive. LR: rolling over ~20 log steps. GWRW: μ's arrivals over the last ~20M frames of finished games.
This is an average over every target, and most targets are easy, so it hides the frontier: see <a href='#reach'>arrival rate per landmark</a> below.</div><div class=chart id=c-ret></div></div>
<div class=card><h3 style=margin-top:0>Frames spent returning <span class=muted>(LR-GoWU and GWRW)</span></h3>
<div class=muted>Cumulative share of frames in the return phase (LR: π_R; GWRW: the route walker μ); the rest is exploration.</div><div class=chart id=c-share></div></div>
</div>

<h2 id=reach>Arrival rate at each landmark</h2>
<p class=muted style=max-width:80ch>Every landmark the return policy was sent to over the whole run, by how deep it is (LR: landmarks in the lineage tree from the start; GWRW: landmarks on the route). Height = arrivals ÷ attempts. Filled dot = first seed, ring = second seed. The dots on the right are the newest places; the dots on the floor were never reached. Hover a dot for the room, position and counts.</p>
<div class=grid id=reach-grid></div>

<h2>Iterations</h2>
<div class="card tw"><table><thead><tr><th>Run family</th><th>What changed</th><th>Frames</th><th>Best R (train)</th><th>Rooms</th>
<th>Eval mean (deployable)</th><th>Verdict</th><th>Architecture</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class=muted>Eval = π_R from <code>env.reset()</code>, sticky actions 0.25, 5 lives, 32 episodes, 30-min cap, one number per seed.
GoWU's comparable number would be its Phase II (robustified) score, which we have not run (paper: 182.7k at 15–20B frames).</p>

<h2>Head-to-head at equal frames</h2>
<div class="card tw"><table><thead><tr><th>Run family</th>{''.join(f"<th>{fmt(x / 1e6)}M</th>" for x in marks)}<th>Test (deployable)</th></tr></thead>
<tbody>{''.join(h2h)}</tbody></table></div>
<p class=muted>Best game score reached in training by that frame count (range over seeds), with the most rooms. GoWU restores emulator
states, so its number is a search result; LR-GoWU and GWRW play every game from reset. GWRW rooms are raw RAM room ids from its
diagnostics (all levels together); LR-GoWU counts level-1 rooms. Tests are not equal: LR-GoWU = π_R alone, 32 episodes, 30-min cap;
GWRW = frozen route controller to its deepest reproducible route, then its explorers, 192 games per run, its own game cap
(4,000 steps per scene seen, 8,000 after hand-off).</p>

<h2>Where the data is</h2>
<div class="card tw"><table><tbody>
<tr><td>Registry (what this page shows)</td><td><code>{E(str(RESEARCH / 'registry.json'))}</code></td></tr>
<tr><td>Algorithm (LaTeX)</td><td><code>{E(str(ALGO / 'lr_gowu.tex'))}</code></td></tr>
<tr><td>Architecture snapshots</td><td><code>{E(str(RESEARCH / 'arch'))}/&lt;id&gt;.md</code></td></tr>
<tr><td>Rebuild this page</td><td><code>python3 scripts/build_research_page.py</code> (in <code>{E(str(REPO))}</code>)</td></tr>
{''.join(locs)}
</tbody></table></div>
<p class=muted>Each run directory holds <code>log.jsonl</code> (training curve), <code>config.json</code>, <code>eval.json</code>,
<code>ckpt.pt</code>, <code>landmarks.json</code>. Built from branch <code>{E(git('rev-parse', '--abbrev-ref', 'HEAD').strip())}</code>
@ <code>{E(git('rev-parse', '--short', 'HEAD').strip())}</code>.</p>
"""
    script = (f"<script id=data type=application/json>{json.dumps(data, separators=(',', ':')).replace('</', '<\\/')}</script>"
              f"<script>{JS}</script>")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(shell("LR-GoWU research", body, script))

    # ---- architecture pages
    (out / "arch").mkdir(exist_ok=True)
    byid = {f["id"]: f for f in fams}
    last_of_kind = {}
    for f in fams:
        # diff against `compare_to` if given, else the previous family of the same kind
        prev = byid.get(f.get("compare_to")) or last_of_kind.get(f["kind"])
        last_of_kind[f["kind"]] = f
        if not f.get("arch"):
            continue
        src = RESEARCH / "arch" / f"{f['arch']}.md"
        doc = md(src.read_text()) if src.exists() else "<p>No snapshot written.</p>"
        cfg = f["runs"][0]["config"] if f["runs"] else {}
        ignore = {"out_dir", "seed", "max_hours"}
        diff = ""
        if prev is not None:
            pc = prev["runs"][0]["config"] if prev["runs"] else {}
            ch = [(k, pc.get(k), cfg.get(k)) for k in sorted(set(pc) | set(cfg)) if k not in ignore and pc.get(k) != cfg.get(k)]
            stat = git("diff", "--stat", f"{prev['commit']}..{f['commit']}", "--", "gowu", "scripts", "configs").strip()
            diff = (f"<h2>Changes since {E(prev['label'])}</h2><h3>Config</h3>" +
                    ("<div class=tw><table><tr><th>key</th><th>before</th><th>now</th></tr>" +
                     "".join(f"<tr><td><code>{E(k)}</code></td><td>{E(json.dumps(a))}</td><td>{E(json.dumps(b))}</td></tr>" for k, a, b in ch) +
                     "</table></div>" if ch else "<p class=muted>No config differences.</p>") +
                    f"<h3>Code (<code>git diff --stat {E(prev['commit'])}..{E(f['commit'])}</code>)</h3>" +
                    (f"<pre>{E(stat)}</pre>" if stat else "<p class=muted>Same commit.</p>"))
        seeds = ", ".join(f"s{r['seed']}: <code>{E(r['dir'])}</code>"
                          + "".join(f" → <code>{E(c)}</code>" for c in r.get("continued", [])) for r in f["runs"])
        body = (f"<p><a href='/{out.name}/'>← LR-GoWU research</a></p><h1>{E(f['label'])}</h1>"
                f"<div class=sub>{E(f['date'])} · commit <code>{E(f['commit'])}</code> · <span class=chip>{E(f['id'])}</span></div>"
                f"<p><b>What changed:</b> {E(f.get('change') or f.get('about', ''))}<br><b>Verdict:</b> {E(f.get('verdict', ''))}</p>"
                f"<p class=muted>Runs: {seeds}</p>"
                f"<div class='card doc'>{doc}</div>{diff}"
                f"<h2>Exact run config</h2><details><summary>config.json (seed {f['runs'][0]['seed'] if f['runs'] else '?'})</summary>"
                f"<pre>{E(json.dumps(cfg, indent=1))}</pre></details>")
        (out / "arch" / f"{f['id']}.html").write_text(shell(f"{f['label']} architecture", body))
    build_algorithm(out)
    return out


def build_algorithm(out: Path):
    """algorithm/: the LaTeX pseudo-algorithms, the compiled PDF (rebuilt with tectonic when the .tex is newer)."""
    tex, pdf, walk = ALGO / "lr_gowu.tex", ALGO / "lr_gowu.pdf", ALGO / "walkthrough.md"
    if shutil.which("tectonic") and (not pdf.exists() or pdf.stat().st_mtime < tex.stat().st_mtime):
        subprocess.run(["tectonic", "-r", "2", tex.name], cwd=ALGO, check=True, capture_output=True)
    d = out / "algorithm"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tex, d / tex.name)
    if pdf.exists():
        shutil.copy2(pdf, d / pdf.name)
    src = tex.read_text()
    m = re.search(r"commit \\texttt\{(\w+)\}", src)
    body = (f"<p><a href='/{out.name}/'>← LR-GoWU research</a></p><h1>LR-GoWU algorithm</h1>"
            f"<div class=sub>Pseudo-algorithms in the style of the GoWU paper's Algorithms 1–2, with terms and definitions"
            f", hyperparameters and a line-by-line walkthrough below the PDF. Describes commit <code>{E(m.group(1) if m else '?')}</code>.</div>"
            f"<div class=ctrl><a href='lr_gowu.pdf' target=_blank><button>Open PDF</button></a>"
            f"<a href='lr_gowu.pdf' download><button>Download PDF</button></a>"
            f"<a href='lr_gowu.tex' download><button>Download .tex</button></a>"
            f"<button id=copy>Copy LaTeX</button><span class=muted>Source: <code>{E(str(tex.relative_to(REPO)))}</code></span></div>"
            + (f"<iframe src='lr_gowu.pdf#view=FitH' title='LR-GoWU algorithm PDF' "
               f"style='width:100%;height:88vh;border:1px solid var(--line);border-radius:10px;background:#fff'></iframe>"
               if pdf.exists() else "<p class=muted>PDF not built (tectonic not found).</p>")
            + (f"<div class='card doc' style='margin-top:24px'>{md(walk.read_text())}</div>" if walk.exists() else "")
            + f"<h2>LaTeX source</h2><pre id=src>{E(src)}</pre>")
    script = ("<script>document.getElementById('copy').onclick=e=>navigator.clipboard.writeText("
              "document.getElementById('src').textContent).then(()=>{e.target.textContent='Copied';"
              "setTimeout(()=>e.target.textContent='Copy LaTeX',1500)})</script>")
    (d / "index.html").write_text(shell("LR-GoWU algorithm", body, script))


def stepAt(rows, x):
    v = None
    for r in rows:
        if r[0] > x:
            break
        v = r[1]
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    print(build(ap.parse_args().out.expanduser()))
