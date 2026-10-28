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
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESEARCH = REPO / "research" / "lr_gowu"
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


def lr_series(run_dir: Path):
    recs = read_log(run_dir / "log.jsonl")
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
                      "landmarks": last.get("n_landmarks"), "fps": last.get("fps"), "hours": last.get("time", 0) / 3600,
                      "timing": last.get("timing")},
            "eval": ev}


def load_config(run_dir: Path):
    p = run_dir / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


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
            if not (d / "log.jsonl").exists():
                continue
            s = gowu_series(d) if kind == "gowu" else lr_series(d)
            s.update(seed=r["seed"], dir=str(d), config=load_config(d))
            runs.append(s)
        fams.append({**f, "kind": kind, "color": f"--s{((f.get('slot') or k + 1) - 1) % 8 + 1}", "step": True, "runs": runs})

    lr_max = max((r["rows"][-1][0] for f in fams if f["kind"] != "gowu" for r in f["runs"]), default=1e8)
    all_max = max(r["rows"][-1][0] for f in fams for r in f["runs"])
    data = {"families": [{k: f[k] for k in ("id", "label", "kind", "color", "step")} |
                         {"runs": [{"seed": r["seed"], "cols": r["cols"], "rows": r["rows"]} for r in f["runs"]]} for f in fams],
            "paper": reg["paper"], "lrMaxFrames": lr_max, "allMaxFrames": all_max}

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

    latest = next((f for f in reversed(fams) if f["kind"] != "gowu" and f["runs"]), None)
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

    locs = []
    for f in fams:
        for r in f["runs"]:
            locs.append(f"<tr><td>{E(f['label'])} · s{r['seed']}</td><td><code>{E(r['dir'])}</code></td></tr>")

    body = f"""
<h1>LR-GoWU research</h1>
<div class=sub>Learned-Return GoWU on Montezuma's Revenge: GoWU's search, but no emulator restores. A goal-conditioned
policy has to walk back to every landmark, so the result is a deployable policy rather than a lucky trajectory.
Baseline: our GoWU reproduction (Mhammedi &amp; Cohan 2026). Plan: <code>docs/PLAN_LR_GOWU.md</code>.</div>
<div class=tiles>{tiles}</div>

<h2>Training curves</h2>
<div class=ctrl>Frames shown: <button data-x=lr aria-pressed=true>LR-GoWU range</button><button data-x=all aria-pressed=false>All (to 400M)</button>
<span class=muted>Click a legend entry to hide it. One line per seed.</span></div>
<div class=legend id=legend></div>
<div class=grid>
<div class=card><h3 style=margin-top:0>Best reward found during training <span class=muted>(log scale)</span></h3>
<div class=muted>GoWU: best restored trajectory. LR-GoWU: best reward any episode reached from reset.</div><div class=chart id=c-best></div></div>
<div class=card><h3 style=margin-top:0>Rooms discovered</h3><div class=muted>Level-1 rooms visited by any particle / episode.</div><div class=chart id=c-rooms></div></div>
<div class=card><h3 style=margin-top:0>Return success <span class=muted>(LR only)</span></h3>
<div class=muted>Share of trips back to the goal landmark that arrive, rolling over ~20 log steps.</div><div class=chart id=c-ret></div></div>
<div class=card><h3 style=margin-top:0>Frames spent returning <span class=muted>(LR only)</span></h3>
<div class=muted>Cumulative share of frames in the return phase; the rest is exploration.</div><div class=chart id=c-share></div></div>
</div>

<h2>Iterations</h2>
<div class="card tw"><table><thead><tr><th>Run family</th><th>What changed</th><th>Frames</th><th>Best R (train)</th><th>Rooms</th>
<th>Eval mean (deployable)</th><th>Verdict</th><th>Architecture</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class=muted>Eval = π_R from <code>env.reset()</code>, sticky actions 0.25, 5 lives, 32 episodes, 30-min cap, one number per seed.
GoWU's comparable number would be its Phase II (robustified) score, which we have not run (paper: 182.7k at 15–20B frames).</p>

<h2>Where the data is</h2>
<div class="card tw"><table><tbody>
<tr><td>Registry (what this page shows)</td><td><code>{E(str(RESEARCH / 'registry.json'))}</code></td></tr>
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
        seeds = ", ".join(f"s{r['seed']}: <code>{E(r['dir'])}</code>" for r in f["runs"])
        body = (f"<p><a href='/{out.name}/'>← LR-GoWU research</a></p><h1>{E(f['label'])}</h1>"
                f"<div class=sub>{E(f['date'])} · commit <code>{E(f['commit'])}</code> · <span class=chip>{E(f['id'])}</span></div>"
                f"<p><b>What changed:</b> {E(f.get('change') or f.get('about', ''))}<br><b>Verdict:</b> {E(f.get('verdict', ''))}</p>"
                f"<p class=muted>Runs: {seeds}</p>"
                f"<div class='card doc'>{doc}</div>{diff}"
                f"<h2>Exact run config</h2><details><summary>config.json (seed {f['runs'][0]['seed'] if f['runs'] else '?'})</summary>"
                f"<pre>{E(json.dumps(cfg, indent=1))}</pre></details>")
        (out / "arch" / f"{f['id']}.html").write_text(shell(f"{f['label']} architecture", body))
    return out


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
