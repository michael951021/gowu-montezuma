"""Independent phase-I watchdog and periodic report writer (no GPU computation)."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path

MILESTONES = {50_000_000: (20_000, 30_000), 100_000_000: (45_000, 60_000),
              200_000_000: (75_000, 85_000), 400_000_000: (98_752.5, 98_752.5)}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def records(path):
    try:
        lines = Path(path).read_text().splitlines()
    except FileNotFoundError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # the trainer may currently be writing the final line
    return out


def inspect(root):
    root = Path(root)
    state = json.loads((root / "state.json").read_text())
    frame_budget = state.get("frames_per_seed", 400_000_000)
    rows, problems, warnings = [], [], []
    for run in state["runs"]:
        log = root / run["out_dir"] / "log.jsonl"
        recs = records(log)
        latest = recs[-1] if recs else {}
        row = {**run, **{k: latest.get(k) for k in ("frames", "best_R", "max_level", "rooms", "fps", "rnd_loss", "exploits",
                                                  "global_U", "global_winner_in_population", "rnd_updates_total")}}
        fps = latest.get("fps")
        row["remaining_hours"] = (max(0, frame_budget - latest.get("frames", 0)) / fps / 3600
                                  if fps and math.isfinite(fps) and fps > 0 else None)
        row["milestones"] = {}
        for frames, reference in MILESTONES.items():
            hit = next((r for r in recs if r.get("frames", 0) >= frames), None)
            if hit:
                row["milestones"][str(frames)] = {"best_R": hit["best_R"], "reference_mean": reference}
        if run["status"] == "running":
            for key in ("best_R", "fps", "rnd_loss", "global_U"):
                if latest.get(key) is not None and not math.isfinite(latest[key]):
                    problems.append(f"seed {run['seed']}: non-finite {key}")
            last_output = log.stat().st_mtime if log.exists() else run["started"]
            if time.time() - last_output > 900:
                problems.append(f"seed {run['seed']}: no training progress for 15 minutes")
        if run.get("returncode") not in (None, 0):
            problems.append(f"seed {run['seed']}: exited {run['returncode']}")
        rows.append(row)
    # Compare each predetermined pair at the SAME frame budget. This is a deliberately loose
    # failure gate, not a statistical equivalence test or a way to select the best seeds.
    early_stop = False
    health_problems = bool(problems)
    for start in range(0, len(state["seeds"]), 2):
        pair = state["seeds"][start:start + 2]
        paired = [r for r in rows if r["seed"] in pair]
        for frames, floor in ((50_000_000, 10_000), (100_000_000, 20_000)):
            reached = [r["milestones"].get(str(frames)) for r in paired]
            if len(paired) == len(pair) and all(reached):
                mean = statistics.mean(r["best_R"] for r in reached)
                if mean < floor:
                    text = f"seeds {pair}: mean reward {mean:.0f} at {frames/1e6:.0f}M is below {floor}"
                    if frames == 100_000_000:
                        problems.append(text)
                        early_stop = not health_problems
                    else:
                        warnings.append(text)
    result = {"checked_at": time.time(), "status": state["status"], "runs": rows,
              "problems": problems, "warnings": warnings, "early_stop": early_stop}
    atomic_json(root / "status.json", result)
    lines = ["# Phase I validation status", "", f"Campaign: **{state['status']}**. Last check: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}.", "",
             "| Seed | State | Frames | Best reward | Max level (RAM, starts at 0) | FPS | Remaining hours |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        eta = f"{r['remaining_hours']:.2f}" if r["remaining_hours"] is not None else "pending"
        lines.append(f"| {r['seed']} | {r['status']} | {(r.get('frames') or 0)/1e6:.1f}M | {r.get('best_R')} | {r.get('max_level')} | {r.get('fps')} | {eta} |")
    lines += ["", "Reference means visually estimated from Fig. 6: 50M 20–30k; 100M 45–60k; 200M 75–85k.",
              "Table 2: mean 98,752.5 at 400M over 100 seeds. These are not per-seed acceptance bounds.", ""]
    lines += [f"- STOP: {p}" for p in problems] + [f"- REVIEW: {w}" for w in warnings]
    status = root / f"STATUS.{os.getpid()}.tmp"
    status.write_text("\n".join(lines) + "\n")
    status.replace(root / "STATUS.md")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=3600)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    next_report, seen = 0.0, set()
    while True:
        report = inspect(args.root)
        milestones = {(r["seed"], m) for r in report["runs"] for m in r["milestones"]}
        if args.once or time.time() >= next_report or milestones - seen or report["problems"]:
            try:
                gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10)
                report["gpus"] = gpu.stdout.strip() or gpu.stderr.strip()
            except (OSError, subprocess.TimeoutExpired) as e:
                report["gpus"] = str(e)
            with (args.root / "monitor.jsonl").open("a") as f:
                f.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            next_report = time.time() + args.interval
            seen = milestones
        if report["problems"] and report["status"] == "running":
            atomic_json(args.root / "stop_requested.json", {"reason": report["problems"],
                                                            "early_stop": report["early_stop"], "time": time.time()})
        if args.once or report["status"] != "running":
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
