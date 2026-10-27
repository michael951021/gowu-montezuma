"""Phase-I campaigns, two seeds at a time; defaults: four fixed seeds x 400M. Native GPU queue only."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

from monitor_phase1 import atomic_json, inspect, records

REPO = Path(__file__).resolve().parents[1]


class EarlyStop(Exception):
    """Designed protocol stop: a pair-mean gate fired and the report is the deliverable."""


def terminate(proc):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)  # let the coordinator save its demo and close its workers
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def summarize(root, state):
    report = inspect(root)
    completed = [r for r in report["runs"] if r["status"] == "completed"]
    rewards = [r["best_R"] for r in completed]
    summary = {"status": state["status"], "reason": state.get("reason"), "completed_seeds": len(completed),
               "expected_seeds": len(state["seeds"]), "frames_per_seed": state.get("frames_per_seed", 400_000_000),
               "mean_reward_at_budget": statistics.mean(rewards) if rewards else None,
               "sample_std": statistics.stdev(rewards) if len(rewards) > 1 else None,
               "median": statistics.median(rewards) if rewards else None,
               "paper_mean": 98_752.5, "paper_mean_frames": 400_000_000, "paper_seeds": 100,
               "caveat": "Small-seed directional check; RAM exploit surrogate is not the authors' visual detector.",
               "runs": report["runs"]}
    sys.path.insert(0, str(root / "source"))
    summary["demo_audits"] = []
    for r in report["runs"]:
        path = root / r["out_dir"] / "demo_best.pkl.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rb") as f:
            demo = pickle.load(f)  # trusted output from this campaign's own frozen training code
        events = [event for node_events in demo["events"] for event in node_events]
        event_sum = sum(event[1] for event in events)
        consecutive, longest = 0, 0
        for event in events:
            consecutive = consecutive + 1 if event[1] == 1000 else 0
            longest = max(longest, consecutive)
        repeated = Counter((event[3][0], event[3][1], event[1]) for event in events)
        summary["demo_audits"].append(dict(seed=r["seed"], reward=demo["R"], event_reward_sum=event_sum,
            reward_sum_matches=abs(event_sum - demo["R"]) < 1e-6, n_reward_events=len(events),
            levels=sorted({ram[0] for ram in demo["ram"]}), longest_consecutive_1000_rewards=longest,
            treasure_loop_review_required=longest >= 30,
            most_repeated=[dict(level=k[0], room=k[1], reward=k[2], count=v) for k, v in repeated.most_common(5)]))
    if state.get("frames_per_seed", 400_000_000) == 400_000_000 and rewards:
        summary["mean_reward_at_400M"] = statistics.mean(rewards)
        summary["mean_ratio_to_paper"] = statistics.mean(rewards) / 98_752.5
    atomic_json(root / "summary.json", summary)
    # Each seed is visible; no best-seed selection or replacement after a failure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in report["runs"]:
        recs = records(root / r["out_dir"] / "log.jsonl")
        if recs:
            ax.plot([p["frames"] / 1e6 for p in recs], [p["best_R"] for p in recs], label=f"seed {r['seed']}")
    ax.axhline(98_752.5, color="black", linestyle="--", label="Paper mean at 400M (100 seeds)")
    ax.scatter([50, 100, 200], [25_000, 52_500, 80_000], color="black", marker="x",
               label="Approximate Fig. 6 reference means")
    ax.set(xlabel="Game frames (M)", ylabel="Best discovered cumulative reward", title=f"Phase I: {state['status']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "curves.png", dpi=160)
    plt.close(fig)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=REPO / "runs" / "phase1_validation")
    ap.add_argument("--monitor-interval", type=float, default=3600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--frames-per-seed", type=int, default=400_000_000)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="config overrides passed to every seed (variant under test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.seeds or len(args.seeds) % 2 or len(set(args.seeds)) != len(args.seeds) or args.frames_per_seed <= 0:
        ap.error("supply an even number of distinct seeds and a positive frame budget")
    if args.dry_run:
        print(json.dumps({"seeds": args.seeds, "frames_per_seed": args.frames_per_seed,
                          "gpus": [0, 1], "workers_per_seed": 7, "overrides": args.set,
                          "monitor_interval_seconds": args.monitor_interval, "phase2": False}, indent=2))
        return
    if os.environ.get("AGENT_JOB_GPU") != "1":
        raise SystemExit("Submit this command through the hub as a (gpu) job; AGENT_JOB_GPU=1 is required.")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "state.json").exists():
        previous = json.loads((root / "state.json").read_text())
        if previous["status"] == "completed":
            return
        if previous.get("stop") == "early_stop":
            # A designed early stop is a terminal, non-defective state; the report is the
            # deliverable. A hub retry must not re-arm the GPUs or read as a failed job.
            print(json.dumps({"existing_campaign": previous["status"], "stop": previous.get("stop"),
                              "reason": previous.get("reason")}, indent=2))
            return
        raise SystemExit("This campaign already has output. No automatic restart/resampling: inspect it and use a new output directory.")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise SystemExit("The queued job must expose exactly two CUDA GPUs.")
    source = root / "source"
    source.mkdir()
    for directory in ("gowu", "scripts", "configs", "docs", "tests"):
        shutil.copytree(REPO / directory, source / directory, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("requirements.txt", "README.md"):
        shutil.copy2(REPO / name, source / name)
    preflight = REPO / "runs" / "preflight"
    shutil.copytree(preflight, root / "preflight")
    revision = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    diff = subprocess.run(["git", "-C", str(REPO), "diff", "HEAD"], capture_output=True, text=True)
    (root / "source_changes.diff").write_text(diff.stdout)
    manifest = {"created": time.time(), "repo": str(REPO), "python": sys.version,
                "torch": torch.__version__, "devices": [torch.cuda.get_device_name(i) for i in range(2)],
                "base_revision": revision.stdout.strip(), "queue_job": os.environ.get("AGENT_JOB"),
                "packages": {p.metadata["Name"]: p.version for p in importlib.metadata.distributions()},
                "sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.rglob("*") if p.is_file()},
                "protocol": {"seeds": args.seeds, "frames_per_seed": args.frames_per_seed,
                             "config": "configs/gowu_mr.yaml", "overrides": args.set,
                             "early_stop": "paired mean below 20k at 100M", "monitor_seconds": args.monitor_interval}}
    atomic_json(root / "manifest.json", manifest)
    state = {"status": "running", "seeds": args.seeds, "frames_per_seed": args.frames_per_seed,
             "started": time.time(), "runs": [
        {"seed": s, "status": "pending", "out_dir": f"seed{s}"} for s in args.seeds]}
    atomic_json(root / "state.json", state)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    monitor_log = (root / "monitor.out").open("a")
    monitor = subprocess.Popen([sys.executable, str(source / "scripts" / "monitor_phase1.py"), "--root", str(root),
                                "--interval", str(args.monitor_interval)], env=env, stdout=monitor_log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    children, logs = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for start in range(0, len(args.seeds), 2):
            pair = args.seeds[start:start + 2]
            children = []
            for gpu, seed in enumerate(pair):
                run = next(r for r in state["runs"] if r["seed"] == seed)
                out = (root / f"seed{seed}.out").open("a")
                logs.append(out)
                cmd = [sys.executable, str(source / "scripts" / "run_gowu.py"), "--config", str(source / "configs" / "gowu_mr.yaml"),
                       "--set", f"seed={seed}", "n_workers=7", f"total_frames={args.frames_per_seed}",
                       f"out_dir={root / run['out_dir']}", "save_every_sec=300", *args.set]
                child = subprocess.Popen(cmd, cwd=source, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((child, run))
                run.update(status="running", started=time.time(), pid=child.pid, gpu=gpu)
            atomic_json(root / "state.json", state)
            while True:
                stop_file = root / "stop_requested.json"
                if stop_file.exists():
                    payload = json.loads(stop_file.read_text())
                    if payload.get("early_stop"):
                        raise EarlyStop(stop_file.read_text())
                    raise RuntimeError(stop_file.read_text())
                if monitor.poll() is not None:
                    raise RuntimeError(f"monitor exited unexpectedly: {monitor.returncode}; refusing unmonitored training")
                done = True
                for child, run in children:
                    rc = child.poll()
                    if rc is None:
                        done = False
                    elif run["status"] == "running":
                        run.update(status="completed" if rc == 0 else "failed", returncode=rc, ended=time.time())
                        recs = records(root / run["out_dir"] / "log.jsonl")
                        if rc != 0 or not recs or recs[-1].get("frames", 0) < args.frames_per_seed:
                            run["status"] = "failed"
                            raise RuntimeError(f"seed {run['seed']} failed or stopped before {args.frames_per_seed} frames; see seed{run['seed']}.out")
                atomic_json(root / "state.json", state)
                check = inspect(root)
                print(json.dumps({"time": time.time(), "progress": [
                    {k: r.get(k) for k in ("seed", "status", "frames", "best_R", "fps")} for r in check["runs"]]}), flush=True)
                if check["problems"]:
                    if check.get("early_stop"):
                        raise EarlyStop("; ".join(check["problems"]))
                    raise RuntimeError("; ".join(check["problems"]))
                if done:
                    break
                time.sleep(30)
        state["status"] = "completed"
    except (Exception, KeyboardInterrupt) as e:
        state.update(status="stopped", reason=str(e),
                     stop="early_stop" if isinstance(e, EarlyStop) else "failure")
        for child, run in children:
            terminate(child)
            if run["status"] == "running":
                run.update(status="stopped", returncode=child.returncode, ended=time.time())
    finally:
        state["ended"] = time.time()
        atomic_json(root / "state.json", state)
        try:
            monitor.wait(timeout=45)
        except subprocess.TimeoutExpired:
            terminate(monitor)
        monitor_log.close()
        for log in logs:
            log.close()
        print(json.dumps(summarize(root, state), indent=2), flush=True)
    if state["status"] != "completed":
        if state.get("stop") != "early_stop":
            raise SystemExit(1)
        print(json.dumps({"campaign": state["status"], "stop": state.get("stop"),
                          "note": "Designed early stop: the report is the deliverable; diagnose it, do not re-run."}, indent=2))


if __name__ == "__main__":
    main()
