"""Print the last log line of each run dir (phase I or II) and optionally plot phase-I curves.
    python scripts/summarize.py runs/gowu_mr_s* [--plot out.png]
"""
import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--plot", default=None)
args = ap.parse_args()
curves = {}
for r in args.runs:
    p = os.path.join(r, "log.jsonl")
    if not os.path.exists(p):
        continue
    recs = [json.loads(line) for line in open(p) if line.strip()]
    if not recs:
        continue
    last = recs[-1]
    if "best_R" in last:
        curves[r] = [(x["frames"], x["best_R"]) for x in recs]
        print(f"{r}: {last['frames']/1e6:.1f}M frames  best_R={last['best_R']:.0f}  rooms={last['rooms']} "
              f"max_level={last['max_level']}  fps={last['fps']}  exploits={last['exploits']}  tree={last['tree_nodes']}")
    else:
        print(f"{r}: {last['frames']/1e6:.1f}M frames  solved={last['n_solved']}/{last['n_segments']} "
              f"progress={last['progress']}  success={last['success']}  fps={last['fps']}")
    ev = os.path.join(r, "eval_watch.jsonl")
    if os.path.exists(ev):
        evs = [json.loads(line) for line in open(ev) if line.strip()]
        if evs:
            b = max(evs, key=lambda e: e["mean"])
            print(f"    eval: last={evs[-1]['mean']:.0f}  best={b['mean']:.0f} @ {b['frames']/1e6:.0f}M  "
                  f"(cap={evs[-1]['max_episode_frames']})")
if curves:
    import numpy as np
    finals = [c[-1][1] for c in curves.values()]
    print(f"phase I: n={len(finals)} mean best_R={np.mean(finals):.0f}  (paper: 98,752 at 400M frames, 100 seeds)")
if args.plot and curves:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for r, c in curves.items():
        f, R = zip(*c)
        plt.plot(np.array(f) / 1e6, R, alpha=0.6, label=os.path.basename(r))
    plt.axhline(98752.5, ls="--", c="k", lw=1, label="GoWU paper @400M")
    plt.xlabel("game frames (M)")
    plt.ylabel("best cumulative reward")
    plt.legend(fontsize=6)
    plt.savefig(args.plot, dpi=130, bbox_inches="tight")
    print("saved", args.plot)
