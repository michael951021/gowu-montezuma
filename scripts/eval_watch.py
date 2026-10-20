"""Background evaluator (GoWU App. G.6): poll <run>/ckpt_latest.pt, run 10 greedy episodes from reset,
keep the 100 best checkpoints in <run>/top/.  Run on the second GPU next to training:

    CUDA_VISIBLE_DEVICES=1 python scripts/eval_watch.py --run runs/backward_mr

NOTE: default cap is the standard 108k frames; pass --uncapped for GoWU-style evaluation.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from gowu.backward.evaluate import DEFAULT_MAX_EPISODE_FRAMES, evaluate, load_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--keep", type=int, default=100)
    ap.add_argument("--poll", type=float, default=30)
    ap.add_argument("--uncapped", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    top_dir = os.path.join(args.run, "top")
    os.makedirs(top_dir, exist_ok=True)
    index_path = os.path.join(top_dir, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else []
    latest = os.path.join(args.run, "ckpt_latest.pt")
    last_mtime = 0
    cap = None if args.uncapped else DEFAULT_MAX_EPISODE_FRAMES
    log = open(os.path.join(args.run, "eval_watch.jsonl"), "a")
    while True:
        if os.path.exists(latest) and os.path.getmtime(latest) > last_mtime:
            last_mtime = os.path.getmtime(latest)
            snap = os.path.join(top_dir, "_candidate.pt")
            shutil.copy(latest, snap)
            ck = torch.load(snap, map_location="cpu", weights_only=False)
            res = evaluate(load_policy(snap, dev), args.episodes, seed=int(time.time()) % 100000, device=dev,
                           max_episode_frames=cap, n_workers=args.workers)
            rec = dict(frames=ck["frames"], updates=ck["updates"], **res)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if len(index) < args.keep or res["mean"] > min(e["mean"] for e in index):
                name = f"u{ck['updates']:07d}.pt"
                os.replace(snap, os.path.join(top_dir, name))
                index.append(dict(file=name, **rec))
                index.sort(key=lambda e: -e["mean"])
                for e in index[args.keep:]:
                    p = os.path.join(top_dir, e["file"])
                    if os.path.exists(p):
                        os.remove(p)
                index = index[:args.keep]
                json.dump(index, open(index_path, "w"), indent=1)
        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
