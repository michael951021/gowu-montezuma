"""Final evaluation (GoWU App. G.6): every checkpoint in <run>/top/ gets 500 rollouts; report the best one
AND the final checkpoint, because picking the max of ~100 noisy estimates inflates scores.

    python scripts/eval_final.py --run runs/backward_mr [--uncapped] [--episodes 500]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from gowu.backward.evaluate import DEFAULT_MAX_EPISODE_FRAMES, evaluate, load_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--uncapped", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--envs", type=int, default=100)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cap = None if args.uncapped else DEFAULT_MAX_EPISODE_FRAMES
    top = os.path.join(args.run, "top")
    files = [e["file"] for e in json.load(open(os.path.join(top, "index.json")))] if os.path.isdir(top) else []
    cands = [os.path.join(top, f) for f in files] + [os.path.join(args.run, "ckpt_latest.pt")]
    results = []
    for i, path in enumerate(cands):
        res = evaluate(load_policy(path, dev), args.episodes, seed=1_000_000 + i, device=dev,
                       max_episode_frames=cap, n_workers=args.workers, n_envs=args.envs)
        results.append(dict(ckpt=path, **res))
        print(json.dumps(results[-1]), flush=True)
    best = max(results[:-1] or results, key=lambda r: r["mean"])
    summary = dict(selected=best, final=results[-1], cap_frames=cap,
                   note="GoWU reports uncapped scores; capped and uncapped are not comparable")
    tag = "uncapped" if cap is None else "capped"
    json.dump(dict(summary=summary, all=results), open(os.path.join(args.run, f"eval_final_{tag}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
