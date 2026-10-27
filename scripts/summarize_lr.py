"""Summarize LR-GoWU runs under a directory: last log record, landmark stats and evaluation per run."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = {}
for d in sorted(p for p in root.iterdir() if (p / "log.jsonl").exists()):
    recs = [json.loads(line) for line in open(d / "log.jsonl")]
    last = recs[-1]
    cfg = json.load(open(d / "config.json"))
    curve = {}
    for r in recs:
        curve[int(r["frames"] // 25e6) * 25] = (r["max_R"], r["rooms"], r["max_level"])
    lms = json.load(open(d / "landmarks.json")) if (d / "landmarks.json").exists() else []
    out[d.name] = dict(
        overrides={k: cfg[k] for k in ("archive_key", "goal_encoder", "select_rule", "explore_policy", "seed")},
        frames=last["frames"], fps=last["fps"], hours=round(last["time"] / 3600, 2),
        max_R=last["max_R"], max_episode_R=last["max_episode_R"], rooms=last["rooms"], max_level=last["max_level"],
        rooms_list=last["rooms_list"], n_landmarks=last["n_landmarks"],
        return_frame_share=round(last["frames_return"] / max(last["frames"], 1), 3),
        reached_landmarks=sum(1 for lm in lms if lm["n_reached"] > 0),
        max_depth_reached=max((lm["depth"] for lm in lms if lm["n_reached"] > 0), default=0),
        curve_maxR_rooms_level_by_25M=curve,
        eval=json.load(open(d / "eval.json")) if (d / "eval.json").exists() else None)
    for e in out[d.name]["eval"] or []:
        e.pop("scores", None)
json.dump(out, open(root / "summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
