"""Compare the preserved baseline with a controlled rerun at the SAME frame budget."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from monitor_phase1 import records


def compare(baseline, candidate, frames=100_000_000, seeds=(0, 1)):
    rows = []
    for seed in seeds:
        row = {"seed": seed}
        for name, root in (("baseline", baseline), ("candidate", candidate)):
            hit = next((r for r in records(Path(root) / f"seed{seed}" / "log.jsonl") if r["frames"] >= frames), None)
            row[name] = ({k: hit.get(k) for k in ("frames", "best_R", "max_level", "rooms", "fps", "global_U")}
                         if hit else None)
        rows.append(row)
    complete = all(r["baseline"] and r["candidate"] for r in rows)
    result = dict(frames=frames, seeds=list(seeds), rows=rows, complete=complete,
                  verdict="Awaiting both seeds at the declared frame budget.")
    if complete:
        old = statistics.mean(r["baseline"]["best_R"] for r in rows)
        new = statistics.mean(r["candidate"]["best_R"] for r in rows)
        result.update(baseline_mean=old, candidate_mean=new, reward_ratio=new / old if old else None)
        if new < 20_000:
            result["verdict"] = "The replay correction did not resolve the gross exploration failure. Do not start the full-budget reproduction."
        elif new < 40_000 or any((r["candidate"]["max_level"] or 0) < 1 for r in rows):
            result["verdict"] = "Still short of the paper's approximate 100M curve or level progression. More diagnosis is needed before a full-budget reproduction."
        else:
            result["verdict"] = "The controlled rerun clears the gross failure and approaches the paper's 100M curve. Four full-budget seeds and exploit audits are still required."
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=Path("runs/phase1_validation"))
    ap.add_argument("--candidate", type=Path, default=Path("runs/phase1_replay_probe"))
    ap.add_argument("--output", type=Path, default=Path("runs/diagnosis"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = compare(args.baseline, args.candidate)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Controlled replay correction comparison", "", "100M game frames per seed; same seeds, hyperparameters, held-action policy and exploit filter.",
             "Only pre-action replay data alignment is changed. These debugging seeds are not an independent confirmation sample.", "",
             "| Seed | Baseline reward | Corrected replay reward | Corrected max level (starts at 0) |", "|---|---:|---:|---:|"]
    for row in result["rows"]:
        a, b = row["baseline"], row["candidate"]
        lines.append(f"| {row['seed']} | {a['best_R'] if a else 'pending'} | {b['best_R'] if b else 'pending'} | {b['max_level'] if b else 'pending'} |")
    lines += ["", result["verdict"], "", "Paper Fig. 6 mean at 100M is visually about 45–60k. Its 98,752.5 Table 2 mean is at 400M and is not a 100M target.",
              "No conclusion here proves exact reproduction or validates phase II."]
    (args.output / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
