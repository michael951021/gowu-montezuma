"""LR-GoWU.  python scripts/run_lr_gowu.py --config configs/lr_gowu_mr.yaml [--set seed=3 archive_key=xy ...]"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gowu.config_io import load_config  # noqa: E402
from gowu.lr.trainer import LRConfig, LRGoWU  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(LRConfig)
    agent = LRGoWU(cfg)
    try:
        agent.run()
        if cfg.eval_episodes > 0:
            for r in agent.final_eval():
                print("eval", json.dumps({k: v for k, v in r.items() if k != "scores"}), flush=True)
    finally:
        agent.pool.close()
