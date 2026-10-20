"""Phase II: backward-algorithm PPO.  python scripts/run_backward.py --config configs/backward_mr.yaml"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gowu.backward.train import BackwardConfig, BackwardTrainer  # noqa: E402
from gowu.config_io import load_config  # noqa: E402

if __name__ == "__main__":
    BackwardTrainer(load_config(BackwardConfig)).run()
