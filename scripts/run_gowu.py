"""Phase I: GoWU exploration.  python scripts/run_gowu.py --config configs/gowu_mr.yaml [--set seed=3 ...]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gowu.config_io import load_config  # noqa: E402
from gowu.explore import GoWU, GowuConfig  # noqa: E402

if __name__ == "__main__":
    cfg = load_config(GowuConfig)
    GoWU(cfg).run()
