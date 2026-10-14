"""Runs active_perception_wbc in MuJoCo with VGN's verdict forced to 'no grasp' for the first
FAIL inferences (FAIL=all: always) — exercises the explore-around-the-table logic
(test_vgn_explore.py).   FAIL=1 python3 vgn_explore_runner.py --mode sim --headless ..."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steve_wbc"))
import active_perception_wbc as ap  # noqa: E402

FAIL = os.environ.get("FAIL", "1")
_calls = {"n": 0}
_real_best = ap.VGNInference.vgn_best


def forced_best(self):
    _calls["n"] += 1
    if FAIL == "all" or _calls["n"] <= int(FAIL):
        print(f"  [test] VGN inference #{_calls['n']} forced to 'no grasp'")
        return None, 0.5
    return _real_best(self)


ap.VGNInference.vgn_best = forced_best
sys.exit(ap.main(sys.argv[1:]))
