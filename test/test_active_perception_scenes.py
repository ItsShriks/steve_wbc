"""
Headless MuJoCo scenarios for active_perception_wbc (integration tests, ~1 min each).

    cd src/steve_wbc && python3 -m pytest test/test_active_perception_scenes.py -v

  single   rolling red ball, grasped from the front
  clutter  red cube among static obstacles: only the cube is picked, the robot must not touch
           an obstacle (the gripper flies over the bottle in the way)
  far      cube out of reach from the front: the robot drives around the table
  control  clutter WITHOUT collision avoidance / transit: must hit an obstacle — proves the
           obstacles really block the naive gripper path

Every run must also keep the base footprint >= --d_safe_base (15 cm) from the environment.
VGN is off to keep the tests fast and independent of the vgn_env conda environment.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

SCRIPT = Path(__file__).resolve().parents[1] / "steve_wbc" / "active_perception_wbc.py"
D_SAFE_BASE_CM = 15.0


def run_scene(*args, timeout=600):
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    p = subprocess.run([sys.executable, str(SCRIPT), "--mode", "sim", "--headless", "--no_vgn", *args],
                       capture_output=True, text=True, timeout=timeout, env=env)
    out = p.stdout + p.stderr
    result = re.search(r"Result: (.+?)  \(final state", out)
    base = re.search(r"Min base clearance: ([\d.]+) cm", out)
    return dict(code=p.returncode, out=out, result=result.group(1) if result else None,
                base_clr=float(base.group(1)) if base else None,
                contacts="[COLLISION]" in out)


def check_success(r):
    assert r["result"] == "success", f"result: {r['result']}\n{r['out'][-3000:]}"
    assert r["code"] == 0
    assert not r["contacts"], "robot touched an obstacle"
    assert r["base_clr"] is not None and r["base_clr"] >= D_SAFE_BASE_CM, \
        f"base came within {r['base_clr']} cm of the environment"


def test_single_moving_ball():
    check_success(run_scene("--scene", "single"))


def test_clutter_picks_cube_without_touching_obstacles():
    r = run_scene("--scene", "clutter")
    check_success(r)
    assert "Obstacle contacts: none" in r["out"]


@pytest.mark.skip(reason="far scene paused for now (quadrant-based approach sides replace it)")
def test_far_cube_drives_around_the_table():
    r = run_scene("--scene", "far")
    check_success(r)
    assert "REPOSITION → SETTLE_ARM" in r["out"], "expected the base to drive around the table"


def test_control_without_collision_avoidance_hits_an_obstacle():
    r = run_scene("--scene", "clutter", "--no_collision", "--no_transit")
    assert r["contacts"], "the naive path should collide — otherwise the clutter test proves nothing"
    assert r["code"] != 0
