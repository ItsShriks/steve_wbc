"""ObjectTracker with odometry drift while the base drives (no simulation):
    cd src/steve_wbc && python3 -m pytest test/test_tracker_drift.py -v

The base drives up to the table at 0.25 m/s while its odometry heading drifts 3 deg per metre: a static object
~2 m away seems to move sideways ~5 cm in the 4 s fit window — over the old 4 cm bar, as in the --sim_realism run of
2026-10-30 (ball fitted rolling 1.9 cm/s; there heading random walk + depth bias). Detections every 60 ms, 3 mm noise.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steve_wbc"))
from active_perception_wbc import ObjectTracker  # noqa: E402

BALL = np.array([2.05, -0.25, 0.825])


def run(v_obj, drive=True, drift_deg_per_m=3.0, allowance=True, dynamic=False, T=4.0, seed=0):
    """Feed detections; returns (tracker, [(t, |v|)])."""
    rng = np.random.default_rng(seed)
    tr = ObjectTracker(dynamic)
    out = []
    for k in range(int(T / 0.06)):
        t = 0.06 * k
        base_true = np.array([0.25 * t if drive else 0.4, 0.0, 0.0])
        dpsi = np.radians(drift_deg_per_m) * base_true[0] if drive else 0.0   # odometry heading error so far
        obj = BALL + np.asarray(v_obj) * t
        # the robot's (drifted) frame: everything rotated by the heading error about the base
        c, s = np.cos(dpsi), np.sin(dpsi)
        rel = obj[:2] - base_true[:2]
        seen = np.array([*(base_true[:2] + [c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]), obj[2]])
        seen += rng.normal(0, 0.003, 3)
        tr.update(seen, t, True, np.array([base_true[0], base_true[1], dpsi]) if allowance else None)
        out.append((t, float(np.linalg.norm(tr.v[:2]))))
    return tr, out


def test_static_object_stays_static_while_driving():
    tr, _ = run([0, 0, 0])
    assert tr.fitted and not np.any(tr.v), f"fake motion {tr.v}, allowance {tr.allowance}"


def test_without_the_allowance_the_drift_fakes_motion():
    """The failure the allowance fixes (the old tracker, no base poses)."""
    tr, _ = run([0, 0, 0], allowance=False)
    assert np.linalg.norm(tr.v[:2]) > 0.01


def test_rolling_object_is_still_detected_while_driving():
    tr, out = run([0, 0.03, 0], T=5.0)
    assert np.linalg.norm(tr.v[:2]) > 0.02
    first = next(t for t, n in out if n > 0.02)
    assert first < 5.0


@pytest.mark.parametrize("v,moving", [([0, 0, 0], False), ([0, 0.02, 0], True)])
def test_standing_still_unchanged(v, moving):
    """Base standing: no allowance — the same decisions as before."""
    tr, _ = run(v, drive=False)
    assert tr.allowance[0] == 0.0 and tr.allowance[1] == 0.0
    assert bool(np.any(tr.v)) == moving


@pytest.mark.parametrize("v", [[0.02, 0, 0], [-0.02, 0, 0], [0, 0.03, 0]])
def test_rolling_along_or_across_the_line_of_sight(v):
    """Rolling away from / towards the robot (along the line of sight, which heading drift cannot fake) is held to
    the small bar; across it, 3 cm/s, to the larger one — both found while the base still drives."""
    tr, out = run(v, T=5.0)
    assert np.linalg.norm(tr.v[:2]) > 0.015, f"{tr.v}, allowance {tr.allowance}"
