"""TableLocalizer on a synthetic LiDAR scan of the sim's table (skirt + legs), no simulation:
    cd src/steve_wbc && python3 -m pytest test/test_table_localizer.py -v
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steve_wbc"))
from table_localizer import TableLocalizer, _rot  # noqa: E402

I2 = np.eye(2)
# the sim table (models/mmo_700_wbc.xml) at (2, 0): skirt boxes + legs (cylinders as boxes)
RECTS = [(np.array([2 - 0.38, 0.0]), I2, np.array([0.02, 0.35])), (np.array([2 + 0.38, 0.0]), I2, np.array([0.02, 0.35])),
         (np.array([2.0, 0.33]), I2, np.array([0.38, 0.02])), (np.array([2.0, -0.33]), I2, np.array([0.38, 0.02]))] + \
        [(np.array([2 + x, y]), I2, np.array([0.025, 0.025])) for x in (-0.35, 0.35) for y in (-0.30, 0.30)]


def scan(base_true, noise=0.012, drop=0.01, seed=0):
    """360 rays from the base at the table's outline (true world, slab ray-box intersection), points in the base
    frame with range noise and dropouts."""
    rng = np.random.default_rng(seed)
    pts = []
    for a in np.radians(np.arange(0, 360, 1.0)):
        u = np.array([np.cos(a + base_true[2]), np.sin(a + base_true[2])])
        best = np.inf
        for c, _, h in RECTS:                            # axis-aligned boxes
            lo, hi = c - h - base_true[:2], c + h - base_true[:2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = lo / u, hi / u
            tn, tf = np.max(np.minimum(t1, t2)), np.min(np.maximum(t1, t2))
            if tn <= tf and tn > 0.05:
                best = min(best, tn)
        if np.isfinite(best) and rng.random() > drop:
            pts.append(_rot(-base_true[2]) @ ((best + rng.normal(0, noise)) * u))
    return np.array(pts)


def odom_of(base_true, dth, d):
    """The drifted odometry pose: odom = R(-dth)(true - d) (the correction C = (dth, d) maps odom -> true)."""
    b = np.array(base_true, float)
    o = b.copy()
    o[:2] = _rot(-dth) @ (b[:2] - d)
    o[2] = b[2] - dth
    return o


def test_converges_to_the_drift():
    base = np.array([1.35, -0.75, 0.6])                   # at the table's front-right corner: two faces in view
    dth, d = np.radians(1.2), np.array([0.03, -0.02])
    loc = TableLocalizer(RECTS)
    for k in range(60):                                   # 12 s of updates at the same pose, a fresh scan each
        loc.update(0.21 * k, odom_of(base, dth, d), scan(base, seed=k))
    corr = loc.base(odom_of(base, dth, d))
    # the drift (3.6 cm, 1.2 deg) down to the scan noise (~60 beams on the table, 1.2 cm each, averaged)
    assert np.linalg.norm(corr[:2] - base[:2]) < 0.010, corr
    assert abs(corr[2] - base[2]) < np.radians(0.4)
    assert loc.localized(0.21 * 59) and loc.n_fit >= 55


def test_steps_are_limited():
    base = np.array([1.1, -0.2, 0.1])
    loc = TableLocalizer(RECTS)
    loc.update(0.0, odom_of(base, np.radians(1.2), np.array([0.05, 0.0])), scan(base))
    assert np.linalg.norm(loc.base(base)[:2] - base[:2]) <= np.linalg.norm([0.05, 0.0])
    assert abs(loc.theta) <= TableLocalizer.MAX_STEP_YAW + 1e-9


def test_no_table_in_view_no_change():
    loc = TableLocalizer(RECTS)
    far = np.random.default_rng(1).uniform(-1, 1, (200, 2)) + [-3.0, 0.0]   # something else, far from the table
    assert not loc.update(0.0, np.array([0.0, 0.0, 0.0]), far)
    assert loc.theta == 0.0 and np.allclose(loc.t, 0)


def test_no_drift_stays_put():
    base = np.array([0.9, 0.3, -0.3])
    loc = TableLocalizer(RECTS)
    for k in range(20):
        loc.update(0.21 * k, base, scan(base, seed=k))
    assert np.linalg.norm(loc.base(base)[:2] - base[:2]) < 0.010 and abs(loc.theta) < np.radians(0.4)


def test_one_face_in_view():
    """From straight in front only the front skirt face is in view: x and the heading converge, the slide along
    the face (y) is barely observable — damped, it must not run away."""
    base = np.array([1.1, -0.2, 0.1])
    dth, d = np.radians(1.2), np.array([0.03, -0.02])
    loc = TableLocalizer(RECTS)
    for k in range(60):
        loc.update(0.21 * k, odom_of(base, dth, d), scan(base, seed=k))
    corr = loc.base(odom_of(base, dth, d))
    assert abs(corr[0] - base[0]) < 0.010 and abs(corr[2] - base[2]) < np.radians(0.4)
    assert abs(corr[1] - base[1]) < 0.025                  # at most the original drift, not more
