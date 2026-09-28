#!/usr/bin/env python3
"""
target_profile.py
=================
What the robot should pick: a colour profile (HSV ranges) plus the object's size and shape.
Profiles are recorded with the real (or simulated) camera by target_calibrator and stored as
YAML in steve_wbc/config/targets/<name>.yaml, so the same object is detected identically in
MuJoCo, Gazebo and on the robot:

    from steve_wbc.target_profile import TargetProfile
    tp = TargetProfile.load("red")               # name in config/targets, or a path
    found, cx, cy, mask = tp.detect(rgb)          # rgb: HxWx3 uint8

HSV follows OpenCV: H 0..179, S and V 0..255. A range may wrap around red (H_lo > H_hi).
Only the largest connected blob counts, so scattered background pixels of a similar colour
do not pull the centroid away.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # detection still works (without the largest-blob filter)
    cv2 = None


def targets_dir():
    """config/targets in the source tree or the installed share directory."""
    here = Path(__file__).resolve().parent
    for d in (here.parent / "config" / "targets", here / "config" / "targets"):
        if d.is_dir():
            return d
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("steve_wbc")) / "config" / "targets"
    except Exception:
        return here.parent / "config" / "targets"


def rgb_to_hsv(rgb):
    """uint8 RGB -> uint8 HSV in OpenCV convention (H 0..179), NumPy only."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm, gm = nz & (mx == r), nz & (mx == g) & (mx != r)
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0.0)
    return np.stack([np.round(h * 30) % 180, np.round(s * 255), np.round(mx * 255)], -1).astype(np.uint8)


@dataclass
class TargetProfile:
    name: str
    hsv_ranges: List[List[int]]      # [[h_lo, s_lo, v_lo, h_hi, s_hi, v_hi], ...]
    size: float = 0.05               # m  diameter (sphere) / edge (box)
    shape: str = "sphere"            # sphere | box | other
    min_pixels: int = 30
    notes: str = ""
    samples: int = 0                 # frames the profile was recorded from
    extra: dict = field(default_factory=dict)

    @property
    def radius(self):
        return 0.5 * self.size

    # ── detection ───────────────────────────────────────────────────────────
    def mask(self, rgb) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV) if cv2 is not None else rgb_to_hsv(rgb)
        h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
        m = np.zeros(h.shape, bool)
        for hl, sl, vl, hh, sh, vh in self.hsv_ranges:
            hue = (h >= hl) & (h <= hh) if hl <= hh else (h >= hl) | (h <= hh)  # wraps around red
            m |= hue & (s >= sl) & (s <= sh) & (v >= vl) & (v <= vh)
        return m

    def detect(self, rgb) -> Tuple[bool, float, float, np.ndarray]:
        """(found, cx, cy, mask of the largest blob)."""
        m = self.mask(rgb)
        if cv2 is not None and m.any():
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            if n > 1:
                k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                m = lab == k
        ys, xs = np.nonzero(m)
        if len(ys) < self.min_pixels:
            return False, 0.0, 0.0, m
        return True, float(np.median(xs)), float(np.median(ys)), m

    def center_from_points(self, pts):
        """Object centre from the 3-D points of its detected pixels (world/odom frame, z up).
        For an object resting on a table and seen from above, the top-most points (the top face
        of a box, the cap of a sphere) are symmetric about the centre from ANY viewpoint, unlike
        the blob centroid, which shifts towards the faces the camera sees. None if too few."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 10:
            return None
        z_top = np.percentile(pts[:, 2], 95)
        top = pts[pts[:, 2] >= z_top - 0.3 * self.size]
        if len(top) < 5:
            return None
        return np.array([*np.median(top[:, :2], axis=0), z_top - self.radius])

    # ── storage ─────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, name_or_path):
        import yaml
        p = Path(name_or_path)
        if not p.suffix:
            p = targets_dir() / f"{name_or_path}.yaml"
        with open(p) as f:
            d = yaml.safe_load(f)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path=None):
        import yaml
        p = Path(path) if path else targets_dir() / f"{self.name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["hsv_ranges"] = [[int(x) for x in r] for r in self.hsv_ranges]
        with open(p, "w") as f:
            f.write(f"# Target profile '{self.name}' — recorded with target_calibrator\n")
            yaml.safe_dump(d, f, sort_keys=False)
        return p


def fit_profile(name, obj_hsv, bg_hsv, size=0.05, shape="sphere", fp_weight=20.0, min_pixels=30):
    """Discriminative HSV ranges: the hue window and S/V floors that find the most object pixels
    while catching the fewest background pixels (score = recall - fp_weight * false-positive
    rate). obj_hsv / bg_hsv: N x 3 uint8 (OpenCV HSV) pixels of the object / everything else in
    the same frames, so e.g. a wooden table with a hue near the target's is excluded."""
    obj, bg = obj_hsv.astype(int), bg_hsv.astype(int)
    q = [0, 1, 2, 5, 10, 20, 35, 50]
    s_grid = sorted({int(np.percentile(obj[:, 1], p)) for p in q})
    v_grid = sorted({int(np.percentile(obj[:, 2], p)) for p in q})
    best = (-np.inf, None)
    for s_lo in s_grid:
        for v_lo in v_grid:
            ho = np.bincount(obj[(obj[:, 1] >= s_lo) & (obj[:, 2] >= v_lo), 0], minlength=180).astype(float)
            hb = np.bincount(bg[(bg[:, 1] >= s_lo) & (bg[:, 2] >= v_lo), 0], minlength=180).astype(float)
            co, cb = np.concatenate([ho, ho]).cumsum(), np.concatenate([hb, hb]).cumsum()
            for width in range(1, 91):  # every circular hue window [start, start + width)
                idx = np.arange(180)
                rec = (co[idx + width - 1] - np.where(idx > 0, co[idx - 1], 0)) / max(len(obj), 1)
                fp = (cb[idx + width - 1] - np.where(idx > 0, cb[idx - 1], 0)) / max(len(bg), 1)
                score = rec - fp_weight * fp - 1e-4 * width  # prefer narrower at equal score
                i = int(np.argmax(score))
                if score[i] > best[0]:
                    best = (score[i], (i, (i + width - 1) % 180, s_lo, v_lo))
    hl, hh, s_lo, v_lo = best[1]
    return TargetProfile(name=name, hsv_ranges=[[hl, s_lo, v_lo, hh, 255, 255]], size=size, shape=shape,
                         min_pixels=min_pixels)


def profile_from_pixels(name, hsv_pixels, size=0.05, shape="sphere", hue_cover=0.97,
                        s_margin=0.8, v_margin=0.6, min_pixels=30):
    """HSV ranges that cover `hue_cover` of the object's pixels (N x 3 uint8, OpenCV HSV).
    The hue interval is the shortest arc on the hue circle holding that share, so red, which
    straddles H = 0/179, becomes one wrapping range. S and V lower limits get a margin for
    shading; upper limits stay open (specular highlights are handled by S)."""
    h = hsv_pixels[:, 0].astype(int)
    hist = np.bincount(h, minlength=180).astype(float)
    need = hue_cover * hist.sum()
    best = None
    for width in range(1, 181):  # shortest circular window holding `need` pixels
        sums = np.convolve(np.concatenate([hist, hist[:width - 1]]), np.ones(width), "valid")[:180]
        i = int(np.argmax(sums))
        if sums[i] >= need:
            best = (i, (i + width - 1) % 180)
            break
    hl, hh = best if best else (0, 179)
    s_lo = int(np.percentile(hsv_pixels[:, 1], 3) * s_margin)
    v_lo = int(np.percentile(hsv_pixels[:, 2], 3) * v_margin)
    return TargetProfile(name=name, hsv_ranges=[[hl, s_lo, v_lo, hh, 255, 255]], size=size, shape=shape,
                         min_pixels=min_pixels)
