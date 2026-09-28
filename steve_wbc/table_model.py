#!/usr/bin/env python3
"""
table_model.py
==============
The table the robot grasps from, as measured on the real robot (table_probe) or written by
hand, instead of the simulation's closed box-like table (top + skirt to the floor):

    center [x, y], yaw     table centre / heading in `frame` (odom: valid while the base
                           driver runs, so probe and WBC must use the same odom session)
    size [lx, ly], top_z   top rectangle (in the table frame) and height of its surface
    thickness              of the top plate
    legs [[x, y, r], ...]  leg positions in the table frame (radius r); [] = unknown
    skirt                  height of closed side panels from the floor (0 = open legs)

Used by
  * MuJoCo  (--table): replaces the MJCF table body, so sim = real geometry
  * the WBC (--table): collision model of the table for arm and base (on the real robot the
    WBC otherwise has no table model at all); the planner's table rectangle and quadrants

Note: the LiDARs (at ~0.22 m) see only legs / skirt, never the top plate — the base collision
check therefore includes the cabinet's height, so the top plate counts for the base.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np


def tables_dir():
    here = Path(__file__).resolve().parent
    for d in (here.parent / "config" / "tables", here / "config" / "tables"):
        if d.is_dir():
            return d
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("steve_wbc")) / "config" / "tables"
    except Exception:
        return here.parent / "config" / "tables"


@dataclass
class TableModel:
    name: str = "table"
    frame: str = "odom"
    center: List[float] = field(default_factory=lambda: [2.0, 0.0])
    yaw: float = 0.0
    size: List[float] = field(default_factory=lambda: [0.80, 0.70])
    top_z: float = 0.80
    thickness: float = 0.05
    legs: List[List[float]] = field(default_factory=list)
    skirt: float = 0.0
    notes: str = ""

    # ── geometry ────────────────────────────────────────────────────────────
    def R(self):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([[c, -s], [s, c]])

    def to_world(self, xy):
        return np.asarray(self.center, float) + self.R() @ np.asarray(xy, float)

    def corners(self):
        hx, hy = 0.5 * self.size[0], 0.5 * self.size[1]
        return np.array([self.to_world(p) for p in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])

    def support_rect(self):
        """Axis-aligned (lo, hi) of the top — the planner's table rectangle (conservative for yaw != 0)."""
        c = self.corners()
        return c.min(0), c.max(0)

    # ── MuJoCo ──────────────────────────────────────────────────────────────
    def apply_to_spec(self, spec):
        """Replace the MJCF body 'table' with this table (geoms keep the names the WBC uses)."""
        import mujoco
        old = spec.body("table")
        if old is not None:
            spec.delete(old)
        q = [np.cos(self.yaw / 2), 0.0, 0.0, np.sin(self.yaw / 2)]
        body = spec.worldbody.add_body(name="table", pos=[*self.center, 0.0], quat=q)
        wood = [0.60, 0.40, 0.20, 1.0]
        box, cyl = mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER
        body.add_geom(name="table_top", type=box, rgba=wood, contype=1, conaffinity=1,
                      size=[0.5 * self.size[0], 0.5 * self.size[1], 0.5 * self.thickness],
                      pos=[0, 0, self.top_z - 0.5 * self.thickness])
        leg_h = self.top_z - self.thickness
        for i, (x, y, r) in enumerate(self.legs):
            body.add_geom(name=f"table_leg_{i}", type=cyl, rgba=wood, contype=1, conaffinity=1,
                          size=[r, 0.5 * leg_h, 0], pos=[x, y, 0.5 * leg_h])
        if self.skirt > 0:
            hx, hy, t = 0.5 * self.size[0], 0.5 * self.size[1], 0.02
            for n, (px, py, sx, sy) in {"front": (-hx + t, 0, t, hy), "rear": (hx - t, 0, t, hy),
                                        "left": (0, hy - t, hx, t), "right": (0, -hy + t, hx, t)}.items():
                body.add_geom(name=f"table_skirt_{n}", type=box, rgba=wood, contype=1, conaffinity=1,
                              size=[sx, sy, 0.5 * self.skirt], pos=[px, py, 0.5 * self.skirt])
        return spec

    # ── storage ─────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, name_or_path):
        import yaml
        p = Path(name_or_path)
        if not p.suffix:
            p = tables_dir() / f"{name_or_path}.yaml"
        with open(p) as f:
            d = yaml.safe_load(f)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path=None):
        import yaml
        p = Path(path) if path else tables_dir() / f"{self.name}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["center"] = [round(float(v), 4) for v in self.center]
        d["size"] = [round(float(v), 4) for v in self.size]
        d["legs"] = [[round(float(v), 4) for v in leg] for leg in self.legs]
        for k in ("yaw", "top_z", "thickness", "skirt"):
            d[k] = round(float(d[k]), 4)
        with open(p, "w") as f:
            f.write(f"# Table '{self.name}' — see steve_wbc/table_model.py (table_probe writes these)\n")
            yaml.safe_dump(d, f, sort_keys=False)
        return p


# ── fitting (table_probe) ────────────────────────────────────────────────────
def fit_table(cloud, lidar, robot_xy=(0.0, 0.0), z_range=(0.5, 1.3), leg_r=0.03):
    """
    Table from measurements in one frame (z up):
      cloud  N x 3 depth-camera points (the top plate is found as the dominant horizontal plane
             in z_range by RANSAC; its outline as the minimum-area rectangle of the inliers)
      lidar  M x 2 LiDAR points at scanner height (legs = small clusters under the top plate;
             a long straight run along an edge = a skirt)
    Returns (TableModel, info dict for plotting) or raises ValueError.
    """
    import cv2
    rng = np.random.default_rng(0)
    pts = cloud[(cloud[:, 2] > z_range[0]) & (cloud[:, 2] < z_range[1])]
    if len(pts) < 200:
        raise ValueError(f"too few depth points at table height ({len(pts)}) — point the head camera at the table")
    best = None  # horizontal plane z = const (tables are level)
    for _ in range(200):
        z0 = pts[rng.integers(len(pts)), 2]
        n = np.sum(np.abs(pts[:, 2] - z0) < 0.01)
        if best is None or n > best[1]:
            best = (z0, n)
    inl = pts[np.abs(pts[:, 2] - best[0]) < 0.015]
    top_z = float(np.median(inl[:, 2]))
    # largest connected patch of the plane (objects on the table stay above it)
    (cx, cy), (w, h), ang = cv2.minAreaRect(inl[:, :2].astype(np.float32))
    yaw = np.radians(ang)
    if w < h:  # table x along the longer side
        w, h, yaw = h, w, yaw + np.pi / 2
    # x axis pointing away from the robot
    d = np.array([cx, cy]) - np.asarray(robot_xy)
    if np.dot([np.cos(yaw), np.sin(yaw)], d) < 0:
        yaw += np.pi
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    t = TableModel(name="table", center=[float(cx), float(cy)], yaw=float(yaw), size=[float(w), float(h)],
                   top_z=top_z)

    # legs / skirt from the LiDAR: points under the top plate (in the table frame)
    loc = (np.asarray(lidar, float).reshape(-1, 2) - [cx, cy]) @ t.R()
    under = loc[(np.abs(loc[:, 0]) < 0.5 * w + 0.05) & (np.abs(loc[:, 1]) < 0.5 * h + 0.05)]
    legs, clusters = [], []
    if len(under):
        order = np.lexsort((under[:, 1], under[:, 0]))
        rest = list(map(tuple, under[order]))
        while rest:  # greedy clustering, 5 cm linkage
            c = [rest.pop(0)]
            changed = True
            while changed:
                changed = False
                for p in rest[:]:
                    if min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in c) < 0.05:
                        c.append(p)
                        rest.remove(p)
                        changed = True
            clusters.append(np.array(c))
        for c in clusters:
            ext = np.ptp(c, axis=0).max() if len(c) > 1 else 0.0
            if ext < 0.12:  # compact: a leg (the LiDAR sees its near half: centre one radius back)
                m = c.mean(0)
                legs.append([float(m[0]), float(m[1]), leg_r])
            elif ext > 0.3:  # long straight run: a skirt / panel
                t.skirt = max(t.skirt, 0.4)
    t.legs = legs
    return t, {"plane_inliers": inl, "lidar_under": under, "clusters": clusters}
