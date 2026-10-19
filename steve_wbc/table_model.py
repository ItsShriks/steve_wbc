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
    """Where table models are read AND written: the package's SOURCE folder when it can be found
    (a colcon build copies config/tables into install/ and would overwrite a model saved there),
    else the installed share folder. Override with $STEVE_TABLES_DIR."""
    import os
    if os.environ.get("STEVE_TABLES_DIR"):
        return Path(os.environ["STEVE_TABLES_DIR"]).expanduser()
    here = Path(__file__).resolve().parent
    cands = [here.parent / "config" / "tables", here / "config" / "tables"]
    try:
        from ament_index_python.packages import get_package_prefix, get_package_share_directory
        share = Path(get_package_share_directory("steve_wbc"))
        ws = Path(get_package_prefix("steve_wbc")).parent.parent      # <ws>/install/steve_wbc -> <ws>
        cands += [ws / "src" / "steve_wbc" / "config" / "tables", share / "config" / "tables"]
    except Exception:
        pass
    for d in cands:
        if d.is_dir():
            return d
    return cands[0]


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
def lidar_clusters(pts, link=0.04):
    """Greedy single-linkage clusters of Nx2 points (grid-hashed, fast enough for a few 1000 pts)."""
    pts = np.unique(np.round(np.asarray(pts, float), 3), axis=0)
    cell = {}
    for i, p in enumerate(pts):
        cell.setdefault((int(p[0] // link), int(p[1] // link)), []).append(i)
    label = -np.ones(len(pts), int)
    k = 0
    for i in range(len(pts)):
        if label[i] >= 0:
            continue
        label[i], stack = k, [i]
        while stack:
            j = stack.pop()
            cx, cy = int(pts[j][0] // link), int(pts[j][1] // link)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for m in cell.get((cx + dx, cy + dy), ()):
                        if label[m] < 0 and np.hypot(*(pts[m] - pts[j])) < link:
                            label[m] = k
                            stack.append(m)
        k += 1
    return [pts[label == c] for c in range(k)]


def fit_table_lidar(lidar, robot, top_z, overhang=0.03, leg_r=0.025, max_yaw_deg=20.0,
                    side=(0.35, 2.0), max_dist=3.5):
    """
    Table from the LiDAR alone (no camera): four legs = four compact clusters forming a rectangle
    in front of the robot, roughly square to it. The LiDARs scan at ~0.22 m, so the top plate is
    invisible: its height `top_z` is given, its outline is the leg rectangle + `overhang`.
      lidar  M x 2 points (frame of `robot`), robot = ((x, y), yaw) in that frame
    Returns (TableModel, info) or raises ValueError. The best rectangle (smallest shape error,
    then nearest) is chosen; info["candidates"] lists the runner-ups to judge ambiguity.
    """
    from itertools import combinations
    (rx, ry), ryaw = robot
    Rr = np.array([[np.cos(ryaw), -np.sin(ryaw)], [np.sin(ryaw), np.cos(ryaw)]])
    loc = (np.asarray(lidar, float) - [rx, ry]) @ Rr          # robot frame: x forward
    front = loc[(loc[:, 0] > 0.3) & (loc[:, 0] < max_dist) & (np.abs(loc[:, 1]) < max_dist)]
    clusters = lidar_clusters(front)
    legs = [c.mean(0) for c in clusters if len(c) >= 3 and np.ptp(c, axis=0).max() < 0.10]
    if len(legs) < 4:
        raise ValueError(f"only {len(legs)} leg-like clusters in front of the robot")
    legs = np.array(legs)[np.argsort(np.hypot(*np.array(legs).T))][:40]   # nearest 40
    cands = []
    for quad in combinations(range(len(legs)), 4):
        P = legs[list(quad)]
        for a, b, c, d in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)):   # (a,b) and (c,d) = diagonals
            m1, m2 = 0.5 * (P[a] + P[b]), 0.5 * (P[c] + P[d])
            d1, d2 = np.hypot(*(P[a] - P[b])), np.hypot(*(P[c] - P[d]))
            err = np.hypot(*(m1 - m2)) + abs(d1 - d2)
            if err > 0.06:
                continue
            s1, s2 = np.hypot(*(P[a] - P[c])), np.hypot(*(P[a] - P[d]))
            if not (side[0] <= min(s1, s2) and max(s1, s2) <= side[1]):
                continue
            ctr = 0.5 * (m1 + m2)
            if ctr[0] < 0.5:
                continue
            e1 = P[c] - P[a] if s1 >= s2 else P[d] - P[a]       # a side of the rectangle
            ang = np.arctan2(e1[1], e1[0])
            yaw = (ang + np.pi / 4) % (np.pi / 2) - np.pi / 4     # table axis closest to robot x
            if abs(np.degrees(yaw)) > max_yaw_deg:
                continue
            cands.append((err, float(np.hypot(*ctr)), ctr, yaw, P))
    if not cands:
        raise ValueError(f"no rectangle of 4 legs found among {len(legs)} leg-like clusters")
    cands.sort(key=lambda c: (round(c[0], 2), c[1]))
    err, dist, ctr, yaw, P = cands[0]
    Rt = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    leg_t = (P - ctr) @ Rt                                        # legs in the table frame
    hx, hy = np.abs(leg_t[:, 0]).max(), np.abs(leg_t[:, 1]).max()
    # back to the input frame
    center = Rr @ ctr + [rx, ry]
    legs_all = [[float(x), float(y), leg_r] for x, y in leg_t]
    # further leg-like clusters UNDER the top (a 5th / centre leg, a support): obstacles for the
    # base as well; anything outside the top outline (chairs, other furniture) is not the table
    extra = []
    for q in legs:
        if np.min(np.hypot(*(P - q).T)) < 0.05:
            continue                                              # one of the four corner legs
        ql = (q - ctr) @ Rt
        if abs(ql[0]) < hx + overhang and abs(ql[1]) < hy + overhang:
            extra.append([float(ql[0]), float(ql[1]), leg_r])
    legs_all += extra
    t = TableModel(name="table", center=[float(v) for v in center], yaw=float(ryaw + yaw),
                   size=[float(2 * (hx + overhang)), float(2 * (hy + overhang))], top_z=float(top_z),
                   legs=legs_all)
    return t, {"lidar_under": front, "clusters": clusters, "shape_err": err, "extra_legs": len(extra),
               "candidates": [(c[0], c[2], np.degrees(c[3])) for c in cands[:5]],
               "legs_robot": P, "leg_like": legs}


def fit_table_template(lidar, robot, template, top_z, match_tol=0.05, max_dist=3.5):
    """
    Pose of a table of KNOWN shape (`template`: a saved TableModel with >= 3 legs, e.g. the lab
    table's 5) from the LiDAR legs it can see: any 3 or more of them — a leg hidden behind another
    one (seen from the scanner) no longer breaks the fit, as with fit_table_lidar's 4-corner
    rectangle. Every pair of leg-like clusters is matched to every pair of template legs of the same
    spacing; the rigid 2-D pose that brings the most template legs onto clusters (then the smallest
    RMS, then the table extending away from the robot) wins.
      lidar  M x 2 points (frame of `robot`), robot = ((x, y), yaw) in that frame
    Returns (TableModel, info) or raises ValueError.
    """
    (rx, ry), ryaw = robot
    Rr = np.array([[np.cos(ryaw), -np.sin(ryaw)], [np.sin(ryaw), np.cos(ryaw)]])
    loc = (np.asarray(lidar, float) - [rx, ry]) @ Rr          # robot frame: x forward
    front = loc[(loc[:, 0] > 0.3) & (loc[:, 0] < max_dist) & (np.abs(loc[:, 1]) < max_dist)]
    clusters = lidar_clusters(front)
    legs = [c.mean(0) for c in clusters if len(c) >= 3 and np.ptp(c, axis=0).max() < 0.10]
    if len(legs) < 3:
        raise ValueError(f"only {len(legs)} leg-like clusters in front of the robot")
    legs = np.array(legs)[np.argsort(np.hypot(*np.array(legs).T))][:40]   # nearest 40
    T = np.array([[lx, ly] for lx, ly, _ in template.legs], float)
    if len(T) < 3:
        raise ValueError(f"template '{template.name}' has {len(T)} legs (needs >= 3)")
    hx, hy = 0.5 * template.size[0], 0.5 * template.size[1]
    cands = []
    for i in range(len(legs)):
        for j in range(len(legs)):
            if i == j:
                continue
            dij = legs[j] - legs[i]
            for k in range(len(T)):
                for m in range(len(T)):
                    if k == m:
                        continue
                    dkm = T[m] - T[k]
                    if abs(np.hypot(*dij) - np.hypot(*dkm)) > match_tol:
                        continue
                    th = np.arctan2(dij[1], dij[0]) - np.arctan2(dkm[1], dkm[0])
                    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                    t = 0.5 * (legs[i] + legs[j]) - R @ (0.5 * (T[k] + T[m]))
                    W = T @ R.T + t                                # template legs, robot frame
                    dist = np.min(np.linalg.norm(W[:, None] - legs[None], axis=2), axis=1)
                    hit = dist < match_tol
                    if hit.sum() < 3:
                        continue
                    # the robot is not under the table (top outline + 10 cm)
                    o = R.T @ (-t)
                    if abs(o[0]) < hx + 0.10 and abs(o[1]) < hy + 0.10:
                        continue
                    rms = float(np.sqrt(np.mean(dist[hit] ** 2)))
                    cands.append((-int(hit.sum()), round(rms, 2), -float(np.hypot(*t)), rms, th, t, hit))
    if not cands:
        raise ValueError(f"the {len(T)}-leg layout of '{template.name}' matches no 3 of the "
                         f"{len(legs)} leg-like clusters")
    cands.sort(key=lambda c: c[:3])
    n_hit, _, _, rms, th, t, hit = cands[0]
    # a clearly different pose with as many legs and about the same error: ambiguous
    rival = next((c for c in cands[1:] if c[0] == n_hit and c[3] < rms + 0.01
                  and (np.hypot(*(c[5] - t)) > 0.10 or abs((c[4] - th + np.pi) % (2 * np.pi) - np.pi) > 0.2)), None)
    table = TableModel(name=template.name, center=[float(v) for v in Rr @ t + [rx, ry]],
                       yaw=float(ryaw + th), size=list(template.size), top_z=float(top_z),
                       thickness=template.thickness, legs=[list(map(float, l)) for l in template.legs],
                       skirt=template.skirt)
    W = T @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T + t
    return table, {"lidar_under": front, "clusters": clusters, "shape_err": rms, "leg_like": legs,
                   "legs_seen": int(hit.sum()), "legs_hidden": [i for i in range(len(T)) if not hit[i]],
                   "legs_robot": W, "ambiguous": rival is not None,
                   "candidates": [(c[3], c[5], np.degrees(c[4])) for c in cands[:5]]}


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
