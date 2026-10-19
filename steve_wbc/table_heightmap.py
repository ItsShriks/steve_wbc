#!/usr/bin/env python3
"""
table_heightmap.py — what stands on the table, from depth frames (static scene)
================================================================================
A 2.5-D occupancy map of the table top for occlusion-aware perception: per cell (2 cm) the height of
whatever stands on the table (max over the frames), and whether the cell has been seen at all. The
tracked target is cut out (a sphere around its estimate), so it never occludes itself. Built from
the same depth frames on the robot (gripper camera + TF) and in MuJoCo (the perception camera's
rendered depth), so the simulation exercises the robot's code path.

Used by active_perception_wbc for
  * line_of_sight(camera, target): share of sight lines camera -> target surface that pass the
    occupied cells (approach-side choice: a side the target is hidden from is not chosen),
  * occluders(...): points on the occupied columns, for the WBC's line-of-sight task (keeps the
    sight line clear of them while the arm is free).

Frames: world = odom (z up); camera = ROS optical (x right, y down, z forward), pose world <- optical.
"""

import numpy as np


class TableHeightMap:
    def __init__(self, lo, hi, top_z, cell=0.02, poly=None, h_min=0.015, h_max=0.45, min_hits=2):
        """lo, hi: axis-aligned table rectangle (world xy); top_z: table surface height; poly: the
        top's outline (Nx2, convex) when the table is turned — cells outside it are ignored."""
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.top_z, self.cell, self.h_min, self.h_max, self.min_hits = float(top_z), cell, h_min, h_max, min_hits
        self.n = np.maximum(np.ceil((self.hi - self.lo) / cell).astype(int), 1)
        self.height = np.zeros(self.n)           # m above the table top (max over frames)
        self.hits = np.zeros(self.n, np.int32)   # frames in which something stood in the cell
        self.seen = np.zeros(self.n, bool)       # the cell (table or object) has been observed
        self.frames = 0
        gx, gy = np.meshgrid(*(self.lo[i] + (np.arange(self.n[i]) + 0.5) * cell for i in range(2)), indexing="ij")
        self.centers = np.stack([gx, gy], -1)
        self.inside = np.ones(self.n, bool)
        if poly is not None:  # convex outline: inside = on the inner side of every edge
            P = np.asarray(poly, float)
            sgn = 1.0 if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) >= 0 else -1.0
            for a, b in zip(P, np.roll(P, -1, axis=0)):
                self.inside &= sgn * ((b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0])) >= -1e-9

    # ── building ────────────────────────────────────────────────────────────
    def _cells(self, xy):
        ij = np.floor((xy - self.lo) / self.cell).astype(int)
        ok = np.all((ij >= 0) & (ij < self.n), axis=1)
        return ij, ok

    def integrate(self, depth, K, R_wc, t_wc, target=None, target_r=0.0, stride=4, z_max=2.0):
        """Add one depth frame [m] (0 / nan = invalid) with intrinsics K and camera pose."""
        D = np.asarray(depth, np.float32)[::stride, ::stride]
        v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
        u, v = u * stride, v * stride
        ok = np.isfinite(D) & (D > 0.1) & (D < z_max)
        z = D[ok]
        P = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1) @ np.asarray(R_wc).T + t_wc
        P = P[(P[:, 2] > self.top_z - 0.03) & (P[:, 2] < self.top_z + self.h_max)]
        ij, ok = self._cells(P[:, :2])
        P, ij = P[ok], ij[ok]
        ok = self.inside[ij[:, 0], ij[:, 1]]
        P, ij = P[ok], ij[ok]
        self.seen[ij[:, 0], ij[:, 1]] = True
        above = P[:, 2] > self.top_z + self.h_min
        if target is not None and target_r > 0:
            above &= np.linalg.norm(P - np.asarray(target, float), axis=1) > target_r
        P, ij = P[above], ij[above]
        if len(P):
            flat = ij[:, 0] * self.n[1] + ij[:, 1]
            cells = np.unique(flat)
            hmax = np.zeros(len(cells))
            np.maximum.at(hmax, np.searchsorted(cells, flat), P[:, 2] - self.top_z)
            self.hits.flat[cells] += 1
            self.height.flat[cells] = np.maximum(self.height.flat[cells], hmax)
        self.frames += 1

    def forget_target(self, target, target_r):
        """Clear cells inside the target's footprint (seen before it was tracked)."""
        d = np.linalg.norm(self.centers - np.asarray(target, float)[:2], axis=-1)
        m = d < target_r
        self.hits[m], self.height[m] = 0, 0.0

    # ── queries ─────────────────────────────────────────────────────────────
    @property
    def occupied(self):
        return (self.hits >= self.min_hits) & self.inside

    def coverage(self):
        """Share of the table top observed so far."""
        return float(self.seen[self.inside].mean()) if self.inside.any() else 0.0

    def occluders(self, near=None, radius=np.inf, levels=(0.5, 1.0)):
        """Points on the occupied columns (cell centre at `levels` of the column height), optionally
        only those within `radius` (xy) of `near`."""
        i, j = np.nonzero(self.occupied)
        if near is not None and len(i):
            keep = np.linalg.norm(self.centers[i, j] - np.asarray(near, float)[:2], axis=1) < radius
            i, j = i[keep], j[keep]
        if not len(i):
            return np.zeros((0, 3))
        xy, h = self.centers[i, j], self.height[i, j]
        return np.concatenate([np.column_stack([xy, self.top_z + f * h]) for f in levels])

    def line_of_sight(self, cam, target, target_r=0.0, n_rim=6, margin=0.01):
        """Share (0..1) of sight lines from `cam` to the target's visible cap (centre + n_rim points
        at 0.7 r around it) that pass above every occupied column (+ margin). 1.0: fully visible."""
        cam, target = np.asarray(cam, float), np.asarray(target, float)
        d = target - cam
        d /= np.linalg.norm(d)
        a = np.cross(d, [0.0, 0.0, 1.0])
        a = a / np.linalg.norm(a) if np.linalg.norm(a) > 1e-6 else np.array([1.0, 0.0, 0.0])
        b = np.cross(d, a)
        ang = np.linspace(0, 2 * np.pi, n_rim, endpoint=False)
        pts = [target - target_r * d] + [target - 0.5 * target_r * d + 0.7 * target_r * (np.cos(t) * a + np.sin(t) * b)
                                         for t in ang]
        occ = self.occupied
        clear = 0
        for p in pts:
            seg = p - cam
            L = np.linalg.norm(seg)
            s = np.linspace(0.0, 1.0, max(int(L / (0.5 * self.cell)), 2))[:-1]
            q = cam + np.outer(s, seg)
            ij, ok = self._cells(q[:, :2])
            q, ij = q[ok], ij[ok]
            blocked = occ[ij[:, 0], ij[:, 1]] & (q[:, 2] < self.top_z + self.height[ij[:, 0], ij[:, 1]] + margin)
            clear += not blocked.any()
        return clear / len(pts)

    def summary(self):
        occ = self.occupied
        return (f"{self.frames} frames, {100 * self.coverage():.0f}% of the top seen, {int(occ.sum())} occupied "
                f"cells" + (f", tallest {100 * self.height[occ].max():.0f} cm" if occ.any() else ""))

    def image(self, scale=6, target=None, cams=()):
        """BGR picture for the records: height (dark -> tall), unseen grey, target red, cameras blue."""
        import cv2
        h = np.where(self.occupied, np.clip(self.height / 0.3, 0, 1), 0.0)
        img = (255 * (1 - 0.85 * h)).astype(np.uint8)
        img = np.repeat(img[..., None], 3, axis=2)
        img[~self.seen] = (150, 150, 150)
        img[~self.inside] = (60, 60, 60)
        img = np.ascontiguousarray(np.flip(np.transpose(img, (1, 0, 2)), 0))  # x right, y up
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        def px(p):
            ij = (np.asarray(p, float)[:2] - self.lo) / self.cell
            return int(ij[0] * scale), int((self.n[1] - ij[1]) * scale)
        if target is not None:
            cv2.circle(img, px(target), 2 * scale, (0, 0, 255), -1)
        for name, c in cams:
            q = px(np.clip(np.asarray(c, float)[:2], self.lo, self.hi))
            cv2.circle(img, q, scale, (255, 0, 0), -1)
            cv2.putText(img, name, (q[0] + 4, q[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        return img
