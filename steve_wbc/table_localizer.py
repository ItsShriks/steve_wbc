"""
table_localizer.py — odometry drift correction against the known static environment (the table's legs and skirt,
anything else in the WBC's environment model) seen by the base LiDARs, continuously while the robot drives.

Wheel odometry drifts — most when the base turns, as on the orbit around the table — and everything the robot
places in its odom frame drifts with it: a static ball seen while driving seemed to roll (sim --sim_realism,
2026-10-30), grasps landed cm off. The table does not move, and its outline at the LiDARs' height is known (MuJoCo:
the model; robot: table_probe's measured table, --table). Every PERIOD the scan (both LiDARs, base frame, placed
with the odometry pose) is registered to that outline:

  correction C (odom -> table-anchored frame):  p_c = R(theta) p_odom + t
  point-to-plane least squares: each scan point within MAX_DIST of a shape's boundary pulls along its normal;
  rotation about the base; Huber weights; damped where the geometry does not constrain (one skirt face seen
  edge-on fixes neither the slide along it nor much of the turn)

C moves a share ALPHA of the way towards each fit (one scan sees the table with ~60 beams at ~1 cm noise: ~0.3 deg
per fit, averaged over successive scans), at most MAX_STEP_XY / MAX_STEP_YAW per update (no jumps for the
controller), and only after a good fit (MIN_POINTS inliers, RMS < MAX_RMS). Without a fit for STALE s the frame is "not localized" (the
tracker's drift allowance goes back to odometry-only).

    loc = TableLocalizer(rects, lidar_z)          # rects: [(centre xy, R 2x2, half xy)] in the table frame
    loc.update(t, base_odom, lidar_pts_base)      # every tick; works every PERIOD
    loc.base(base_odom), loc.p(p_odom), loc.R(R_odom), loc.inv_p(p_c)
"""

import numpy as np


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class TableLocalizer:
    PERIOD = 0.2          # s   between fits
    MAX_DIST = 0.10       # m   scan points farther from every shape are not the table (people, walls, ...)
    MIN_POINTS = 25       # inliers for a fit
    MAX_RMS = 0.015       # m   fit residual accepted
    MAX_STEP_XY = 0.01    # m   correction change per update at the base (5 cm/s at PERIOD)
    MAX_STEP_YAW = np.radians(0.2)
    ALPHA = 0.25          # share of the way to each fit (low-pass over scans)
    STALE = 1.0           # s   no good fit this long: not localized
    HUBER = 0.01          # m
    ITERS = 6

    def __init__(self, rects, lidar_z=None):
        self.c = np.array([r[0] for r in rects], float).reshape(-1, 2)
        self.Rr = np.array([r[1] for r in rects], float).reshape(-1, 2, 2)
        self.h = np.array([r[2] for r in rects], float).reshape(-1, 2)
        self.lidar_z = lidar_z
        self.theta, self.t = 0.0, np.zeros(2)
        self.t_last, self.t_fix = -np.inf, -np.inf
        self.n_fit, self.n_rejected, self.last = 0, 0, None   # last: (inliers, rms) of the last fit

    def __bool__(self):
        return len(self.c) > 0

    # ── the frame ────────────────────────────────────────────────────────────────
    def p(self, p):
        """A point (2- or 3-D, z unchanged) from odom into the corrected frame."""
        q = np.array(p, float)
        q[:2] = _rot(self.theta) @ q[:2] + self.t
        return q

    def inv_p(self, p):
        q = np.array(p, float)
        q[:2] = _rot(-self.theta) @ (q[:2] - self.t)
        return q

    def v(self, v):
        """A vector (2- or 3-D): rotated only."""
        q = np.array(v, float)
        q[:2] = _rot(self.theta) @ q[:2]
        return q

    def inv_v(self, v):
        q = np.array(v, float)
        q[:2] = _rot(-self.theta) @ q[:2]
        return q

    def R(self, R):
        """A 3x3 orientation (world <- body) from odom into the corrected frame."""
        Rz = np.eye(3)
        Rz[:2, :2] = _rot(self.theta)
        return Rz @ np.asarray(R, float)

    def base(self, base):
        b = np.array(base, float)
        b[:2] = _rot(self.theta) @ b[:2] + self.t
        b[2] = b[2] + self.theta
        return b

    def localized(self, now):
        return now - self.t_fix < self.STALE

    # ── geometry ─────────────────────────────────────────────────────────────────
    def closest(self, P, sensor=None):
        """For points P (N x 2): (closest boundary point, outward normal there, distance) over all shapes — with
        `sensor` (xy) only on faces it can see (a noisy point inside a 4 cm skirt board must not snap to the board's
        back face: that biased the heading by 0.2-0.5 deg)."""
        q = np.einsum("nkj,kji->nki", P[:, None, :] - self.c[None], self.Rr)    # local coords (N, K, 2)
        h = self.h[None]
        N_, K = q.shape[:2]
        inside = np.all(np.abs(q) <= h, axis=-1)
        # outside: the clamped point
        cl = np.clip(q, -h, h)
        diff = q - cl
        nd = np.linalg.norm(diff, axis=-1, keepdims=True)
        n_cl = diff / np.maximum(nd, 1e-12)
        d_cl = nd[..., 0]
        # inside: the nearest of the four faces (axis a, side sg)
        faces = []
        for a in (0, 1):
            for sg in (-1.0, 1.0):
                b = q.copy()
                b[..., a] = sg * h[..., a]
                n = np.zeros_like(q)
                n[..., a] = sg
                faces.append((b, n, h[..., a] - sg * q[..., a]))
        if sensor is not None:
            sl = np.einsum("kj,kji->ki", np.asarray(sensor, float)[None, :] - self.c, self.Rr)[None]   # (1, K, 2)
            d_cl = np.where(np.einsum("nki,nki->nk", n_cl, sl - cl) > 0, d_cl, np.inf)
            faces = [(b, n, np.where(np.einsum("nki,nki->nk", n, sl - b) > 0, d, np.inf)) for b, n, d in faces]
        fd = np.stack([f[2] for f in faces], -1)                                # (N, K, 4)
        fi = np.argmin(fd, -1)
        fb = np.stack([f[0] for f in faces], -2)                                # (N, K, 4, 2)
        fn = np.stack([f[1] for f in faces], -2)
        ii = np.indices(fi.shape)
        b_in, n_in, d_in = fb[ii[0], ii[1], fi], fn[ii[0], ii[1], fi], fd[ii[0], ii[1], fi]
        b = np.where(inside[..., None], b_in, cl)
        n = np.where(inside[..., None], n_in, n_cl)
        d = np.where(inside, d_in, d_cl)
        k = np.argmin(d, axis=1)
        r = np.arange(N_)
        Rk = self.Rr[k]
        bw = np.einsum("nij,nj->ni", Rk, b[r, k]) + self.c[k]
        nw = np.einsum("nij,nj->ni", Rk, n[r, k])
        return bw, nw, d[r, k]

    # ── fit ──────────────────────────────────────────────────────────────────────
    def update(self, now, base_odom, pts_base):
        """Register the scan (points in the base frame, or a callable giving them — called only when a fit is due;
        the base at base_odom in odom); True after a good fit."""
        if not self or now - self.t_last < self.PERIOD:
            return False
        if callable(pts_base):
            pts_base = pts_base()
        if pts_base is None or len(pts_base) < self.MIN_POINTS:
            return False
        self.t_last = now
        b = np.asarray(base_odom, float)
        Q = np.asarray(pts_base, float)[:, :2] @ _rot(b[2]).T + b[:2]        # scan in odom
        th, t = self.theta, self.t.copy()
        inl, rms = 0, np.inf
        for _ in range(self.ITERS):
            W = Q @ _rot(th).T + t
            pivot = _rot(th) @ b[:2] + t                                      # the base, corrected
            B, N, d = self.closest(W, pivot)
            m = d < self.MAX_DIST
            inl = int(m.sum())
            if inl < self.MIN_POINTS:
                break
            W, B, N = W[m], B[m], N[m]
            r = np.einsum("ij,ij->i", N, W - B)                               # point-to-plane residuals
            rel = W - pivot
            J = np.column_stack([N[:, 0] * -rel[:, 1] + N[:, 1] * rel[:, 0], N[:, 0], N[:, 1]])   # d r / (dth, dx, dy)
            w = np.where(np.abs(r) <= self.HUBER, 1.0, self.HUBER / np.maximum(np.abs(r), 1e-12))
            H = J.T @ (J * w[:, None])
            g = J.T @ (w * r)
            lam = 1e-3 * np.trace(H) + 1e-6
            dx = -np.linalg.solve(H + lam * np.eye(3), g)
            dth, dt = dx[0], dx[1:]
            t = _rot(dth) @ (t - pivot) + pivot + dt                          # rotate about the base, then shift
            th = th + dth
            rms = float(np.sqrt(np.mean(r ** 2)))
            if abs(dth) < 1e-5 and np.linalg.norm(dt) < 1e-5:
                break
        self.last = (inl, rms)
        if inl < self.MIN_POINTS or not np.isfinite(rms) or rms > self.MAX_RMS:
            self.n_rejected += 1
            return False
        # move the correction towards the fit, limited (smooth for the controller)
        dth = float(np.clip(self.ALPHA * (th - self.theta), -self.MAX_STEP_YAW, self.MAX_STEP_YAW))
        base_now = _rot(self.theta) @ b[:2] + self.t
        base_fit = _rot(th) @ b[:2] + t
        step = self.ALPHA * (base_fit - base_now)
        n = np.linalg.norm(step)
        if n > self.MAX_STEP_XY:
            step *= self.MAX_STEP_XY / n
        # new C: rotate by dth about the corrected base, then shift by step
        self.t = _rot(dth) @ (self.t - base_now) + base_now + step
        self.theta += dth
        self.t_fix = now
        self.n_fit += 1
        return True


def env_rects(model, data, boxes, lidar_z):
    """The static environment's cross-section at the LiDAR height: [(centre xy, R 2x2, half xy)] of the upright
    boxes among the WBC's environment collision boxes (body, centre in body, R in body, half, name, moving)
    spanning lidar_z — table legs (cylinders as boxes), skirt, standing obstacles; mocap bodies left out."""
    rects = []
    for body, c, R, h, name, moving in boxes:
        if moving or model.body_mocapid[body] >= 0:
            continue
        Rw = data.xmat[body].reshape(3, 3) @ R
        cw = data.xpos[body] + data.xmat[body].reshape(3, 3) @ c
        k = int(np.argmax(np.abs(Rw[2])))                        # the box axis closest to vertical
        if abs(Rw[2, k]) < 0.99:
            continue                                             # not upright: no clean cross-section
        if abs(lidar_z - cw[2]) > h[k]:
            continue                                             # not at the LiDAR height
        ax = [i for i in range(3) if i != k]
        R2 = Rw[:2][:, ax]
        R2 = R2 / np.linalg.norm(R2, axis=0, keepdims=True)
        if np.linalg.det(R2) < 0:
            R2[:, 1] *= -1
        rects.append((cw[:2].copy(), R2, np.array([h[ax[0]], h[ax[1]]])))
    return rects


# ── the WBC sees the corrected frame through these (everything else passes through unchanged) ─────────────
class FrameBackend:
    """The robot backend with the base pose in the corrected frame; feeds the localizer (LiDAR every PERIOD).
    Commands are body-frame velocities: unchanged."""

    def __init__(self, backend, loc, lidar_points):
        self.__dict__.update(raw=backend, loc=loc, lidar=lidar_points)

    def _corrected(self, st):
        return None if st is None else type(st)(self.loc.base(st.base), st.q.copy())

    def read_state(self):
        st = self.raw.read_state()
        if st is not None:
            self.loc.update(self.raw.now(), st.base, self.lidar)
        return self._corrected(st)

    def wait_for_state(self, timeout):
        return self._corrected(self.raw.wait_for_state(timeout))

    def __getattr__(self, name):
        return getattr(self.raw, name)


class FrameWorld:
    """The world (sim / robot) with positions in the corrected frame: detections, object points, the ground truth
    report, camera poses, grasp detector views (in: back to odom) and results (out), table-map frames. The table
    geometry (support_*, transit_height, relocate) is already in the table frame: unchanged."""

    def __init__(self, world, loc):
        self.__dict__.update(raw=world, loc=loc)

    def __getattr__(self, name):
        if name == "last_view":
            v = getattr(self.raw, "last_view", None)
            return None if v is None else (v[0], v[1], self.loc.R(v[2]), self.loc.p(v[3]))
        return getattr(self.raw, name)

    def __setattr__(self, name, value):
        if name == "vgn_target" and value is not None:
            value = self.loc.inv_p(value)
        setattr(self.raw, name, value)

    def _pts(self, P):
        if P is None:
            return None
        Q = np.array(P, float)
        Q[:, :2] = Q[:, :2] @ _rot(self.loc.theta).T + self.loc.t
        return Q

    def detect(self, *args, **kw):
        p = self.raw.detect(*args, **kw)
        return None if p is None else self.loc.p(p)

    def object_points(self):
        return self._pts(self.raw.object_points())

    def ground_truth(self):
        p = self.raw.ground_truth()
        return None if p is None else self.loc.p(p)

    def camera_pose(self, frame):
        r = self.raw.camera_pose(frame)
        return None if r is None else (self.loc.R(r[0]), self.loc.p(r[1]))

    def mapping_frame(self):
        f = self.raw.mapping_frame()
        return None if f is None else (f[0], f[1], self.loc.R(f[2]), self.loc.p(f[3]))

    def vgn_init(self, center):
        return self.raw.vgn_init(self.loc.inv_p(center))

    def vgn_view(self, center, obj_shift=None):
        return self.raw.vgn_view(self.loc.inv_p(center), None if obj_shift is None else self.loc.inv_v(obj_shift))

    def _grasp(self, r):
        if r is None:
            return None
        best, dur = r
        if best is not None:
            best = (self.loc.p(best[0]), self.loc.R(best[1]), *best[2:])
        return best, dur

    def vgn_best(self):
        return self._grasp(self.raw.vgn_best())

    def vgn_poll(self):
        return self._grasp(self.raw.vgn_poll())
