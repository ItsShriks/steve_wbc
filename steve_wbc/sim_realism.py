"""
sim_realism.py — the real robot's imperfections for the MuJoCo simulation of active_perception_wbc
(--sim_realism): what the perfect simulation leaves out and the robot (MMO-700 + UR5e over Wi-Fi, D415 on the
printed wrist collar, Ability Hand over ah_node) does not.

  component  what                                                         real source
  ─────────  ───────────────────────────────────────────────────────────  ─────────────────────────────────────
  depth      D415 depth noise (sigma_z = z^2 * subpixel / (f * baseline),  stereo matching; Intel D400 tuning
             half of it spatially correlated), depth scale bias per run,   guide: RMS ~ z^2 * 0.08..0.2 px /
             invalid pixels at edges / random holes / below min_range,     (f * B)
             1 mm quantisation
  rgb        exposure gain + white balance per run, gain flicker, pixel    auto exposure, sensor noise
             noise, motion blur from the camera's speed (exposure time)
  detector   red_ball_detector over ROS: detections arrive ~80 ms late     RealSense USB + detector + DDS;
             (stamped on arrival like RosWorld.detect: the last one is     RosWorld.detect / --target_timeout
             returned again until --target_timeout), misses, outliers,
             camera pose from TF at a slightly wrong time
  calib      hand-eye (camera extrinsic) error, hand mount error on the    gripper_camera_calibrator accuracy,
             flange (the WBC keeps the nominal model)                      3D-printed collar
  odom       wheel odometry scale bias + random walk: the robot's world    MPO-700 caster odometry (odom frame)
             (odom) frame drifts away from the true one — detections,
             the EE and the ground-truth report are all in the drifted
             frame, the table model stays where it was measured
  base       base velocity gain error per axis (true motion != command)   drive tuning, wheel slip
  net        robot state 1 tick old, commands late with jitter / lost,     PC <-> robot over the lab Wi-Fi
             small arm velocity noise                                      (UR velocity mode jitter)
  lidar      S300 range noise and dropouts                                SICK S300 spec (~1-2 cm)
  object     ball: rolling resistance, a slightly tilted table, table      a real ball / table
             unevenness while rolling (not a perfect constant velocity)
  hand       Ability Hand command latency and finite finger speed          ah_node + serial, hand motors

Every random per-run error is drawn from --realism_seed and printed at the start; --realism_scale scales all
noise / error magnitudes (latencies stay), --realism_off disables single components (to find which one breaks
a run). The numbers below are the nominal profile — measure and update them on the robot where possible.
"""

import math
from collections import deque
from dataclasses import dataclass, fields

import mujoco
import numpy as np

try:
    from steve_wbc.steve_figure8_wbc import NU, MujocoSimBackend, RobotState
except ImportError:  # plain script next to the other modules
    from steve_figure8_wbc import NU, MujocoSimBackend, RobotState

COMPONENTS = ("depth", "rgb", "detector", "calib", "odom", "base", "net", "lidar", "object", "hand")


@dataclass
class RealismConfig:
    # depth (D415)
    depth_subpixel: float = 0.12     # px   RMS disparity noise
    depth_baseline: float = 0.055    # m    D415 stereo baseline
    depth_f_px: float = 600.0        # px   depth focal length (D415 65 deg HFOV at 848 px: ~665, 640 px: ~500)
    depth_corr: float = 0.5          # fraction of the noise variance that is spatially correlated (blobs ~16 px)
    depth_scale_sd: float = 0.008    # per run: depth scale bias (D415 calibration, temperature)
    depth_max: float = 4.0           # m
    depth_holes: float = 0.01        # fraction of random invalid pixels
    depth_edge_drop: float = 0.6     # probability that a pixel on a depth edge (> 2 cm jump) is invalid
    depth_quant: float = 0.001       # m    16-bit depth in mm
    # colour
    rgb_noise: float = 3.0           # grey levels, per pixel
    rgb_gain_sd: float = 0.08        # per run: exposure
    rgb_wb_sd: float = 0.03          # per run: white balance per channel
    rgb_flicker_sd: float = 0.02     # per frame: gain
    exposure: float = 0.015          # s    motion blur
    # detector path (red_ball_detector -> PointStamped -> RosWorld.detect)
    det_latency: float = 0.08        # s    capture -> arrival at the WBC
    det_latency_sd: float = 0.015    # s
    det_miss: float = 0.05           # probability that a frame gives no detection
    det_outlier: float = 0.01        # probability of a wrong detection (a reflection, a red patch)
    det_outlier_dist: float = 0.08   # m    its offset (random direction, horizontal)
    tf_offset_sd: float = 0.010      # s    per run: camera pose looked up at the wrong time (clocks, stamps)
    tf_jitter_sd: float = 0.004      # s    per frame
    # calibration
    cam_pos_sd: float = 0.004        # m    hand-eye calibration, per axis
    cam_rot_sd: float = 0.7          # deg  per axis
    mount_pos_sd: float = 0.0015     # m    hand on the flange (printed collar), per axis
    mount_rot_sd: float = 1.0        # deg  per axis (mostly the roll about the flange axis in reality)
    # odometry
    odom_scale_sd: float = 0.01      # per run: x / y scale bias
    odom_yaw_scale_sd: float = 0.02  # per run: yaw scale bias
    odom_xy_rw: float = 0.01         # m / sqrt(m) random walk
    odom_yaw_rw: float = 0.01        # rad / sqrt(m + rad)
    # base actuation
    base_gain_sd: float = 0.03       # per run, per axis (vx, vy, wz)
    # network (control at --rate, default 50 Hz: a tick is 20 ms)
    state_delay_ticks: int = 1       # robot state this many ticks old
    cmd_extra_ticks: int = 0         # on top of --sim_latency_ticks
    cmd_jitter_p: float = 0.15       # probability of one more tick
    cmd_drop_p: float = 0.02         # probability that a command never arrives (the last one stays)
    arm_vel_noise: float = 0.01      # rad/s low-pass (0.2 s) joint velocity noise while the arm moves
    # LiDAR
    lidar_sd: float = 0.012          # m
    lidar_drop: float = 0.01         # fraction of missing points
    # object (ball)
    obj_roll_decel: float = 0.002    # m/s^2 rolling resistance (also holds a ball still against the tilt below)
    obj_tilt_sd: float = 0.001       # m/s^2 per run, per axis: table tilt (0.001 = 0.006 deg)
    obj_rough_sd: float = 0.003      # m/s^2 table unevenness while rolling (low-pass 0.5 s)
    # hand (Ability Hand)
    hand_latency: float = 0.10       # s    GripperCommand -> ability_hand_gripper -> ah_node -> serial
    hand_speed: float = 200.0        # deg/s finger speed (measure on the hand: full close 95 deg in ~0.5 s?)

    STOCHASTIC = ("depth_subpixel", "depth_scale_sd", "depth_holes", "depth_edge_drop", "rgb_noise", "rgb_gain_sd",
                  "rgb_wb_sd", "rgb_flicker_sd", "det_latency_sd", "det_miss", "det_outlier", "tf_offset_sd",
                  "tf_jitter_sd", "cam_pos_sd", "cam_rot_sd", "mount_pos_sd", "mount_rot_sd", "odom_scale_sd",
                  "odom_yaw_scale_sd", "odom_xy_rw", "odom_yaw_rw", "base_gain_sd", "cmd_jitter_p", "cmd_drop_p",
                  "arm_vel_noise", "lidar_sd", "lidar_drop", "obj_tilt_sd", "obj_rough_sd")

    def scaled(self, s):
        """Noise / error magnitudes times s (probabilities capped at 1), latencies unchanged."""
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        for k in self.STOCHASTIC:
            kw[k] = min(kw[k] * s, 1.0) if k.endswith(("_p", "_miss", "_outlier", "_holes", "_drop")) else kw[k] * s
        return RealismConfig(**kw)


def _rotvec(v):
    """Rotation matrix of a rotation vector."""
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(v, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class SimRealism:
    def __init__(self, args, dt):
        self.c = RealismConfig().scaled(args.realism_scale)
        off = {s.strip() for s in (args.realism_off or "").split(",") if s.strip()}
        if off - set(COMPONENTS):
            raise SystemExit(f"--realism_off: unknown {sorted(off - set(COMPONENTS))}, choose from {COMPONENTS}")
        self.on = set(COMPONENTS) - off
        self.dt = dt
        self.rng = np.random.default_rng(args.realism_seed)
        self.min_range = float(args.min_detect_range)
        c, n = self.c, self.rng.normal
        on = lambda comp, v: v if comp in self.on else np.zeros_like(np.asarray(v, float))
        # per-run draws (always drawn, so one component off leaves the others' draws unchanged)
        self.cam_t = on("calib", n(0, c.cam_pos_sd, 3))
        self.cam_R = _rotvec(on("calib", np.radians(n(0, c.cam_rot_sd, 3))))
        self.mount_t = on("calib", n(0, c.mount_pos_sd, 3))
        self.mount_rv = on("calib", np.radians(n(0, c.mount_rot_sd, 3)))
        self.depth_scale = 1.0 + float(on("depth", n(0, c.depth_scale_sd)))
        self.rgb_gain = 1.0 + float(on("rgb", n(0, c.rgb_gain_sd)))
        self.rgb_wb = 1.0 + on("rgb", n(0, c.rgb_wb_sd, 3))
        self.tf_offset = float(on("detector", n(0, c.tf_offset_sd)))
        self.odom_scale = 1.0 + on("odom", np.array([*n(0, c.odom_scale_sd, 2), n(0, c.odom_yaw_scale_sd)]))
        self.base_gain = 1.0 + on("base", n(0, c.base_gain_sd, 3))
        self.tilt_acc = on("object", n(0, c.obj_tilt_sd, 2))
        # state
        self.b_bel = self.b_true = None  # believed (odom) / true base pose
        self.det_q = {}                  # camera -> deque of (arrival time, point)
        self.det_last = {}               # camera -> (arrival time, point) newest arrived
        self.n_det = dict(frames=0, miss=0, outlier=0)
        self.rough = np.zeros(2)
        self.obj_body, self.obj_z0, self.grip = -1, None, None
        self.hand_cmd = None             # (time the command reaches the hand, target ctrl)

    def has(self, comp):
        return comp in self.on

    def summary(self):
        mrot = np.degrees(np.linalg.norm(self.mount_rv))
        crot = np.degrees(np.linalg.norm(_rotvec_inv(self.cam_R)))
        lines = [f"[sim realism] components: {', '.join(c for c in COMPONENTS if c in self.on) or 'none'}"]
        if self.has("calib"):
            lines.append(f"  camera extrinsic error {np.linalg.norm(self.cam_t) * 1e3:.1f} mm / {crot:.2f} deg, "
                         f"hand mount error {np.linalg.norm(self.mount_t) * 1e3:.1f} mm / {mrot:.2f} deg")
        if self.has("depth"):
            lines.append(f"  depth scale {(self.depth_scale - 1) * 100:+.2f} %, noise at 0.5 / 1.0 m: "
                         f"{self._sd_z(0.5) * 1e3:.1f} / {self._sd_z(1.0) * 1e3:.1f} mm, min range {self.min_range:.2f} m")
        if self.has("rgb"):
            lines.append(f"  exposure gain {self.rgb_gain:.2f}, white balance {np.round(self.rgb_wb, 3)}")
        if self.has("detector"):
            lines.append(f"  detections {self.c.det_latency * 1e3:.0f}+-{self.c.det_latency_sd * 1e3:.0f} ms late, "
                         f"miss {self.c.det_miss * 100:.0f} %, outliers {self.c.det_outlier * 100:.1f} %, "
                         f"TF time offset {self.tf_offset * 1e3:+.1f} ms")
        if self.has("odom"):
            lines.append(f"  odometry scale x/y/yaw {np.round((self.odom_scale - 1) * 100, 2)} %, random walk "
                         f"{self.c.odom_xy_rw * 100:.1f} cm/sqrt(m)")
        if self.has("base"):
            lines.append(f"  base velocity gain vx/vy/wz {np.round((self.base_gain - 1) * 100, 1)} %")
        if self.has("net"):
            lines.append(f"  state {self.c.state_delay_ticks} tick(s) old, commands +{self.c.cmd_extra_ticks} tick(s), "
                         f"jitter {self.c.cmd_jitter_p * 100:.0f} %, lost {self.c.cmd_drop_p * 100:.0f} %")
        if self.has("lidar"):
            lines.append(f"  LiDAR noise {self.c.lidar_sd * 100:.1f} cm, dropouts {self.c.lidar_drop * 100:.0f} %")
        if self.has("object"):
            lines.append(f"  ball: rolling resistance {self.c.obj_roll_decel * 100:.1f} cm/s^2, table tilt "
                         f"{np.round(self.tilt_acc * 100, 2)} cm/s^2")
        if self.has("hand"):
            lines.append(f"  hand: {self.c.hand_latency * 1e3:.0f} ms latency, fingers {self.c.hand_speed:.0f} deg/s")
        print("\n".join(lines))

    # ── model: the hand where it really sits on the flange (the WBC's kinematic model keeps the nominal one) ──
    def perturb_model(self, model):
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        if s < 0 or not self.has("calib"):
            return
        b = model.site_bodyid[s]
        model.body_pos[b] += self.mount_t
        q, dq = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, self.mount_rv / max(np.linalg.norm(self.mount_rv), 1e-12),
                                  float(np.linalg.norm(self.mount_rv)))
        mujoco.mju_mulQuat(q, model.body_quat[b], dq)
        model.body_quat[b] = q

    def attach_world(self, model, data, obj_body, grip):
        self.obj_body, self.grip = obj_body, grip
        self.obj_z0 = float(data.xpos[obj_body][2]) if obj_body >= 0 else None
        self.obj_mass = float(model.body_mass[obj_body]) if obj_body >= 0 else 0.0

    # ── odometry: the robot's world frame drifts away from the true one ─────────
    def odom_reset(self, base):
        self.b_true, self.b_bel = np.array(base, float), np.array(base, float)

    def odom_update(self, base):
        prev, self.b_true = self.b_true, np.array(base, float)
        d = self.b_true - prev
        dyaw = (d[2] + np.pi) % (2 * np.pi) - np.pi
        c, s = math.cos(prev[2]), math.sin(prev[2])
        dx, dy = c * d[0] + s * d[1], -s * d[0] + c * d[1]  # true body-frame increment
        if self.has("odom"):
            ds = math.hypot(dx, dy)
            dx, dy, dyaw = dx * self.odom_scale[0], dy * self.odom_scale[1], dyaw * self.odom_scale[2]
            dx += self.rng.normal(0, self.c.odom_xy_rw * math.sqrt(ds))
            dy += self.rng.normal(0, self.c.odom_xy_rw * math.sqrt(ds))
            dyaw += self.rng.normal(0, self.c.odom_yaw_rw * math.sqrt(ds + abs(dyaw)))
        th = self.b_bel[2] + 0.5 * dyaw
        self.b_bel += [math.cos(th) * dx - math.sin(th) * dy, math.sin(th) * dx + math.cos(th) * dy, dyaw]

    def drift(self):
        """(dyaw, d): believed = rot_z(dyaw) @ true + d (planar)."""
        if self.b_bel is None:
            return 0.0, np.zeros(3)
        dth = float(self.b_bel[2] - self.b_true[2])
        d = np.zeros(3)
        d[:2] = self.b_bel[:2] - _rz(dth)[:2, :2] @ self.b_true[:2]
        return dth, d

    def to_bel(self, p):
        dth, d = self.drift()
        return _rz(dth) @ np.asarray(p, float) + d

    def drift_report(self):
        dth, _ = self.drift()
        if self.b_bel is None:
            return ""
        e = np.linalg.norm(self.b_bel[:2] - self.b_true[:2])
        return f"odometry drift at the end: {e * 100:.1f} cm, {np.degrees(dth):+.2f} deg"

    # ── camera ──────────────────────────────────────────────────────────────────
    def cam_pose(self, model, data, cid):
        """The camera pose (MuJoCo camera frame: looks along -z) the robot would use: from TF at a slightly
        wrong time, through the hand-eye calibration, in the drifted odom frame."""
        pos, R = data.cam_xpos[cid].copy(), data.cam_xmat[cid].reshape(3, 3).copy()
        if self.has("detector"):
            dt = self.tf_offset + self.rng.normal(0, self.c.tf_jitter_sd)
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_CAMERA, cid, vel, 0)  # [rot, lin] world
            pos, R = pos + vel[3:] * dt, _rotvec(vel[:3] * dt) @ R
        # extrinsic error in the optical frame (z forward, y down) = MuJoCo camera frame turned 180 deg about x
        F = np.diag([1.0, -1.0, -1.0])
        pos = pos + R @ F @ self.cam_t
        R = R @ F @ self.cam_R @ F
        dth, d = self.drift()
        return _rz(dth) @ pos + d, _rz(dth) @ R

    def _sd_z(self, z):
        return z * z * self.c.depth_subpixel / (self.c.depth_f_px * self.c.depth_baseline)

    def noisy_depth(self, depth):
        if not self.has("depth"):
            depth = depth.astype(np.float32).copy()
            depth[depth < self.min_range] = 0.0
            return depth
        c, rng = self.c, self.rng
        z = depth.astype(np.float32) * self.depth_scale
        H, W = z.shape
        sd = self._sd_z(z)
        white = rng.standard_normal((H, W), dtype=np.float32)
        g = rng.standard_normal((H // 16 + 2, W // 16 + 2), dtype=np.float32)
        from scipy.ndimage import zoom
        coarse = zoom(g, 16, order=1)[:H, :W]
        z = z + sd * (math.sqrt(1 - c.depth_corr) * white + math.sqrt(c.depth_corr) * coarse)
        edge = np.zeros((H, W), bool)
        jump_x = np.abs(np.diff(depth, axis=1)) > 0.02
        jump_y = np.abs(np.diff(depth, axis=0)) > 0.02
        edge[:, 1:] |= jump_x
        edge[:, :-1] |= jump_x
        edge[1:, :] |= jump_y
        edge[:-1, :] |= jump_y
        bad = (edge & (rng.random((H, W)) < c.depth_edge_drop)) | (rng.random((H, W)) < c.depth_holes)
        bad |= (depth < self.min_range) | (depth > c.depth_max)
        if c.depth_quant > 0:
            z = np.round(z / c.depth_quant) * c.depth_quant
        z[bad] = 0.0
        return z.astype(np.float32)

    def noisy_rgb(self, rgb, model, data, cid, f_px):
        if not self.has("rgb"):
            return rgb
        c, rng = self.c, self.rng
        img = rgb.astype(np.float32)
        # motion blur: image speed from the camera's rotation (+ translation at ~0.5 m) during the exposure
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_CAMERA, cid, vel, 1)  # camera frame
        bx = f_px * (abs(vel[1]) + abs(vel[3]) / 0.5) * c.exposure  # about y / along x: horizontal
        by = f_px * (abs(vel[0]) + abs(vel[4]) / 0.5) * c.exposure
        if bx >= 1.5 or by >= 1.5:
            from scipy.ndimage import uniform_filter1d
            if bx >= 1.5:
                img = uniform_filter1d(img, int(round(bx)), axis=1)
            if by >= 1.5:
                img = uniform_filter1d(img, int(round(by)), axis=0)
        gain = self.rgb_gain * (1.0 + rng.normal(0, c.rgb_flicker_sd))
        img = img * (gain * self.rgb_wb) + rng.normal(0, c.rgb_noise, img.shape).astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ── detector path: late, sometimes missing or wrong, returned until --target_timeout like RosWorld ─────
    def det_frame_ok(self):
        """False: this frame gives no detection (missed)."""
        self.n_det["frames"] += 1
        if self.has("detector") and self.rng.random() < self.c.det_miss:
            self.n_det["miss"] += 1
            return False
        return True

    def det_submit(self, cam, t, det):
        """det: (centre, object points) of one frame, or None (nothing detected)."""
        if det is None:
            return
        if not self.has("detector"):
            self.det_last[cam] = (t, det)
            return
        c = self.c
        if self.rng.random() < c.det_outlier:  # a wrong blob elsewhere: centre and points
            a = self.rng.uniform(0, 2 * np.pi)
            off = c.det_outlier_dist * np.array([math.cos(a), math.sin(a), 0.0])
            det = (det[0] + off, None if det[1] is None else det[1] + off)
            self.n_det["outlier"] += 1
        lat = max(0.03, self.rng.normal(c.det_latency, c.det_latency_sd))
        self.det_q.setdefault(cam, deque()).append((t + lat, det))

    def det_latest(self, cam, t, timeout):
        q = self.det_q.get(cam)
        while q and q[0][0] <= t:
            self.det_last[cam] = q.popleft()
        last = self.det_last.get(cam)
        if last is None or t - last[0] >= timeout:
            return None
        return last[1][0].copy(), last[1][1]

    # ── LiDAR ──────────────────────────────────────────────────────────────────
    def lidar_range(self, r):
        if not self.has("lidar") or not np.isfinite(r):
            return r
        return max(0.0, r + self.rng.normal(0, self.c.lidar_sd))

    def lidar_points(self, pts):
        if not self.has("lidar") or pts is None or not len(pts):
            return pts
        keep = self.rng.random(len(pts)) >= self.c.lidar_drop
        pts = pts[keep]
        r = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts * (1.0 + self.rng.normal(0, self.c.lidar_sd, (len(pts), 1)) / np.maximum(r, 1e-6))

    # ── hand ───────────────────────────────────────────────────────────────────
    def hand_command(self, target, t):
        if not self.has("hand") or self.grip is None or not self.grip:
            return False
        self.hand_cmd = (t + self.c.hand_latency, np.asarray(target, float))
        return True

    # ── per tick, before the physics ────────────────────────────────────────────
    def pre_step(self, model, data):
        t = data.time
        if self.hand_cmd is not None and t >= self.hand_cmd[0]:
            ids = self.grip.ids
            step = np.radians(self.c.hand_speed) * self.dt
            cur = data.ctrl[ids]
            data.ctrl[ids] = cur + np.clip(self.hand_cmd[1] - cur, -step, step)
        if self.has("object") and self.obj_body >= 0:
            data.xfrc_applied[self.obj_body, :] = 0.0
            p = data.xpos[self.obj_body]
            if abs(p[2] - self.obj_z0) > 0.005:  # lifted / falling: no table forces
                return
            v = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, self.obj_body, v, 0)
            vxy = v[3:5]
            speed = float(np.linalg.norm(vxy))
            a = self.tilt_acc.copy()
            k = math.exp(-self.dt / 0.5)
            if speed > 0.002:
                self.rough = k * self.rough + math.sqrt(1 - k * k) * self.rng.normal(0, self.c.obj_rough_sd, 2)
                a += self.rough - self.c.obj_roll_decel * vxy / speed
            elif np.linalg.norm(a) <= self.c.obj_roll_decel:
                return  # resting: the rolling resistance holds it against the tilt
            # a force at the centre of a rolling solid sphere accelerates it by 5/7 F / m
            data.xfrc_applied[self.obj_body, :2] = self.obj_mass * a * 7.0 / 5.0


def _rotvec_inv(R):
    """Rotation vector of a rotation matrix."""
    th = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class RealisticSimBackend(MujocoSimBackend):
    """MujocoSimBackend with the robot's state and commands as the PC sees them over the Wi-Fi: the base pose
    from drifting odometry, state late, commands late / lost, base gains off, arm velocity noise."""

    def __init__(self, model, args, dt, realism):
        realism.perturb_model(model)  # (a copy of the model: the WBC's kinematics keep the nominal hand)
        super().__init__(model, args, dt)
        self.realism = r = realism
        r.odom_reset(super().read_state().base)
        self.n = 0
        self.states = deque([self._measured()])
        self.cmd_q, self.u_hold = [], np.zeros(NU)
        self.arm_noise = np.zeros(NU - 3)

    def _measured(self):
        st = super().read_state()
        return RobotState(self.realism.b_bel.copy(), st.q.copy())

    def read_state(self):
        return self.states[0]

    def send(self, u):
        r, c = self.realism, self.realism.c
        u = np.asarray(u, float).copy()
        if r.has("net"):
            if r.rng.random() >= c.cmd_drop_p:
                delay = c.cmd_extra_ticks + (1 if r.rng.random() < c.cmd_jitter_p else 0)
                self.cmd_q.append((self.n + delay, u))
            due = [q for q in self.cmd_q if q[0] <= self.n]
            if due:
                self.u_hold = max(due, key=lambda q: q[0])[1]
                self.cmd_q = [q for q in self.cmd_q if q[0] > self.n]
            u = self.u_hold.copy()
            k = math.exp(-self.dt / 0.2)
            self.arm_noise = k * self.arm_noise + math.sqrt(1 - k * k) * r.rng.normal(0, c.arm_vel_noise, NU - 3)
            u[3:] += self.arm_noise * min(1.0, float(np.max(np.abs(u[3:]))) / 0.05)
        if r.has("base"):
            u[:3] *= r.base_gain
        super().send(u)

    def tick(self):
        self.realism.pre_step(self.model, self.data)
        super().tick()
        self.n += 1
        self.realism.odom_update(super().read_state().base)
        self.states.append(self._measured())
        delay = self.realism.c.state_delay_ticks if self.realism.has("net") else 0
        while len(self.states) > delay + 1:
            self.states.popleft()

    def stop(self):
        super().stop()
        self.cmd_q, self.u_hold = [], np.zeros(NU)
        self.arm_noise[:] = 0.0

    def shutdown(self):
        r = self.realism
        print(f"[sim realism] {r.drift_report()}; detector frames {r.n_det['frames']}, missed {r.n_det['miss']}, "
              f"outliers {r.n_det['outlier']}")
        super().shutdown()
