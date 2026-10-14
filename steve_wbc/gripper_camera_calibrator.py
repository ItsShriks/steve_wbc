#!/usr/bin/env python3
"""
gripper_camera_calibrator.py
============================
Where is the camera on the gripper? Eye-in-hand calibration of the gripper camera's mount
(ur5etool0 -> gripper_camera_color_optical_frame) with the target ball lying on the table — no
marker needed. The result goes to config/gripper_camera.yaml, which gripper_camera.launch.py
publishes as a static TF.

Put the ball (target profile, e.g. my_ball) on the table, keep the BASE still, then move the ARM
to 6-10 poses that see the ball from different directions: tilt the camera, turn wrist 3 by
+-45 deg, vary the distance (0.35-0.7 m; the D415 sees nothing closer than ~0.31 m) and where
the ball is in the image. At every pose, with the arm still, press Enter.

    ros2 launch steve_wbc gripper_camera.launch.py                       # terminal 1
    ros2 run steve_wbc gripper_camera_calibrator --target my_ball        # terminal 2
    ros2 run steve_wbc gripper_camera_calibrator --load ~/ros_logs/gripper_camera_calib/<t>/captures.json

Per capture (median of --frames depth frames): the ball centre in the camera frame (sphere fit
with the profile's radius to its depth points), the table plane (RANSAC on the other points) and
base_link <- ur5etool0 from the robot's TF (arm still: two lookups 0.4 s apart must agree).
Unknowns: the mount (6), the ball position (3), the table height (1). Residuals: the ball must
stay put, the table must be level and at one height, the ball lies on it (radius above).
Several starts (camera looking along +-tool z, turned 0/90/180/270 deg); reported: per-capture
ball residuals, the RMS, and the 1-sigma uncertainty of the mount.

Keys: Enter = capture   u = undo last   s = solve + save   q = quit
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

try:
    from steve_wbc.target_profile import TargetProfile
    from steve_wbc.table_model import tables_dir
except ImportError:
    from target_profile import TargetProfile
    from table_model import tables_dir

NAME = "gripper_camera"
SIG_BALL, SIG_N, SIG_H, SIG_ON = 0.004, 0.02, 0.004, 0.01   # m, rad, m, m


# ══════════════════════════════════════════════════════════════════════════════
# Geometry from one RGB-D view (camera optical frame)
# ══════════════════════════════════════════════════════════════════════════════
def unproject(D, K, mask=None, step=1):
    v, u = np.mgrid[0:D.shape[0]:step, 0:D.shape[1]:step]
    z = D[v, u]
    ok = (z > 0.15) & (z < 2.0) & np.isfinite(z)
    if mask is not None:
        ok &= mask[v, u]
    return np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)


def sphere_center(P, r):
    """Centre of a sphere of known radius r through the surface points P (the visible cap)."""
    m = np.median(P, axis=0)
    c0 = m + r * m / np.linalg.norm(m)          # one radius behind the surface, along the ray
    res = least_squares(lambda c: np.linalg.norm(P - c, axis=1) - r, c0, loss="soft_l1", f_scale=0.003)
    return res.x, float(np.std(res.fun))


def fit_plane(P, thr=0.006, iters=300, seed=0):
    """RANSAC + least squares. (n, d, inliers): n . p = d on the plane, n pointing to the camera."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - a) @ n) < thr).sum())
        if k > best_n:
            best, best_n = (n, a), k
    n, a = best
    S = P[np.abs((P - a) @ n) < thr]
    for _ in range(3):
        mu = S.mean(0)
        n = np.linalg.eigh(np.cov((S - mu).T))[1][:, 0]
        S = P[np.abs((P - mu) @ n) < thr]
    if n @ mu > 0:
        n = -n
    return n, float(n @ mu), len(S)


def analyse_view(D, rgb, K, target):
    """Ball centre + table plane in the camera frame from one (median) depth image."""
    found, _, _, blob = target.detect(rgb)
    if not found:
        raise ValueError("ball not detected in the colour image")
    if blob.shape != D.shape:
        raise ValueError(f"colour {blob.shape} and depth {D.shape} differ — aligned depth?")
    zb = D[blob]
    valid = float(np.mean((zb > 0.15) & np.isfinite(zb)))
    if valid < 0.5:
        raise ValueError(f"only {valid * 100:.0f} % of the ball's depth pixels are valid — camera too close "
                         "(D415: > ~0.45 m at 1280x720, > ~0.31 m at 848x480) or ball too shiny")
    Pb = unproject(D, K, blob)
    # the ball's cap is the nearest surface in its blob: drop depth outliers behind it
    z_near = np.percentile(Pb[:, 2], 5)
    Pb = Pb[Pb[:, 2] < z_near + 1.2 * target.size]
    if len(Pb) < 80:
        raise ValueError(f"only {len(Pb)} depth points on the ball — too close (< ~0.31 m) / too far?")
    c, sd = sphere_center(Pb, target.radius)
    if sd > 0.006:
        raise ValueError(f"ball depth points do not fit a {target.size * 100:.1f} cm sphere (sd {sd * 1000:.1f} mm)")
    import cv2
    around = cv2.dilate(blob.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    Pt = unproject(D, K, ~around, step=3)
    if len(Pt) < 1500:
        raise ValueError(f"only {len(Pt)} table points")
    n, d, k = fit_plane(Pt)
    if k < 1000:
        raise ValueError(f"table plane: only {k} inliers")
    return dict(c=c.tolist(), sd=sd, n=n.tolist(), d=d, plane_inliers=k, n_ball=int(len(Pb)))


# ══════════════════════════════════════════════════════════════════════════════
# Solve
# ══════════════════════════════════════════════════════════════════════════════
def _unpack(x):
    return Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6], x[6:9], x[9]


def residuals(x, caps, r, top_z=None):
    Rx, tx, b, h = _unpack(x)
    if top_z is not None:
        h = top_z
    out = []
    for cp in caps:
        R, p = np.asarray(cp["R"]), np.asarray(cp["p"])
        Rbc, tbc = R @ Rx, R @ tx + p
        out += list((Rbc @ np.asarray(cp["c"]) + tbc - b) / SIG_BALL)              # ball stays put
        n = np.asarray(cp["n"])
        out += list((Rbc @ n - [0.0, 0.0, 1.0]) / SIG_N)                            # table level
        out.append(((Rbc @ (cp["d"] * n) + tbc)[2] - h) / SIG_H)                    # one height
    out.append((b[2] - h - r) / SIG_ON)                                              # ball on it
    return np.array(out)


def solve(caps, r, top_z=None):
    """Best of several starts. Returns dict with X (4x4), ball, height, per-capture errors, sigma."""
    best = None
    for flip in (False, True):
        for k in range(4):
            R0 = Rotation.from_euler("z", k * np.pi / 2)
            if flip:
                R0 = Rotation.from_euler("x", np.pi) * R0
            Rx0 = R0.as_matrix()
            bs = [np.asarray(cp["R"]) @ (Rx0 @ np.asarray(cp["c"])) + np.asarray(cp["p"]) for cp in caps]
            b0 = np.mean(bs, axis=0)
            x0 = np.concatenate([R0.as_rotvec(), np.zeros(3), b0, [b0[2] - r]])
            sol = least_squares(residuals, x0, args=(caps, r, top_z), loss="soft_l1", f_scale=2.0)
            if best is None or sol.cost < best.cost:
                best = sol
    Rx, tx, b, h = _unpack(best.x)
    if top_z is not None:
        h = top_z
    errs = []
    for cp in caps:
        R, p = np.asarray(cp["R"]), np.asarray(cp["p"])
        Rbc, tbc = R @ Rx, R @ tx + p
        e_b = np.linalg.norm(Rbc @ np.asarray(cp["c"]) + tbc - b)
        e_n = np.degrees(np.arccos(np.clip((Rbc @ np.asarray(cp["n"]))[2], -1, 1)))
        e_h = (Rbc @ (cp["d"] * np.asarray(cp["n"])) + tbc)[2] - h
        errs.append((e_b, e_n, e_h))
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * max(1.0, 2 * best.cost / max(1, len(best.fun) - len(best.x)))
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sig = np.full(len(best.x), np.inf)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = Rx, tx
    e = np.array(errs)
    return dict(X=X, ball=b, height=h, errs=e, rms_cm=float(np.sqrt(np.mean(e[:, 0] ** 2)) * 100),
                sig_t_mm=sig[3:6] * 1000, sig_r_deg=np.degrees(sig[:3]))


def report(s, caps):
    for i, (eb, en, eh) in enumerate(s["errs"]):
        print(f"   #{i + 1}: ball {eb * 100:4.1f} cm   table tilt {en:4.1f} deg   height {eh * 100:+4.1f} cm"
              f"   (camera {np.linalg.norm(caps[i]['c']):.2f} m from the ball)")
    X = s["X"]
    rpy = Rotation.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True)
    print(f"   mount ur5etool0 -> camera: xyz {np.round(X[:3, 3], 4)} m  rpy {np.round(rpy, 1)} deg   "
          f"(+-{np.round(s['sig_t_mm'], 1)} mm, +-{np.round(s['sig_r_deg'], 2)} deg)")
    print(f"   ball at {np.round(s['ball'], 3)} (base_link), table top {s['height']:.3f} m   "
          f"ball RMS {s['rms_cm']:.2f} cm")


def save(s, caps, out, target):
    X = s["X"]
    q = Rotation.from_matrix(X[:3, :3]).as_quat()
    import yaml
    data = {"parent": "ur5etool0", "child": f"{NAME}_color_optical_frame",
            "xyz": [round(float(v), 5) for v in X[:3, 3]], "quat_xyzw": [round(float(v), 6) for v in q],
            "rpy_deg": [round(float(v), 2) for v in Rotation.from_matrix(X[:3, :3]).as_euler("xyz", degrees=True)],
            "rms_cm": round(s["rms_cm"], 2), "captures": len(caps), "target": target.name,
            "sigma_xyz_mm": [round(float(v), 1) for v in s["sig_t_mm"]],
            "sigma_rpy_deg": [round(float(v), 2) for v in s["sig_r_deg"]],
            "time": time.strftime("%Y-%m-%d %H:%M")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Gripper camera mount — written by gripper_camera_calibrator, read by gripper_camera.launch.py\n"
                   + yaml.safe_dump(data, sort_keys=False))
    print(f"saved {out}\n-> restart `ros2 launch steve_wbc gripper_camera.launch.py` to publish the mount TF")


# ══════════════════════════════════════════════════════════════════════════════
# ROS capture
# ══════════════════════════════════════════════════════════════════════════════
class Capturer:
    SFX = "/aligned_depth_to_color"

    def __init__(self, camera, base, tool, timeout):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
        self.rclpy = rclpy
        rclpy.init()
        self.node = n = rclpy.create_node("gripper_camera_calibrator")
        # TF on its own node + thread: it keeps receiving the robot's /tf while the prompt waits
        self.tf = Buffer()
        self.tf_node = rclpy.create_node("gripper_camera_calibrator_tf")
        self._tf_listener = TransformListener(self.tf, self.tf_node, spin_thread=True)
        self.base, self.tool = base, tool
        ns = camera
        if ns == "auto":
            t0, found = time.monotonic(), []
            while not found and time.monotonic() - t0 < timeout:
                rclpy.spin_once(n, timeout_sec=0.2)
                found = sorted(t for t, _ in n.get_topic_names_and_types()
                               if t.endswith(self.SFX + "/camera_info") and n.count_publishers(t) > 0)
            if not found:
                rclpy.try_shutdown()
                sys.exit("[ERROR] no */aligned_depth_to_color/camera_info — gripper_camera.launch.py running "
                         "(and this terminal on the robot's ROS_DOMAIN_ID / DDS profile)?")
            found.sort(key=lambda t: NAME not in t)
            ns = found[0][:-len(self.SFX + "/camera_info")]
        print(f"camera: {ns}")
        t0 = time.monotonic()
        from rclpy.time import Time
        while not self.tf.can_transform(base, tool, Time()):
            if time.monotonic() - t0 > timeout:
                rclpy.try_shutdown()  # ends the TF thread, else the process would not exit
                sys.exit(f"[ERROR] no TF {base} <- {tool} from the robot within {timeout:.0f} s — robot bringup "
                         "running? this terminal on the robot's ROS_DOMAIN_ID / DDS profile?")
            time.sleep(0.2)
        print(f"robot TF {base} <- {tool}: ok")
        self.K, self.rgb, self.depth, self.want = None, None, [], 0
        n.create_subscription(CameraInfo, f"{ns}{self.SFX}/camera_info",
                              lambda m: setattr(self, "K", np.array(m.k).reshape(3, 3)), 10)
        n.create_subscription(Image, f"{ns}/color/image_raw", self._rgb, qos_profile_sensor_data)
        n.create_subscription(Image, f"{ns}{self.SFX}/image_raw", self._depth, qos_profile_sensor_data)

    def _rgb(self, m):
        if self.want and self.rgb is None and m.encoding in ("rgb8", "bgr8"):
            im = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.rgb = im[..., ::-1].copy() if m.encoding == "bgr8" else im.copy()

    def _depth(self, m):
        if self.want and len(self.depth) < self.want:
            if m.encoding in ("16UC1", "mono16"):
                d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
            else:
                d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
            self.depth.append(d)

    def spin(self, sec):
        t0 = time.monotonic()
        while time.monotonic() - t0 < sec:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)

    def _tool_pose(self):
        from rclpy.time import Time
        tr = self.tf.lookup_transform(self.base, self.tool, Time())
        q, t = tr.transform.rotation, tr.transform.translation
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([t.x, t.y, t.z])

    def capture(self, frames, target):
        self.spin(0.3)
        try:
            R1, p1 = self._tool_pose()
            self.spin(0.4)
            R2, p2 = self._tool_pose()
        except Exception as e:
            raise ValueError(f"no TF {self.base} <- {self.tool} from the robot: {e}")
        if np.linalg.norm(p1 - p2) > 0.002 or np.degrees(np.linalg.norm(Rotation.from_matrix(R1.T @ R2).as_rotvec())) > 0.2:
            raise ValueError("arm still moving — wait until it stands still")
        self.rgb, self.depth, self.want = None, [], frames
        t0 = time.monotonic()
        while (len(self.depth) < frames or self.rgb is None or self.K is None) and time.monotonic() - t0 < 5:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)
        self.want = 0
        if len(self.depth) < frames or self.rgb is None or self.K is None:
            raise ValueError("no camera frames within 5 s")
        R3, p3 = self._tool_pose()
        if np.linalg.norm(p3 - p2) > 0.002:
            raise ValueError("arm moved during the capture")
        D = np.median(np.stack(self.depth), axis=0)
        v = analyse_view(D, self.rgb, self.K, target)
        v.update(R=R3.tolist(), p=p3.tolist())
        return v


# ══════════════════════════════════════════════════════════════════════════════
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="my_ball", help="target profile of the ball on the table")
    p.add_argument("--camera", default="auto", help="RealSense namespace (auto: the gripper camera)")
    p.add_argument("--base", default="base_link")
    p.add_argument("--tool", default="ur5etool0")
    p.add_argument("--frames", type=int, default=5, help="depth frames per capture (median)")
    p.add_argument("--min", type=int, default=6, help="captures needed before saving")
    p.add_argument("--top_z", type=float, default=None, help="table top height in base_link if known [m]")
    p.add_argument("--load", default=None, help="solve saved captures (captures.json) instead of capturing")
    p.add_argument("--out", default=None, help=f"result (default config/{NAME}.yaml in the steve_wbc source)")
    p.add_argument("--timeout", type=float, default=15.0)
    a = p.parse_args(argv)
    target = TargetProfile.load(a.target)
    out = Path(a.out) if a.out else tables_dir().parent / f"{NAME}.yaml"

    if a.load:
        caps = json.loads(Path(a.load).expanduser().read_text())
        s = solve(caps, target.radius, a.top_z)
        report(s, caps)
        if input(f"save to {out}? [y/N] ").strip().lower() == "y":
            save(s, caps, out, target)
        return 0

    log = Path.home() / "ros_logs" / "gripper_camera_calib" / time.strftime("%Y%m%d_%H%M%S")
    log.mkdir(parents=True, exist_ok=True)
    cap = Capturer(a.camera, a.base, a.tool, a.timeout)
    print(f"ball profile '{target.name}' ({target.size * 100:.1f} cm). Move the arm (base still), then:\n"
          "  Enter = capture   u = undo   s = solve + save   q = quit")
    caps, s = [], None
    while True:
        k = input(f"[{len(caps)} captures] > ").strip().lower()
        if k == "q":
            break
        if k == "u":
            if caps:
                caps.pop()
                print("  last capture removed")
            continue
        if k == "s":
            if len(caps) < a.min:
                print(f"  need at least {a.min} captures (from different directions)")
                continue
            s = solve(caps, target.radius, a.top_z)
            report(s, caps)
            if s["rms_cm"] > 1.0:
                print("  [WARN] ball RMS > 1 cm: remove bad captures (u) / add more varied ones")
            if input(f"  save to {out}? [y/N] ").strip().lower() == "y":
                save(s, caps, out, target)
            continue
        try:
            v = cap.capture(a.frames, target)
        except ValueError as e:
            print(f"  not captured: {e}")
            continue
        caps.append(v)
        (log / "captures.json").write_text(json.dumps(caps))
        print(f"  #{len(caps)}: ball {np.linalg.norm(v['c']):.2f} m away ({v['n_ball']} points, sphere fit "
              f"{v['sd'] * 1000:.1f} mm), table {v['plane_inliers']} points, tool at {np.round(v['p'], 3)}")
        if len(caps) >= 4:
            s = solve(caps, target.radius, a.top_z)
            print(f"  so far: ball RMS {s['rms_cm']:.2f} cm, mount xyz {np.round(s['X'][:3, 3], 3)} m "
                  f"(+-{np.round(s['sig_t_mm'], 0)} mm)")
    print(f"captures saved in {log}/captures.json (re-solve with --load)")
    cap.node.destroy_node()
    cap.rclpy.try_shutdown()   # also stops the TF thread
    return 0


if __name__ == "__main__":
    sys.exit(main())
