#!/usr/bin/env python3
"""
vgn_grasp_real.py
=================
VGN grasp on the real robot: gripper/wrist-camera depth -> TSDF -> VGN -> the best grasp on the
object that sits on the table (e.g. a ball), saved for the WBC (`steve_figure8_wbc --goto_file`).

Runs where VGN runs (the PC with vgn_env); reads the camera and TF over ROS (Wi-Fi works with
the Steve DDS profiles). Park the arm so the camera looks down at the object from ~0.4-0.6 m
(D415: no depth closer than ~0.31 m at 848x480), then:

    ros2 run steve_wbc vgn_grasp_real                        # table measured in this view (no model)
    ros2 run steve_wbc vgn_grasp_real --table lab_table      # a table model in the current odom

The camera (--camera auto: the RealSense with aligned depth, gripper camera preferred) needs its
pose in TF: odom <- ... <- ur5etool0 (robot) <- gripper_camera_color_optical_frame (static mount
TF from gripper_camera.launch.py after gripper_camera_calibrator).

Steps
  1. depth frames (aligned to colour) + camera_info + TF  <frame> <- camera optical frame
  1b. without --table: the dominant horizontal plane in view is the table top; its visible
     rectangle is saved as table model --save_table (seen_table) for the WBC's collision checks
  2. object = largest compact cluster of depth points above the table top (inside its outline)
  3. VGN on a 0.30 m TSDF centred on the object in x/y with the table plane 5 cm above the volume's
     floor (as in VGN's training scenes); only grasps within `--near` of it count
  4. gripper point (pinch) = VGN grasp + finger depth along the approach axis; orientation:
     top-down, rotated about z to VGN's finger axis (as in the MuJoCo pipeline)
Output
  * ~/ros_logs/vgn_grasp.json  (pinch, pregrasp, quaternion, score, width, object, frame)
  * /vgn/grasp_pose (PoseStamped, latched) and /vgn/markers for RViz
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from steve_wbc.table_model import TableModel, lidar_clusters
except ImportError:
    from table_model import TableModel, lidar_clusters

try:
    from steve_perception.vgn import VGNBridge
except ImportError:  # steve_perception not built (e.g. plain script): use the source-tree sibling
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "steve_perception"))
    from steve_perception.vgn import VGNBridge

R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)


def quat_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [float(w), float(x), float(y), float(z)]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fit_table_plane(cloud, table, band=0.06, iters=4):
    """Robust least-squares plane z = a x + b y + c through the depth points near the model's top
    height and inside its outline. Returns (normal (up), point on the plane, inlier count)."""
    loc = (cloud[:, :2] - np.asarray(table.center)) @ table.R()
    inside = np.all(np.abs(loc) < 0.5 * np.asarray(table.size) - 0.02, axis=1)
    pts = cloud[inside & (np.abs(cloud[:, 2] - table.top_z) < band)]
    if len(pts) < 200:
        raise ValueError(f"only {len(pts)} depth points near the table top — table in view?")
    for _ in range(iters):  # drop points on objects (above the plane) and outliers
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        res = pts[:, 2] - A @ coef
        keep = np.abs(res) < max(0.006, 2.5 * np.std(res))
        pts = pts[keep]
    a, b, c = coef
    n = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
    p0 = np.array([pts[:, 0].mean(), pts[:, 1].mean(), a * pts[:, 0].mean() + b * pts[:, 1].mean() + c])
    return n, p0, len(pts)


def fit_horizontal_plane(cloud, max_tilt=20.0, thr=0.008, iters=400, seed=0):
    """Dominant near-horizontal plane (RANSAC + least squares) in a world cloud: (normal up, point,
    inlier mask). For the table top when no table model is given."""
    rng = np.random.default_rng(seed)
    cos_max, best, best_k = np.cos(np.radians(max_tilt)), None, -1
    for _ in range(iters):
        a, b, c = cloud[rng.choice(len(cloud), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < cos_max:
            continue
        k = int((np.abs((cloud - a) @ n) < thr).sum())
        if k > best_k:
            best, best_k = (n, a), k
    if best is None or best_k < 500:
        raise ValueError("no horizontal plane (table top) in view")
    n, a = best
    inl = np.abs((cloud - a) @ n) < thr
    for _ in range(3):
        mu = cloud[inl].mean(0)
        n = np.linalg.eigh(np.cov((cloud[inl] - mu).T))[1][:, 0]
        inl = np.abs((cloud - mu) @ n) < thr
    if n[2] < 0:
        n = -n
    return n, cloud[inl].mean(0), inl


def table_from_plane(pts, top_z, name):
    """Table model of the VISIBLE part of the top plate (min-area rectangle of its points)."""
    import cv2
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
    return TableModel(name=name, frame="odom", center=[float(cx), float(cy)], yaw=float(np.radians(ang)),
                      size=[float(w), float(h)], top_z=float(top_z), thickness=0.03, legs=[], skirt=0.0,
                      notes=f"vgn_grasp_real: visible part of the top plate, {time.strftime('%Y-%m-%d %H:%M')}")


def level_correction(n, p0, top_z):
    """Rotation R_c (about p0) + height shift that make the fitted plane horizontal at top_z:
    x' = R_c (x - p0) + p0 + [0, 0, top_z - p0_z]."""
    up = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, up)
    s, c = np.linalg.norm(v), float(np.dot(n, up))
    if s < 1e-9:
        R_c = np.eye(3)
    else:
        k = v / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R_c = np.eye(3) + s * Kx + (1 - c) * Kx @ Kx
    return R_c, np.array([0.0, 0.0, top_z - p0[2]])


def find_object(cloud, table, margin=0.03, h_max=0.25, max_ext=0.15):
    """Largest compact cluster of points above the table top and inside its outline (world)."""
    loc = (cloud[:, :2] - np.asarray(table.center)) @ table.R()
    lim = 0.5 * np.asarray(table.size) - margin
    m = (np.abs(loc[:, 0]) < lim[0]) & (np.abs(loc[:, 1]) < lim[1]) & \
        (cloud[:, 2] > table.top_z + 0.01) & (cloud[:, 2] < table.top_z + h_max)
    pts = cloud[m]
    if len(pts) < 30:
        raise ValueError(f"only {len(pts)} depth points above the table top — is the object in view?")
    clusters = lidar_clusters(pts[:, :2], link=0.015)
    best = None
    for c in clusters:
        if len(c) < 20 or np.ptp(c, axis=0).max() > max_ext:
            continue
        if best is None or len(c) > len(best):
            best = c
    if best is None:
        raise ValueError("no compact object on the table (larger than 15 cm, or too few points)")
    xy = best.mean(0)
    r = 0.5 * np.ptp(best, axis=0).max() + 0.01
    obj = pts[np.hypot(*(pts[:, :2] - xy).T) < r]
    z_top = float(np.percentile(obj[:, 2], 95))
    return np.array([xy[0], xy[1], 0.5 * (table.top_z + z_top)]), z_top, len(best)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default=None,
                   help="table model (config/tables/<name>.yaml) in the CURRENT odom; default: measure the "
                        "table top in this view")
    p.add_argument("--save_table", default="seen_table",
                   help="without --table: save the measured (visible) table top as this table model")
    p.add_argument("--camera", default="auto",
                   help="RealSense namespace (topics <ns>/aligned_depth_to_color/...); auto: found, gripper first")
    p.add_argument("--frame", default="odom", help="world frame (must be the table model's)")
    p.add_argument("--frames", type=int, default=3, help="depth frames to fuse (same pose)")
    p.add_argument("--finger_depth", type=float, default=0.05,
                   help="VGN grasp origin -> fingertip centre along the approach axis [m]")
    p.add_argument("--pregrasp", type=float, default=0.10, help="pre-grasp distance above the grasp [m]")
    p.add_argument("--near", type=float, default=0.05, help="grasps within this xy radius of the object [m]")
    p.add_argument("--out", default=str(Path.home() / "ros_logs" / "vgn_grasp.json"))
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--no_level", action="store_true",
                   help="do not correct the camera pose with the table plane (level at top_z)")
    p.add_argument("--max_level", type=float, default=8.0,
                   help="refuse corrections tilting more than this [deg]")
    a = p.parse_args(argv)

    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
    from rclpy.time import Time
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    from tf2_ros import Buffer, TransformListener
    from visualization_msgs.msg import Marker, MarkerArray

    table = TableModel.load(a.table) if a.table else None
    rclpy.init()
    n = rclpy.create_node("vgn_grasp_real")
    tf = Buffer()
    TransformListener(tf, n)
    if a.camera == "auto":
        sfx, found, t0 = "/aligned_depth_to_color/camera_info", [], time.monotonic()
        while not found and time.monotonic() - t0 < a.timeout:
            rclpy.spin_once(n, timeout_sec=0.2)
            found = sorted(t for t, _ in n.get_topic_names_and_types() if t.endswith(sfx) and n.count_publishers(t) > 0)
        if not found:
            sys.exit("[ERROR] no */aligned_depth_to_color/camera_info — camera running (gripper_camera.launch.py)? "
                     "this terminal on the robot's ROS_DOMAIN_ID / DDS profile?")
        found.sort(key=lambda t: ("gripper" not in t, "wrist" not in t))
        a.camera = found[0][:-len(sfx)]
        print(f"camera: {a.camera}")
    st = {"K": None, "depth": []}
    n.create_subscription(CameraInfo, f"{a.camera}/aligned_depth_to_color/camera_info",
                          lambda m: st.__setitem__("K", np.array(m.k).reshape(3, 3)), 10)

    def depth_cb(m):
        if len(st["depth"]) >= a.frames:
            return
        if m.encoding in ("16UC1", "mono16"):
            d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
        else:
            d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
        st["depth"].append((d, m.header.frame_id))
    n.create_subscription(Image, f"{a.camera}/aligned_depth_to_color/image_raw", depth_cb, qos_profile_sensor_data)

    t0 = time.monotonic()
    while (st["K"] is None or len(st["depth"]) < a.frames) and time.monotonic() - t0 < a.timeout:
        rclpy.spin_once(n, timeout_sec=0.1)
    if st["K"] is None or not st["depth"]:
        sys.exit(f"[ERROR] no depth / camera_info on {a.camera} within {a.timeout:.0f} s")
    K, cam_frame = st["K"], st["depth"][0][1]
    try:
        for _ in range(20):
            rclpy.spin_once(n, timeout_sec=0.05)
        tr = tf.lookup_transform(a.frame, cam_frame, Time())
    except Exception as e:
        sys.exit(f"[ERROR] no TF {a.frame} <- {cam_frame}: {e}\n        gripper camera: the mount TF comes from "
                 "gripper_camera.launch.py once gripper_camera_calibrator saved config/gripper_camera.yaml")
    R_wc = quat_R(tr.transform.rotation)
    t_wc = np.array([tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z])
    # median over the frames (the arm is still): less speckle
    D = np.median(np.stack([d for d, _ in st["depth"]]), axis=0)
    print(f"{len(st['depth'])} depth frames {D.shape[1]}x{D.shape[0]}, camera at {np.round(t_wc, 3)} "
          f"looking {np.round(R_wc[:, 2], 2)} ({a.frame})")

    # point cloud in the world frame
    H, W = D.shape
    v, u = np.mgrid[0:H:2, 0:W:2]
    z = D[v, u]
    ok = (z > 0.2) & (z < 2.0)
    pc = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
    cloud = pc @ R_wc.T + t_wc
    if np.median(z[ok]) < 0.33:
        print(f"  [WARN] median depth {np.median(z[ok]):.2f} m: closer than the D415's range (~0.31 m at 848x480)")
    if table is None:
        # no model: the table top is the dominant horizontal plane in view; level the camera pose on
        # it (tilt only — its height is the measurement) and save the visible top as a table model
        try:
            n_pl, p_pl, inl = fit_horizontal_plane(cloud)
        except ValueError as e:
            sys.exit(f"[ERROR] {e}")
        tilt = np.degrees(np.arccos(np.clip(n_pl[2], -1, 1)))
        if tilt > a.max_level:
            sys.exit(f"[ERROR] table plane tilted {tilt:.1f} deg in odom: camera mount TF wrong? (--max_level)")
        R_c, _ = level_correction(n_pl, p_pl, p_pl[2])
        if not a.no_level:
            cloud = (cloud - p_pl) @ R_c.T + p_pl
            t_wc = R_c @ (t_wc - p_pl) + p_pl
            R_wc = R_c @ R_wc
        table = table_from_plane(cloud[inl], p_pl[2], a.save_table)
        path = table.save()
        print(f"table top measured: z {table.top_z:.3f} m, visible {table.size[0]:.2f} x {table.size[1]:.2f} m, "
              f"tilt {tilt:.1f} deg {'(levelled)' if not a.no_level else ''} -> table model '{table.name}' ({path})")
    else:
        top_pts = cloud[np.abs(cloud[:, 2] - table.top_z) < 0.03]
        print(f"table plane check: {len(top_pts)} points within 3 cm of the model top ({table.top_z:.2f} m)"
              + (f", their median z {np.median(top_pts[:, 2]):.3f} m" if len(top_pts) else " — camera pose / table model wrong?"))
    if a.table and not a.no_level:
        # the table top is level at top_z: correct the camera pose's tilt + height with it (the wrist
        # camera's mount in the URDF is approximate); x/y offsets are not observable from a plane
        try:
            n_pl, p_pl, n_in = fit_table_plane(cloud, table)
        except ValueError as e:
            sys.exit(f"[ERROR] {e}")
        R_c, dz = level_correction(n_pl, p_pl, table.top_z)
        tilt = np.degrees(np.arccos(np.clip(n_pl[2], -1, 1)))
        print(f"table plane fit ({n_in} pts): tilted {tilt:.1f} deg (normal {np.round(n_pl, 3)}), "
              f"{(p_pl[2] - table.top_z) * 100:+.1f} cm vs the model -> camera pose corrected")
        if tilt > a.max_level or abs(dz[2]) > 0.06:
            sys.exit(f"[ERROR] correction too large ({tilt:.1f} deg, {dz[2] * 100:.1f} cm): wrong table model / "
                     "camera mount — measure the mount, or --max_level")
        cloud = (cloud - p_pl) @ R_c.T + p_pl + dz
        t_wc = R_c @ (t_wc - p_pl) + p_pl + dz
        R_wc = R_c @ R_wc
    try:
        obj, z_top, n_pts = find_object(cloud, table)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    print(f"object at {np.round(obj, 3)} (top {z_top:.3f} m, {n_pts} points)")

    vgn = VGNBridge()
    if not vgn.available:
        sys.exit("[ERROR] VGN not available (vgn_env / model) — run this on the PC with vgn_env")
    # 30 cm TSDF: centred on the object in x/y, table plane 5 cm above its floor (VGN's training scenes)
    centre = np.array([obj[0], obj[1], table.top_z - 0.05 + 0.5 * vgn.tsdf_size])
    for d, _ in st["depth"]:
        vgn.add_real_view(d, t_wc, R_wc, K, centre)
    if not vgn.run_inference():
        sys.exit("[ERROR] VGN found no grasp")
    g = vgn.best_grasp_world(near=obj, xy_radius=a.near, z_window=0.10)
    if g is None:
        sys.exit(f"[ERROR] no VGN grasp within {a.near * 100:.0f} cm of the object")
    pos, rot, width, score = g
    approach = rot[:, 2] / np.linalg.norm(rot[:, 2])
    pinch = pos + a.finger_depth * approach
    ax = rot[:, 1]
    R_goal = rot_z(np.arctan2(ax[1], ax[0]) - np.pi / 2) @ R_TOPDOWN   # top-down, VGN finger axis
    pregrasp = pinch + np.array([0.0, 0.0, a.pregrasp])
    res = {"frame": a.frame, "table": table.name, "object": obj.tolist(), "pinch": pinch.tolist(),
           "pregrasp": pregrasp.tolist(), "quat_wxyz": R_to_quat_wxyz(R_goal), "score": float(score),
           "width": float(width), "vgn_pos": pos.tolist(), "vgn_approach": approach.tolist(),
           "camera": t_wc.tolist(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"grasp: pinch {np.round(pinch, 3)}  (object {np.round(obj, 3)}, off by "
          f"{np.linalg.norm(pinch[:2] - obj[:2]) * 100:.1f} cm in xy)  score {score:.2f}  "
          f"width {width * 100:.1f} cm  approach {np.round(approach, 2)}\nsaved {a.out}")

    # RViz
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pp = n.create_publisher(PoseStamped, "/vgn/grasp_pose", latched)
    mp = n.create_publisher(MarkerArray, "/vgn/markers", latched)
    ps = PoseStamped()
    ps.header.frame_id = a.frame
    ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (float(v) for v in pinch)
    w, x, y, zq = res["quat_wxyz"]
    ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = w, x, y, zq
    pp.publish(ps)
    ma = MarkerArray()
    for i, (pt, rgba, s) in enumerate(((obj, (1, 0, 0, 0.8), 0.06), (pinch, (0, 1, 0, 0.9), 0.02),
                                       (pregrasp, (0, 0.6, 1, 0.9), 0.02))):
        m = Marker()
        m.header.frame_id, m.ns, m.id, m.type = a.frame, "vgn", i, Marker.SPHERE
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pt)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
        ma.markers.append(m)
    mp.publish(ma)
    print("RViz: PoseStamped /vgn/grasp_pose, MarkerArray /vgn/markers (red object, green grasp, blue pre-grasp)")
    t_end = time.monotonic() + 2.0
    while time.monotonic() < t_end:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
