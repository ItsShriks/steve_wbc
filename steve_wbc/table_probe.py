#!/usr/bin/env python3
"""
table_probe.py
==============
Measure the real table with the robot's own sensors and save it as a table model
(steve_wbc/config/tables/<name>.yaml) for the WBC and MuJoCo — instead of assuming the
simulation's closed box-like table.

Park the robot facing the table (~0.5-1 m away), point the pan-tilt camera at it, then:

    ros2 run steve_wbc table_probe --name lab_table            # measure + save + debug plot
    ros2 run steve_wbc table_probe --check lab_table           # overlay a saved model on live data

  * top plate: dominant horizontal plane in the head camera's depth cloud (RANSAC) and the
    minimum-area rectangle of its points -> height, size, position, heading
  * legs / skirt: LiDAR points under the top plate (compact clusters = legs, long runs = skirt)
  * saved in `odom` (valid while the base driver keeps running) + a top-view PNG with the
    LiDAR points, the fitted table, the robot footprint and its 15 cm safety field, and RViz
    markers on /table_probe/markers
  * --check: per leg the distance to the nearest LiDAR point and the base clearance, to see
    at a glance whether the model still matches reality

Then:  ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false table:=lab_table
       (MuJoCo with the same table: ... simulator:=mujoco table:=lab_table)
"""

import argparse
import sys
import time

import numpy as np

try:
    from steve_wbc.table_model import TableModel, fit_table, fit_table_lidar, tables_dir
    from steve_manipulation.safety_stop import SCAN_TOPICS, FOOTPRINT_MAX, FOOTPRINT_MIN, D_SAFE, ScanPoints
except ImportError:
    from table_model import TableModel, fit_table, fit_table_lidar, tables_dir
    from steve_manipulation.safety_stop import SCAN_TOPICS, FOOTPRINT_MAX, FOOTPRINT_MIN, D_SAFE, ScanPoints

HEAD = "/pan_tilt_camera/pan_tilt_camera"
# Known lab table: geometry fixed, only its pose is measured (LiDAR legs) — no camera needed.
# Height from the earlier depth-camera probe; override with --top_z.
TABLE_TOP_Z = 0.827
TABLE_LEGS = 5   # 4 corners + 1 in the middle of one long side
R_OPT_TO_XFWD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quat_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Probe:
    def __init__(self, a):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_ros import Buffer, TransformListener
        self.rclpy, self.a = rclpy, a
        rclpy.init()
        n = self.node = rclpy.create_node("table_probe")
        self.tf = Buffer()
        self._tfl = TransformListener(self.tf, n)
        self.scans = ScanPoints(n, a.lidar_topics, frame="base_link")
        self.K, self.depth = None, None
        # camera disabled: table height is hardcoded (TABLE_TOP_Z), the pose comes from the LiDAR
        # n.create_subscription(CameraInfo, f"{a.camera}/aligned_depth_to_color/camera_info",
        #                       lambda m: setattr(self, "K", np.array(m.k).reshape(3, 3)), 10)
        # n.create_subscription(Image, f"{a.camera}/aligned_depth_to_color/image_raw", self._depth,
        #                       qos_profile_sensor_data)

    def _depth(self, msg):
        if msg.encoding in ("16UC1", "mono16"):
            d = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        else:
            d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        self.depth = (d, msg.header.frame_id)

    def T(self, target, source):
        from rclpy.time import Time
        tf = self.tf.lookup_transform(target, source, Time())
        t = tf.transform.translation
        return quat_R(tf.transform.rotation), np.array([t.x, t.y, t.z])

    def spin(self, secs):
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)

    def collect(self, secs=2.0, step=4):
        """LiDAR (x, y) and depth cloud (x, y, z) in odom, accumulated over `secs`."""
        self.spin(1.0)  # TF + first messages
        lid, cloud = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            self.spin(0.2)
            Rb, tb = self.T(self.a.frame, "base_link")
            p = self.scans.points()
            if p is not None:
                lid.append((np.column_stack([p, np.zeros(len(p))]) @ Rb.T + tb)[:, :2])
            if self.depth is not None and self.K is not None:
                d, frame = self.depth
                ys, xs = np.mgrid[0:d.shape[0]:step, 0:d.shape[1]:step]
                z = d[ys, xs]
                ok = np.isfinite(z) & (z > 0.15) & (z < 4.0)
                K = self.K
                pc = np.stack([(xs[ok] - K[0, 2]) * z[ok] / K[0, 0], (ys[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
                if self.a.camera_conv == "x_forward":
                    pc = pc @ R_OPT_TO_XFWD.T
                Rc, tc = self.T(self.a.frame, self.a.camera_frame or frame)
                cloud.append(pc @ Rc.T + tc)
        if not lid:
            sys.exit(f"[ERROR] no LiDAR scans on {self.a.lidar_topics} (steve_doctor --ros)")
        Rb, tb = self.T(self.a.frame, "base_link")
        robot = (tb[:2], np.arctan2(Rb[1, 0], Rb[0, 0]))
        return np.vstack(lid), (np.vstack(cloud) if cloud else np.zeros((0, 3))), robot


def footprint(robot):
    (x, y), yaw = robot
    lo, hi = np.array(FOOTPRINT_MIN), np.array(FOOTPRINT_MAX)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    rect = lambda lo, hi: np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]], [lo[0], lo[1]]])
    return (rect(lo, hi) @ R.T + [x, y]), (rect(lo - D_SAFE, hi + D_SAFE) @ R.T + [x, y])


def plot(path, table, lidar, info, robot, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax.scatter(lidar[:, 0], lidar[:, 1], s=2, c="0.55", label="LiDAR points")
    if info and len(info.get("plane_inliers", [])):
        pi = info["plane_inliers"][::10]
        ax.scatter(pi[:, 0], pi[:, 1], s=1, c="tab:orange", alpha=0.3, label="table top (depth)")
    c = np.vstack([table.corners(), table.corners()[:1]])
    ax.plot(c[:, 0], c[:, 1], "tab:brown", lw=2, label=f"table top {table.size[0]:.2f} x {table.size[1]:.2f} m")
    for i, (lx, ly, r) in enumerate(table.legs):
        w = table.to_world([lx, ly])
        ax.add_patch(plt.Circle(w, r, color="tab:red", alpha=0.7))
        ax.annotate(f"leg {i}", w, textcoords="offset points", xytext=(4, 4), fontsize=8)
    fp, field = footprint(robot)
    ax.plot(fp[:, 0], fp[:, 1], "tab:blue", lw=2, label="robot footprint")
    ax.plot(field[:, 0], field[:, 1], "tab:red", ls="--", lw=1, label=f"{D_SAFE * 100:.0f} cm safety field")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title)
    # side view: table height vs robot parts
    bx.bar(["LiDAR", "base top", "table top", "cabinet top"], [0.22, 0.37, table.top_z, 1.1],
           color=["0.6", "tab:blue", "tab:brown", "tab:blue"])
    bx.axhline(table.top_z, color="tab:brown", ls=":")
    bx.set_ylabel("height [m]")
    bx.set_title("the LiDARs cannot see the top plate;\nthe cabinet is higher than it")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def publish_markers(node, table, frame):
    from visualization_msgs.msg import Marker, MarkerArray
    from rclpy.qos import DurabilityPolicy, QoSProfile
    pub = getattr(node, "_table_marker_pub", None)
    if pub is None:
        pub = node._table_marker_pub = node.create_publisher(
            MarkerArray, "/table_probe/markers", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    ma = MarkerArray()

    def marker(i, kind, xyz, scale, rgba, yaw=0.0):
        m = Marker()
        m.header.frame_id, m.ns, m.id, m.type = frame, "table", i, kind
        m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in xyz)
        m.pose.orientation.z, m.pose.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m
    ma.markers.append(marker(0, Marker.CUBE, [*table.center, table.top_z - table.thickness / 2],
                             [*table.size, table.thickness], (0.6, 0.4, 0.2, 0.8), table.yaw))
    for i, (lx, ly, r) in enumerate(table.legs):
        w = table.to_world([lx, ly])
        ma.markers.append(marker(1 + i, Marker.CYLINDER, [*w, table.top_z / 2], [2 * r, 2 * r, table.top_z],
                                 (0.8, 0.2, 0.2, 0.8)))
    pub.publish(ma)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="table", help="model name (config/tables/<name>.yaml)")
    p.add_argument("--check", metavar="TABLE", help="compare a saved model with the live LiDAR")
    p.add_argument("--frame", default="odom", help="frame the model is stored in")
    p.add_argument("--camera", default=HEAD, help="RealSense namespace with aligned depth")
    p.add_argument("--camera_frame", default="", help="override the depth image's frame_id")
    p.add_argument("--camera_conv", choices=["optical", "x_forward"], default="optical",
                   help="frame convention of the depth image's frame (Gazebo head camera: x_forward)")
    p.add_argument("--lidar_topics", nargs="+", default=list(SCAN_TOPICS["real"]))
    p.add_argument("--seconds", type=float, default=2.0, help="measurement time")
    p.add_argument("--lidar_only", action="store_true",
                   help="(now the default, kept for compatibility) no camera: find the 4 legs in the "
                        "LiDAR scan (rectangle in front of the robot, roughly square to it)")
    p.add_argument("--top_z", type=float, default=TABLE_TOP_Z,
                   help=f"table top height above the floor [m] (default: hardcoded {TABLE_TOP_Z})")
    p.add_argument("--overhang", type=float, default=0.03, help="top plate beyond the legs [m]")
    p.add_argument("--show", metavar="TABLE",
                   help="only show a saved model in RViz (MarkerArray /table_probe/markers, frame of "
                        "the model) until Ctrl+C — no measurement")
    if argv is None:  # strip --ros-args (when started from a launch file)
        try:
            from rclpy.utilities import remove_ros_args
            argv = remove_ros_args(sys.argv)[1:]
        except ImportError:
            argv = sys.argv[1:]
    a = p.parse_args(argv)

    if a.show:
        # keeps publishing the model; reloads it whenever the file changes (a new table_probe run)
        # and waits if it does not exist yet — started by hardware_bringup.launch.py
        import rclpy
        from rclpy.executors import ExternalShutdownException
        path = tables_dir() / f"{a.show}.yaml"
        rclpy.init()
        node = rclpy.create_node("table_probe_show")
        table, mtime, waiting = None, None, False
        try:
            while rclpy.ok():
                try:
                    m = path.stat().st_mtime
                except OSError:
                    m = None
                    if not waiting:
                        print(f"waiting for {path} (run table_probe --lidar_only ...)")
                        waiting = True
                if m is not None and m != mtime:
                    try:
                        table, mtime, waiting = TableModel.load(a.show), m, False
                        print(f"showing '{a.show}' ({table.notes}): centre {np.round(table.center, 3)} "
                              f"{table.frame}, top {table.top_z:.2f} m, {len(table.legs)} legs — RViz: "
                              f"Fixed Frame {table.frame}, MarkerArray /table_probe/markers")
                    except Exception as e:  # half-written file: retry next second
                        print(f"cannot read {path}: {e}")
                if table is not None:
                    publish_markers(node, table, table.frame)
                t_end = time.monotonic() + 1.0
                while time.monotonic() < t_end:
                    rclpy.spin_once(node, timeout_sec=0.1)
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        return 0

    probe = Probe(a)
    lidar, cloud, robot = probe.collect(a.seconds)
    if a.check:
        table = TableModel.load(a.check)
        print(f"model '{a.check}': centre {np.round(table.center, 3)}, top z {table.top_z:.3f} m, "
              f"{len(table.legs)} legs")
        for i, (lx, ly, r) in enumerate(table.legs):
            w = table.to_world([lx, ly])
            d = np.min(np.linalg.norm(lidar - w, axis=1)) - r
            print(f"  leg {i} at {np.round(w, 3)}: nearest LiDAR point {d * 100:+.1f} cm "
                  f"{'OK' if abs(d) < 0.05 else '<- model and reality differ (table moved? re-probe)'}")
        info, out = None, tables_dir() / f"{a.check}_check.png"
    else:  # LiDAR only, hardcoded top height (was: `elif a.lidar_only:`)
        try:
            table, info = fit_table_lidar(lidar, robot, a.top_z, a.overhang)
        except ValueError as e:
            sys.exit(f"[ERROR] {e}")
        table.name, table.frame = a.name, a.frame
        # in the robot frame (what one would measure with a tape)
        (rx, ry), ryaw = robot
        Rr = np.array([[np.cos(ryaw), -np.sin(ryaw)], [np.sin(ryaw), np.cos(ryaw)]])
        c_r = (np.asarray(table.center) - [rx, ry]) @ Rr
        print(f"table found (LiDAR, {len(info['leg_like'])} leg-like clusters, shape error "
              f"{info['shape_err'] * 100:.1f} cm): centre {c_r[0]:.2f} m ahead, {c_r[1]:+.2f} m left, "
              f"rotated {np.degrees(table.yaw - ryaw):+.1f}°, top {table.size[0]:.2f} x {table.size[1]:.2f} m, "
              f"{len(table.legs)} legs ({info['extra_legs']} besides the 4 corners)")
        for i, (e, ctr, yaw) in enumerate(info["candidates"][1:4], 1):
            print(f"  alternative {i}: centre {ctr[0]:.2f} m ahead {ctr[1]:+.2f} m left, {yaw:+.1f}°, "
                  f"shape error {e * 100:.1f} cm")
        if len(table.legs) != TABLE_LEGS:
            print(f"  [WARN] {len(table.legs)} legs found, the lab table has {TABLE_LEGS} — a leg hidden "
                  f"from the LiDAR? Move the robot so all legs are in view and re-run.")
        info["plane_inliers"] = []
        if a.top_z is None:
            out = tables_dir() / f"{a.name}_lidar_detect.png"
            print("  not saved: give the top height with --top_z <m> to save the model")
        else:
            table.notes = f"table_probe --lidar_only {time.strftime('%Y-%m-%d %H:%M')}: top_z {a.top_z}"
            path = table.save()
            print(f"saved {path}")
            out = path.with_suffix(".png")
    # camera path (depth RANSAC top plate) — disabled while the table height is hardcoded
    # else:
    #     try:
    #         table, info = fit_table(cloud, lidar, robot[0])
    #     except ValueError as e:
    #         sys.exit(f"[ERROR] {e}")
    #     table.name, table.frame = a.name, a.frame
    #     table.notes = (f"table_probe {time.strftime('%Y-%m-%d %H:%M')}: {len(info['plane_inliers'])} top "
    #                    f"points, {len(info['lidar_under'])} LiDAR points under the top")
    #     path = table.save()
    #     print(f"saved {path}\n  centre {np.round(table.center, 3)} yaw {np.degrees(table.yaw):.1f}°  "
    #           f"size {np.round(table.size, 3)} m  top z {table.top_z:.3f} m  legs {len(table.legs)}  "
    #           f"skirt {table.skirt:.2f} m")
    #     if not table.legs and not table.skirt:
    #         print("  [WARN] no legs seen by the LiDAR — table outside the scan plane? The model still "
    #               "blocks the base via the top plate.")
    #     out = path.with_suffix(".png")
    fp, _ = footprint(robot)
    clear = min(np.min(np.linalg.norm(table.corners() - q, axis=1)) for q in fp[:4])
    plot(out, table, lidar, info, robot, f"{table.name} ({a.frame}) — robot {clear:.2f} m from the nearest corner")
    publish_markers(probe.node, table, a.frame)
    print(f"debug plot: {out}   RViz: MarkerArray /table_probe/markers")
    probe.spin(0.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
