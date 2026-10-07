#!/usr/bin/env python3
"""
table_approach.py
=================
Drive the base straight ahead until the first obstacle in front is at a safe distance, then take
that obstacle as the table and save a table model (config/tables/<name>.yaml, frame odom) for
the WBC (--table) — no camera needed.

    ros2 run steve_wbc table_approach --top_z 0.74 --name lab_table          # drive + save
    ros2 run steve_wbc table_approach --top_z 0.74 --no_drive                  # only measure

Stand at the EM stop. Requires the bringup with enable_joystick:=false (teleop floods /cmd_vel).

Stopping
  * obstacles are LiDAR points ahead of the bumper within a corridor `--corridor` wider than the
    base on each side (the LiDARs scan at ~0.22 m: they see table LEGS, never the top plate —
    the corridor makes sure the front legs of a table wider than the robot stop it, so the
    robot never drives under the top plate with its cabinet / arm)
  * stops at `--stop_dist` (bumper -> nearest point), slows down before; never drives more than
    `--max_travel`; stops at once on EM stop / red scanner field, stale scans or odometry
Table model (robot frame at the stop)
  * front legs: the two nearest compact clusters at about the same distance, left and right
    -> near edge (leg - radius - overhang) and width; rear legs (same y, further back) -> depth,
    else `--depth`
  * no leg pair found (closed table / panel): the nearest obstacle line is the near edge, width
    `--width`, centred on the robot
  * the top height cannot be seen by the LiDAR: `--top_z` (measure it)
"""

import argparse
import sys
import time

import numpy as np

try:
    from steve_wbc.table_model import TableModel, lidar_clusters
    from steve_manipulation.safety_stop import (EM_STOP_TOPIC, FOOTPRINT_MAX, FOOTPRINT_MIN,
                                                SCAN_TOPICS, EmStopMonitor, ScanPoints)
except ImportError:
    from table_model import TableModel, lidar_clusters
    from steve_manipulation.safety_stop import (EM_STOP_TOPIC, FOOTPRINT_MAX, FOOTPRINT_MIN,
                                                SCAN_TOPICS, EmStopMonitor, ScanPoints)

FRONT = FOOTPRINT_MAX[0]                       # bumper x in base_link
HALF_W = max(FOOTPRINT_MAX[1], -FOOTPRINT_MIN[1])


def front_obstacle(pts, corridor):
    """(distance bumper -> nearest point in the corridor ahead, that point) or (inf, None)."""
    if pts is None or len(pts) == 0:
        return np.inf, None
    m = (pts[:, 0] > FRONT) & (np.abs(pts[:, 1]) < HALF_W + corridor)
    if not np.any(m):
        return np.inf, None
    i = np.argmin(pts[m, 0])
    return float(pts[m][i, 0] - FRONT), pts[m][i]


def table_from_scan(pts, a):
    """Table model in base_link from the scan at the stop (see module doc)."""
    ahead = pts[(pts[:, 0] > FRONT) & (pts[:, 0] < FRONT + 2.0) & (np.abs(pts[:, 1]) < 1.5)]
    legs = [c.mean(0) for c in lidar_clusters(ahead) if len(c) >= 3 and np.ptp(c, axis=0).max() < 0.10]
    legs = np.array(legs) if legs else np.zeros((0, 2))
    d0, p0 = front_obstacle(pts, a.corridor)
    # front pair: nearest legs within 8 cm of the nearest leg's x, on both sides of the robot axis
    pair = None
    if len(legs) >= 2:
        near = legs[np.argsort(legs[:, 0])]
        for i in range(len(near)):
            same = near[np.abs(near[:, 0] - near[i, 0]) < 0.08]
            left, right = same[same[:, 1] > 0], same[same[:, 1] < 0]
            if len(left) and len(right) and near[i, 0] - FRONT < d0 + 0.3:
                pair = (left[np.argmin(left[:, 1])], right[np.argmax(right[:, 1])])
                break
    r = a.leg_r
    if pair is not None:
        (lx, ly), (rx, ry) = pair
        x_legs = 0.5 * (lx + rx)
        width = (ly - ry) + 2 * (r + a.overhang)
        cy = 0.5 * (ly + ry)
        # rear legs: same y (+-6 cm) as the front ones, 0.3-1.6 m further back
        rear = [q for q in legs if 0.3 < q[0] - x_legs < 1.6 and min(abs(q[1] - ly), abs(q[1] - ry)) < 0.06]
        depth_legs = float(np.median([q[0] - x_legs for q in rear])) if rear else None
        depth = depth_legs + 2 * (r + a.overhang) if depth_legs else a.depth
        x_edge = x_legs - r - a.overhang
        dl = depth_legs if depth_legs else depth - 2 * (r + a.overhang)
        leg_t = [[-dl / 2, (ly - cy), r], [-dl / 2, (ry - cy), r]]
        if rear:
            leg_t += [[dl / 2, (ly - cy), r], [dl / 2, (ry - cy), r]]
        how = (f"front legs at x {x_legs:.2f} m, y {ly:+.2f} / {ry:+.2f} m"
               + (f"; rear legs {depth_legs:.2f} m behind" if rear else f"; rear legs not seen, depth {depth:.2f} m assumed"))
    else:
        if p0 is None:
            raise ValueError("nothing in front of the robot")
        x_edge, width, cy, depth = float(p0[0]) - 0.02, a.width, 0.0, a.depth
        leg_t = []
        how = f"no leg pair: nearest obstacle at x {p0[0]:.2f} m taken as the near edge"
    t = TableModel(name=a.name, frame="base_link", center=[x_edge + depth / 2, cy], yaw=0.0,
                   size=[depth, width], top_z=a.top_z, legs=leg_t)
    return t, how, legs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="lab_table")
    p.add_argument("--top_z", type=float, required=True, help="table top height above the floor [m] (measure it)")
    p.add_argument("--no_drive", action="store_true", help="do not move, only measure and save")
    p.add_argument("--speed", type=float, default=0.08, help="max approach speed [m/s]")
    p.add_argument("--stop_dist", type=float, default=0.30, help="bumper -> nearest obstacle at the stop [m]")
    p.add_argument("--corridor", type=float, default=0.35, help="obstacle corridor beyond the base sides [m]")
    p.add_argument("--max_travel", type=float, default=1.5, help="never drive further [m]")
    p.add_argument("--overhang", type=float, default=0.03, help="top plate beyond the legs [m]")
    p.add_argument("--leg_r", type=float, default=0.025)
    p.add_argument("--depth", type=float, default=0.60, help="table depth if the rear legs are not seen [m]")
    p.add_argument("--width", type=float, default=0.80, help="table width if no leg pair is seen [m]")
    p.add_argument("--lidar_topics", nargs="+", default=list(SCAN_TOPICS["real"]))
    a = p.parse_args(argv)

    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    rclpy.init()
    node = rclpy.create_node("table_approach")
    scans = ScanPoints(node, a.lidar_topics, frame="base_link")
    em = EmStopMonitor(node, EM_STOP_TOPIC)
    odom = {}

    def odom_cb(m):
        q = m.pose.pose.orientation
        odom["pose"] = (m.pose.pose.position.x, m.pose.pose.position.y,
                        np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2)))
        odom["t"] = time.monotonic()
    node.create_subscription(Odometry, "/odom", odom_cb, 10)
    pub = node.create_publisher(Twist, "/cmd_vel", 10)

    def spin(secs):
        t0 = time.monotonic()
        while time.monotonic() - t0 < secs:
            rclpy.spin_once(node, timeout_sec=0.02)

    def stop():
        for _ in range(5):
            pub.publish(Twist())
            time.sleep(0.02)

    spin(1.5)
    if "pose" not in odom or scans.points() is None:
        sys.exit("[ERROR] no /odom or LiDAR scans — is the bringup running?")
    x0 = np.array(odom["pose"][:2])
    d, _ = front_obstacle(scans.points(), a.corridor)
    print(f"nearest obstacle ahead (corridor ±{HALF_W + a.corridor:.2f} m): {d:.2f} m from the bumper")

    status = "measured without driving"
    if not a.no_drive:
        if d <= a.stop_dist:
            status = "already at the safe distance"
        else:
            print(f"driving forward (max {a.speed} m/s) until {a.stop_dist:.2f} m ... Ctrl+C / EM stop to abort")
            status = "?"
            try:
                while True:
                    spin(0.05)
                    pts, now = scans.points(), time.monotonic()
                    red, why = em.status()
                    if red:
                        status = f"EM stop / scanner field: {why}"
                        break
                    if pts is None or now - odom.get("t", 0) > 0.3:
                        status = "LiDAR or odometry stale"
                        break
                    d, _ = front_obstacle(pts, a.corridor)
                    travelled = float(np.hypot(*(np.array(odom["pose"][:2]) - x0)))
                    if d <= a.stop_dist:
                        status = f"stopped {d:.2f} m before the obstacle"
                        break
                    if travelled >= a.max_travel:
                        status = f"max travel {a.max_travel} m reached — no obstacle close enough"
                        break
                    v = float(np.clip(0.5 * (d - a.stop_dist), 0.02, a.speed))  # slow down near it
                    tw = Twist()
                    tw.linear.x = v
                    pub.publish(tw)
            except KeyboardInterrupt:
                status = "aborted (Ctrl+C)"
            finally:
                stop()
        print(status)
        if status.startswith(("EM", "LiDAR", "aborted", "max travel")):
            node.destroy_node()
            rclpy.try_shutdown()
            return 1

    spin(1.0)  # settle, fresh scans
    pts = np.vstack([scans.points() for _ in range(1)])
    try:
        t, how, legs = table_from_scan(pts, a)
    except ValueError as e:
        sys.exit(f"[ERROR] {e}")
    # base_link -> odom
    bx, by, byaw = odom["pose"]
    R = np.array([[np.cos(byaw), -np.sin(byaw)], [np.sin(byaw), np.cos(byaw)]])
    c_r = np.array(t.center)
    t.center = [float(v) for v in R @ c_r + [bx, by]]
    t.yaw, t.frame = float(byaw), "odom"
    t.notes = f"table_approach {time.strftime('%Y-%m-%d %H:%M')}: {how}; top_z given"
    path = t.save()
    print(f"{how}\nnear edge {c_r[0] - t.size[0] / 2 - FRONT:.2f} m ahead of the bumper, "
          f"width {t.size[1]:.2f} m, depth {t.size[0]:.2f} m, top {t.top_z:.2f} m, centre y {c_r[1]:+.2f} m\nsaved {path}")
    node.destroy_node()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
