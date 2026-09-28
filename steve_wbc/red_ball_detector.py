#!/usr/bin/env python3
"""
red_ball_detector.py
====================
ROS 2 perception front end for active_perception_wbc over ROS (Gazebo / real robot):
detects the red object in RGB-D cameras and publishes its CENTRE as
geometry_msgs/PointStamped in `target_frame` (default odom) on /steve_perception/target_point.

Per camera: synchronised colour + depth image, the same red-blob detector as the MuJoCo
pipeline (perception_core.detect_red_object), median depth over the blob, pinhole
unprojection with the camera_info intrinsics, then the object centre one radius further
along the viewing ray (exact for a sphere), transformed with TF.

Frame conventions (parameter `<camera>.frame`):
  optical  header frame is a REP-103 optical frame (z forward, x right, y down)
  x_forward header frame looks along +x (y left, z up) — the Gazebo camera plugin on
            pan_tilt_camera_color_optical_frame, whose URDF frame is not optical
            (on the real robot steve_hardware_bringup adds *_corrected frames instead)

Optionally points the pan-tilt head (pan_tilt_controller) so the head camera sees the
table while the robot drives: the Gazebo controller otherwise holds the head slightly
upward.

    ros2 run steve_wbc red_ball_detector --ros-args -p use_sim_time:=true
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    from steve_wbc.target_profile import TargetProfile
except ImportError:
    from target_profile import TargetProfile

# Gazebo topics (the plugin namespaces them twice: <ns>/<camera_name>/...)
DEFAULT_CAMERAS = {
    "pan_tilt_camera": ("/pan_tilt_camera/pan_tilt_camera", "x_forward"),
    "wrist_camera": ("/wrist_camera/wrist_camera", "optical"),
}
# topic names under a camera namespace: (colour, aligned depth, camera_info)
TOPIC_STYLES = {
    "gazebo": ("image_raw", "depth/image_raw", "camera_info"),
    "realsense": ("color/image_raw", "aligned_depth_to_color/image_raw", "color/camera_info"),
}
# optical (x right, y down, z forward) -> x-forward camera frame (x forward, y left, z up)
R_OPT_TO_XFWD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quat_to_mat(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def image_to_array(msg: Image) -> np.ndarray:
    if msg.encoding in ("32FC1",):
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ("16UC1", "mono16"):  # millimetres (RealSense)
        return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[msg.encoding]
    img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // ch if msg.step else msg.width, ch)
    img = img[:, :msg.width, :3]
    return img[:, :, ::-1] if msg.encoding.startswith("bgr") else img


class RedBallDetector(Node):
    def __init__(self):
        super().__init__("red_ball_detector")
        p = self.declare_parameter
        self.target_frame = p("target_frame", "odom").value
        # what to look for: a profile recorded with target_calibrator (colour + size)
        self.target = TargetProfile.load(p("target", "red").value)
        self.radius = self.target.radius
        self.min_range = p("min_range", 0.12).value
        self.max_range = p("max_range", 4.0).value
        cameras = p("cameras", list(DEFAULT_CAMERAS)).value
        style = p("topic_style", "gazebo").value  # gazebo | realsense (real robot)
        self.get_logger().info(f"target '{self.target.name}': HSV {self.target.hsv_ranges}, "
                               f"{self.target.size * 100:.1f} cm {self.target.shape}")

        self.tf = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf, self)
        self.pub = self.create_publisher(PointStamped, p("target_topic", "/steve_perception/target_point").value, 10)

        self.cams = {}
        for name in cameras:
            ns_default, frame_default = DEFAULT_CAMERAS.get(name, (f"/{name}", "optical"))
            ns = p(f"{name}.namespace", ns_default).value
            # real RealSense drivers publish proper optical frames; Gazebo's head camera does not
            conv = p(f"{name}.frame", frame_default if style == "gazebo" else "optical").value
            img_t, depth_t, info_t = TOPIC_STYLES[style]
            self.cams[name] = dict(K=None, conv=conv, n=0)
            self.create_subscription(CameraInfo, f"{ns}/{info_t}",
                                     lambda m, c=name: self._info_cb(c, m), 10)
            sync = ApproximateTimeSynchronizer(
                [Subscriber(self, Image, f"{ns}/{img_t}"), Subscriber(self, Image, f"{ns}/{depth_t}")],
                queue_size=5, slop=0.05)
            sync.registerCallback(lambda rgb, depth, c=name: self._rgbd_cb(c, rgb, depth))
            self.cams[name]["sync"] = sync
            self.get_logger().info(f"camera '{name}': {ns}/{{image_raw,depth/image_raw}}  frame={conv}")

        # Pan-tilt head: hold it at the given angles (empty topic disables)
        self.head_topic = p("head_topic", "/pan_tilt_controller/joint_trajectory").value
        self.head_q = [p("head_pan", 0.0).value, p("head_tilt", 0.35).value]
        if self.head_topic:
            self.head_pub = self.create_publisher(JointTrajectory, self.head_topic, 10)
            self.create_timer(1.0, self._hold_head)
            self.get_logger().info(f"holding head at pan={self.head_q[0]:.2f} tilt={self.head_q[1]:.2f} rad")
        self.create_timer(5.0, self._report)

    # ── callbacks ───────────────────────────────────────────────────────────
    def _info_cb(self, cam, msg: CameraInfo):
        self.cams[cam]["K"] = np.array(msg.k, float).reshape(3, 3)

    def _hold_head(self):
        if self.head_pub.get_subscription_count() == 0:
            return
        traj = JointTrajectory()
        traj.joint_names = ["pan_tilt_pan_motor_joint", "pan_tilt_tilt_motor_joint"]
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in self.head_q]
        pt.time_from_start.sec = 1
        traj.points.append(pt)
        self.head_pub.publish(traj)

    def _report(self):
        self.get_logger().info("detections: " + ", ".join(f"{c}={v['n']}" for c, v in self.cams.items()))

    def _rgbd_cb(self, cam, rgb_msg: Image, depth_msg: Image):
        c = self.cams[cam]
        if c["K"] is None:
            return
        rgb, depth = image_to_array(rgb_msg), image_to_array(depth_msg)
        found, _, _, mask = self.target.detect(rgb)
        if not found or depth.shape != mask.shape:
            return
        d = depth[mask]
        d = d[np.isfinite(d) & (d > self.min_range) & (d < self.max_range)]
        if len(d) < self.target.min_pixels:
            return
        z = float(np.median(d))
        ys, xs = np.nonzero(mask)
        K = c["K"]
        # blob pixel -> surface point facing the camera -> centre one radius further on
        ray = np.array([(np.median(xs) - K[0, 2]) / K[0, 0], (np.median(ys) - K[1, 2]) / K[1, 1], 1.0])
        p_opt = ray * z
        p_opt = p_opt + self.radius * p_opt / np.linalg.norm(p_opt)
        p_cam = R_OPT_TO_XFWD @ p_opt if c["conv"] == "x_forward" else p_opt

        try:
            tf = self.tf.lookup_transform(self.target_frame, rgb_msg.header.frame_id,
                                          Time.from_msg(rgb_msg.header.stamp), Duration(seconds=0.05))
        except TransformException:
            try:  # sim-time stamps can race the TF buffer; the latest transform is close enough
                tf = self.tf.lookup_transform(self.target_frame, rgb_msg.header.frame_id, Time())
            except TransformException as e:
                self.get_logger().warn(f"TF {rgb_msg.header.frame_id} -> {self.target_frame}: {e}",
                                       throttle_duration_sec=2.0)
                return
        t, q = tf.transform.translation, tf.transform.rotation
        R, tr = quat_to_mat(q), np.array([t.x, t.y, t.z])
        p = R @ p_cam + tr
        # all blob pixels in 3-D: centre from the object's top, independent of the viewpoint
        # (same estimator as in MuJoCo; needs target_frame with z up, e.g. odom)
        zz = depth[ys, xs]
        ok = np.isfinite(zz) & (zz > self.min_range) & (zz < self.max_range)
        pts = np.stack([(xs[ok] - K[0, 2]) * zz[ok] / K[0, 0], (ys[ok] - K[1, 2]) * zz[ok] / K[1, 1], zz[ok]], 1)
        if c["conv"] == "x_forward":
            pts = pts @ R_OPT_TO_XFWD.T
        top = self.target.center_from_points(pts @ R.T + tr)
        if top is not None:
            p = top

        out = PointStamped()
        out.header.stamp = rgb_msg.header.stamp
        out.header.frame_id = self.target_frame
        out.point.x, out.point.y, out.point.z = (float(v) for v in p)
        self.pub.publish(out)
        c["n"] += 1


def main(args=None):
    rclpy.init(args=args)
    node = RedBallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
