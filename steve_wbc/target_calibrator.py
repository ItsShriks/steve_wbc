#!/usr/bin/env python3
"""
target_calibrator.py
====================
Record what the target object looks like with the REAL camera (or the simulated one) and save
it as a target profile (steve_wbc/config/targets/<name>.yaml) for the detectors and the WBC.

    # real robot: wrist D405 (default) or pan-tilt L515, with depth for a size check
    ros2 run steve_wbc target_calibrator --name blue_cube --size 0.05 --shape box
    ros2 run steve_wbc target_calibrator --name ball --topic /pan_tilt_camera/pan_tilt_camera/color/image_raw
    # MuJoCo (sim scene, wrist camera in scan pose) / a saved image
    ros2 run steve_wbc target_calibrator --name red --sim
    ros2 run steve_wbc target_calibrator --name ball --image frame.png
    # check a stored profile on the live image
    ros2 run steve_wbc target_calibrator --check blue_cube

Window keys:  c  capture: draw a box around the object (GrabCut cuts it out of the background)
              u  undo last capture          + / -  looser / tighter S,V limits
              s  save the profile           q      quit
Capture the object from several angles, distances and under the lighting you will run in —
every capture is added to the profile. The overlay shows the detection live: green = detected
object, magenta = background pixels that would ALSO be detected (tighten or re-capture).

Headless:  --roi X Y W H [--no_gui]  uses one frame and that box (scripts / tests).
"""

import argparse
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("[ERROR] target_calibrator needs OpenCV: sudo apt install python3-opencv")

try:
    from steve_wbc.target_profile import TargetProfile, fit_profile
except ImportError:
    from target_profile import TargetProfile, fit_profile

WRIST_TOPIC = "/wrist_camera/wrist_camera/color/image_raw"


# ── frame sources ────────────────────────────────────────────────────────────
class ImageSource:
    def __init__(self, path):
        bgr = cv2.imread(path)
        if bgr is None:
            sys.exit(f"[ERROR] cannot read {path}")
        self.rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def frame(self):
        return self.rgb, None

    def object_roi(self):
        return None


class SimSource:
    """MuJoCo: the robot in front of the table, wrist camera in the scan pose."""

    def __init__(self, cam, object_type):
        import mujoco
        try:
            from steve_wbc.active_perception_wbc import SCAN_SEED_Q, TARGET_POS
            from steve_wbc.steve_figure8_wbc import BASE_JOINTS, ARM_JOINTS, resolve_model
        except ImportError:
            from active_perception_wbc import SCAN_SEED_Q, TARGET_POS
            from steve_figure8_wbc import BASE_JOINTS, ARM_JOINTS, resolve_model
        self.mj = mujoco
        m = self.m = mujoco.MjModel.from_xml_path(str(resolve_model(None)))
        d = self.d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        jq = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n, v in zip(BASE_JOINTS, (1.16, 0.0, 0.0)):
            d.qpos[jq(n)] = v
        for n, v in zip(ARM_JOINTS, SCAN_SEED_Q):
            d.qpos[jq(n)] = v
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        m.geom_type[g] = mujoco.mjtGeom.mjGEOM_BOX if object_type == "box" else mujoco.mjtGeom.mjGEOM_SPHERE
        qa = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")]
        d.qpos[qa:qa + 7] = [*TARGET_POS["single"], 0.825, 1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        # aim the arm (shoulder pan) at the object like the WBC does
        sh = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")]
        obj = d.qpos[qa:qa + 3]
        d.qpos[jq(ARM_JOINTS[0])] = np.arctan2(obj[1] - sh[1], obj[0] - sh[0])
        mujoco.mj_forward(m, d)
        self.cam, self.obj_geom, self.jq = cam, g, jq
        self.renderer = mujoco.Renderer(m, 480, 640)

    def set_object(self, kind):
        """MuJoCo test object: sphere or box (profiles for several objects in one recording)."""
        self.m.geom_type[self.obj_geom] = (self.mj.mjtGeom.mjGEOM_BOX if kind == "box"
                                           else self.mj.mjtGeom.mjGEOM_SPHERE)
        self.m.geom_size[self.obj_geom] = [0.025, 0.025, 0.025]  # 5 cm, like the WBC scenes
        self.mj.mj_forward(self.m, self.d)

    def frame(self):
        self.renderer.update_scene(self.d, camera=self.cam)
        return self.renderer.render().copy(), None

    def object_roi(self):
        """Bounding box of the object in the image (for --roi auto)."""
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.d, camera=self.cam)
        seg = self.renderer.render()[..., 0]
        self.renderer.disable_segmentation_rendering()
        ys, xs = np.nonzero(seg == self.obj_geom)
        if not len(xs):
            return None
        pad = 6
        return (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                xs.max() - xs.min() + 2 * pad, ys.max() - ys.min() + 2 * pad)


class RosSource:
    def __init__(self, topic, depth_topic):
        import rclpy
        from sensor_msgs.msg import CameraInfo, Image
        from rclpy.qos import qos_profile_sensor_data
        self.rclpy = rclpy
        rclpy.init()
        self.node = rclpy.create_node("target_calibrator")
        self.rgb, self.depth, self.K = None, None, None
        self.node.create_subscription(Image, topic, self._rgb, qos_profile_sensor_data)
        if depth_topic:
            self.node.create_subscription(Image, depth_topic, self._depth, qos_profile_sensor_data)
            info = depth_topic.rsplit("/", 1)[0] + "/camera_info"
            self.node.create_subscription(CameraInfo, info, lambda m: setattr(self, "K", np.array(m.k).reshape(3, 3)), 10)
        t0 = time.monotonic()
        while self.rgb is None and time.monotonic() - t0 < 10:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self.rgb is None:
            sys.exit(f"[ERROR] no image on {topic} (camera driver running? steve_doctor --ros)")

    @staticmethod
    def _to_array(msg):
        if msg.encoding in ("16UC1", "mono16"):
            return np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        if msg.encoding == "32FC1":
            return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        ch = 4 if msg.encoding in ("rgba8", "bgra8") else 3
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // ch, ch)[:, :msg.width, :3]
        return img[..., ::-1].copy() if msg.encoding.startswith("bgr") else img.copy()

    def _rgb(self, msg):
        self.rgb = self._to_array(msg)

    def _depth(self, msg):
        self.depth = self._to_array(msg)

    def frame(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.03)
        return self.rgb, self.depth

    def object_roi(self):
        return None


# ── profile from captures ────────────────────────────────────────────────────
def object_pixels(rgb, roi):
    """HSV pixels of the object inside roi (x, y, w, h), separated from the background by GrabCut."""
    x, y, w, h = [int(v) for v in roi]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(rgb.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (x, y, max(w, 2), max(h, 2)), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    obj = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[obj], obj


class Calibration:
    def __init__(self, args):
        self.a, self.samples, self.margin = args, [], 1.0  # samples: (rgb, roi, hsv pixels, object mask)
        self.profile = None

    def add(self, rgb, roi):
        px, obj = object_pixels(rgb, roi)
        if len(px) < 20:
            print("  capture ignored: GrabCut found (almost) no object in the box")
            return
        self.samples.append((rgb.copy(), roi, px, obj))
        self.update()

    def undo(self):
        if self.samples:
            self.samples.pop()
            self.update()

    def update(self):
        if not self.samples:
            self.profile = None
            return
        obj = np.vstack([smp[2] for smp in self.samples])
        bg = []
        for rgb, _, _, m in self.samples:  # everything else in the captured frames
            far = cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0  # not the object's rim
            bg.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[far])
        bg = np.vstack(bg)
        if len(bg) > 200000:
            bg = bg[np.random.default_rng(0).choice(len(bg), 200000, replace=False)]
        a = self.a
        self.profile = fit_profile(a.name, obj, bg, a.size, a.shape)
        for r in self.profile.hsv_ranges:  # +/- keys: looser / tighter S,V floors
            r[1], r[2] = int(r[1] * self.margin), int(r[2] * self.margin)
        self.profile.samples = len(self.samples)
        rec, fp = self.evaluate()
        print(f"  {len(self.samples)} capture(s): HSV {self.profile.hsv_ranges}  "
              f"object pixels found {rec * 100:.0f}%  background false positives {fp * 100:.2f}%")

    def evaluate(self):
        """Mean recall on the captured objects and false-positive share of the background."""
        rec, fp = [], []
        for rgb, roi, _, obj in self.samples:
            m = self.profile.mask(rgb)
            rec.append((m & obj).sum() / max(obj.sum(), 1))
            fp.append((m & ~obj).sum() / max((~obj).sum(), 1))
        return float(np.mean(rec)), float(np.mean(fp))

    def save(self):
        if self.profile is None:
            print("  nothing captured yet")
            return
        rec, fp = self.evaluate()
        self.profile.notes = (f"recall {rec * 100:.0f}%, background false positives {fp * 100:.2f}% "
                              f"on {len(self.samples)} capture(s); {time.strftime('%Y-%m-%d %H:%M')}")
        p = self.profile.save(self.a.out)
        for i, (rgb, roi, _, _) in enumerate(self.samples):  # keep the captures for later tuning
            img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x, y, w, h = [int(v) for v in roi]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
            cv2.imwrite(str(p.with_name(f"{p.stem}_capture{i}.png")), img)
        print(f"  saved {p} (+ {len(self.samples)} capture image(s))")


def overlay(rgb, depth, profile, K, size):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if profile is None:
        return img
    full = profile.mask(rgb)
    found, cx, cy, blob = profile.detect(rgb)
    img[full & ~blob] = (255, 0, 255)                         # would also be detected
    if found:
        cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
        cv2.drawMarker(img, (int(cx), int(cy)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
        txt = f"{profile.name}: {blob.sum()} px"
        if depth is not None and K is not None:
            z = depth[blob & (depth > 0.05)]
            if len(z):
                zm = float(np.median(z))
                ys, xs = np.nonzero(blob)
                w_m = (xs.max() - xs.min()) * zm / K[0, 0]
                txt += f"  {zm:.2f} m away, ~{w_m * 100:.1f} cm wide (profile {size * 100:.1f} cm)"
        cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    else:
        cv2.putText(img, f"{profile.name}: not detected", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return img


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--topic", default=WRIST_TOPIC, help="ROS colour image topic")
    src.add_argument("--sim", action="store_true", help="MuJoCo scene instead of a camera")
    src.add_argument("--image", help="a saved image instead of a camera")
    p.add_argument("--depth_topic", default=None, help="aligned depth (size check); default: derived from --topic")
    p.add_argument("--sim_cam", default="wrist_cam", help="MuJoCo camera (wrist_cam | pan_tilt_cam)")
    p.add_argument("--sim_objects", nargs="+", choices=["sphere", "box"], default=None,
                   help="--sim --roi auto: capture each of these objects (default: --shape)")
    p.add_argument("--name", default="target", help="profile name (file config/targets/<name>.yaml)")
    p.add_argument("--size", type=float, default=0.05, help="object diameter / edge [m]")
    p.add_argument("--shape", choices=["sphere", "box", "other"], default="sphere")
    p.add_argument("--out", default=None, help="profile path (default config/targets/<name>.yaml)")
    p.add_argument("--check", metavar="PROFILE", help="show the detection of a stored profile (no recording)")
    p.add_argument("--roi", nargs="+", metavar="X Y W H", help="headless capture box X Y W H ('auto' in --sim: the object)")
    p.add_argument("--no_gui", action="store_true", help="with --roi: compute and save without a window")
    a = p.parse_args(argv)

    if a.image:
        source = ImageSource(a.image)
    elif a.sim:
        source = SimSource(a.sim_cam, "box" if a.shape == "box" else "sphere")
    else:
        depth = a.depth_topic or a.topic.replace("/color/image_raw", "/aligned_depth_to_color/image_raw")
        source = RosSource(a.topic, depth)

    cal = Calibration(a)
    if a.check:
        cal.profile = TargetProfile.load(a.check)
        a.size = cal.profile.size
    if a.roi and a.roi != ["auto"] and len(a.roi) != 4:
        p.error("--roi needs X Y W H (or 'auto' with --sim)")
    if a.roi:
        kinds = (a.sim_objects or [a.shape]) if a.sim and a.roi == ["auto"] else [None]
        for kind in kinds:
            if kind is not None:
                source.set_object(kind)
            rgb, _ = source.frame()
            roi = source.object_roi() if a.roi[0] == "auto" else tuple(int(v) for v in a.roi)
            if roi is None:
                sys.exit("[ERROR] --roi auto: object not visible")
            cal.add(rgb, roi)
        if a.no_gui:
            cal.save()
            return 0

    win = "target_calibrator  [c capture | u undo | +/- looser/tighter | s save | q quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        rgb, depth = source.frame()
        cv2.imshow(win, overlay(rgb, depth, cal.profile, getattr(source, "K", None), a.size))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c") and not a.check:
            frozen = rgb.copy()
            roi = cv2.selectROI(win, cv2.cvtColor(frozen, cv2.COLOR_RGB2BGR), showCrosshair=False)
            if roi[2] > 2 and roi[3] > 2:
                cal.add(frozen, roi)
        elif k == ord("u"):
            cal.undo()
        elif k in (ord("+"), ord("=")):
            cal.margin *= 0.85   # lower S,V limits further down: looser
            cal.update()
        elif k == ord("-"):
            cal.margin /= 0.85
            cal.update()
        elif k == ord("s") and not a.check:
            cal.save()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
