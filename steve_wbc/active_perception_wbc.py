#!/usr/bin/env python3
"""
active_perception_wbc.py
========================
Active perception + VGN grasping for Steve (MMO-700 + UR5e + Robotiq 2F-85) with a
UNIFIED joint-velocity whole-body controller.

Every phase is one constrained QP over all 9 joints (base vx, vy, wz + 6 arm joints,
see steve_figure8_wbc.WholeBodyController) — only the active tasks change:

  phase          EE task                     base task              posture (null space)
  ─────────────  ──────────────────────────  ─────────────────────  ─────────────────────
  DRIVE          —                           LiDAR-reactive forward drive pose
  BACKUP         —                           reverse                drive pose
  REPOSITION     —                           smooth orbit around    drive pose
                                             the table (TableOrbit)
                 (only if the grasp is not reachable from here with the base >= d_safe_base
                  from the table, or VGN found no grasp here; IK check per side, _plan_approach)
  SETTLE_ARM     —                           hold                   scan seed pose
  COARSE_DETECT  —                           hold                   scan pose, pan sweeps
                 (only if the object was not already tracked)
  VGN_SCAN       —                           hold                   scan pose, pan -> view i
  VGN_WAIT       —                           hold                   (--vgn_mode scan: inference)
  PRE_GRASP      MPC: meet the predicted     — (QP decides)         table-reach pose
                 path 10 cm above it
  DESCEND        MPC: onto the object        soft hold              table-reach pose
  GRASP          MPC: move with the object   soft hold              table-reach pose
  LIFT           MPC: +lift height           soft hold              table-reach pose
  HOLD           MPC: stay up (--hold_time)  soft hold              table-reach pose
                 (success only if the object is still in the gripper afterwards)

Perception runs from the very first tick — head (pan-tilt) camera while driving,
then wrist + head camera — into a gated least-squares constant-velocity tracker, so
the object's path is known long before the grasp. A receding-horizon MPC
(InterceptionMPC) plans the gripper trajectory to intercept that predicted path with
matched velocity; the WBC tracks the MPC's first step.

Hard constraints in every phase: joint position / velocity / acceleration limits and
collision avoidance (arm vs table, obstacles, floor and own body — relaxed margin only
for the final descent onto the table; base footprint >= --d_safe_base (0.15 m) from the
environment, never relaxed). Ground truth is used only to report estimation errors.

Compared with active_perception_dynamic_wbc_vgn.py (porta_ros2_ws): there the torque
OSC was never called, reaching/grasping/lifting used arm-only IK, the base only drove
and the "tracking" read the object's ground-truth position.

Run
---
  # MuJoCo (on macOS use `mjpython` for the viewer)
  python3 active_perception_wbc.py [--headless] [--no_vgn] [--object_type box] [--log run.csv]
  ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=true

  # Physical robot / Gazebo: LiDAR from --scan_topic, object from --target_topic
  # (geometry_msgs/PointStamped in odom or base_link), gripper via steve_manipulation
  ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false dry_run:=true
"""

import os
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear

try:
    from steve_wbc.steve_figure8_wbc import (
        BASE_LAG_BUFFER, NU, BaseReference, DryRunBackend, MujocoSimBackend, Reference, RosBackend, RobotState,
        WholeBodyController, WholeBodyKinematics, build_arg_parser, exp_so3, load_model, log_so3,
        make_safety, orientation_error, parse_args, resolve_model, rot_z, summarize, wrap_angle,
    )
    from steve_wbc.perception_core import (IMG_H, IMG_W, PerceptionConfig, detect_red_object,
                                           read_lidar, render_rgbd, unproject_pixel_to_world)
    from steve_wbc.target_profile import TargetProfile
    from steve_wbc.table_model import TableModel
    from steve_wbc.table_heightmap import TableHeightMap
except ImportError:  # plain script next to the other modules (e.g. macOS without ROS)
    from steve_figure8_wbc import (
        BASE_LAG_BUFFER, NU, BaseReference, DryRunBackend, MujocoSimBackend, Reference, RosBackend, RobotState,
        WholeBodyController, WholeBodyKinematics, build_arg_parser, exp_so3, load_model, log_so3,
        make_safety, orientation_error, parse_args, resolve_model, rot_z, summarize, wrap_angle,
    )
    from perception_core import (IMG_H, IMG_W, PerceptionConfig, detect_red_object,
                                 read_lidar, render_rgbd, unproject_pixel_to_world)
    from target_profile import TargetProfile
    from table_model import TableModel
    from table_heightmap import TableHeightMap

try:
    from steve_perception.vgn import VGNBridge
except ImportError:  # steve_perception not built (e.g. plain script): use the source-tree sibling
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "steve_perception"))
    from steve_perception.vgn import VGNBridge


def make_grasp_backend(a, target):
    """The grasp scan's detector (--grasp_backend): VGN (TSDF of all views, vgn_env subprocess) or
    Contact-GraspNet per view on the target's Grounded-SAM mask (model servers, robot only)."""
    if a.grasp_backend == "cgn":
        from steve_perception.cgn import CGNBridge
        return CGNBridge(a.grasp_prompt, target.radius, a.gsam_server, a.cgn_server)
    return VGNBridge()

# ── Arm poses ────────────────────────────────────────────────────────────────
DRIVE_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]   # retracted for driving
SCAN_SEED_Q = [0.0, -1.8, 1.4, -1.2, -1.57, 0.0]             # forward-looking wide-FOV scan
ARM_HOME_Q = [0.0, -1.0, 1.5, -2.05, -1.57, 0.0]             # table reach, gripper down
R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)

# ── Driving (SICK S300, cf. neo_sick_s300-2) ─────────────────────────────────
# Front LiDAR distances relative to the base safety distance --d_safe_base (default 0.15 m),
# which the base must never undercut: the QP enforces it as a hard constraint on the base
# footprint (sim), these thresholds make the drive stop before it.
LIDAR_SLOW_MARGIN = 0.30      # m  above d_safe_base -> slow speed
LIDAR_STOP_MARGIN = 0.05      # m  above d_safe_base -> stop (braking room)
LIDAR_EMERGENCY_MARGIN = 0.05 # m  below d_safe_base -> back up
SLOW_SPEED, BACKUP_SPEED = 0.10, 0.08

# ── Grasping ─────────────────────────────────────────────────────────────────
PREGRASP_HEIGHT = 0.10        # m above the object
LIFT_HEIGHT = 0.15            # m
EE_APPROACH_SPEED = 0.15      # m/s  max EE speed (relative to the object) on approach
EE_DESCEND_SPEED = 0.05       # m/s
EE_ROT_SPEED = 0.6            # rad/s
GRASP_TIME = 1.0              # s gripper closing while tracking
HOLD_TOL = 0.03               # m object centre within this of the grasp point after the hold
TRANSIT_CORRIDOR = 0.08       # m  obstacles this close to the gripper's xy path are flown over
TRANSIT_CLEARANCE = 0.10      # m  pinch point above the tallest such obstacle
TRANSIT_XY_TOL = 0.03         # m  over the object: drop from transit to pre-grasp height
STATIC_SPEED = 0.01           # m/s fitted speeds below this are detection noise: object static
MIN_TRACK_DISPLACEMENT = 0.04 # m  fitted motion over the window must exceed this (> viewpoint bias)
MIN_FIT_SPAN = 2.0            # s  still detections needed to (re)fit the velocity
SEARCH_STILL_DQ = 0.15        # rad/s arm joint speed above which search detections are glimpses
TRACK_STILL_DQ = 0.15         # rad/s arm joint speed below which detections may fit the velocity

# ── VGN modes (--vgn_mode) ─────────────────────────────────────────────────────
#   scan  : stop at the table, pan the wrist camera over --vgn_views views, run inference and
#           wait for it (VGN_WAIT), then approach                                    [original]
#   async : same views, inference runs in the background while the gripper approaches
#   drive : views are taken while driving up to the table (arm in scan pose, object-motion
#           compensated); inference starts in the background as soon as VGN_DRIVE_MAX_VIEWS are
#           in (still driving) or on arrival; a short stopped scan only tops up if the drive
#           yielded too few views
# The worker's measured inference time is charged 1:1 to simulation time in every mode, so the
# modes compare fairly even when the headless simulation runs faster than real time.
VGN_DRIVE_RANGE = 2.3         # m  base-object distance below which views are taken while driving
VGN_DRIVE_SPACING = 0.08      # m  base travel between drive views (or VGN_DRIVE_PERIOD)
VGN_DRIVE_PERIOD = 1.0        # s
VGN_DRIVE_MAX_VIEWS = 6       # reached while driving -> inference starts right away
VGN_DRIVE_MIN_VIEWS = 3       # fewer -> top up with a stopped scan
VGN_ARM_READY = 0.10          # rad arm within this of the scan pose before drive views are taken

# ── Approach-side planning around the support (sim: table geometry known) ────────
STANDOFF_MARGIN = 0.01        # m  base stand-off: footprint this far beyond d_safe_base (+lag buffer)
GRASP_ETA = 8.0               # s  from arriving at a side to the grasp (object prediction)
PREGRASP_TIMEOUT = 12.0       # s  pre-grasp not reached from this side -> replan (another side)
RING_MARGIN = 0.05            # m  ring around the table: clear at any heading, + this
REPOSITION_SPEED = 0.25       # m/s along the ring (--orbit_speed; capped by --max_base_v)
ORBIT_ACC = 0.25              # m/s^2 ramps of repositioning paths (trapezoidal speed; QP limit 0.5)

# ── Continuous operation (--continuous): after an episode the robot watches the table and goes for
# the object again once it has been put down somewhere else ──
WATCH_SETTLE = 1.5            # s  a new placement: the object still (within WATCH_STILL_TOL) this long
WATCH_STILL_TOL = 0.025       # m
WATCH_RETRACK = 1.0           # s  track not confirmed this long (picked up, carried): start a new track
WATCH_SEARCH_AFTER = 4.0      # s  nothing tracked this long -> sweep, then search the other sides
RELEASE_OPEN_WAIT = 1.5       # s  gripper open -> watching
SIM_RELOCATE_AFTER = 3.0      # s  sim: the object is put somewhere else this long into WATCH

# ── Visibility task (--no_visibility): camera axis kept on the object while the EE is free ──
VIS_STATES = ("DRIVE", "SETTLE_ARM", "COARSE_DETECT", "REPOSITION", "VGN_SCAN", "VGN_WAIT", "WATCH")
VIS_READY_DEG = 4.0           # deg camera axis -> object for "arm ready" (settle, scan / drive views)
VIS_FOV_DEG = 20.0            # deg "object in view" for the summary (D415 colour: 69 x 42 deg FOV)

# ── Occlusion-aware line of sight (--no_los): what stands on the table, from the camera's depth
# frames (TableHeightMap), chooses the approach side (target not hidden) and feeds the QP's
# line-of-sight term; an object not found from one side is searched from the others ──
MAP_STATES = ("DRIVE", "SETTLE_ARM", "COARSE_DETECT", "REPOSITION", "VGN_WAIT")  # VGN_SCAN: its views
MAP_PERIOD = 0.5              # s  depth frames into the table map
LOS_OK = 0.6                  # share of clear sight lines: the object counts as visible from a side
LOS_NEAR = 0.6                # m  map columns within this (xy) of the object enter the QP's LOS term
# Table quadrants as seen from the side the robot arrives on (bottom = near half, top = far half,
# left / right from the robot's view) -> side of the table to grasp from. "here" = stay.
VIEW_TIMEOUT = 3.0            # s  real camera: no depth frame + TF for a VGN view -> skip the view
NO_GRASP = "no grasp candidate found"
GOTO_DONE = "reached grasp pose"   # --goto_grasp_only
MPC_LEASH = 0.04              # m    EE reference at most this far ahead of the real EE
MPC_LEASH_ANG = 0.25          # rad  ... and this much rotation ahead
LEASHED_STATES = ("PRE_GRASP", "RETREAT")   # free-space moves; never near the object (contact detection)
AT_GRASP_PUSHED = 0.03        # m  --goto_grasp_only: EE pushed this far off the grasp pose -> back up
# table region of the object (seen from the side the robot arrives on) -> side to grasp from:
# near half -> stay; far half by thirds: left -> the robot's left side, right -> its right side,
# centre -> the opposite (back) side
QUADRANT_SIDE = {"bottom-left": "here", "bottom-right": "here", "top-left": "left", "top-center": "back",
                 "top-right": "right"}
REACH_TOL = 0.01              # m  IK residual accepted as reachable
GRASP_YAWS = (0.0, np.pi / 2)  # top-down grasp rotations tried (fingers open along y / along x)
SIDES = {  # outward normal of the table side, base yaw facing the table
    "front": (np.array([-1.0, 0.0]), 0.0),
    "back": (np.array([1.0, 0.0]), np.pi),
    "right": (np.array([0.0, -1.0]), np.pi / 2),
    "left": (np.array([0.0, 1.0]), -np.pi / 2),
}

# ── Scenes ───────────────────────────────────────────────────────────────────
# single : one red ball (or --object_type box) rolling along +y
# clutter: a red cube among static obstacles on the table. Only the cube is picked; the
#          obstacles enter the WBC as collision constraints (static, collidable geoms of the
#          MuJoCo model) and any robot contact with them fails the run.
# far    : a red cube near the far edge of the table, out of reach from the front with the base
#          outside its safety distance -> the robot has to drive around the table
TARGET_POS = {"single": (2.05, -0.25), "clutter": (2.00, -0.08), "far": (2.32, 0.10),
              "occluded": (2.22, 0.0)}
CLUTTER_OBSTACLES = [
    # name,        geom type,  size (MuJoCo convention),  centre xyz,              rgba
    # tall bottle (top 1.12 m) between the robot and the cube, on the path the gripper takes
    # without collision avoidance (--no_collision hits it)
    # (placed so the grasp pose itself keeps ~7 cm from it)
    ("obst_bottle", "cylinder", (0.030, 0.160),           (1.87,  0.06, 0.960), (0.15, 0.30, 0.90, 1)),
    # can next to the cube (+y)
    ("obst_can",    "cylinder", (0.035, 0.060),           (2.02,  0.11, 0.860), (0.15, 0.70, 0.25, 1)),
    # block diagonally behind the cube
    ("obst_block",  "box",      (0.030, 0.030, 0.050),    (2.13, -0.19, 0.850), (0.90, 0.80, 0.10, 1)),
]
# occluded: a static ball in the far half of the table behind a 32 cm box (as seen from the front
# side, where the robot arrives). The wrist camera looks down steeply (front scan pose: x 1.71,
# z 1.34 m): the sight lines to the ball cross the box at z ~1.02 m, its top is at 1.12 m — hidden
# from the front, visible from the other sides; 15 cm between box and ball leave the gripper room.
# Tests: searching another side, the table map, side choice by line of sight.
OCCLUDED_OBSTACLES = [
    ("obst_box",    "box",      (0.020, 0.200, 0.160),    (2.07,  0.00, 0.960), (0.55, 0.55, 0.60, 1)),
]
SCENE_OBSTACLES = {"clutter": CLUTTER_OBSTACLES, "occluded": OCCLUDED_OBSTACLES}


def build_model(path, scene, intruder=False, table=None):
    """MuJoCo model of the scene: the robot/table model, plus the clutter obstacles; `table`
    (TableModel) replaces the MJCF's box-like table with a measured one."""
    def edit(spec):
        if table is not None:
            table.apply_to_spec(spec)
        if scene in SCENE_OBSTACLES:
            add_obstacles(spec)

    def add_obstacles(spec):
        gtype = {"cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "box": mujoco.mjtGeom.mjGEOM_BOX,
                 "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
        for name, kind, size, pos, rgba in SCENE_OBSTACLES[scene]:
            body = spec.worldbody.add_body(name=name, pos=list(pos))  # no joint: static
            body.add_geom(name=f"{name}_geom", type=gtype[kind], size=[*size, 0, 0][:3], rgba=list(rgba),
                          contype=1, conaffinity=1, friction=[1.0, 0.005, 0.0001])
    return load_model(path, intruder, edit if (scene in SCENE_OBSTACLES or table is not None) else None)


# ══════════════════════════════════════════════════════════════════════════════
# World interfaces: perception, LiDAR, gripper  (MuJoCo  |  ROS 2)
# ══════════════════════════════════════════════════════════════════════════════
class VGNInference:
    """VGN inference shared by SimWorld and RosWorld (self.vgn: VGNBridge, self.target: profile).
    Inference runs in the vgn_env worker process. Its duration is measured in wall-clock time and
    returned with the result, so the caller can charge it to simulation time."""
    vgn = None
    vgn_target = None  # (position) VGN grasps must lie near, in the TSDF frame
    save_dir = None    # robot: <run folder>/grasp_scan (--save_dir): every scan's views + grasps as images
    scan_no = 0

    def _save_scan(self, best):
        if self.save_dir is None or not hasattr(self.vgn, "scan_data"):
            return
        self.scan_no += 1
        n, folder = self.scan_no, self.save_dir
        views, grasps, name = self.vgn.scan_data()  # references only: the next scan starts new lists

        def write():  # JPEG + compressed npz per view take ~0.1-0.5 s: off the control loop
            from steve_perception.utils.run_record import save_scan
            try:
                save_scan(folder, n, views, grasps, best, name)
                print(f"  grasp scan {n} saved: {folder}/scan{n:02d}_*")
            except Exception as e:  # a record must never stop the robot
                print(f"  [WARN] grasp scan {n} not saved: {type(e).__name__}: {e}")
        threading.Thread(target=write, name=f"save_scan{n}").start()  # not a daemon: finishes at exit

    def vgn_available(self):
        return self.vgn is not None and self.vgn.available

    def vgn_checking(self):
        """True while a requested grasp detector's environment check is still running."""
        return self.vgn is not None and self.vgn.available and self.vgn.health(timeout=60.0) is None

    def vgn_n_views(self):
        return self.vgn.n_views

    def vgn_best(self):
        """Blocking: (grasp or None, inference duration [s])."""
        t0 = time.monotonic()
        ok = self.vgn.run_inference()
        best = self.vgn.best_grasp_world(*self._vgn_near()) if ok else None
        dur = time.monotonic() - t0
        self._save_scan(best)
        return best, dur

    def vgn_start(self):
        """Non-blocking: start inference in the background. False if it cannot start."""
        self._vgn_t0 = time.monotonic()
        return self.vgn.start_inference()

    def vgn_poll(self):
        """None while running; else (grasp or None, inference duration [s])."""
        r = self.vgn.poll_inference()
        if r is None:
            return None
        best = self.vgn.best_grasp_world(*self._vgn_near()) if r else None
        dur = time.monotonic() - self._vgn_t0
        self._save_scan(best)
        return best, dur

    def _vgn_near(self):
        """Only VGN grasps on the target object: within its radius + 3 cm horizontally and
        +8 cm vertically (VGN's grasp centre sits above the object's centre)."""
        if self.vgn_target is None:
            return ()
        return self.vgn_target, self.target.radius + 0.03, self.target.radius + 0.08


class SimWorld(VGNInference):
    """Wrist-camera perception, rangefinder LiDAR and gripper actuator in MuJoCo."""

    def __init__(self, backend: MujocoSimBackend, args):
        self.b, self.a = backend, args
        m, d = backend.model, backend.data
        self.cfg = PerceptionConfig(cam_name="wrist_cam")
        self.renderer = mujoco.Renderer(m, IMG_H, IMG_W)
        self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        self.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        self.target = TargetProfile.load(args.target)  # what the detector looks for
        self.half_h = 0.025  # sphere radius / cube half-size (geom size below)
        self.vgn_target = None  # (position, xy radius, z window) VGN grasps must lie in
        if args.grasp_backend != "vgn" and not args.no_vgn:
            sys.exit(f"--grasp_backend {args.grasp_backend}: robot / Gazebo only (--mode ros); MuJoCo uses VGN")
        self.vgn = VGNBridge() if not args.no_vgn else None
        if args.save_dir not in ("", "auto"):  # simulation: records only into an explicit folder
            from steve_perception.utils.run_record import save_folder
            self.save_dir = save_folder(args.save_dir, "grasp_scan")
        self._place_object()
        # Obstacle contact monitor: any contact between a robot geom and an obstacle fails the run
        name = lambda i, t: mujoco.mj_id2name(m, t, i) or ""
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        root = lambda b: b if m.body_parentid[b] in (0, b) else root(m.body_parentid[b])
        self.robot_body = np.array([root(b) == base for b in range(m.nbody)])
        self.obstacle_geoms = {g for g in range(m.ngeom)
                               if name(m.geom_bodyid[g], mujoco.mjtObj.mjOBJ_BODY).startswith("obst_")}
        self.obstacle_hits = {}  # "robot geom/obstacle" -> first contact time
        if self.obstacle_geoms:
            print(f"  obstacles: {sorted(name(m.geom_bodyid[g], mujoco.mjtObj.mjOBJ_BODY) for g in self.obstacle_geoms)}")
        # The pan-tilt joints have no actuators in the MJCF: hold them with a stiff spring
        # (the real Dynamixels hold their position) so the head camera does not sag.
        for jn in ("pan_tilt_tilt_motor_joint", "pan_tilt_pan_motor_joint"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if j >= 0:
                m.jnt_stiffness[j] = 50.0
                m.qpos_spring[m.jnt_qposadr[j]] = d.qpos[m.jnt_qposadr[j]]
                m.dof_damping[m.jnt_dofadr[j]] = 5.0

    def _place_object(self):
        m, d, a = self.b.model, self.b.data, self.a
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        if g >= 0:
            m.geom_type[g] = (mujoco.mjtGeom.mjGEOM_BOX if a.object_type == "box"
                              else mujoco.mjtGeom.mjGEOM_SPHERE)
            m.geom_size[g] = [0.025, 0.025, 0.025]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        if j >= 0:
            qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
            vy = a.object_speed
            # resting on the table top (z = 0.80); the old script placed the 5 cm cube 1.25 cm into it
            xy = a.target_xy if a.target_xy is not None else TARGET_POS[a.scene]
            d.qpos[qa:qa + 7] = [*xy, 0.80 + self.half_h, 1, 0, 0, 0]
            d.qvel[va:va + 6] = [0.0, vy, 0.0, -vy / 0.025 if a.object_type == "sphere" else 0.0, 0, 0]
            mujoco.mj_forward(m, d)

    def front_distance(self):
        return read_lidar(self.b.model, self.b.data, "front_lidar")

    def transit_height(self, p_from, p_to, margin=TRANSIT_CORRIDOR):
        """Top (world z) of the tallest obstacle whose footprint comes within `margin` of the
        straight xy path p_from -> p_to, or -inf if the path is clear."""
        m, d = self.b.model, self.b.data
        a, b = np.asarray(p_from[:2], float), np.asarray(p_to[:2], float)
        ab = b - a
        top = -np.inf
        for g in self.obstacle_geoms:
            c, h = d.geom_xpos[g], m.geom_aabb[g, 3:]  # obstacles are upright: local AABB ~ world
            s = np.clip((c[:2] - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
            if np.linalg.norm(a + s * ab - c[:2]) < max(h[0], h[1]) + margin:
                top = max(top, c[2] + h[2])
        return top

    def check_obstacles(self, t):
        """Record robot-obstacle contacts of the current physics state."""
        if not self.obstacle_geoms:
            return
        m, d = self.b.model, self.b.data
        for c in d.contact[:d.ncon]:
            for g_obs, g_rob in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if g_obs in self.obstacle_geoms and self.robot_body[m.geom_bodyid[g_rob]]:
                    rob = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g_rob)
                           or mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g_rob]))
                    obs = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g_obs])
                    key = f"{rob}/{obs}"
                    if key not in self.obstacle_hits:
                        self.obstacle_hits[key] = t
                        print(f"  [COLLISION] t={t:.1f}s robot '{rob}' touched obstacle '{obs}'")

    CAMERAS = ("wrist_cam", "pan_tilt_cam")

    def detect(self, cam="wrist_cam"):
        """Object centre in world frame from one RGB-D frame of `cam`, or None."""
        m, d = self.b.model, self.b.data
        mujoco.mj_forward(m, d)
        rgb, depth = render_rgbd(self.renderer, m, d, cam)
        if cam == self.cfg.cam_name:
            self._store_map_frame(depth, cam)
        found, cx, cy, blob = self.target.detect(rgb)
        if not found:
            return None
        pt, raw_d = unproject_pixel_to_world(m, d, cam, cx, cy, depth)
        if pt is None or raw_d < self.a.min_detect_range:
            return None  # at close range the gripper occludes part of the object -> biased
        # all detected pixels in 3-D: the centre from the object's top (viewpoint-independent)
        c = self.target.center_from_points(self._pixels_to_world(cam, blob, depth))
        if c is not None:
            return c
        # The blob centre pixel sees the surface point facing the camera; the object
        # centre lies one radius further along the viewing ray (exact for a sphere).
        # Shifting straight down instead biases the estimate with the viewpoint, which a
        # moving wrist camera turns into a phantom object velocity.
        cam = d.cam_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)]
        ray = pt - cam
        return pt + self.target.radius * ray / np.linalg.norm(ray)

    save_dir = None  # --save_dir <path>: records (table map, scans) of a simulated run

    def _store_map_frame(self, depth, cam_name):
        m, d = self.b.model, self.b.data
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        f = 0.5 * IMG_H / np.tan(np.deg2rad(m.cam_fovy[c]) / 2.0)
        K = np.array([[f, 0.0, IMG_W / 2.0], [0.0, f, IMG_H / 2.0], [0.0, 0.0, 1.0]])
        R = d.cam_xmat[c].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo camera -> optical frame
        self._map_frame = (depth.copy(), K, R, d.cam_xpos[c].copy())

    def mapping_frame(self):
        """The newest perception-camera depth frame for the table map (once), or None."""
        f, self._map_frame = getattr(self, "_map_frame", None), None
        return f

    def mapping_stop(self):
        pass

    def support_top_z(self):
        m, d = self.b.model, self.b.data
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        return None if g < 0 else float(d.geom_xpos[g][2] + m.geom_size[g][2])

    def support_poly(self):
        return None

    def _pixels_to_world(self, cam_name, mask, depth):
        """World points of the mask's pixels (MuJoCo camera: looks along -z, depth along it)."""
        m, d = self.b.model, self.b.data
        c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        f = 0.5 * IMG_H / np.tan(np.deg2rad(m.cam_fovy[c]) / 2.0)
        ys, xs = np.nonzero(mask)
        z = depth[ys, xs]
        ok = z > 1e-6
        xs, ys, z = xs[ok], ys[ok], z[ok]
        p_cam = np.stack([(xs - IMG_W / 2.0) * z / f, (IMG_H / 2.0 - ys) * z / f, -z], axis=1)
        return d.cam_xpos[c] + p_cam @ d.cam_xmat[c].reshape(3, 3).T

    def ground_truth(self):
        return self.b.data.xpos[self.obj_body].copy()

    def support_rect(self):
        """(lo, hi) xy corners of the table the object stands on."""
        m, d = self.b.model, self.b.data
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        if g < 0:
            return None
        return d.geom_xpos[g][:2] - m.geom_size[g][:2], d.geom_xpos[g][:2] + m.geom_size[g][:2]

    def support_exit_time(self, p, v):
        """Seconds until the predicted object path leaves the table top (inf if never)."""
        m, d = self.b.model, self.b.data
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        if g < 0:
            return np.inf
        lo = d.geom_xpos[g][:2] - m.geom_size[g][:2]
        hi = d.geom_xpos[g][:2] + m.geom_size[g][:2]
        t_exit = np.inf
        for ax in range(2):
            if v[ax] > 1e-4:
                t_exit = min(t_exit, (hi[ax] - p[ax]) / v[ax])
            elif v[ax] < -1e-4:
                t_exit = min(t_exit, (lo[ax] - p[ax]) / v[ax])
        return max(t_exit, 0.0)

    def gripper(self, close):
        self.b.data.ctrl[self.grip_act] = 255.0 if close else 0.0

    def relocate(self, xy):
        """--continuous test: put the object down at xy on the table (at rest)."""
        m, d = self.b.model, self.b.data
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        top = self.support_top_z()
        d.qpos[qa:qa + 7] = [*xy, (0.80 if top is None else top) + self.half_h, 1, 0, 0, 0]
        d.qvel[va:va + 6] = 0.0
        mujoco.mj_forward(m, d)

    # VGN (simulation only)
    def vgn_init(self, center):
        self.vgn.init_tsdf(center)

    def vgn_view(self, center, obj_shift=None):
        mujoco.mj_forward(self.b.model, self.b.data)
        return self.vgn.collect_view(self.b.model, self.b.data, self.renderer, self.cfg.cam_name, center,
                                     obj_shift)


class WristDepthGrabber:
    """Real wrist-camera depth views for VGN: aligned depth + camera_info of the RealSense node and
    the camera pose from TF (odom <- optical frame; the URDF's wrist_camera_* frames). The image
    topic is subscribed only while views are taken (images are large over the network). A view is
    a frame that ARRIVED after the request (the caller asks once the arm is still), with the latest
    TF (the robot is still; no clock sync between robot and PC needed).
    With a table model each view's pose is levelled on the table top (as vgn_grasp_real): tilt and
    height of an approximate camera mount are corrected, x/y are not observable from a plane."""

    SFX = "/aligned_depth_to_color"
    RGB_WAIT = 0.3  # s a view waits for an optional colour frame after its depth frame

    def __init__(self, node, tf, args, table, rgb=False, rgb_required=False):
        from sensor_msgs.msg import CameraInfo
        self.node, self.tf, self.a, self.table = node, tf, args, table
        self.log = node.get_logger()
        self.ns = args.vgn_camera
        self.K, self.frame_id, self._sub, self._t_req, self._got = None, None, None, None, None
        # rgb: also a colour frame per view (<ns>/color/image_raw, same pixels as the aligned depth);
        # rgb_required: the view waits for it (CGN's target mask), else up to RGB_WAIT after the depth
        self.want_rgb, self.rgb_required = rgb or rgb_required, rgb_required
        self._rgb_sub, self._rgb_got, self.rgb, self._got_t = None, None, None, None
        self._info_sub = None
        if self.ns != "auto":
            self._info_sub = node.create_subscription(CameraInfo, f"{self.ns}{self.SFX}/camera_info",
                                                      self._info_cb, 10)

    def _info_cb(self, m):
        self.K, self.frame_id = np.array(m.k).reshape(3, 3), m.header.frame_id

    def _resolve(self):
        """--vgn_camera auto: the published */aligned_depth_to_color/camera_info, gripper camera first."""
        if self._info_sub is not None:
            return True
        from sensor_msgs.msg import CameraInfo
        found = sorted(t for t, _ in self.node.get_topic_names_and_types()
                       if t.endswith(self.SFX + "/camera_info") and self.node.count_publishers(t) > 0)
        if not found:
            return False
        found.sort(key=lambda t: ("gripper" not in t, "wrist" not in t))
        self.ns = found[0][:-len(self.SFX + "/camera_info")]
        self.log.info(f"VGN depth camera: {self.ns}")
        self._info_sub = self.node.create_subscription(CameraInfo, found[0], self._info_cb, 10)
        return True

    def _depth_cb(self, m):
        # _t_req None: no frame requested (between views, the subscription stays alive)
        if self._got is None and self._t_req is not None and time.monotonic() >= self._t_req:
            self._got, self._got_t = m, time.monotonic()

    def _rgb_cb(self, m):
        if self._rgb_got is None and self._t_req is not None and time.monotonic() >= self._t_req:
            self._rgb_got = m

    def request(self):
        """Start (or keep) the image subscription; only frames arriving from now on count."""
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        if not self._resolve():
            return
        self._t_req, self._got, self._rgb_got = time.monotonic() + 0.1, None, None  # in flight: older
        if self._sub is None:
            self._sub = self.node.create_subscription(Image, f"{self.ns}{self.SFX}/image_raw", self._depth_cb,
                                                      qos_profile_sensor_data)
        if self.want_rgb and self._rgb_sub is None:
            self._rgb_sub = self.node.create_subscription(Image, f"{self.ns}/color/image_raw", self._rgb_cb,
                                                          qos_profile_sensor_data)

    def stop(self):
        if self._sub is not None:
            self.node.destroy_subscription(self._sub)
            self._sub = None
        if self._rgb_sub is not None:
            self.node.destroy_subscription(self._rgb_sub)
            self._rgb_sub = None

    def poll(self):
        """(depth [m], R_world_optical, t_world_optical, K) once a fresh frame + TF are there, else None.
        With rgb: also a fresh colour frame, left in self.rgb (HxWx3 RGB uint8)."""
        m = self._got
        if m is None or self.K is None:
            return None
        if self.want_rgb and self._rgb_got is None and (
                self.rgb_required or time.monotonic() - self._got_t < self.RGB_WAIT):
            return None
        self._got = None
        self.rgb = None
        if self.want_rgb and self._rgb_got is not None:
            c, self._rgb_got = self._rgb_got, None
            ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(c.encoding)
            if ch is None:
                self.log.warn(f"colour encoding {c.encoding}: no image for the target mask")
            else:
                img = np.frombuffer(c.data, np.uint8).reshape(c.height, c.step // ch, ch)[:, :c.width, :3]
                self.rgb = np.ascontiguousarray(img[:, :, ::-1] if c.encoding.startswith("bgr") else img)
        if m.encoding in ("16UC1", "mono16"):
            d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
        else:
            d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
        from rclpy.time import Time
        try:
            tr = self.tf.lookup_transform("odom", m.header.frame_id or self.frame_id, Time())
        except Exception as e:
            self.log.warn(f"VGN view skipped: no TF odom <- {m.header.frame_id}: {e}")
            return None
        q, tt = tr.transform.rotation, tr.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        t = np.array([tt.x, tt.y, tt.z])
        if self.table is not None and not self.a.vgn_no_level:
            R, t = self._level(d, R, t)
        return d, R, t, self.K

    def _level(self, D, R, t):
        """Camera pose corrected so the table top in this view is level at top_z (see the class doc)."""
        try:
            from steve_wbc.vgn_grasp_real import fit_table_plane, level_correction
        except ImportError:
            from vgn_grasp_real import fit_table_plane, level_correction
        K = self.K
        v, u = np.mgrid[0:D.shape[0]:3, 0:D.shape[1]:3]
        z = D[v, u]
        ok = (z > 0.2) & (z < 2.0)
        pc = np.stack([(u[ok] - K[0, 2]) * z[ok] / K[0, 0], (v[ok] - K[1, 2]) * z[ok] / K[1, 1], z[ok]], 1)
        try:
            n, p0, n_in = fit_table_plane(pc @ R.T + t, self.table)
        except ValueError as e:
            print(f"  VGN view: no table-plane levelling ({e})")
            return R, t
        R_c, dz = level_correction(n, p0, self.table.top_z)
        tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        if tilt > self.a.vgn_max_level or abs(dz[2]) > 0.06:
            print(f"  VGN view: table plane {tilt:.1f} deg / {dz[2] * 100:+.1f} cm off the model — too large, "
                  "not corrected (camera mount / table model / URDF shoulder offset?)")
            return R, t
        print(f"  VGN view levelled on the table: {tilt:.1f} deg, {dz[2] * 100:+.1f} cm ({n_in} plane points)")
        return R_c @ R, R_c @ (t - p0) + p0 + dz


class RosWorld(VGNInference):
    """LiDAR from a LaserScan, object from a PointStamped, gripper via GripperCommand,
    VGN views from the wrist camera (depth + TF)."""

    def __init__(self, node, args, get_base, dry_run, table=None):
        from geometry_msgs.msg import PointStamped
        from sensor_msgs.msg import LaserScan
        self.node, self.a, self.get_base, self.dry_run = node, args, get_base, dry_run
        self.log = node.get_logger()
        self._scan, self._target, self._target_t = np.inf, None, -np.inf
        # The MPO-700 LiDARs sit on the base corners, yawed 45° and mounted upside down, so
        # the scan's 0° is not "ahead": use TF to measure straight ahead in base_link.
        from tf2_ros import Buffer, TransformListener
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, node)
        self._scan_T = None  # scan frame -> base_link (static, looked up once)
        node.create_subscription(LaserScan, args.scan_topic, self._scan_cb, 10)
        node.create_subscription(PointStamped, args.target_topic, self._target_cb, 10)
        self._grip = None
        if not dry_run:
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient
            self._GripperCommand = GripperCommand
            self._grip = ActionClient(node, GripperCommand, args.gripper_action)
        # VGN: runs on this PC (vgn_env), views from the real wrist camera
        self.table = table
        self.target = TargetProfile.load(args.target)  # object size for the grasp filter
        self.vgn = make_grasp_backend(args, self.target) if not args.no_vgn else None
        from steve_perception.utils.run_record import save_folder
        self.save_dir = save_folder(args.save_dir, "grasp_scan")  # scans + table maps
        self._depth = (WristDepthGrabber(node, self._tf, args, table, rgb=self.save_dir is not None,
                                         rgb_required=args.grasp_backend == "cgn")
                       if self.vgn is not None else None)
        if self.vgn is not None and self.vgn.available:
            self.vgn.start_health_check()

    def mapping_frame(self):
        """A fresh gripper-camera depth frame for the table map, or None (asked for once, delivered
        on a later call). No table levelling: a few mm of mount error do not matter at 2 cm cells,
        and its plane fit would cost the control loop ~50 ms per frame."""
        g = getattr(self, "_map_grab", None)
        if g is None:
            g = self._map_grab = WristDepthGrabber(self.node, self._tf, self.a, None)
        if g._sub is None or g._t_req is None:
            g.request()
            return None
        r = g.poll()
        if r is None:
            return None
        g._t_req = None
        d, R, t, K = r
        return d, K, R, t

    def mapping_stop(self):
        g = getattr(self, "_map_grab", None)
        if g is not None:
            g.stop()
            g._t_req = None

    def support_top_z(self):
        return None if getattr(self, "table", None) is None else float(self.table.top_z)

    def support_poly(self):
        return None if getattr(self, "table", None) is None else self.table.corners()

    def camera_pose(self, frame):
        """(R, t) of `frame` in odom from the latest TF, or None."""
        from rclpy.time import Time
        try:
            tr = self._tf.lookup_transform("odom", frame, Time())
        except Exception:
            return None
        q, tt = tr.transform.rotation, tr.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return R, np.array([tt.x, tt.y, tt.z])

    def vgn_available(self):
        if self.vgn is None:
            return False
        if self.vgn.health(timeout=60.0) is None:   # environment check still running: not yet
            return False
        return self.vgn.available

    def vgn_init(self, center):
        self.vgn.init_tsdf(np.asarray(center, float))

    def vgn_view(self, center, obj_shift=None):
        """None while waiting for a fresh depth frame (the caller retries), else the view count."""
        if self._depth._sub is None or self._depth._t_req is None:
            self._depth.request()
            return None
        r = self._depth.poll()
        if r is None:
            return None
        d, R, t, K = r
        if obj_shift is not None:  # object moved / odometry drifted since the TSDF was set up
            t = t - np.asarray(obj_shift, float)
        n = self.vgn.add_real_view(d, t, R, K, center, rgb=self._depth.rgb)
        self.last_view = (d, K, R, t)  # also into the WBC's table map
        self._depth._t_req = None   # the next call requests the next frame
        return n

    def vgn_start(self):
        self._depth.stop()
        return super().vgn_start()

    def vgn_best(self):
        self._depth.stop()
        return super().vgn_best()

    def _scan_cb(self, msg):
        ang = msg.angle_min + msg.angle_increment * np.arange(len(msg.ranges))
        r = np.asarray(msg.ranges, float)
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        if self._scan_T is None:
            try:
                from rclpy.time import Time
                tf = self._tf.lookup_transform("base_link", msg.header.frame_id, Time())
                t, q = tf.transform.translation, tf.transform.rotation
                x, y, z, w = q.x, q.y, q.z, q.w
                R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w)],
                              [2 * (x * y + z * w), 1 - 2 * (x * x + z * z)]])
                self._scan_T = (R, np.array([t.x, t.y]))
                if self.vgn is None:  # TF no longer needed: stop the robot's ~500 Hz /tf stream here
                    self._tf_listener.unregister()
                self.log.info(f"LiDAR {msg.header.frame_id} -> base_link found: distances are measured "
                              f"straight ahead in a {2 * self.a.lidar_corridor:.2f} m wide corridor")
            except Exception:  # TF not up yet: sector around the scan's 0° as before
                half = np.deg2rad(self.a.lidar_sector_deg) / 2
                ok &= np.abs(ang) <= half
                self._scan = float(r[ok].min()) if ok.any() else np.inf
                return
        # scan points in base_link; distance ahead of the LiDAR, like the MuJoCo rangefinder
        R, t = self._scan_T
        pts = (R @ np.vstack([r * np.cos(ang), r * np.sin(ang)])).T + t
        x0 = t[0]
        ahead = ok & (pts[:, 0] > x0 + 0.01) & (np.abs(pts[:, 1]) < self.a.lidar_corridor)
        self._scan = float((pts[ahead, 0] - x0).min()) if ahead.any() else np.inf

    def _target_cb(self, msg):
        p = np.array([msg.point.x, msg.point.y, msg.point.z])
        frame = msg.header.frame_id.lstrip("/")
        if frame.endswith(("base_link", "base_footprint")):
            b = self.get_base()
            if b is None:
                return
            p = np.array([b[0], b[1], 0.0]) + rot_z(b[2]) @ p
        elif frame not in ("odom", "map", "world", ""):
            self.log.warn(f"target in unsupported frame '{frame}' (use odom or base_link)")
            return
        self._target, self._target_t = p, time.monotonic()

    def front_distance(self):
        return self._scan

    CAMERAS = ("target_topic",)

    def detect(self, cam=None):
        fresh = time.monotonic() - self._target_t < self.a.target_timeout
        return self._target.copy() if fresh and self._target is not None else None

    def ground_truth(self):
        return None

    def support_exit_time(self, p, v):
        return np.inf  # support surface unknown on the real robot

    def gripper(self, close):
        pos = 0.7929 if close else 0.0
        if self._grip is None:
            self.log.info(f"[dry-run] gripper -> {pos:.3f}")
            return
        if not self._grip.server_is_ready():
            self.log.warn(f"gripper action {self.a.gripper_action} not available")
            return
        goal = self._GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = pos, self.a.grip_effort
        self._grip.send_goal_async(goal)

    obstacle_hits = {}  # no contact monitoring outside the simulation

    def transit_height(self, p_from, p_to, margin=None):
        return -np.inf  # obstacles unknown: the WBC's collision constraints only

    def support_rect(self):
        # measured table (--table, table_probe) or unknown: no approach-side planning
        return self.table.support_rect() if getattr(self, "table", None) is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# Task-level state machine on top of the WBC
# ══════════════════════════════════════════════════════════════════════════════
def grasp_for_heading(R, yaw):
    """Of the two equivalent top-down grasps of a parallel gripper (R and R turned 180 deg about the
    vertical), the one closer to the base heading's natural wrist pose (rot_z(yaw) @ R_TOPDOWN) —
    else, e.g. from the table's back side, wrist 3 must turn ~180 deg and the WBC compensates by
    sliding and turning the base."""
    ref = rot_z(yaw) @ R_TOPDOWN
    R2 = rot_z(np.pi) @ R
    return R if np.trace(ref.T @ R) >= np.trace(ref.T @ R2) else R2


def table_quadrant(xy, lo, hi, arrival_side):
    """Region of the table that point xy lies in, seen from `arrival_side` (the side the robot
    came from), and the table side to grasp from (QUADRANT_SIDE): near (bottom) half -> stay; far
    (top) half split in thirds across: left / right -> that side, centre -> the opposite side.
    Returns (region, side)."""
    n_out = SIDES[arrival_side][0]                 # outward normal of the robot's side
    into = -n_out                                  # from the robot into the table
    left = np.array([-into[1], into[0]])           # the robot's left, facing the table
    d = np.asarray(xy, float)[:2] - 0.5 * (np.asarray(lo) + np.asarray(hi))
    width = abs((np.asarray(hi) - np.asarray(lo)) @ left)      # table extent across
    across = d @ left
    if d @ into <= 0:
        quad = f"bottom-{'left' if across > 0 else 'right'}"
    else:
        quad = f"top-{'left' if across > width / 6 else 'right' if across < -width / 6 else 'center'}"
    pref = QUADRANT_SIDE[quad]
    if pref in ("left", "right", "back"):          # robot-relative -> table side (front/back/left/right)
        want = {"left": left, "right": -left, "back": into}[pref]
        pref = max(SIDES, key=lambda s: SIDES[s][0] @ want)
    else:
        pref = arrival_side
    return quad, pref


class ObjectTracker:
    """Constant-velocity estimate: least-squares line fit over a sliding window of
    detections, with outlier gating. Objects stay on the table (no z velocity)."""

    GATE = 0.05      # m   reject detections this far from the prediction
    WINDOW = 4.0     # s   fit window
    V_MAX = 0.15     # m/s

    def __init__(self):
        self.p, self.v, self.t = None, np.zeros(3), None
        self.hist = []
        self.rejected = 0

    def update(self, p, t, still=True):
        """Add a detection. `still`: taken while the camera was (nearly) still. Every detection
        updates the position; the velocity is only fitted from still detections, so the
        viewpoint-dependent bias of a camera moving with the arm/base (its centre estimate is
        taken from the visible face) never turns into fake object motion."""
        if self.t is not None and t - self.t > self.WINDOW:
            # not confirmed for a whole fit window (occluded, or the robot drove elsewhere):
            # the old estimate is stale, start a fresh track instead of gating against it
            self.p, self.v, self.hist = None, np.zeros(3), []
        if self.p is not None and np.linalg.norm(p - self.predict(t)) > self.GATE:
            self.rejected += 1
            return False
        self.hist = [h for h in self.hist if t - h[0] <= self.WINDOW] + [(t, p.copy(), still)]
        T = np.array([h[0] for h in self.hist])
        P = np.array([h[1] for h in self.hist])
        S = np.array([h[2] for h in self.hist])
        Ts, Ps = T[S], P[S]
        # refit only with enough still data; otherwise keep the velocity known so far (a short
        # span would fail the displacement test and wipe a real velocity)
        if len(Ts) >= 5 and Ts[-1] - Ts[0] >= MIN_FIT_SPAN:
            A = np.column_stack([np.ones_like(Ts), Ts - t])
            coef, *_ = np.linalg.lstsq(A, Ps, rcond=None)   # P ≈ p(t) + v (T - t)
            v = coef[1].copy()
            v[2] = 0.0
            n = np.linalg.norm(v)
            # standard error of the fitted xy velocity from the fit residuals
            res = Ps[:, :2] - A @ coef[:, :2]
            sigma = np.sqrt(np.sum(res ** 2) / max(2 * len(Ts) - 4, 1))
            se_v = sigma / (np.sqrt(len(Ts)) * max(np.std(Ts), 1e-6))
            span = Ts[-1] - Ts[0]
            if n < max(STATIC_SPEED, 3.0 * se_v) or n * span < MIN_TRACK_DISPLACEMENT:
                # not significant, or smaller than the viewpoint bias of a non-spherical object
                v[:] = 0.0
            elif n > self.V_MAX:
                v *= self.V_MAX / n
            self.v = v
        # position: all detections, moved to time t with the current velocity (robust median)
        self.p = np.median(P - np.outer(T - t, self.v), axis=0)
        self.t = t
        return True

    def predict(self, t):
        return self.p + self.v * (t - self.t)


class InterceptionMPC:
    """
    Receding-horizon interception planner for the EE reference.

    Each axis is a double integrator driven by acceleration a_k (step h):
        p_{k+1} = p_k + h v_k + h²/2 a_k,     v_{k+1} = v_k + h a_k
    Over the horizon k = 1..N the plan minimises
        Σ_k  w_p |p_k - o_k|²  +  w_v |v_k - ȯ_k|²  +  w_a |a_k|²     s.t.  |a_k| ≤ a_max
    where o_k is the object's PREDICTED path (constant-velocity tracker, fitted from
    perception since the start of the run) plus the phase offset (pre-grasp height,
    grasp point, ...). Matching ȯ makes the gripper arrive moving with the object.
    Condensed, each axis is a bounded least-squares problem solved exactly (BVLS);
    the first step becomes the WBC's EE reference, then the horizon recedes.
    """

    def __init__(self, p, R, horizon=15, h=0.1, w_p=1.0, w_v=0.3, w_a=0.02):
        self.p, self.v, self.R = p.copy(), np.zeros(3), R.copy()
        self.N, self.h = horizon, h
        self.k = np.arange(1, horizon + 1)
        Gp, Gv = np.zeros((horizon, horizon)), np.zeros((horizon, horizon))
        for i, kk in enumerate(self.k):
            for j in range(kk):
                Gp[i, j] = (kk - j - 0.5) * h * h
                Gv[i, j] = h
        self.sw = np.sqrt([w_p, w_v, w_a])
        self.A = np.vstack([self.sw[0] * Gp, self.sw[1] * Gv, self.sw[2] * np.eye(horizon)])
        self.Gp = Gp
        self.plan = None

    def step(self, obj_p, obj_v, offset, goal_R, a_max, v_max, dt):
        a0, plan = np.zeros(3), np.zeros((self.N, 3))
        for ax in range(3):
            o = obj_p[ax] + offset[ax] + obj_v[ax] * self.k * self.h
            free_p = self.p[ax] + self.k * self.h * self.v[ax]
            b = np.concatenate([self.sw[0] * (o - free_p),
                                self.sw[1] * (obj_v[ax] - self.v[ax]) * np.ones(self.N),
                                np.zeros(self.N)])
            acc = lsq_linear(self.A, b, bounds=(-a_max, a_max), method="bvls").x
            a0[ax] = acc[0]
            plan[:, ax] = free_p + self.Gp @ acc
        self.plan = plan
        v_new = self.v + a0 * dt
        n = np.linalg.norm(v_new - obj_v)
        if n > v_max:  # speed limit relative to the object
            v_new = obj_v + (v_new - obj_v) * (v_max / n)
        step_p = 0.5 * (self.v + v_new) * dt
        self.p, self.v = self.p + step_p, v_new
        rv = log_so3(goal_R @ self.R.T)
        ang = np.linalg.norm(rv)
        rstep = rv if ang <= EE_ROT_SPEED * dt else rv * (EE_ROT_SPEED * dt / ang)
        self.R = exp_so3(rstep) @ self.R
        return Reference(self.p.copy(), self.v.copy(), self.R.copy(), rstep / dt, "mpc")

    def leash(self, p_ee, R_ee, max_lead=MPC_LEASH, max_ang=MPC_LEASH_ANG):
        """Reference governor: the reference never runs more than max_lead [m] / max_ang [rad] ahead
        of the real EE. Else a slow arm (low speed limits, a late control loop, a large wrist
        rotation) falls behind until the tracking error looks like a blocked gripper; real blocks
        still end in the pre-grasp timeout / the descent's stuck check."""
        d = self.p - p_ee
        n = np.linalg.norm(d)
        if n > max_lead:
            self.p = p_ee + d * (max_lead / n)
        rv = log_so3(self.R @ R_ee.T)
        ang = np.linalg.norm(rv)
        if ang > max_ang:
            self.R = exp_so3(rv * (max_ang / ang)) @ R_ee


class TableOrbit:
    """Smooth base path around a rectangular table [lo, hi] (REPOSITION).

    Positions p = Q(s) + d(s) N(s): Q / N run along the table outline (4 edges and 4 corners) with
    outward normal N, s is the arc length on the ring at offset `o` (edges joined by quarter
    circles: no corners to stop at), and the distance d to the table blends smoothly from the
    start's to `o` (clear at any heading), and from `o` to the goal's — never closer to the table
    than start or goal. The base faces the table all the way (yaw = direction of -N; the omni base
    moves sideways), the speed follows a min-jerk profile (rest -> v_max -> rest), and the WBC's base
    task gets pose + velocity feed-forward at every tick."""

    def __init__(self, lo, hi, o):
        self.lo, self.hi, self.o = np.asarray(lo, float), np.asarray(hi, float), float(o)
        w, h = self.hi - self.lo
        (x0, y0), (x1, y1) = self.lo, self.hi
        q = 0.5 * np.pi * self.o
        # ccw: (kind, length on the ring, start point / corner, edge direction, normal angle at start)
        self.segs = [("edge", w, (x0, y0), (1, 0), -0.5 * np.pi), ("arc", q, (x1, y0), None, -0.5 * np.pi),
                     ("edge", h, (x1, y0), (0, 1), 0.0), ("arc", q, (x1, y1), None, 0.0),
                     ("edge", w, (x1, y1), (-1, 0), 0.5 * np.pi), ("arc", q, (x0, y1), None, 0.5 * np.pi),
                     ("edge", h, (x0, y1), (0, -1), np.pi), ("arc", q, (x0, y0), None, np.pi)]
        self.cum = np.concatenate([[0.0], np.cumsum([sg[1] for sg in self.segs])])
        self.P = self.cum[-1]

    def QN(self, s):
        """Outline point Q and outward normal N at ring arc length s."""
        s = s % self.P
        k = min(int(np.searchsorted(self.cum, s, side="right")) - 1, 7)
        kind, _, a, t, phi = self.segs[k]
        u = s - self.cum[k]
        if kind == "edge":
            return np.asarray(a, float) + u * np.asarray(t, float), np.array([np.cos(phi), np.sin(phi)])
        ang = phi + u / self.o
        return np.asarray(a, float), np.array([np.cos(ang), np.sin(ang)])

    def project(self, p):
        """(s, d) of a point outside the table: its ring arc length and distance to the table."""
        p = np.asarray(p, float)[:2]
        (x0, y0), (x1, y1) = self.lo, self.hi
        q = np.clip(p, self.lo, self.hi)
        v = p - q
        d = float(np.linalg.norm(v))
        below, above, left, right = p[1] < y0, p[1] > y1, p[0] < x0, p[0] > x1
        if (below or above) and (left or right):              # beside a corner: on its arc
            k = {(True, False, False, True): 1, (False, True, False, True): 3,
                 (False, True, True, False): 5, (True, False, True, False): 7}[(below, above, left, right)]
            ang = (np.arctan2(v[1], v[0]) - self.segs[k][4]) % (2 * np.pi)
            return self.cum[k] + self.o * min(ang, 0.5 * np.pi), d
        if below:
            return self.cum[0] + (q[0] - x0), d
        if right:
            return self.cum[2] + (q[1] - y0), d
        if above:
            return self.cum[4] + (x1 - q[0]), d
        return self.cum[6] + (y1 - q[1]), d                 # left (or inside: nearest is ambiguous)

    def path(self, start, goal, v_max, blend=0.6):
        """OrbitPath from start (x, y, yaw) to goal (x, y, yaw), the shorter way round."""
        s0, d0 = self.project(start)
        s1, d1 = self.project(goal)
        D = (s1 - s0) % self.P
        if D > 0.5 * self.P:
            D -= self.P
        L = abs(D)
        la, lb = min(blend, 0.45 * L), min(blend, 0.45 * L)
        sm = lambda x: 3 * x ** 2 - 2 * x ** 3
        sig = np.linspace(0.0, 1.0, max(200, int(L / 0.005)))
        a = sm(np.clip(sig * L / max(la, 1e-9), 0, 1))
        b = sm(np.clip((sig * L - (L - lb)) / max(lb, 1e-9), 0, 1))
        d = self.o + (d0 - self.o) * (1 - a) + (d1 - self.o) * b
        QN = [self.QN(s0 + x * D) for x in sig]
        pts = np.array([Q + di * N for (Q, N), di in zip(QN, d)])
        face = np.unwrap([np.arctan2(-N[1], -N[0]) for _, N in QN])  # base +x towards the table
        face += 2 * np.pi * np.round((start[2] - face[0]) / (2 * np.pi))  # same 2-pi branch as the start yaw
        yaw = face + wrap_angle(start[2] - face[0]) * (1 - a) + wrap_angle(goal[2] - face[-1]) * b
        return OrbitPath(pts, yaw, v_max)


class OrbitPath:
    """Time-parameterised base path: at(t) -> (BaseReference, done). Trapezoidal speed along its
    length — ramps of `acc`, cruise at <= v_max: ~0.95 of v_max on average for a 2 m path, where
    the earlier min-jerk profile averaged 0.53 (2 m at 0.1 m/s: ~21 s instead of 37.5 s)."""

    def __init__(self, pts, yaw, v_max, acc=ORBIT_ACC):
        self.pts, self.yaw = pts, np.asarray(yaw, float)
        self.c = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        self.length = L = float(self.c[-1])
        turn = float(np.abs(np.diff(self.yaw)).sum())
        t_lin = 2.0 * np.sqrt(L / acc) if L < v_max ** 2 / acc else L / v_max + v_max / acc
        self.T = max(2.0, t_lin, turn / 0.4)   # <= v_max, <= 0.4 rad/s on average
        # cruise speed covering L in T with ramps of acc: v^2/acc - v T + L = 0 (smaller root)
        self.acc = acc
        self.v_c = 0.5 * (acc * self.T - np.sqrt(max((acc * self.T) ** 2 - 4.0 * acc * L, 0.0)))
        self.t_r = self.v_c / acc

    def __bool__(self):
        return True

    def at(self, t):
        T, a, vc, tr = self.T, self.acc, self.v_c, self.t_r
        t = min(max(t, 0.0), T)
        tau = t / T
        if t < tr:
            c, dc = 0.5 * a * t * t, a * t
        elif t < T - tr:
            c, dc = 0.5 * a * tr * tr + vc * (t - tr), vc
        else:
            c, dc = self.length - 0.5 * a * (T - t) ** 2, a * (T - t)
        i = int(np.clip(np.searchsorted(self.c, c) - 1, 0, len(self.c) - 2))
        f = (c - self.c[i]) / max(self.c[i + 1] - self.c[i], 1e-9)
        p = self.pts[i] + f * (self.pts[i + 1] - self.pts[i])
        yw = self.yaw[i] + f * (self.yaw[i + 1] - self.yaw[i])
        ds = max(self.c[i + 1] - self.c[i], 1e-9)
        v = (self.pts[i + 1] - self.pts[i]) / ds * dc
        wz = (self.yaw[i + 1] - self.yaw[i]) / ds * dc
        return BaseReference(np.array([p[0], p[1], yw]), np.array([v[0], v[1], wz])), tau >= 1.0


class ActivePerceptionWBC:
    def __init__(self, backend, world, kin, args):
        self.b, self.w, self.kin, self.a = backend, world, kin, args
        self.dt = 1.0 / args.rate
        self.sim = isinstance(backend, MujocoSimBackend)
        self.d_safe = args.d_safe

    # ── helpers ─────────────────────────────────────────────────────────────
    def _enter(self, state, msg=""):
        print(f"\n[{self.state} → {state}] {msg}")
        self.state, self.t_state = state, self.t

    def _settled(self, st, q_ref, tol=0.02):
        return np.max(np.abs(np.asarray(q_ref) - st.q)) < tol and np.max(np.abs(self.u_prev)) < 0.03

    def _pan_to(self, st, p_world):
        """Shoulder-pan angle that points the arm at a world point."""
        self.kin.set_state(st)
        sh = self.kin.data.xpos[mujoco.mj_name2id(self.kin.model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")]
        d = rot_z(st.base[2]).T @ (p_world - sh)
        return float(np.arctan2(d[1], d[0]))

    def _setup_camera(self, st):
        """Camera of the visibility task. MuJoCo: the perception camera of the model; robot: the
        camera's optical frame from TF (calibrated mount), expressed on the gripper body of the
        model once at the start (the robot stands still)."""
        # cam_ok: camera known (its angle to the object is measured either way, for the summary);
        # vis_on: the visibility task steers it (not with --no_visibility)
        self.cam_ok = self.vis_on = False
        self.vis_deg, self.vis_ticks, self.vis_good = None, 0, 0
        self.still_since = None  # time since which the arm moves < 0.03 rad/s
        m = self.kin.model
        if self.sim:
            name = self.w.cfg.cam_name
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                print(f"[visibility] no camera '{name}' in the model -> off")
                return
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.cam_quat[cid])
            self.kin.set_camera(m.cam_bodyid[cid], m.cam_pos[cid], -R.reshape(3, 3)[:, 2])  # MuJoCo: -z
            src = f"MuJoCo camera '{name}'"
        else:
            pose, t_end = None, time.monotonic() + 10.0
            while pose is None and time.monotonic() < t_end:
                pose = self.w.camera_pose(self.a.look_camera_frame)
                if pose is None:
                    st = self.b.read_state() or st  # keeps the TF listener fed
                    time.sleep(0.05)
            if pose is None:
                print(f"[visibility] no TF odom -> {self.a.look_camera_frame} within 10 s (camera driver + mount "
                      "TF running?) -> visibility task off")
                return
            R_c, t_c = pose
            self.kin.set_state(st)
            b = m.site_bodyid[self.kin.site_id]  # gripper body (the pinch site's): rigid with the flange
            R_b, p_b = self.kin.data.xmat[b].reshape(3, 3), self.kin.data.xpos[b]
            self.kin.set_camera(b, R_b.T @ (t_c - p_b), R_b.T @ R_c[:, 2])
            src = f"'{self.a.look_camera_frame}' from TF"
        self.kin.set_state(st)
        c, z, _, _ = self.kin.camera(st.base[2])
        p_ee, R_ee = self.kin.ee_pose()
        self.cam_ok, self.vis_on = True, not self.a.no_visibility
        print(f"[visibility{'' if self.vis_on else ' OFF, angle measured only'}] camera {src}: {np.linalg.norm(c - p_ee) * 100:.0f} cm from the pinch point, axis "
              f"{np.degrees(np.arccos(np.clip(z @ R_ee[:, 2], -1, 1))):.0f} deg from the approach axis; "
              f"{'kept on' if self.vis_on else 'angle to'} the object in {', '.join(VIS_STATES)}")

    def _front_clearance(self):
        """Base clearance ahead: front LiDAR, and the environment model (table) when there is one."""
        front, coll = self.w.front_distance(), self.ctrl.coll
        if coll is not None and np.isfinite(getattr(coll, "base_clearance", np.inf)):
            front = min(front, coll.base_clearance)
        return front

    def _setup_map(self):
        """Table map for the occlusion-aware line of sight (needs the camera and a table)."""
        self.hmap, self.los_pts, self.map_t, self.map_cut, self.searched = None, None, -np.inf, False, []
        rect = self.w.support_rect()
        top = self.w.support_top_z() if rect is not None else None
        if self.a.no_los or not self.cam_ok or top is None:
            if not self.a.no_los:
                print("[line of sight] off: needs the camera and a table model (--table)")
            return
        self.hmap = TableHeightMap(*rect, top, poly=self.w.support_poly())
        print(f"[line of sight] table map {self.hmap.n[0]} x {self.hmap.n[1]} cells of {self.hmap.cell * 100:.0f} cm "
              f"at z {top:.2f} m: approach side by line of sight, sight line kept clear by the QP, an object "
              f"not found is searched from the other sides")

    def _update_map(self):
        """Depth frames into the table map while the EE is free (every MAP_PERIOD)."""
        if self.state not in MAP_STATES:
            self.w.mapping_stop()
            return
        if self.t - self.map_t < MAP_PERIOD:
            return
        f = self.w.mapping_frame()
        if f is not None:
            self.map_t = self.t
            self._integrate_map(*f)

    def _integrate_map(self, depth, K, R, t):
        tgt = self.tracker.predict(self.t) if self.tracker.p is not None else None
        r = self.w.target.radius + 0.025
        if tgt is not None and not self.map_cut:  # cells of the object seen before it was tracked
            self.hmap.forget_target(tgt, r)
            self.map_cut = True
        self.hmap.integrate(depth, K, R, t, target=tgt, target_r=r)
        self.los_pts = self.hmap.occluders(near=tgt, radius=LOS_NEAR) if tgt is not None else None

    def _los_from(self, pose, obj):
        """Share of clear sight lines from the camera at the scan pose with the base at `pose`."""
        q = np.array(SCAN_SEED_Q, float)
        q[0] = self._pan_to(RobotState(pose, q), obj)
        self.kin.set_state(RobotState(pose, q))
        c = self.kin.camera(pose[2])[0]
        return self.hmap.line_of_sight(c, obj, self.w.target.radius), c

    def _standoff(self):
        return self.a.d_safe_base + BASE_LAG_BUFFER + STANDOFF_MARGIN + self.ctrl.coll.base_xy_max[0]

    def _search_next_side(self, st):
        """Object not found from here (hidden behind something on the table?): drive around the
        table to the nearest side not searched yet and look again. False if none is left."""
        if self.a.no_los or not self._can_plan():
            return False
        lo, hi = self.w.support_rect()
        here = self._side_of(st.base, lo, hi)
        if here not in self.searched:
            self.searched.append(here)
        best, centre = None, 0.5 * (lo + hi)
        for name in [s for s in SIDES if s not in self.searched]:
            n, yaw = SIDES[name]
            xy = np.where(n != 0, np.where(n > 0, hi, lo) + n * self._standoff(), centre)
            goal = np.array([xy[0], xy[1], yaw])
            path, length = self._ring_path(st.base, goal, lo, hi)
            if best is None or length < best[0]:
                best = (length, name, path)
        if best is None:
            return False
        length, name, path = best
        self.side, self.base_path = name, path
        self._enter("REPOSITION", f"object not found from the {here} side"
                    + (f" (table map: {self.hmap.summary()})" if self.hmap is not None else "")
                    + f" -> searching from the {name} side ({length:.1f} m)")
        return True

    def _look_target(self):
        """Point the camera keeps on its axis: the tracked (static) object, else a fresh glimpse."""
        if not self.cam_ok or self.state not in VIS_STATES:
            return None
        if self.tracker.p is not None and self.t - self.last_seen < self.a.lost_timeout_static:
            return self.tracker.predict(self.t)
        if self.glimpse is not None and self.t - self.glimpse[1] < 2.0:
            return self.glimpse[0]
        return None

    def _camera_ready(self, still_for=None):
        """Visibility task on: camera on the object (< VIS_READY_DEG) and, with still_for, the arm
        still (< 0.03 rad/s) for that long. The posture's joint targets are not waited for: posture
        and visibility share a QP level and settle on a compromise, so the shoulder pan can stop short
        of its posture target for good (a 0.03 rad pan condition never fired: sim, 2026-10-15)."""
        if self.vis_deg is None or self.vis_deg >= VIS_READY_DEG:
            return False
        return still_for is None or (self.still_since is not None and self.t - self.still_since >= still_for)

    def _tracked(self):
        """Object path known well enough to act on (recent, several detections)."""
        return self.tracker.p is not None and len(self.tracker.hist) >= 5 and self.t - self.last_seen < 1.0

    def _perceive(self):
        """Perception runs in every phase from the start of the run: the head (pan-tilt)
        camera while driving, then wrist and head camera alternately. Every detection
        goes through the gated least-squares tracker."""
        if self.n_tick % self.a.render_every:
            return
        cams = self.w.CAMERAS
        if self.state in ("DRIVE", "BACKUP"):
            cams = [c for c in cams if c != "wrist_cam"] or cams
        elif self.state in ("DESCEND", "GRASP", "LIFT"):
            cams = [c for c in cams if c != "pan_tilt_cam"] or cams
        cam = cams[(self.n_tick // self.a.render_every) % len(cams)]
        p = self.w.detect(cam)
        if p is None:
            return
        top = self.w.support_top_z() if self.w.support_rect() is not None else None
        if top is not None and self.state not in ("LIFT", "HOLD", "RELEASE"):
            # the object rests on the table: its centre at top + radius. Anything else is a bad
            # detection (camera closer than its depth range at the grasp pose, partly hidden by the
            # gripper, ...) — accepted after an occlusion it re-starts the track somewhere wrong
            dz = p[2] - (top + self.w.target.radius)
            if abs(dz) > self.a.detect_z_tol:
                self.n_bad_z += 1
                if self.n_bad_z in (1, 10, 100) or self.n_bad_z % 500 == 0:
                    print(f"  detection at z={p[2]:.3f} m ignored: {dz * 100:+.0f} cm from resting on the table "
                          f"(--detect_z_tol {self.a.detect_z_tol:.2f}; {self.n_bad_z} so far)")
                return
        # while searching, a camera swung by the arm sees the object biased along the sweep
        # (which the tracker would fit as motion): only a glimpse to turn towards
        if (self.state in ("SETTLE_ARM", "COARSE_DETECT")
                and np.max(np.abs(self.u_prev[3:])) > SEARCH_STILL_DQ):
            self.glimpse = (p, self.t)
            return
        # the viewpoint bias changes when the arm swings the camera, hardly when the base
        # translates it (e.g. while driving up to the table)
        still = np.max(np.abs(self.u_prev[3:])) < TRACK_STILL_DQ
        if self.tracker.update(p, self.t, still):
            self.last_seen = self.t
            self.n_seen[cam] = self.n_seen.get(cam, 0) + 1

    def _mpc(self, offset, goal_R, a_max, v_max, obj_moves=True):
        if self.state in LEASHED_STATES:  # free-space moves only: near the object a blocked arm
            self.mpc.leash(self.ee, self.R_ee)  # must show up as tracking error (stop / replan)
        obj = self.tracker.predict(self.t)
        return self.mpc.step(obj, self.tracker.v if obj_moves else np.zeros(3), np.asarray(offset, float),
                             goal_R, a_max, v_max, self.dt)

    # ── state machine: returns (ee_ref, base_ref, q_nom) ────────────────────
    def update(self, st):
        a, t, dt = self.a, self.t, self.dt
        hold = BaseReference(self.base_hold, np.zeros(3))
        ts = t - self.t_state
        if self.state not in ("DONE", "ABORT"):
            self._perceive()
            if self.w.vgn_available():
                self._vgn_update()
            if self.explore_why is not None and self.state in ("VGN_WAIT", "PRE_GRASP"):
                why, self.explore_why = self.explore_why, None
                self._explore(st, why)
                return None, BaseReference(st.base.copy(), np.zeros(3)), list(st.q)

        if self.state == "DRIVE":
            front = self._front_clearance()
            yaw = st.base[2]
            stop_d = a.d_safe_base + LIDAR_STOP_MARGIN
            speed = (a.drive_speed if front > a.d_safe_base + LIDAR_SLOW_MARGIN
                     else SLOW_SPEED if front > stop_d else 0.0)
            travelled = np.linalg.norm(st.base[:2] - self.drive_start[:2])
            if speed == 0.0 or travelled >= a.max_drive:
                self.base_hold, self.backup_start = st.base.copy(), st.base.copy()
                why = f"LiDAR {front:.3f} m" if front <= stop_d else f"travelled {travelled:.2f} m"
                self._enter("BACKUP" if front < a.d_safe_base - LIDAR_EMERGENCY_MARGIN else "SETTLE_ARM",
                            why + (f" | object tracked since t={self.tracker.hist[0][0]:.1f} s"
                                   if self.tracker.hist else " | object not seen yet"))
                return None, BaseReference(self.base_hold, np.zeros(3)), DRIVE_POSE
            q = self._drive_vgn_views(st) if a.vgn_mode == "drive" else DRIVE_POSE
            return None, BaseReference(st.base.copy(), np.array([speed * np.cos(yaw), speed * np.sin(yaw), 0.0])), q

        if self.state == "BACKUP":
            # LiDAR and the table model (the LiDARs do not see the top plate): until the base is
            # clear of both again, at most 0.5 m
            front = self._front_clearance()
            backed = np.linalg.norm(st.base[:2] - self.backup_start[:2])
            if front >= a.d_safe_base + LIDAR_STOP_MARGIN or backed > 0.5:
                self.base_hold = st.base.copy()
                self._enter("SETTLE_ARM", f"clearance {front * 100:.0f} cm restored" if front >= a.d_safe_base
                            + LIDAR_STOP_MARGIN else f"[WARN] backed up {backed:.2f} m, clearance still only "
                                                     f"{front * 100:.0f} cm (< {a.d_safe_base * 100:.0f} cm)")
                return None, BaseReference(self.base_hold, np.zeros(3)), DRIVE_POSE
            yaw = st.base[2]
            return None, BaseReference(st.base.copy(), -BACKUP_SPEED * np.array([np.cos(yaw), np.sin(yaw), 0.0])), DRIVE_POSE

        if self.state == "REPOSITION":
            # drive around the table on a smooth orbit (TableOrbit: rounded ring, facing the table,
            # trapezoidal speed; arm tucked) — the WBC's base task tracks pose + velocity feed-forward,
            # the QP's base constraint still guards d_safe_base. The orbit clock is the state time
            # (t_state skips safety stops), so the reference waits during a stop.
            if not self.base_path:
                self._enter("SETTLE_ARM", f"at the {self.side} side")
                return None, BaseReference(self.base_hold, np.zeros(3)), DRIVE_POSE
            ref, done = self.base_path.at(ts)
            err = np.linalg.norm(ref.pose[:2] - st.base[:2])
            if done and err < 0.02 and abs(wrap_angle(ref.pose[2] - st.base[2])) < 0.03:
                self.base_hold, self.base_path = ref.pose.copy(), []
                self._enter("SETTLE_ARM", f"at the {self.side} side")
                return None, BaseReference(self.base_hold, np.zeros(3)), DRIVE_POSE
            if ts > self.base_path.T + 30.0:
                self._enter("ABORT", f"repositioning timed out (base {err * 100:.0f} cm from the orbit)")
                return None, BaseReference(st.base.copy(), np.zeros(3)), DRIVE_POSE
            return None, ref, DRIVE_POSE

        if self.state == "SETTLE_ARM":
            q = list(SCAN_SEED_Q)
            if self.tracker.p is not None:  # look at where the object is predicted to be
                q[0] = self._pan_to(st, self.tracker.predict(t))
            # the pan target follows the moving object, so allow residual motion
            ready = (self._camera_ready(still_for=0.3) and np.max(np.abs(self.u_prev)) < 0.06
                     if self.vis_deg is not None else
                     self._settled(st, q, tol=0.04) and np.max(np.abs(self.u_prev)) < 0.06)
            if ready or ts > 6.0:
                if self._tracked():
                    self._object_found(st, "object path known from perception since the start")
                else:
                    self.pan_cmd, self.pan_dir = st.q[0], -1.0
                    self._enter("COARSE_DETECT", "object not tracked yet -> sweeping shoulder_pan")
            return None, hold, q

        if self.state == "COARSE_DETECT":
            cfg = PerceptionConfig()
            if self.pan_cmd <= cfg.scan_start:
                self.pan_dir = 1.0
            elif self.pan_cmd >= cfg.scan_end:
                self.pan_dir = -1.0
            if self.glimpse is not None and t - self.glimpse[1] < 2.0:
                # seen while sweeping: stop and look there, still detections feed the tracker
                self.pan_cmd = self._pan_to(st, self.glimpse[0])
            else:
                self.pan_cmd += self.pan_dir * cfg.scan_vel * dt
            q = list(SCAN_SEED_Q)
            q[0] = self.pan_cmd
            if self._tracked():
                self._object_found(st, "object found")
            elif ts > (a.side_search_timeout if self._can_plan() and not a.no_los else a.search_timeout):
                if not self._search_next_side(st):
                    self._end_episode("object not found" + (f" from the {', '.join(self.searched)} side(s)"
                                                            if self.searched else ""), failed=True)
            return None, hold, q

        if self.state == "VGN_SCAN":
            q = list(SCAN_SEED_Q)
            if self.view_i < len(self.view_pans):
                q[0] = self.view_pans[self.view_i]
                if (self._camera_ready(still_for=0.0 if abs(st.q[0] - q[0]) < 0.10 else 1.0)
                        if self.vis_deg is not None else
                        abs(st.q[0] - q[0]) < 0.03 and np.max(np.abs(self.u_prev)) < 0.03):
                    n = self._vgn_capture(st, "scan")
                    if n is None:  # real camera: waiting for a depth frame taken at this pose
                        self.view_wait_t0 = t if self.view_wait_t0 is None else self.view_wait_t0
                        if t - self.view_wait_t0 > VIEW_TIMEOUT:
                            print(f"  VGN view at pan={st.q[0]:.3f} rad skipped: no depth frame + TF "
                                  f"within {VIEW_TIMEOUT:.0f} s")
                            self.view_i, self.view_wait_t0 = self.view_i + 1, None
                        return None, hold, q
                    self.view_wait_t0 = None
                    print(f"  VGN view {n} at pan={st.q[0]:.3f} rad ({self.side or 'here'} side)")
                    self.view_i += 1
                return None, hold, q
            if self.w.vgn_n_views() == 0:  # no grasp can be known: as if VGN found none
                self.vgn_t_start, self.vgn_result, self.vgn_ready_t = t, (None,), t
                self._enter("VGN_WAIT", "VGN: no depth view could be taken")
                return None, hold, q
            if a.vgn_mode == "scan" and a.mode == "ros":
                # robot: the control loop must keep running while VGN works (streamed commands, fresh
                # robot state: a blocking call of several seconds trips --state_timeout and the drives'
                # command timeout) -> inference in the background, the robot holds in VGN_WAIT
                self.vgn_result, self.vgn_used, self.vgn_ready_t = None, False, np.inf
                self._vgn_launch()
                if not self.vgn_pending:  # could not start: as if no grasp was found
                    self.vgn_result, self.vgn_ready_t = (None,), t
                self._enter("VGN_WAIT", "robot holds until the grasp detector has answered")
            elif a.vgn_mode == "scan":  # simulation: run inference and wait for it (charged to sim time)
                print("  running VGN inference ...")
                self.vgn_t_start = t
                self.w.vgn_target = self._vgn_target()
                best, dur = self.w.vgn_best()
                self.vgn_result, self.vgn_ready_t = (best,), t + dur
                self._enter("VGN_WAIT", f"inference took {dur:.1f} s -> robot waits that long")
            else:
                self._vgn_launch()
                self._enter("PRE_GRASP", "MPC interception of the predicted object path "
                                         "(VGN inference in the background)")
            return None, hold, q

        if self.state == "VGN_WAIT":  # scan mode: the robot stands while VGN runs
            if t >= self.vgn_ready_t:
                self._enter("PRE_GRASP", "MPC interception of the predicted object path")
            return None, hold, list(SCAN_SEED_Q if not self.view_pans else
                                    [self.view_pans[-1], *SCAN_SEED_Q[1:]])

        if self.state in ("PRE_GRASP", "DESCEND", "GRASP", "AT_GRASP", "RETREAT"):
            # a static object keeps its estimate while occluded (by obstacles or the own gripper);
            # a moving one drifts from the constant-velocity prediction, so it is given up sooner
            static = not np.any(self.tracker.v)
            lost_after = a.lost_timeout_static if static else a.lost_timeout
            if t - self.last_seen > lost_after and self.state == "PRE_GRASP":
                why = f"object lost for {t - self.last_seen:.1f} s"
                if static and self._can_plan():  # hidden from this side: try another one
                    self._replan(st, why)
                else:
                    self._end_episode(why, failed=True)
                return None, BaseReference(st.base.copy(), np.zeros(3)), ARM_HOME_Q
            obj, v_obj = self.tracker.predict(t), self.tracker.v
            if self.state != "GRASP":
                t_exit = self.w.support_exit_time(obj, v_obj)
                if t_exit < 1.5:
                    self._end_episode(f"predicted path leaves the table in {t_exit:.1f} s — "
                                      "object cannot be intercepted on the table", failed=True)
                    return None, BaseReference(st.base.copy(), np.zeros(3)), ARM_HOME_Q

            if self.state == "PRE_GRASP":
                off = [0.0, 0.0, PREGRASP_HEIGHT]
                if self.transit_z is not None:  # fly over obstacles first, then drop to pre-grasp
                    if np.linalg.norm(self.ee[:2] - obj[:2]) < TRANSIT_XY_TOL:
                        print(f"  over the object at transit height -> pre-grasp height")
                        self.transit_z = None
                    else:
                        off[2] = max(PREGRASP_HEIGHT, self.transit_z - obj[2])
                ref = self._mpc(off, self.grasp_R, a_max=0.6, v_max=EE_APPROACH_SPEED)
                goal = obj + off
                if self.a.vgn_explore != "off" and (self.vgn_pending or
                                                    (self.vgn_result is not None and not self.vgn_used)):
                    self.t_state = t - min(ts, 1.0)  # the pre-grasp timeouts count from the VGN result
                if (np.linalg.norm(self.ee - goal) < 0.01 and np.linalg.norm(self.mpc.v - v_obj) < 0.01
                        and ts > 1.0 and not self._vgn_hover()):
                    a.d_safe = a.d_safe_grasp  # fingers may come close to the table now
                    self.base_hold = st.base.copy()  # the arm does the final centimetres
                    self._enter("DESCEND", f"moving with the object above it | margin -> {a.d_safe_grasp * 100:.1f} cm")
                elif ts > PREGRASP_TIMEOUT and self._can_plan():
                    # e.g. oscillating at the edge of the workspace: another side instead
                    self._replan(st, f"pre-grasp not reached in {PREGRASP_TIMEOUT:.0f} s")
                elif ts > 30.0:
                    self._end_episode("pre-grasp not reached", failed=True)
                return ref, None, ARM_HOME_Q

            if self.state == "RETREAT":  # --goto_grasp_only: back up above the object, then done
                up = [0.0, 0.0, PREGRASP_HEIGHT]
                ref = self._mpc(up, self.grasp_R, a_max=0.3, v_max=EE_DESCEND_SPEED)
                if np.linalg.norm(self.ee - (obj + up)) < 0.015 or ts > 6.0:
                    e = np.array(self.at_grasp_err or [np.nan])
                    if self.result != "contact at the grasp pose":
                        self.result = GOTO_DONE
                    self._end_episode(f"back at the pre-grasp height | during {a.hold_time:.1f} s at the grasp pose "
                                      f"the EE was {np.nanmean(e) * 100:.1f} cm (mean), {np.nanmax(e) * 100:.1f} cm "
                                      "(max) from the grasp point")
                return ref, hold, ARM_HOME_Q
            off = [0.0, 0.0, a.grasp_dz]
            ref = self._mpc(off, self.grasp_R, a_max=0.3, v_max=EE_DESCEND_SPEED)
            goal = obj + off
            if self.state == "DESCEND":
                stuck = ts > 2.0 and np.linalg.norm(self.u_prev[3:]) < 0.02  # stopped by the table margin
                reached = np.linalg.norm(self.ee - goal) < 0.006 or stuck or ts > 12.0
                if reached and a.goto_grasp_only:  # demo: stay at the grasp pose, never close the gripper
                    self.at_grasp_err = []
                    self._enter("AT_GRASP", f"EE {np.linalg.norm(self.ee - goal) * 100:.1f} cm from the grasp point"
                                + (" (limited by table clearance)" if stuck else "")
                                + f" | staying with the object for {a.hold_time:.1f} s, gripper not used")
                elif reached:
                    self.w.gripper(True)
                    gt = self.w.ground_truth()
                    self._enter("GRASP", f"EE {np.linalg.norm(self.ee - goal) * 100:.1f} cm from grasp point"
                                + (f", {np.linalg.norm((self.ee - off) - gt) * 100:.1f} cm from true object"
                                   if gt is not None else "")
                                + (" (limited by table clearance)" if stuck else ""))
            elif self.state == "AT_GRASP":
                self.at_grasp_err.append(float(np.linalg.norm(self.ee - goal)))
                # contact: the EE cannot follow its own reference. A jump of the object estimate moves the
                # goal, not the EE's lag behind the smooth MPC reference -> no false "contact"
                lag = float(np.linalg.norm(self.ee - ref.p))
                if ts > 0.5 and lag > AT_GRASP_PUSHED:   # contact (something unseen)
                    a.d_safe = self.d_safe
                    self.result = "contact at the grasp pose"
                    self._enter("RETREAT", f"EE held {lag * 100:.1f} cm off its reference — "
                                           "contact with something the robot does not know? -> backing up")
                elif ts >= max(a.hold_time, 0.5):
                    a.d_safe = self.d_safe
                    self._enter("RETREAT", f"margin -> {self.d_safe * 100:.1f} cm, back up to the pre-grasp height")
            elif ts >= GRASP_TIME:
                a.d_safe = self.d_safe
                self.lift_goal = self.ee + [0, 0, LIFT_HEIGHT]
                self._enter("LIFT", f"margin -> {self.d_safe * 100:.1f} cm")
            # soft base hold: a base still driving while the fingers close bumps the object away
            return ref, hold, ARM_HOME_Q

        if self.state == "LIFT":
            ref = self.mpc.step(self.lift_goal, np.zeros(3), np.zeros(3), self.grasp_R, 0.4, 0.08, dt)
            if np.linalg.norm(self.ee - self.lift_goal) < 0.01 or ts > 10.0:
                gt = self.w.ground_truth()
                if gt is not None and gt[2] <= 0.90:
                    self.result = "object not lifted"
                    self._end_episode(f"object z = {gt[2]:.3f} m -> not lifted")
                elif a.hold_time > 0:
                    self._enter("HOLD", (f"object z = {gt[2]:.3f} m -> LIFTED; " if gt is not None else "")
                                + f"holding it for {a.hold_time:.1f} s")
                else:
                    self.result = "success" if gt is not None else "done"
                    self._end_episode("lift complete")
            return ref, hold, ARM_HOME_Q

        if self.state == "HOLD":  # keep the object up: a grasp that slips drops it here
            ref = self.mpc.step(self.lift_goal, np.zeros(3), np.zeros(3), self.grasp_R, 0.4, 0.08, dt)
            gt = self.w.ground_truth()
            if gt is not None:
                self.hold_slip = max(self.hold_slip, np.linalg.norm(gt - (self.ee - [0, 0, a.grasp_dz])))
            if ts >= a.hold_time:
                if gt is None:
                    self.result = "done"
                    self._end_episode("hold complete")
                else:
                    held = gt[2] > 0.90 and np.linalg.norm(gt - (self.ee - [0, 0, a.grasp_dz])) < HOLD_TOL
                    self.result = "success" if held else "dropped after the lift"
                    self._end_episode(f"after {a.hold_time:.1f} s: object z = {gt[2]:.3f} m, "
                                      f"max {self.hold_slip * 100:.1f} cm from the grasp point -> "
                                      + ("HELD" if held else "DROPPED / SLIPPED"))
            return ref, hold, ARM_HOME_Q

        if self.state == "RELEASE":  # --continuous with a gripper: hand the object over, then watch
            ref = self.mpc.step(self.lift_goal, np.zeros(3), np.zeros(3), self.grasp_R, 0.4, 0.08, dt)
            if ts >= a.handover_time and not self.released:
                self.released = True
                self.w.gripper(False)
                print("  gripper opened")
            if ts >= a.handover_time + RELEASE_OPEN_WAIT:
                self._watch(st, "gripper open")
            return ref, hold, ARM_HOME_Q

        if self.state == "WATCH":  # --continuous: wait for the object to be put down somewhere new
            if self.tracker.p is not None and t - self.last_seen > WATCH_RETRACK:
                self.tracker = ObjectTracker()  # picked up / carried away: start a new track
            if self.sim:
                self._sim_relocate(ts)
            q = list(SCAN_SEED_Q)
            look = self.tracker.predict(t) if self.tracker.p is not None else self.watch_ref
            if look is not None:
                q[0] = self._pan_to(st, look)
            if self.tracker.p is None:
                if t - self.watch_lost_t > WATCH_SEARCH_AFTER:
                    self.pan_cmd, self.pan_dir = st.q[0], -1.0
                    self._enter("COARSE_DETECT", "object not in view -> sweeping shoulder_pan")
                return None, hold, q
            self.watch_lost_t = t
            why = self._placed_anew(t)
            if why and not self.vgn_pending:  # (a late inference of the last episode is dropped first)
                self._new_episode(st, why)
            return None, hold, q

        return None, BaseReference(st.base.copy(), np.zeros(3)), list(st.q)

    # ── continuous operation (--continuous) ─────────────────────────────────
    def _reset_episode(self):
        """Per-episode state: everything a grasp attempt decides or learns about the object."""
        self.grasp_R, self.result = R_TOPDOWN.copy(), "aborted"
        self.transit_z = None  # z of the obstacle-clearing transit in PRE_GRASP (None: direct)
        self.placed, self.bad_sides, self.base_path = False, set(), []
        self.arrival_side = None  # table side the robot first arrived on (quadrant frame)
        self.glimpse = None  # (position, t) of a detection made while the camera was swinging
        self.view_pans, self.vgn_t0, self.vgn_t_first, self.vgn_t_start = [], None, None, None
        self.vgn_result, self.vgn_ready_t, self.vgn_used = None, np.inf, False
        self.vgn_views, self.vgn_last_view = {}, None
        self.vgn_status, self.vgn_score = "not run", None
        self.vgn_hover_t0, self.vgn_hover_gave_up = None, False
        self.vgn_center, self.view_wait_t0 = None, None
        self.explored, self.explore_why = [], None  # table sides where VGN found no grasp
        self.hold_slip, self.at_grasp_err, self.released = 0.0, None, False
        self.mpc = None

    def _object_found(self, st, why):
        """Object tracked after a search: plan the grasp — or, while watching, first make sure it has
        been put down somewhere new (WATCH)."""
        if self.watching:
            self.watch_lost_t = self.t
            self._enter("WATCH", f"{why} -> is it put down somewhere new?")
        else:
            self._begin_grasp_planning(st, why)

    def _end_episode(self, msg, failed=False):
        """An episode is over (self.result set; failed: it could not be completed). Without --continuous
        (or after --max_episodes) the run ends: DONE / ABORT. Otherwise the object is handed over
        (gripper) and the robot watches for its next placement."""
        a = self.a
        if not self.watching:
            self.episodes.append((self.episode, self.t, self.result if not failed or self.result != "aborted"
                                  else f"failed: {msg.split(' — ')[0]}"))
        if not a.continuous or (a.max_episodes and self.episode >= a.max_episodes):
            self._enter("ABORT" if failed else "DONE", msg)
            return
        a.d_safe = self.d_safe
        if not self.watching and self.tracker.p is not None:
            # the next placement must be elsewhere: where the object was seen with the camera still
            still = [p for (_, p, s) in self.tracker.hist if s]
            self.watch_ref = np.median(still, axis=0) if still else self.tracker.predict(self.t)
        if not failed and not a.goto_grasp_only and self.mpc is not None:
            self._enter("RELEASE", f"{msg} | take the object out of the gripper: it opens in "
                                   f"{a.handover_time:.0f} s")
        else:
            self._watch(self.st, msg)

    def _watch(self, st, msg):
        self.watching, self.watch_lost_t, self.sim_relocated = True, self.t, False
        self.base_hold, self.base_path, self.searched = st.base.copy(), [], []
        self.mpc, self.transit_z = None, None
        self._enter("WATCH", f"{msg} -> episode {self.episode} over; watching for the object to be put down "
                    + (f"at least {self.a.watch_moved * 100:.0f} cm from {np.round(self.watch_ref[:2], 2)}"
                       if self.watch_ref is not None else "on the table")
                    + " (Ctrl+C ends)")

    def _placed_anew(self, t):
        """Reason string if the tracked object rests at a new place (still for WATCH_SETTLE, at least
        --watch_moved from where the last episode left it), else None."""
        if not self._tracked() or np.any(self.tracker.v):
            return None
        # only detections taken with the camera still: a moving camera biases them (TF lag)
        recent = [p for (tp, p, still) in self.tracker.hist if t - tp <= WATCH_SETTLE and still]
        if t - self.tracker.hist[0][0] < WATCH_SETTLE or len(recent) < 5:
            return None
        p = np.median(recent, axis=0)
        if max(np.linalg.norm(q[:2] - p[:2]) for q in recent) > WATCH_STILL_TOL:
            return None
        if self.watch_ref is None:
            return f"object at {np.round(p, 3)}"
        moved = float(np.linalg.norm(p[:2] - self.watch_ref[:2]))
        if moved < self.a.watch_moved:
            return None
        return f"object put down at {np.round(p, 3)}, {moved * 100:.0f} cm from where it was"

    def _new_episode(self, st, why):
        self.episode += 1
        self.watching, self.watch_ref = False, None
        self._reset_episode()
        if self.hmap is not None:
            self._setup_map()  # a fresh table map: hands and arms seen while the object was moved
        self.side = self._side_of(st.base, *self.w.support_rect()) if self.w.support_rect() is not None else None
        print(f"\n{'═' * 74}\n  Episode {self.episode}\n{'═' * 74}")
        self._begin_grasp_planning(st, why)

    def _sim_relocate(self, ts):
        """Simulation: 'someone' puts the object down somewhere else on the table, SIM_RELOCATE_AFTER s
        into WATCH — random, 0.1 m inside the table edges, at least 0.2 m from the last place."""
        if self.sim_relocated or ts < SIM_RELOCATE_AFTER or self.w.support_rect() is None:
            return
        self.sim_relocated = True
        lo, hi = self.w.support_rect()
        rng = np.random.default_rng(self.a.sim_seed + self.episode)
        old = self.watch_ref[:2] if self.watch_ref is not None else 0.5 * (lo + hi)
        for _ in range(100):
            xy = rng.uniform(lo + 0.10, hi - 0.10)
            if np.linalg.norm(xy - old) > 0.20:
                break
        self.w.relocate(xy)
        print(f"  [sim] object put down at ({xy[0]:.2f}, {xy[1]:.2f})")

    def _begin_grasp_planning(self, st, why):
        if self.w.vgn_checking():
            # VGN requested but its environment check (vgn_env / model servers) is still running:
            # wait (the caller retries every tick) rather than silently grasping without it
            if not self.vgn_check_note:
                self.vgn_check_note = True
                print("  waiting for the grasp detector (environment check still running) ...")
            return
        obj = self.tracker.predict(self.t)
        gt = self.w.ground_truth()
        err = f" | estimate error {np.linalg.norm(obj - gt) * 100:.1f} cm" if gt is not None else ""
        print(f"  object at {np.round(obj, 3)}, velocity {np.round(self.tracker.v, 3)} m/s "
              f"({len(self.tracker.hist)} detections in window, {self.tracker.rejected} rejected){err}")
        if not self.placed and self._can_plan():
            self.placed = True
            plan = self._plan_approach(st, obj)
            if plan is None:
                if self.explored:
                    self.result = NO_GRASP
                    self._end_episode(f"{NO_GRASP.upper()} — VGN found no grasp from the "
                                      f"{', '.join(self.explored)} side(s); no other side is reachable", failed=True)
                else:
                    self._end_episode(f"object not reachable from any side of the table with the base "
                                      f"{self.a.d_safe_base * 100:.0f} cm away", failed=True)
                return
            if plan["path"]:
                self.side, self.base_path = plan["side"], plan["path"]
                self._enter("REPOSITION", f"{why}; not reachable from here -> around the table to the "
                                          f"{plan['side']} side ({plan['cost']:.1f} m)")
                return
            self.side = plan["side"]
        if self.w.vgn_available() and (self.vgn_pending or self.vgn_result is not None):
            self._enter("PRE_GRASP", f"{why}; VGN already {'running' if self.vgn_pending else 'done'}")
        elif self.w.vgn_available():
            have = self.w.vgn_n_views() if self.vgn_t0 is not None else 0
            if self.a.vgn_mode == "drive" and have >= self._drive_views()[0]:
                self._vgn_launch()
                self._enter("PRE_GRASP", f"{why}; {have} VGN views from the drive, inference in the background")
                return
            need = (self.a.vgn_views if self.explored  # a full scan from every new side
                    else max(self.a.vgn_views - have, 1) if have else self.a.vgn_views)
            c = self._pan_to(st, obj)
            self.view_pans = list(np.linspace(c - 0.20, c + 0.20, need)) if need > 1 else [c]
            self.view_i = 0
            if self.vgn_t0 is None:
                self._vgn_init(obj)
            self._enter("VGN_SCAN", f"{why}; {need} TSDF views around pan={c:.2f} rad"
                        + (f" (+{have} fused from the {', '.join(self.explored)} side(s))" if self.explored and have
                           else f" (+{have} from the drive)" if have else ""))
        else:
            self._enter("PRE_GRASP", f"{why}; MPC interception of the predicted path (VGN off)")

    # ── VGN: views, background inference, result ───────────────────────────
    def _vgn_init(self, obj):
        self.vgn_center = np.asarray(obj, float).copy()
        self.w.vgn_init(obj)
        self.vgn_t0 = self.t  # TSDF frame = object at this time

    def _vgn_fused_sides(self):
        """Views of several table sides in one TSDF (--vgn_explore fuse, after the first side)."""
        return bool(self.explored) and self.a.vgn_explore == "fuse" and self.vgn_center is not None

    def _vgn_shift(self):
        """Object motion since the TSDF was initialised (drive mode compensates it). Across table
        sides: the tracked object's offset from the TSDF centre — object motion, and on the real
        robot the odometry drift of the drive around the table (camera and detections share it)."""
        if self.a.vgn_mode == "drive":
            return self.tracker.v * (self.t - self.vgn_t0)
        if self._vgn_fused_sides():
            return self.tracker.predict(self.t) - self.vgn_center
        return None

    def _vgn_target(self, t_ref=None):
        """The target in the TSDF frame (VGN grasps must lie near it)."""
        if self._vgn_fused_sides():
            return self.vgn_center
        return self.tracker.predict(self.t if t_ref is None else t_ref)

    def _vgn_capture(self, st, where):
        """Take a VGN view; None while a real camera's frame is still on its way."""
        n = self.w.vgn_view(self.tracker.predict(self.t), self._vgn_shift())
        if n is not None and self.hmap is not None and getattr(self.w, "last_view", None) is not None:
            self._integrate_map(*self.w.last_view)
        if n is None:
            return None
        if self.side:
            where = f"{where}@{self.side}"
        self.vgn_views[where] = self.vgn_views.get(where, 0) + 1
        if self.vgn_t_first is None:
            self.vgn_t_first = self.t
        self.vgn_last_view = (self.t, st.base[:2].copy())
        return n

    def _drive_views(self):
        """drive mode: (views that let the robot skip the stopped scan, views that start inference
        while still driving) — --vgn_views if given (1: the first good view starts it), else 3 / 6."""
        n_max = self.a.vgn_views if self.a.vgn_views_given else VGN_DRIVE_MAX_VIEWS
        return min(VGN_DRIVE_MIN_VIEWS, n_max), n_max

    def _drive_vgn_views(self, st):
        """drive mode: arm posture while driving, taking VGN views once the object is near."""
        if not self._tracked():
            return DRIVE_POSE
        obj = self.tracker.predict(self.t)
        if np.linalg.norm(obj[:2] - st.base[:2]) > self.a.vgn_drive_range:
            return DRIVE_POSE
        q = list(SCAN_SEED_Q)
        q[0] = self._pan_to(st, obj)
        n = self.w.vgn_n_views() if self.vgn_t0 is not None else 0
        ready = (self._camera_ready() if self.vis_deg is not None  # driving: the arm keeps adjusting
                 else np.max(np.abs(np.asarray(q) - st.q)) < VGN_ARM_READY)
        n_min, n_max = self._drive_views()
        if ready and n < n_max:
            last = self.vgn_last_view
            if (last is None or np.linalg.norm(st.base[:2] - last[1]) >= VGN_DRIVE_SPACING
                    or self.t - last[0] >= VGN_DRIVE_PERIOD):
                if self.vgn_t0 is None:
                    self._vgn_init(obj)
                n = self._vgn_capture(st, "drive")
                if n is None:
                    return q
                print(f"  VGN view {n} while driving (object {np.linalg.norm(obj[:2] - st.base[:2]):.2f} m "
                      f"from the base)")
                if n >= n_max and not self.vgn_pending and self.vgn_result is None:
                    self._vgn_launch()  # while still driving
        return q

    def _vgn_launch(self):
        self.vgn_t_start = self.t
        # the target in the TSDF frame (object at vgn_t0 when motion-compensated, else now)
        t_ref = self.vgn_t0 if self.a.vgn_mode == "drive" and self.vgn_t0 is not None else self.t
        self.w.vgn_target = self._vgn_target(t_ref)
        self.vgn_pending = self.w.vgn_start()
        print(f"  VGN inference started in the background ({self.w.vgn_n_views()} views)"
              if self.vgn_pending else "  VGN inference could not start")

    def _vgn_update(self):
        """Every tick: collect a finished background inference, use the result once the
        measured inference time has also passed in simulation time."""
        if self.vgn_pending:
            r = self.w.vgn_poll()
            if r is not None:
                best, dur = r
                self.vgn_pending = False
                if self.watching:  # a late result of the episode that just ended
                    return
                self.vgn_result, self.vgn_ready_t = (best,), self.vgn_t_start + dur
        if self.vgn_result is not None and not self.vgn_used and self.t >= self.vgn_ready_t:
            self.vgn_used = True
            self._vgn_apply(self.vgn_result[0])

    def _vgn_hover(self):
        """At the pre-grasp: keep hovering above the object (moving with it) while a background
        VGN inference is still running, up to --vgn_max_wait. True while hovering."""
        if not (self.vgn_pending or (self.vgn_result is not None and not self.vgn_used)):
            return False
        if np.any(self.tracker.v) and self.t - self.last_seen > 0.5 * self.a.lost_timeout:
            # a moving object hidden under the own gripper: waiting longer would lose it
            if not self.vgn_hover_gave_up:
                self.vgn_hover_gave_up = True
                print("  moving object out of sight while waiting for VGN -> grasping without it")
            return False
        if self.vgn_hover_t0 is None:
            self.vgn_hover_t0 = self.t
            print(f"  at the pre-grasp before VGN is ready -> hovering (max {self.a.vgn_max_wait:.1f} s)")
        if self.t - self.vgn_hover_t0 < self.a.vgn_max_wait or self.a.vgn_explore != "off":
            # exploring: never grasp without VGN's verdict (no grasp -> another side)
            self.vgn_wait += self.dt  # robot held for VGN
            return True
        if not self.vgn_hover_gave_up:
            self.vgn_hover_gave_up = True
            print("  VGN still not ready -> grasping without it")
        return False

    def _vgn_apply(self, best):
        dur = self.vgn_ready_t - self.vgn_t_start
        if best is None:
            self.vgn_status = "no grasp"
            late = self.state in ("DESCEND", "GRASP", "LIFT", "DONE", "ABORT")
            # a static object: look from another side; a moving one keeps the tracked top-down grasp
            # (it moves on while the robot drives around the table)
            if self.a.vgn_explore != "off" and not late and not np.any(self.tracker.v):
                side = self.side or "current"
                self.explore_why = f"VGN: no grasp candidate on the target from the {side} side"
                print(f"  VGN (after {dur:.1f} s, {self.w.vgn_n_views()} views): no grasp on the target "
                      f"-> explore another side of the table")
                return
            print(f"  VGN (after {dur:.1f} s): no grasp above threshold -> tracked position, top-down")
            return
        pos, rot, width, score = best
        shift = self._vgn_shift()
        if shift is not None:  # TSDF frame -> the object's current position
            pos = pos + shift
        gt = self.w.ground_truth()
        late = self.state in ("DESCEND", "GRASP", "LIFT", "DONE", "ABORT")
        self.vgn_status = f"late ({self.state})" if late else "used"
        self.vgn_score = score
        print(f"  VGN grasp (after {dur:.1f} s, t={self.t:.1f}) {np.round(pos, 3)}  score={score:.2f}  "
              f"width={width * 100:.1f} cm"
              + (f"  error {np.linalg.norm(pos - gt) * 100:.1f} cm" if gt is not None else "")
              + (f"  -> arrived during {self.state}, not used" if late else ""))
        if not late and self.a.vgn_orientation == "yaw":
            # keep top-down approach, rotate about z to VGN's finger axis (assumed VGN y-axis)
            ax = rot[:, 1]
            self.grasp_R = grasp_for_heading(rot_z(np.arctan2(ax[1], ax[0]) - np.pi / 2) @ R_TOPDOWN,
                                             self.base_hold[2])

    def _explore(self, st, why):
        """VGN found no grasp from this side: drive around the table to the next side (the views
        of every side stay in the TSDF with --vgn_explore fuse), scan, try again. No side left:
        'no grasp candidate found' — the robot stops, nothing is grasped."""
        side = self.side
        if side is None and self.w.support_rect() is not None:
            side = self._side_of(st.base, *self.w.support_rect())
        self.explored.append(side or "current")
        if side:
            self.bad_sides.add(side)
        self.vgn_result, self.vgn_used, self.vgn_pending, self.vgn_ready_t = None, False, False, np.inf
        self.vgn_hover_t0, self.vgn_hover_gave_up = None, False
        if self.a.vgn_explore == "fresh":
            self.vgn_t0, self.vgn_center = None, None  # next side: a new TSDF
        self.grasp_R = R_TOPDOWN.copy()
        self.placed, self.mpc, self.transit_z = False, None, None
        self.a.d_safe = self.d_safe
        left = [s for s in SIDES if s not in self.bad_sides]
        if not self._can_plan() or not left or len(self.explored) >= self.a.vgn_explore_max_sides:
            reason = ("no table model (--table): cannot plan other sides" if not self._can_plan()
                      else f"explored {len(self.explored)} side(s)")
            self.result = NO_GRASP
            self._end_episode(f"{NO_GRASP.upper()} — {why}; {reason}", failed=True)
            return
        print(f"  {why} -> exploring another side (not tried yet: {', '.join(left)})")
        self._begin_grasp_planning(st, why)

    # ── approach-side planning around the table ─────────────────────────────
    def _can_plan(self):
        return (not self.a.no_reposition and self.ctrl.coll is not None
                and self.w.support_rect() is not None)

    def _reach_residual(self, base, points):
        """Largest top-down IK residual [m] over `points` with the base fixed at `base`; inf if
        the gripper reaches them only in configurations that collide with obstacles/the table."""
        kin, worst, q = self.kin, 0.0, np.array(ARM_HOME_Q, float)
        names = ("pre-grasp", "grasp", "lift", "transit")
        for i, p in enumerate(points):
            def f(x):
                kin.set_state(RobotState(base, x))
                pe, Re = kin.ee_pose()
                return np.concatenate([pe - p, 0.3 * orientation_error(self.grasp_R, Re)])
            sols = []
            for seed in (q, np.array(ARM_HOME_Q, float), np.array(SCAN_SEED_Q, float)):
                s0 = seed.copy()
                s0[0] = self._pan_to(RobotState(base, s0), p)
                r = least_squares(f, np.clip(s0, kin.q_min, kin.q_max), bounds=(kin.q_min, kin.q_max))
                sols.append((np.linalg.norm(f(r.x)[:3]), np.linalg.norm(r.x - q), r.x))
            converged = sorted([sl for sl in sols if sl[0] < REACH_TOL], key=lambda sl: sl[1])
            if not converged:  # out of reach: report how far
                worst = max(worst, min(sl[0] for sl in sols))
                continue
            block = None
            for res, _, x in converged:  # closest-to-previous collision-free solution
                block = self._arm_collision(base, x, grasp_point=(i == 1))
                if block is None:
                    q, worst = x, max(worst, res)
                    break
            if block is not None:
                self._reach_block = f"{block} at {names[min(i, 3)]}"
                return np.inf
        return worst

    def _arm_collision(self, base, q, grasp_point=False):
        """None if the arm configuration keeps clear of obstacles (d_safe_obstacle) and the table
        (at the grasp point the table is left to the QP's relaxed descent margin); else the pair."""
        coll = self.ctrl.coll
        self.kin.set_state(RobotState(base, q))
        for pr in coll._pairs(base[2]):
            dist, label = pr[0], pr[6]
            if pr[2] == coll.base_body:
                continue  # base clearance is set by the stand-off
            obstacle = label.rsplit("/", 1)[-1].startswith("obst_")
            if not obstacle and grasp_point:
                continue
            if dist < (self.a.d_safe_obstacle if obstacle else 0.0):
                return f"{label} {dist * 100:.1f} cm"
        return None

    def _approach_points(self, base, obj):
        """Gripper points the grasp from this base pose needs (incl. an obstacle transit)."""
        a = self.a
        pts = [obj + [0, 0, PREGRASP_HEIGHT], obj + [0, 0, a.grasp_dz], obj + [0, 0, LIFT_HEIGHT]]
        sh = np.array([base[0], base[1]]) + rot_z(base[2])[:2, :2] @ [0.205, 0.0]  # shoulder xy
        top = -np.inf if a.no_transit else self.w.transit_height(np.array([*sh, 0.0]), obj)
        if top + TRANSIT_CLEARANCE > obj[2] + PREGRASP_HEIGHT:
            pts.append(np.array([obj[0], obj[1], top + TRANSIT_CLEARANCE]))
        return pts

    def _side_of(self, base, lo, hi):
        x, y = base[:2]
        gaps = {"front": lo[0] - x, "back": x - hi[0], "right": lo[1] - y, "left": y - hi[1]}
        return max(gaps, key=gaps.get)

    def _ring(self, lo, hi):
        """Corners (ccw) of a rectangle around the table on which the base clears it at any yaw."""
        fp = self.ctrl.coll
        r = np.hypot(*np.maximum(np.abs(fp.base_xy_min), np.abs(fp.base_xy_max)))
        o = self.a.d_safe_base + BASE_LAG_BUFFER + r + RING_MARGIN
        return np.array([[lo[0] - o, lo[1] - o], [hi[0] + o, lo[1] - o],
                         [hi[0] + o, hi[1] + o], [lo[0] - o, hi[1] + o]]), o

    def _ring_path(self, start, goal, lo, hi):
        """Smooth orbit start -> goal around the table (TableOrbit), the shorter way round:
        (OrbitPath, length [m])."""
        _, o = self._ring(lo, hi)
        path = TableOrbit(lo, hi, o).path(np.asarray(start, float), np.asarray(goal, float), self._orbit_v())
        return path, path.length

    def _orbit_v(self):
        return min(self.a.orbit_speed, self.a.max_base_v)

    def _plan_approach(self, st, obj):
        """Cheapest base pose, among staying and the four table sides, from which the whole grasp
        is reachable with the base >= d_safe_base from the table. None if there is none."""
        lo, hi = self.w.support_rect()
        fp = self.ctrl.coll
        standoff = self.a.d_safe_base + BASE_LAG_BUFFER + STANDOFF_MARGIN + fp.base_xy_max[0]
        here = self._side_of(st.base, lo, hi)
        if self.arrival_side is None:
            self.arrival_side = here   # quadrants stay fixed to the side the robot came from
        quad, preferred = table_quadrant(obj[:2], lo, hi, self.arrival_side)
        moving = bool(np.any(self.tracker.v))
        if moving:  # it rolls on while the robot drives around: intercept from here if reachable
            preferred = here
        print(f"  object in the {quad.upper()} part of the table (seen from the {self.arrival_side} side) -> "
              + ("moving: intercept from here if reachable" if moving
                 else f"grasp from here ({here})" if preferred == here else f"grasp from the {preferred} side"))
        cands = []
        for name, (n, yaw) in SIDES.items():
            if name in self.bad_sides:
                continue
            # where the object will be when the grasp from this side can start
            eta = GRASP_ETA
            if name != here:
                eta += self._ring_path(st.base, np.array([*obj[:2], yaw]), lo, hi)[1] / self._orbit_v()
            o = self.tracker.predict(self.t + eta)
            xy = np.clip(o[:2], lo, hi)
            xy = np.where(n != 0, np.where(n > 0, hi, lo) + n * standoff, xy)
            if name == here:  # the QP moves the base to the stand-off itself; keep the heading
                goal, path, length = np.array([xy[0], xy[1], st.base[2]]), [], 0.0
            else:
                goal = np.array([xy[0], xy[1], yaw])
                path, length = self._ring_path(st.base, goal, lo, hi)
            cands.append((length, name, goal, path, o))
        los, cams = {}, []
        if self.hmap is not None and self.hmap.occupied.any():
            for _, name, pose, _, o in cands:
                los[name], c = self._los_from(pose, o)
                cams.append((f"{name} {100 * los[name]:.0f}%", c))
            print(f"  line of sight to the object (table map: {self.hmap.summary()}): "
                  + ", ".join(f"{n} {100 * v:.0f}%" for n, v in los.items()))
            if self.w.save_dir is not None:
                import cv2
                self.n_maps = getattr(self, "n_maps", 0) + 1
                cv2.imwrite(str(self.w.save_dir / f"table_map_{self.n_maps:02d}.png"),
                            self.hmap.image(target=obj, cams=cams))
        report, chosen = [], None
        R0 = self.grasp_R.copy()
        # the side of the object's table region first (near half: stay); the others (here first, then
        # by path length) only if it is not reachable; sides where VGN found no grasp are not candidates
        # sides the object is hidden from (line of sight < LOS_OK) last
        for cost, name, pose, path, o in sorted(cands, key=lambda c: (los.get(c[1], 1.0) < LOS_OK,
                                                                      c[1] != preferred, c[0])):
            if np.any(o[:2] < lo + 0.03) or np.any(o[:2] > hi - 0.03):
                report.append(f"{name}: object off the table by then")
                continue
            for yaw in GRASP_YAWS:  # rotate the grasp so the open fingers clear the obstacles
                self.grasp_R = grasp_for_heading(rot_z(yaw) @ R0, pose[2])
                res = self._reach_residual(pose, self._approach_points(pose, o))
                report.append(f"{name}{'(here)' if not path else ''}/{np.degrees(yaw):.0f}°: "
                              + (f"collision ({self._reach_block})" if not np.isfinite(res)
                                 else f"{res * 100:.1f} cm"))
                if res < REACH_TOL:
                    chosen = dict(side=name, pose=pose, path=path, cost=cost, yaw=yaw)
                    break
            if chosen:
                break
        self.grasp_R = grasp_for_heading(rot_z(chosen["yaw"]) @ R0, chosen["pose"][2]) if chosen else R0
        self.kin.set_state(st)
        print(f"  approach planning (IK residual per side): {', '.join(report)}"
              f" -> {chosen['side'] if chosen else 'none reachable'}")
        return chosen

    def _replan(self, st, why):
        """The grasp got blocked from this side: try another one."""
        self.bad_sides.add(self.side)
        self.placed, self.mpc, self.transit_z = False, None, None
        self.a.d_safe = self.d_safe
        print(f"  {why} -> {self.side} side marked unusable, replanning")
        self._begin_grasp_planning(st, why)

    # ── main loop ─────────────────────────────────────────────────────────────
    def run(self):
        a = self.a
        st = self.b.wait_for_state(a.wait_timeout)
        if st is None:
            return 1
        self.ctrl = WholeBodyController(self.kin, a, q_nom=DRIVE_POSE, yaw_nom=st.base[2])
        self.u_prev = np.zeros(NU)
        self.state = "DRIVE"
        self._setup_camera(st)
        self._setup_map()
        self.state, self.t, self.t0 = "DRIVE", 0.0, self.b.now()
        self.t_state, self.n_tick = 0.0, 0
        self.base_hold, self.drive_start = st.base.copy(), st.base.copy()
        self.tracker, self.last_seen, self.n_seen, self.n_bad_z = ObjectTracker(), -np.inf, {}, 0
        self.min_obst_clr = np.inf
        self.min_base_clr = np.inf  # base footprint <-> environment, must stay >= d_safe_base
        self.side, self.vgn_pending, self.vgn_wait = None, False, 0.0
        self.vgn_check_note = False  # "waiting for the environment check" printed
        self._reset_episode()
        # --continuous: episode number, [(episode, t, result)], watching for the next placement
        self.episode, self.episodes, self.watching, self.watch_ref, self.st = 1, [], False, None, st
        self.kin.set_state(st)
        p0, R0 = self.kin.ee_pose()
        self.ee = p0
        rows, next_print = [], 0.0
        safety = make_safety(self.b, a)
        stop_t0 = None  # start of the current safety stop
        print(f"\n{'═' * 74}\n  Steve active perception + grasping — unified joint-velocity WBC  [{a.mode}]"
              f"\n{'═' * 74}\n  VGN: {('on' + ('' if a.grasp_backend == 'vgn' else f' ({a.grasp_backend})')) if self.w.vgn_available() else 'starting (environment check)' if self.w.vgn_checking() else 'off'}   object: "
              f"{getattr(a, 'object_type', '-')}   collision env: {a.collision_env}")

        try:
            while self.b.ok() and self.state not in ("DONE", "ABORT"):
                st = self.b.read_state()
                if st is None:
                    self._enter("ABORT", "no valid robot state")
                    break
                if safety is not None:
                    s = safety.check(self.b.now())
                    if s == "abort":
                        self._enter("ABORT", "safety stop timed out")
                        break
                    if s == "stop":  # zero command sent
                        if stop_t0 is None:
                            stop_t0 = self.b.now()
                        self.u_prev = np.zeros(NU)
                        self.b.tick()
                        continue
                    if stop_t0 is not None:
                        # the object moved on during the stop (tracker keeps world time); only
                        # the phase timers skip the stop
                        stopped_for = self.b.now() - stop_t0
                        self.t_state += stopped_for
                        if self.vgn_hover_t0 is not None:
                            self.vgn_hover_t0 += stopped_for
                        stop_t0 = None
                self.t = self.b.now() - self.t0
                self.kin.set_state(st)
                self.ee, R_ee = self.kin.ee_pose()
                self.R_ee = R_ee
                if self.mpc is None and self.state == "PRE_GRASP":
                    self.mpc = InterceptionMPC(self.ee, R_ee)
                    obj = self.tracker.predict(self.t)
                    top = -np.inf if self.a.no_transit else self.w.transit_height(self.ee, obj)
                    if top + TRANSIT_CLEARANCE > obj[2] + PREGRASP_HEIGHT:
                        self.transit_z = top + TRANSIT_CLEARANCE
                        print(f"  obstacle (top z={top:.2f} m) between gripper and object -> "
                              f"transit at z={self.transit_z:.2f} m")

                self.st = st
                ee_ref, base_ref, q_nom = self.update(st)
                if ee_ref is None and self.state in ("PRE_GRASP", "DESCEND", "GRASP", "AT_GRASP", "RETREAT", "LIFT",
                                                     "RELEASE"):
                    continue  # state just changed; next tick produces the reference
                self.ctrl.q_nom = np.asarray(q_nom, float)
                self.ctrl.yaw_nom = self.base_hold[2]
                tic = time.perf_counter()
                if self.hmap is not None:
                    self._update_map()
                look = self._look_target()
                use = self.vis_on and look is not None
                u, info = self.ctrl.step(st, ee_ref, self.u_prev, self.dt, base_ref,
                                         look=look if use else None,
                                         los_obstacles=self.los_pts if use and self.hmap is not None else None,
                                         look_radius=self.w.target.radius)
                solve_ms = (time.perf_counter() - tic) * 1e3
                self.vis_deg = info.get("vis_deg") if look is not None and self.vis_on else None
                ang = self.vis_deg
                if look is not None and not self.vis_on:  # --no_visibility: measured only (kin is at st)
                    c_cam, z_cam, _, _ = self.kin.camera(st.base[2])
                    los = look - c_cam
                    ang = float(np.degrees(np.arccos(np.clip(z_cam @ los / np.linalg.norm(los), -1.0, 1.0))))
                if np.max(np.abs(u[3:])) < 0.03:
                    self.still_since = self.t if self.still_since is None else self.still_since
                else:
                    self.still_since = None
                if ang is not None:
                    self.vis_ticks += 1
                    self.vis_good += ang < VIS_FOV_DEG
                if not np.all(np.isfinite(u)):
                    self._enter("ABORT", "non-finite command")
                    break
                if ee_ref is not None and np.linalg.norm(info["e_p"]) > a.max_track_err:
                    why = f"EE tracking error {np.linalg.norm(info['e_p']) * 100:.1f} cm"
                    if self.state in ("PRE_GRASP", "DESCEND") and self._can_plan():
                        self._replan(st, why + " (gripper blocked)")
                        u = np.zeros(NU)  # stop; the next tick continues in the new state
                    else:
                        self._enter("ABORT", why)
                        break

                coll = self.ctrl.coll
                if coll is not None:
                    self.min_base_clr = min(self.min_base_clr, getattr(coll, "base_clearance", np.inf))
                    for obs, dist in getattr(coll, "d_min_by_obstacle", {}).items():
                        if obs.startswith("obst_"):
                            self.min_obst_clr = min(self.min_obst_clr, dist)
                self.b.send(u)
                if self.sim and ee_ref is not None:
                    self.b.draw_live(ee_ref.p, self.ee)
                if ee_ref is not None:
                    rows.append([self.t, *ee_ref.p, *self.ee, np.linalg.norm(info["e_p"]),
                                 np.linalg.norm(info["e_o"]), *u, *st.base, *st.q, solve_ms, info["d_min"]])
                if self.t >= next_print:
                    next_print += 1.0
                    gt = self.w.ground_truth()
                    est = (f" obj_est_err={np.linalg.norm(self.tracker.predict(self.t) - gt) * 100:.1f}cm"
                           if gt is not None and self.tracker.p is not None else "")
                    clr = f" clr={info['d_min'] * 100:.1f}cm" if np.isfinite(info["d_min"]) else ""
                    print(f"  t={self.t:5.1f}s [{self.state:13s}] base=({st.base[0]:+.2f},{st.base[1]:+.2f},"
                          f"{np.degrees(st.base[2]):+4.0f}°) u_b=({u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f}) "
                          f"|dq|={np.abs(u[3:]).max():.2f}{clr}{est}"
                          + (f" vis={self.vis_deg:.0f}°" if self.vis_deg is not None else "")
                          + (f" los={info['n_los']}" if info.get("n_los") else ""))
                self.u_prev = u
                if self.state in ("VGN_SCAN", "VGN_WAIT"):
                    self.vgn_wait += self.dt  # robot held at the table for VGN
                self.n_tick += 1
                self.b.tick()
        except KeyboardInterrupt:
            print("\n[STOP] Ctrl+C")
        finally:
            self.a.d_safe = self.d_safe
            self.b.stop()

        hits = self.w.obstacle_hits
        if hits and self.result == "success":
            self.result = "lifted, but collided with obstacles"
        if self.min_base_clr < a.d_safe_base - 1e-3 and self.result in ("success", "done", GOTO_DONE):
            self.result = "base safety distance violated"
        print(f"\n{'─' * 74}\n  Result: {self.result}  (final state {self.state}, t={self.t:.1f} s)")
        if a.continuous:
            print(f"  Episodes ({len(self.episodes)} finished):")
            for n, te, r in self.episodes:
                print(f"    {n:2d}  t={te:6.1f} s  {r}")
        if self.explored:
            print(f"  Explored: VGN found no grasp from the {', '.join(self.explored)} side(s)"
                  + (f", grasp pose reached from the {self.side} side" if self.result in ("success", "done", GOTO_DONE)
                     else ""))
        print(f"  Detections used: {self.n_seen}   rejected by the gate: {self.tracker.rejected}")
        if getattr(self, "hmap", None) is not None:
            print(f"  Table map: {self.hmap.summary()}" + (f"; searched from {', '.join(self.searched)}"
                                                         if self.searched else ""))
        if getattr(self, "vis_ticks", 0):
            print(f"  Camera on the object (axis within {VIS_FOV_DEG:.0f}°): {100.0 * self.vis_good / self.vis_ticks:.0f}% "
                  f"of {self.vis_ticks * self.dt:.1f} s while the EE was free — visibility task "
                  f"{'on' if self.vis_on else 'off'}")
        if self.w.vgn_available() or self.vgn_status != "not run":
            views = ", ".join(f"{n} {k}" for k, n in self.vgn_views.items()) or "none"
            timing = ""
            if self.vgn_t_start is not None and np.isfinite(self.vgn_ready_t):
                timing = (f" | first view t={self.vgn_t_first:.1f}s, inference t={self.vgn_t_start:.1f}"
                          f"->{self.vgn_ready_t:.1f}s")
            print(f"  VGN [{a.vgn_mode}]: views {views}{timing} | robot waited {self.vgn_wait:.1f} s at the "
                  f"table for VGN | result {self.vgn_status}"
                  + (f" (score {self.vgn_score:.2f})" if self.vgn_score is not None else ""))
        if np.isfinite(self.min_base_clr):
            ok = self.min_base_clr >= a.d_safe_base - 1e-3
            print(f"  Min base clearance: {self.min_base_clr * 100:.1f} cm   (d_safe_base "
                  f"{a.d_safe_base * 100:.0f} cm{'' if ok else ' — VIOLATED'})")
        if isinstance(self.w, SimWorld) and self.w.obstacle_geoms:
            print(f"  Obstacle contacts: {hits if hits else 'none'}")
            if np.isfinite(self.min_obst_clr):
                print(f"  Min clearance to obstacles (collision model): {self.min_obst_clr * 100:.1f} cm")
        summarize(rows, a)
        if safety is not None:
            print(safety.summary())
        if a.continuous:
            return 0 if self.episodes and all(r in ("success", "done", GOTO_DONE) for _, _, r in self.episodes) else 1
        return 0 if self.result in ("success", "done", GOTO_DONE) else 1


class _Tee:
    """Console output also to a file (the run's states, decisions and result stay available)."""

    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, data):
        self.stream.write(data)
        self.f.write(data)
        self.f.flush()

    def flush(self):
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def save_console(mode):
    """--mode ros: tee stdout + stderr to ~/ros_logs/active_perception/<time>.log — for `ros2 run`;
    under `ros2 launch` the workspace's logging hook already saves the output
    (~/ros_logs/<launch file>_<time>/, see steve_hardware_bringup/env-hooks)."""
    if mode != "ros" or os.environ.get("STEVE_LAUNCH_LOG_DIR"):
        return
    d = Path.home() / "ros_logs" / "active_perception"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{time.strftime('%Y%m%d_%H%M%S')}.log"
    f = open(path, "w", buffering=1)
    f.write(f"# {' '.join(sys.argv)}\n")
    sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
    print(f"console output also saved to {path}")


def main(argv=None):
    p = build_arg_parser("Steve active perception + VGN grasping — unified joint-velocity WBC")
    g = p.add_argument_group("active perception")
    # --table (build_arg_parser): measured table model, name in steve_wbc/config/tables or a path
    # (table_probe); sim: replaces the MJCF table, robot: the WBC's table collision model
    g.add_argument("--target_xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="sim: object position on the table (default: per scene); the table spans "
                        "x 1.6..2.4, y -0.35..0.35, the robot arrives from -x")
    g.add_argument("--target", default="red",
                   help="target profile: name in steve_wbc/config/targets or a path (target_calibrator)")
    g.add_argument("--scene", choices=["single", "clutter", "far", "occluded"], default="single",
                   help="sim: one object | a red cube among static obstacles (only the cube is picked) | "
                        "a cube out of reach from the front (drive around the table)")
    g.add_argument("--no_reposition", action="store_true",
                   help="never drive around the table to another approach side")
    g.add_argument("--no_transit", action="store_true",
                   help="approach the object in a straight line, not over the obstacles in the way")
    g.add_argument("--object_type", choices=["sphere", "box"], default=None,
                   help="sim object (default: sphere; clutter scene: box)")
    g.add_argument("--object_speed", type=float, default=None,
                   help="sim object speed along +y [m/s] (default: 0.02; clutter scene: 0)")
    g.add_argument("--no_vgn", action="store_true", help="skip the VGN scan (coarse position, top-down)")
    g.add_argument("--vgn_views", type=int, default=None,
                   help="views per grasp scan (default 4); drive mode: inference starts once this many are "
                        "taken while driving (default 6) — 1: as soon as the object is in view and in range")
    g.add_argument("--no_los", action="store_true",
                   help="no occlusion-aware line of sight (table map from depth, side choice, QP term, "
                        "searching the other sides)")
    g.add_argument("--side_search_timeout", type=float, default=12.0,
                   help="with a table model: object not found from a side within this [s] -> search from the "
                        "next side (no table model: --search_timeout, then abort)")
    g.add_argument("--no_visibility", action="store_true",
                   help="no visibility task: the camera is aimed only through the posture's shoulder pan")
    g.add_argument("--look_camera_frame", default="gripper_camera_color_optical_frame",
                   help="robot: optical frame of the camera the visibility task aims (TF)")
    g.add_argument("--vgn_drive_range", type=float, default=VGN_DRIVE_RANGE,
                   help="drive mode: base-object distance below which views are taken while driving [m] "
                        "(real D415: ~1.2, its depth gets coarse farther away)")
    g.add_argument("--grasp_backend", choices=["vgn", "cgn"], default="vgn",
                   help="grasp detector of the scan: vgn (one TSDF of all views, vgn_env) | cgn (robot: "
                        "Contact-GraspNet per view on the target's Grounded-SAM mask; model servers "
                        "`ros2 run steve_perception model_server gsam|cgn`, started by the launch file)")
    g.add_argument("--grasp_prompt", default="red ball", help="cgn: text prompt naming the target (Grounded SAM)")
    g.add_argument("--gsam_server", default="tcp://127.0.0.1:5556", help="cgn: Grounded-SAM model server")
    g.add_argument("--cgn_server", default="tcp://127.0.0.1:5555", help="cgn: Contact-GraspNet model server")
    g.add_argument("--save_dir", default="auto",
                   help="robot: save every grasp scan (views with target mask + grasps, .jpg/.npz) to "
                        "<dir>/grasp_scan — auto: the launch's log folder ($STEVE_LAUNCH_LOG_DIR) or "
                        "~/ros_logs/perception_<time>; '' : off")
    g.add_argument("--vgn_mode", choices=["scan", "async", "drive"], default="scan",
                   help="scan: stop, scan, wait for inference (original) | async: same scan, inference "
                        "in the background | drive: views while driving up, inference in the background")
    g.add_argument("--vgn_max_wait", type=float, default=5.0,
                   help="async/drive: max time to hover at the pre-grasp for a running VGN inference [s] "
                        "(0: never wait, grasp without a late result)")
    g.add_argument("--vgn_orientation", choices=["topdown", "yaw"], default="topdown",
                   help="topdown, or top-down rotated to VGN's finger axis")
    g.add_argument("--vgn_explore", choices=["fuse", "fresh", "off"], default="fuse",
                   help="VGN finds no grasp on the target: drive around the table to the next side and scan "
                        "again — fuse: all sides' views in one TSDF | fresh: a new TSDF per side | off: grasp "
                        "the tracked position top-down anyway. No side left: 'no grasp candidate found'")
    g.add_argument("--vgn_explore_max_sides", type=int, default=4, help="table sides to try at most")
    g.add_argument("--vgn_camera", default="auto",
                   help="robot: RealSense namespace of the VGN depth camera (auto: gripper camera first, then "
                        "wrist, e.g. /gripper_camera)")
    g.add_argument("--vgn_no_level", action="store_true",
                   help="robot: do not level each VGN view's camera pose on the table plane (--table)")
    g.add_argument("--vgn_max_level", type=float, default=8.0,
                   help="robot: largest table-plane tilt correction applied to a VGN view [deg]")
    g.add_argument("--render_every", type=int, default=3, help="perception every N control ticks")
    g.add_argument("--detect_z_tol", type=float, default=0.05,
                   help="with a table (sim, or --table): ignore detections whose centre is further than this from resting on "
                        "the table top (top_z + radius) [m] — close-range / mis-registered depth")
    g.add_argument("--min_detect_range", type=float, default=0.20,
                   help="ignore detections closer than this to the camera [m] (occlusion)")
    g.add_argument("--drive_speed", type=float, default=0.25, help="[m/s]")
    g.add_argument("--max_drive", type=float, default=1.5, help="max drive distance [m]")
    # --grasp_dz (build_arg_parser): pinch point relative to the object centre [m]; here -0.015
    # (set_defaults below): the pads then close at the equator
    g.add_argument("--camera_tilt", type=float, default=0.0,
                   help="tilt the camera forward by this much in the drive and scan poses [deg] (wrist 1); "
                        "for a camera looking along the flange axis, which otherwise sees only the floor/table "
                        "straight below (robot: ~30). The MJCF wrist camera is already tilted: 0")
    g.add_argument("--goto_grasp_only", action="store_true",
                   help="demo without a gripper: go to the grasp pose, stay there with the object (static or "
                        "moving) for --hold_time, never close the gripper or lift, then back up to the "
                        "pre-grasp height. Result 'reached grasp pose'")
    g.add_argument("--hold_time", type=float, default=3.0,
                   help="after the lift, keep holding the object this long; success only if it stays "
                        "in the gripper [s] (0: success at the end of the lift)")
    g.add_argument("--continuous", action="store_true",
                   help="never stop after an episode: hand the object over (gripper: opens after "
                        "--handover_time), then watch the table and go for the object again once it has been "
                        "put down somewhere new (>= --watch_moved away); searches around the table if it is "
                        "out of view. Ctrl+C ends. Sim: the object is moved automatically")
    g.add_argument("--max_episodes", type=int, default=0, help="--continuous: stop after this many (0: never)")
    g.add_argument("--watch_moved", type=float, default=0.10,
                   help="--continuous: a new placement is at least this far from the last one [m]")
    g.add_argument("--handover_time", type=float, default=5.0,
                   help="--continuous with a gripper: after the hold, open the gripper after this long [s]")
    g.add_argument("--sim_seed", type=int, default=0, help="--continuous, sim: seed of the object placements")
    g.add_argument("--orbit_speed", type=float, default=REPOSITION_SPEED,
                   help="base speed around the table [m/s] (capped by --max_base_v)")
    g.add_argument("--d_safe_grasp", type=float, default=0.005, help="collision margin for the final descent [m]")
    g.add_argument("--search_timeout", type=float, default=30.0, help="[s]")
    g.add_argument("--lost_timeout", type=float, default=3.0, help="abort if a moving object is unseen for [s]")
    g.add_argument("--lost_timeout_static", type=float, default=15.0,
                   help="abort if a static object is unseen for [s] (occluded while approaching)")
    g = p.add_argument_group("active perception — ROS")
    g.add_argument("--scan_topic", default="/lidar_1/scan_filtered", help="front LiDAR (sensor_msgs/LaserScan)")
    g.add_argument("--lidar_sector_deg", type=float, default=60.0,
                   help="sector around the scan's 0° used until the LiDAR's TF is available")
    g.add_argument("--lidar_corridor", type=float, default=0.35,
                   help="half-width of the straight-ahead corridor (base_link y) checked by the LiDAR [m]")
    g.add_argument("--target_topic", default="/steve_perception/target_point",
                   help="object position (geometry_msgs/PointStamped, odom or base_link frame)")
    g.add_argument("--target_timeout", type=float, default=0.5, help="max age of a target message [s]")
    g.add_argument("--gripper_action", default="/gripper_controller/gripper_cmd")
    g.add_argument("--grip_effort", type=float, default=50.0)
    p.set_defaults(sim_q0=DRIVE_POSE, max_base_v=0.30, max_base_w=0.50, max_arm_vel=0.8, grasp_dz=-0.015)
    a = parse_args(argv, p)
    a.vgn_views_given = a.vgn_views is not None
    a.vgn_views = a.vgn_views if a.vgn_views_given else 4
    save_console(a.mode)
    if a.mode == "kin":
        sys.exit("active_perception_wbc needs perception: use --mode sim or --mode ros")
    static_cube = a.scene in ("clutter", "far")
    if a.object_type is None:
        a.object_type = "box" if static_cube else "sphere"
    if a.object_speed is None:
        a.object_speed = 0.0 if static_cube or a.scene == "occluded" else 0.02  # a rolling cube would plough into the obstacles

    if a.camera_tilt:  # drive + scan poses look forward-down (wrist 1 negative = camera forward)
        for pose in (DRIVE_POSE, SCAN_SEED_Q):
            pose[3] -= np.radians(a.camera_tilt)
        print(f"camera tilted {a.camera_tilt:.0f} deg forward: drive pose {np.round(DRIVE_POSE, 3)}, "
              f"scan pose {np.round(SCAN_SEED_Q, 3)}")
    table = TableModel.load(a.table) if a.table else None
    if table is not None:
        print(f"table '{table.name}' ({table.frame}): top {table.size[0]:.2f} x {table.size[1]:.2f} m at "
              f"z {table.top_z:.2f} m, {len(table.legs)} legs, skirt {table.skirt:.2f} m")
        if a.mode == "ros":
            a.collision_env = "model"  # the measured table is the real environment
    elif a.mode == "ros":
        print("[WARN] no table model (--table, see table_probe): the WBC does not know the table — "
              "no arm/base collision check against it; the LiDARs cannot see its top plate")
    model = build_model(resolve_model(a.model), a.scene if a.mode == "sim" else "single",
                        intruder=a.sim_intruder is not None, table=table)
    kin = WholeBodyKinematics(model)
    dt = 1.0 / a.rate
    if a.mode == "sim":
        backend = MujocoSimBackend(model, a, dt)
        world = SimWorld(backend, a)
        if world.obstacle_geoms:
            backend.on_step = lambda: world.check_obstacles(backend.data.time)
        if backend.viewer:
            backend.viewer.cam.lookat[:] = [1.4, 0.0, 0.8]
            backend.viewer.cam.distance, backend.viewer.cam.elevation = 3.5, -25
    else:
        try:
            ros = RosBackend(a, dt)
        except ImportError:
            sys.exit("[ERROR] rclpy not found — source ROS 2 first")
        backend = DryRunBackend(ros, dt) if a.dry_run else ros
        # base pose for base_link -> odom target transforms (the shadow state in a dry run)
        get_base = ((lambda: backend.st.base.copy()) if a.dry_run
                    else (lambda: None if ros.base is None else ros.base.copy()))
        world = RosWorld(ros.node, a, get_base, a.dry_run, table)
    try:
        return ActivePerceptionWBC(backend, world, kin, a).run()
    finally:
        backend.shutdown()


if __name__ == "__main__":
    sys.exit(main())
