#!/usr/bin/env python3
"""
steve_figure8_wbc.py
====================
Unified velocity-level (kinematic) Whole-Body Controller for Steve Butler
(Neobotix MMO-700 omnidirectional base + UR5e), tracking a figure-8 with the
end-effector. The SAME controller runs against three backends:

  --mode sim   MuJoCo physics (mmo_700_wbc.xml) with viewer   -> develop / tune
  --mode kin   pure kinematic integration, no physics          -> quick checks
  --mode ros   real robot (or Gazebo) via ROS 2                -> deployment
               add --dry_run to read the real state but publish nothing

Whole-body formulation
----------------------
Decision variable (9 DOF), base velocities expressed in the BASE frame so they
map 1:1 onto /cmd_vel:

    u = [vx_b, vy_b, wz_b, dq1 ... dq6]

Whole-body Jacobian (exact, from the MuJoCo kinematic tree, incl. the lever
arm of base yaw on the EE):

    J_wb = [ J_xy(world) * R_z(yaw) | J_yaw | J_arm ]          (6 x 9)

Primary task: EE pose tracking
    v_des = [ p_dot_d + Kp (p_d - p) ;  w_d + Ko e_o ]

Secondary task (posture), projected into the null space of J_wb so it cannot
disturb the EE task:
    u_sec = N_W * [0, 0, Ky (yaw_0 - yaw), Kq (q_0 - q)],   N_W = I - J_W^+ J

Strict task hierarchy (default, --hierarchy strict): one constrained least-squares
problem per priority level, each lower level restricted to the solutions that keep
what the levels above achieved (|A_i u - A_i u_i| <= --hqp_tol):

    1. EE pose        min || J_wb u - v_des ||^2_{W_t}
    2. base pose      min || u_base - u_base_des ||^2        (only if a base task is set)
    3. posture        min || u - u_post ||^2_{W_u}           (u_post unprojected)
    all levels s.t. lb <= u <= ub, collision constraints C u >= c

A level that cannot be solved is dropped and the higher levels' solution kept, so a
lower task never degrades a higher one; W_t / W_u only weigh rows WITHIN a level.

Weighted alternative (--hierarchy weighted): one constrained weighted QP (bounded
least squares, solved exactly by BVLS):

    min_u  || J_wb u - v_des ||^2_{W_t}  +  || u - u_sec ||^2_{W_u}
           [ + w_b || u_base - u_base_des ||^2   optional soft base-pose task ]
    s.t.   lb <= u <= ub

    lb/ub = intersection of
        velocity limits      (base |v|, |w|; arm |dq|)
        acceleration limits  (u_prev +- a_max * dt)
        joint-position limits (k_lim * (q_lim -+ margin - q))

W_u decides how motion is shared between base and arm; the null-space
posture task makes the base take over sustained EE displacement while the
arm stays near its start configuration.

Run  (see steve_wbc/README.md and the launch files for the full workflow)
---
  ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=true simulator:=mujoco
  ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=true             # Gazebo
  ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false dry_run:=true
  ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false amplitude:=0.08 frequency:=0.25 loops:=1

  ros2 run steve_wbc steve_figure8_wbc --mode sim --amplitude 0.25 --log wbc_sim.csv

  # Without ROS (e.g. macOS; use `mjpython` instead of `python3` for the viewer)
  python3 steve_figure8_wbc.py --mode sim --amplitude 0.25 --log wbc_sim.csv
  python3 steve_figure8_wbc.py --mode kin
"""

import argparse
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:
    from scipy.optimize import lsq_linear, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ── Model lookup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

try:  # measured / hand-written table (config/tables/<name>.yaml), --table
    from steve_wbc.table_model import TableModel
except ImportError:
    sys.path.insert(0, str(SCRIPT_DIR))
    from table_model import TableModel

try:  # safety stop shared with steve_manipulation (manipulator)
    from steve_manipulation.safety_stop import (EM_STOP_TOPIC, SCAN_TOPICS, ClearTimer, EmStopMonitor,
                                                LidarSafety, ScanPoints)
except ImportError:  # plain python3 without the ROS workspace: the module from the source tree
    sys.path.insert(0, str(SCRIPT_DIR.parents[1] / "steve_manipulation"))
    from steve_manipulation.safety_stop import (EM_STOP_TOPIC, SCAN_TOPICS, ClearTimer, EmStopMonitor,
                                                LidarSafety, ScanPoints)
MODEL_NAME = "mmo_700_wbc.xml"


def _share_model():
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("steve_wbc")) / "models" / MODEL_NAME
    except Exception:  # not installed / ROS not sourced
        return None


# Works with and without ROS (e.g. plain python3 / mjpython on macOS)
MODEL_CANDIDATES = [
    SCRIPT_DIR.parent / "models" / MODEL_NAME,                            # src/steve_wbc/steve_wbc/
    SCRIPT_DIR.parent / "src" / "steve_wbc" / "models" / MODEL_NAME,      # steve_ros2_ws/scripts/
    SCRIPT_DIR.parent / "robots" / MODEL_NAME,                            # porta_ros2_ws/scripts/
    SCRIPT_DIR.parent.parent / "porta_ros2_ws" / "robots" / MODEL_NAME,
    _share_model(),                                                       # installed: share/steve_wbc/models
]

# Simulated person for testing the LiDAR safety stop (--sim_intruder): a mocap cylinder the
# LiDAR sees; no contacts and not in the WBC's collision model, so only the safety stop reacts.
INTRUDER = "intruder"
INTRUDER_PARK = (50.0, 50.0)   # xy while the person is away


def load_model(path, intruder=False, spec_edit=None):
    """MuJoCo model from `path`, optionally with the test intruder and further scene edits."""
    if not intruder and spec_edit is None:
        return mujoco.MjModel.from_xml_path(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    if intruder:
        body = spec.worldbody.add_body(name=INTRUDER, mocap=True, pos=[*INTRUDER_PARK, 0.85])
        body.add_geom(name=f"{INTRUDER}_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.15, 0.85, 0],
                      rgba=[1.0, 0.55, 0.0, 0.8], contype=0, conaffinity=0)
    if spec_edit is not None:
        spec_edit(spec)
    return spec.compile()


BASE_JOINTS = ["base_x", "base_y", "base_yaw"]
ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
EE_SITE = "pinch"
NU = 9  # 3 base + 6 arm

# Start pose for simulation (arm raised, wrist forward) — same as the old script
SIM_ARM_Q0 = [0.0, -1.2, 1.2, -1.6, -1.57, 0.0]

# Wrist pointing along the base +X axis (used by --orientation forward)
R_FORWARD = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)   # gripper pointing down


# ══════════════════════════════════════════════════════════════════════════════
# Small math helpers
# ══════════════════════════════════════════════════════════════════════════════
def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9:
        return np.eye(3) + skew(rv)
    K = skew(rv / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def log_so3(R):
    th = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th * w / (2.0 * np.sin(th))


def orientation_error(R_des, R_cur):
    """World-frame orientation error (Siciliano); robust, no singularity at pi."""
    return 0.5 * sum(np.cross(R_cur[:, i], R_des[:, i]) for i in range(3))


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# ══════════════════════════════════════════════════════════════════════════════
# Robot state + whole-body kinematics
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RobotState:
    base: np.ndarray  # [x, y, yaw] in world / odom frame
    q: np.ndarray     # 6 arm joint angles


class WholeBodyKinematics:
    """Kinematic model on its own MjData — never touches a physics simulation."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, self.data, 0)

        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        base_j = [jid(n) for n in BASE_JOINTS]
        arm_j = [jid(n) for n in ARM_JOINTS]
        if min(base_j + arm_j) < 0:
            raise RuntimeError("Model is missing base_x/base_y/base_yaw or UR5e joints")

        self.base_qpos = [model.jnt_qposadr[j] for j in base_j]
        self.base_dof = [model.jnt_dofadr[j] for j in base_j]
        self.arm_qpos = [model.jnt_qposadr[j] for j in arm_j]
        self.arm_dof = [model.jnt_dofadr[j] for j in arm_j]
        self.q_min = model.jnt_range[arm_j, 0].copy()
        self.q_max = model.jnt_range[arm_j, 1].copy()

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
        if self.site_id < 0:
            raise RuntimeError(f"Site '{EE_SITE}' not found in model")
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def set_state(self, st: RobotState):
        self.data.qpos[self.base_qpos] = st.base
        self.data.qpos[self.arm_qpos] = st.q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # needed by mj_jacSite

    def ee_pose(self):
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return p, R

    def to_decision(self, J, yaw):
        """Map a (k x nv) MuJoCo Jacobian to decision coordinates [v_b(2), wz, dq(6)]."""
        c, s = np.cos(yaw), np.sin(yaw)
        R2 = np.array([[c, -s], [s, c]])  # v_world = R2 @ v_base
        return np.hstack([J[:, self.base_dof[:2]] @ R2, J[:, [self.base_dof[2]]], J[:, self.arm_dof]])

    def jacobian(self, yaw):
        """6x9 whole-body Jacobian of the EE site."""
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site_id)
        return self.to_decision(np.vstack([self._jp, self._jr]), yaw)

    def point_jacobian(self, point, body, yaw):
        """3x9 translational Jacobian of a world point rigidly attached to `body`."""
        mujoco.mj_jac(self.model, self.data, self._jp, None, point, body)
        return self.to_decision(self._jp, yaw)


# ══════════════════════════════════════════════════════════════════════════════
# Collision avoidance — velocity-damper inequality constraints
# ══════════════════════════════════════════════════════════════════════════════
# The base constraint targets this much more than --d_safe_base, so the lagging base (cmd_vel
# tracking, ~0.15 s) stops before the safety distance itself
BASE_LAG_BUFFER = 0.03


class CollisionAvoidance:
    """
    Collision avoidance as hard velocity-damper constraints in the QP
    (Faverjon & Tournassoud 1987; Kanehiro et al. 2008). For every monitored
    pair closer than d_i:

        n^T (J_a - J_b) u  >=  -xi * (d - d_s) / (d_i - d_s)

    n: unit vector from the obstacle to the robot, J_a/J_b: Jacobians of the
    two closest points (J_b = 0 for static obstacles). The allowed approach
    speed falls linearly from xi at d_i to 0 at d_s (and becomes a push-away
    demand below d_s), so distances cannot shrink below d_s.

    Geometry is approximated conservatively and distances are computed in
    closed form (mj_geomDistance is not reliable for these mesh/primitive pairs):
      * arm + gripper  -> spheres covering each collision geom
                          (capsules along their axis, boxes/meshes over their AABB)
      * obstacles      -> oriented boxes (table parts; cylinders/meshes via their AABB)
                          and the floor plane
      * base vs. env   -> 2-D: base footprint rectangle vs. obstacle footprints
                          (both are vertical prisms at the base's height)

    Monitored:
      arm   vs  floor                                         always
      arm   vs  own body (base, cabinet, pan-tilt)  [self]    always
      arm   vs  static environment of the MuJoCo model        --collision_env model
      base  vs  static environment of the MuJoCo model        --collision_env model
    """

    SELF_EXCLUDED_ARM_BODIES = ("shoulder_link", "upper_arm_link")  # mounted on the cabinet
    SELF_OBSTACLE_BODIES = ("cabinet", "pan_tilt")                  # visual-only in the MJCF

    def __init__(self, kin: WholeBodyKinematics, args):
        self.kin, self.a = kin, args
        m = self.m = kin.model
        self.bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"#{b}"
        self.gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or self.bname(m.geom_bodyid[g])

        def subtree(root):
            out = set()
            for b in range(m.nbody):
                p = b
                while p > 0 and p != root:
                    p = m.body_parentid[p]
                if p == root:
                    out.add(b)
            return out

        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.base_body = body("base_link")
        robot, arm = subtree(self.base_body), subtree(body("shoulder_link"))
        collidable = lambda g: bool(m.geom_contype[g] or m.geom_conaffinity[g])
        movable = {b for b in range(1, m.nbody) if b not in robot and m.body_jntnum[b] > 0}
        T = mujoco.mjtGeom

        # ── robot spheres (body id, centre in body frame, radius, geom) ─────
        sb, sp, sr, sg = [], [], [], []
        for g in range(m.ngeom):
            if not collidable(g) or m.geom_bodyid[g] not in arm:
                continue
            for c, r in self._spheres(g):
                sb.append(m.geom_bodyid[g]); sp.append(c); sr.append(r); sg.append(g)
        self.s_body, self.s_pos = np.array(sb), np.array(sp)
        self.s_rad, self.s_geom = np.array(sr), np.array(sg)
        self.s_self = np.array([self.bname(b) not in self.SELF_EXCLUDED_ARM_BODIES for b in sb])

        # ── boxes: (body, centre in body frame, R in body frame, half sizes, name, moves_with_robot)
        self.boxes = []
        if args.collision_env == "model":
            for g in range(m.ngeom):
                if (collidable(g) and m.geom_bodyid[g] not in robot and m.geom_bodyid[g] not in movable
                        and m.geom_type[g] != T.mjGEOM_PLANE):
                    self.boxes.append((*self._geom_box(g), self.gname(g), False))
        self.n_env_boxes = len(self.boxes)
        for g in range(m.ngeom):  # self-collision obstacles
            bn = self.bname(m.geom_bodyid[g])
            if m.geom_rbound[g] > 0 and (bn.startswith(self.SELF_OBSTACLE_BODIES)
                                         or (m.geom_bodyid[g] == self.base_body and collidable(g)
                                             and m.geom_type[g] == T.mjGEOM_MESH)):
                self.boxes.append((*self._geom_box(g), bn, True))

        self.floor = any(collidable(g) and m.geom_type[g] == T.mjGEOM_PLANE for g in range(m.ngeom))

        # ── base footprint (base frame) from the base collision mesh ───────
        bg = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.base_body and collidable(g)
              and m.geom_type[g] == T.mjGEOM_MESH]
        corners = np.vstack([self._box_corners(*self._geom_box(g)[1:4]) for g in bg])
        self.base_xy_min, self.base_xy_max = corners[:, :2].min(0), corners[:, :2].max(0)
        # two parts with their own footprint (base frame) and height band:
        #   low  = the base body: keeps --d_safe_base (LiDAR safety field) from legs, walls, ...
        #   high = cabinet + the arm's mount on it: keeps --d_safe_top from things above the base
        #          only, e.g. a table's top plate — the base itself fits under the overhang
        self.base_z = (corners[:, 2].min(), corners[:, 2].max())
        kin.set_state(RobotState(np.zeros(3), np.zeros(6)))
        hp = []
        for bn in ("cabinet_link", "ur5e_base"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bn)
            if b < 0:
                continue
            Rb, pb = kin.data.xmat[b].reshape(3, 3), kin.data.xpos[b] - kin.data.xpos[self.base_body]
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == b:
                    hp.append(pb + self._box_corners(*self._geom_box(g)[1:4]) @ Rb.T)
        hp = np.vstack(hp) if hp else corners
        self.top_xy_min, self.top_xy_max = hp[:, :2].min(0), hp[:, :2].max(0)
        self.top_z = (hp[:, 2].min(), hp[:, 2].max())
        self.env_footprints = []  # (polygon xy, name) of env boxes overlapping the base height
        self._ignored = set()
        self._checked_start = False
        print(f"[collision] {len(self.s_rad)} arm spheres | {self.n_env_boxes} env boxes "
              f"(env: {args.collision_env}) | {len(self.boxes) - self.n_env_boxes} self boxes | "
              f"floor | d_safe={args.d_safe} m  d_influence={args.d_influence} m | "
              f"base d_safe={args.d_safe_base} m")

    # ── geometry helpers ────────────────────────────────────────────────────
    def _geom_box(self, g):
        """Oriented bounding box of a geom in its BODY frame: (body, centre, R, half)."""
        m, T = self.m, mujoco.mjtGeom
        R_g = np.zeros(9)
        mujoco.mju_quat2Mat(R_g, m.geom_quat[g])
        R_g = R_g.reshape(3, 3)
        s = m.geom_size[g]
        if m.geom_type[g] == T.mjGEOM_BOX:
            c, h = np.zeros(3), s.copy()
        elif m.geom_type[g] in (T.mjGEOM_CYLINDER, T.mjGEOM_CAPSULE):
            c, h = np.zeros(3), np.array([s[0], s[0], s[1] + (s[0] if m.geom_type[g] == T.mjGEOM_CAPSULE else 0)])
        elif m.geom_type[g] == T.mjGEOM_SPHERE:
            c, h = np.zeros(3), np.full(3, s[0])
        else:  # mesh, ellipsoid, ...: local AABB
            c, h = m.geom_aabb[g, :3].copy(), m.geom_aabb[g, 3:].copy()
        return m.geom_bodyid[g], m.geom_pos[g] + R_g @ c, R_g, h

    @staticmethod
    def _box_corners(c, R, h):
        s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return c + (s * h) @ R.T

    def _spheres(self, g):
        """Spheres (centre in BODY frame, radius) that cover geom g."""
        m, T = self.m, mujoco.mjtGeom
        _, c, R, h = self._geom_box(g)
        t = m.geom_type[g]
        if t in (T.mjGEOM_CAPSULE, T.mjGEOM_CYLINDER):
            r, hl = m.geom_size[g, 0], m.geom_size[g, 1]
            n = max(2, int(np.ceil(2 * hl / r)) + 1)
            rad = r if t == T.mjGEOM_CAPSULE else np.hypot(r, hl / (n - 1))
            return [(c + R @ np.array([0, 0, z]), rad) for z in np.linspace(-hl, hl, n)]
        if t == T.mjGEOM_SPHERE:
            return [(c, m.geom_size[g, 0])]
        # box / mesh: grid of cells over the (local) box, each covered by one sphere
        r0 = max(min(h.min(), 0.03), 0.01)
        n = np.maximum(1, np.ceil(h / r0)).astype(int)
        cell = h / n
        rad = np.linalg.norm(cell)
        axes = [np.linspace(-h[i] + cell[i], h[i] - cell[i], n[i]) for i in range(3)]
        return [(c + R @ np.array([x, y, z]), rad) for x in axes[0] for y in axes[1] for z in axes[2]]

    @staticmethod
    def _convex_hull(P):
        P = sorted(map(tuple, np.round(P, 9)))
        cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, hi = [], []
        for p in P:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
                lo.pop()
            lo.append(p)
        for p in reversed(P):
            while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
                hi.pop()
            hi.append(p)
        return np.array(lo[:-1] + hi[:-1])

    @staticmethod
    def _poly_distance(A, B):
        """Signed distance between convex polygons A, B (ccw). Returns d, pA, pB."""
        def seg_closest(p, a, b):
            ab = b - a
            t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
            return a + t * ab
        # separating-axis test for overlap depth
        depth, axis = np.inf, None
        for P in (A, B):
            for i in range(len(P)):
                e = P[(i + 1) % len(P)] - P[i]
                nrm = np.array([e[1], -e[0]]) / max(np.linalg.norm(e), 1e-12)
                pa, pb = A @ nrm, B @ nrm
                ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
                if ov <= 0:
                    depth = None
                    break
                if ov < depth:
                    depth, axis = ov, nrm
            if depth is None:
                break
        if depth is not None:  # overlapping
            if (A.mean(0) - B.mean(0)) @ axis < 0:
                axis = -axis
            pA = A[np.argmin(A @ axis)]
            return -depth, pA, pA - depth * axis
        best = (np.inf, None, None)
        for P, Q, flip in ((A, B, False), (B, A, True)):
            for p in P:
                for i in range(len(Q)):
                    q = seg_closest(p, Q[i], Q[(i + 1) % len(Q)])
                    d = np.linalg.norm(p - q)
                    if d < best[0]:
                        best = (d, q, p) if flip else (d, p, q)
        return best

    # ── constraint assembly ────────────────────────────────────────────────
    def _pairs(self, yaw):
        """All (distance, robot point, robot body, obstacle point, obstacle body|None, normal, label)."""
        m, d, a = self.m, self.kin.data, self.a
        X = d.xpos[self.s_body] + np.einsum("nij,nj->ni", d.xmat[self.s_body].reshape(-1, 3, 3), self.s_pos)
        out = []

        # arm spheres vs floor (z = 0)
        if self.floor:
            dz = X[:, 2] - self.s_rad
            for gi in np.unique(self.s_geom):
                idx = np.where(self.s_geom == gi)[0]
                i = idx[np.argmin(dz[idx])]
                out.append((dz[i], X[i], self.s_body[i], X[i] * [1, 1, 0], None,
                            np.array([0.0, 0.0, 1.0]), f"{self.gname(gi)}/floor"))

        # arm spheres vs boxes
        for bi, (bb, bc, bR, bh, name, moving) in enumerate(self.boxes):
            C = d.xpos[bb] + d.xmat[bb].reshape(3, 3) @ bc
            Rw = d.xmat[bb].reshape(3, 3) @ bR
            # the shoulder / upper arm sit on the cabinet, so they skip cabinet and base — but the
            # pan-tilt tower (0.35 m behind the shoulder) is in reach of the upper arm: all links
            mask = (np.ones(len(X), bool) if not moving or name.startswith("pan_tilt") else self.s_self)
            if not mask.any():
                continue
            L = (X[mask] - C) @ Rw
            Q = np.clip(L, -bh, bh)
            diff = L - Q
            dist = np.linalg.norm(diff, axis=1)
            inside = dist < 1e-9
            if inside.any():  # centre inside the box: push out through the nearest face
                pen = bh - np.abs(L[inside])
                ax = np.argmin(pen, axis=1)
                dd = np.zeros_like(L[inside])
                dd[np.arange(len(ax)), ax] = np.sign(L[inside][np.arange(len(ax)), ax]) + (L[inside][np.arange(len(ax)), ax] == 0)
                diff[inside] = dd * 1e-9
                dist[inside] = -pen[np.arange(len(ax)), ax]
            sd = dist - self.s_rad[mask]
            ids = np.where(mask)[0]
            geoms = self.s_geom[mask]
            for gi in np.unique(geoms):
                k = np.where(geoms == gi)[0]
                j = k[np.argmin(sd[k])]
                if sd[j] >= a.d_influence or (gi, bi) in self._ignored:
                    continue
                n = Rw @ (diff[j] / np.linalg.norm(diff[j]))
                i = ids[j]
                out.append((sd[j], X[i], self.s_body[i], C + Rw @ Q[j], bb if moving else None, n,
                            f"{self.gname(gi)}/{name}", (gi, bi)))

        # base vs env footprints (2-D): the base body against what reaches into its height band,
        # the cabinet + arm mount against what is only above it (a table top)
        if self.n_env_boxes:
            bp = d.xpos[self.base_body]
            bR = d.xmat[self.base_body].reshape(3, 3)

            def footprint(lo, hi):
                rect = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
                return bp[:2] + rect @ bR[:2, :2].T
            parts = (("base", footprint(self.base_xy_min, self.base_xy_max), self.base_z, a.d_influence_base),
                     ("basetop", footprint(self.top_xy_min, self.top_xy_max), self.top_z, a.d_influence))
            for bb, bc, bR_, bh, name, moving in self.boxes[:self.n_env_boxes]:
                cw = self._box_corners(d.xpos[bb] + d.xmat[bb].reshape(3, 3) @ bc,
                                       d.xmat[bb].reshape(3, 3) @ bR_, bh)
                for part, A, (z0, z1), d_inf in parts:
                    zlo, zhi = bp[2] + z0, bp[2] + z1
                    if cw[:, 2].max() < zlo or cw[:, 2].min() > zhi:
                        continue
                    if part == "basetop" and cw[:, 2].min() <= bp[2] + self.base_z[1]:
                        continue  # reaches down to the base: handled by the base part
                    c2 = cw[:, :2]
                    if np.linalg.norm(c2.mean(0) - A.mean(0)) > 2.0 + d_inf:
                        continue
                    dist, pA, pB = self._poly_distance(A, self._convex_hull(c2))
                    if dist >= d_inf:
                        continue
                    v = pA - pB  # from obstacle to base; for overlaps pB = pA - depth*axis, so also outward
                    n2 = v / max(np.linalg.norm(v), 1e-12)
                    z = 0.5 * (zlo + zhi)
                    out.append((dist, np.array([*pA, z]), self.base_body, np.array([*pB, z]), None,
                                np.array([*n2, 0.0]), f"{part}/{name}"))
        return out

    def constraints(self, yaw):
        """C u >= c for the current kin state (set_state already called)."""
        a = self.a
        pairs = self._pairs(yaw)
        if not self._checked_start:
            # self pairs already inside d_safe at start are modelling overlaps, not avoidable
            for p in pairs:
                if len(p) > 7 and p[4] is not None and p[0] < a.d_safe:
                    self._ignored.add(p[7])
                    print(f"[collision] ignoring self pair overlapping at start: {p[6]} ({p[0] * 100:.1f} cm)")
                elif p[0] < a.d_safe:
                    print(f"[collision][WARN] start state within d_safe: {p[6]} ({p[0] * 100:.1f} cm)")
            self._checked_start = True
            pairs = self._pairs(yaw)
        rows, rhs = [], []
        d_min, closest = np.inf, ""
        self.d_min_by_obstacle = {}  # obstacle name -> min distance (pairs within d_influence)
        self.base_clearance = np.inf  # base footprint vs environment (pairs within d_influence_base)
        for p in pairs:
            dist, pa, ba, pb, bb, n, label = p[:7]
            obs = label.rsplit("/", 1)[-1]
            if ba == self.base_body and label.startswith("basetop/"):
                # cabinet / arm mount vs something above the base (table top): no LiDAR field there
                d_s, d_i = a.d_safe_top, a.d_influence
                self.base_clearance = min(self.base_clearance, dist)
            elif ba == self.base_body:  # (by body: the gripper also has a body named "base")
                # the base keeps its own, never relaxed safety distance (LiDAR safety field),
                # aimed BASE_LAG_BUFFER inside it: the drives follow cmd_vel with a lag
                d_s, d_i = a.d_safe_base + BASE_LAG_BUFFER, a.d_influence_base
                self.base_clearance = min(self.base_clearance, dist)
            else:
                d_s, d_i = a.d_safe, a.d_influence
                if obs.startswith("obst_"):  # objects on the table: no relaxed grasp margin
                    d_s = max(d_s, a.d_safe_obstacle)
                if dist < d_min:
                    d_min, closest = dist, label
            self.d_min_by_obstacle[obs] = min(dist, self.d_min_by_obstacle.get(obs, np.inf))
            if dist >= d_i:
                continue
            J = self.kin.point_jacobian(pa, ba, yaw)
            if bb is not None:
                J = J - self.kin.point_jacobian(pb, bb, yaw)
            rows.append(n @ J)
            rhs.append(-a.xi * (dist - d_s) / (d_i - d_s))
        return np.array(rows).reshape(-1, NU), np.array(rhs), d_min, closest


# ══════════════════════════════════════════════════════════════════════════════
# Reference: approach (min-jerk) + figure-8 with smooth speed ramp in/out
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Reference:
    p: np.ndarray
    v: np.ndarray
    R: np.ndarray
    w: np.ndarray
    phase: str
    done: bool = False


class Figure8Reference:
    """
    Lemniscate of Gerono:
        p(phi) = c + A_a sin(phi) cos(phi) a_hat + A_b sin(phi) b_hat
    preceded by a min-jerk approach from the start pose (p0, R0) to the centre
    and the goal orientation R_goal.

    phi(t) uses a smooth frequency ramp so the EE starts and ends at rest,
    and the total phase is exactly 2*pi*loops (ends back at the centre).
    """

    def __init__(self, p0, R0, center, a_hat, b_hat, amp_a, amp_b, omega, loops,
                 ramp_time=3.0, R_goal=None):
        self.A_a, self.A_b = amp_a, amp_b
        self.omega = omega
        self.a_hat, self.b_hat = np.asarray(a_hat, float), np.asarray(b_hat, float)
        self.p0 = p0.copy()
        self.center = np.asarray(center, float).copy()

        self.R0 = R0.copy()
        R_goal = R0 if R_goal is None else R_goal
        self.rv = log_so3(R_goal @ R0.T)  # world-frame rotation start -> goal

        dist = np.linalg.norm(self.center - p0)
        ang = np.linalg.norm(self.rv)
        self.T_app = 0.0 if (dist < 1e-3 and ang < 1e-2) else max(3.0, dist / 0.05, ang / 0.2)

        self.T_ramp = min(ramp_time, np.pi * loops / self.omega)
        self.phi_total = 2.0 * np.pi * loops
        self.T_cruise = (self.phi_total - self.omega * self.T_ramp) / self.omega
        self.T_8 = 2.0 * self.T_ramp + self.T_cruise
        self.T_total = self.T_app + self.T_8

    @classmethod
    def from_args(cls, p0, R0, yaw0, args):
        """Figure-8 in a plane attached to the START base heading:
        vertical  : a_hat = base lateral, b_hat = world up      (8 stands upright)
        horizontal: a_hat = base forward, b_hat = base lateral  (8 lies flat)"""
        fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
        lat = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        a_hat, b_hat = (lat, up) if args.plane == "vertical" else (fwd, lat)
        center = p0.copy()
        if getattr(args, "center_xy", None) is not None:   # e.g. over the table (--table)
            center[:2] = args.center_xy
        if args.cz is not None:
            center[2] = args.cz
        # Orientation: hold the start orientation, turn the wrist to point along base +X, or down
        R_goal = {"hold": R0, "forward": rot_z(yaw0) @ R_FORWARD,
                  "down": rot_z(yaw0) @ R_TOPDOWN}[args.orientation]
        return cls(p0, R0, center, a_hat, b_hat, args.amplitude, args.amplitude,
                   args.frequency, args.loops, args.ramp_time, R_goal)

    def phase_at(self, t):
        """Figure-8 phase and rate at absolute time t (0 during approach)."""
        if t < self.T_app:
            return 0.0, 0.0
        if t - self.T_app >= self.T_8:
            return self.phi_total, 0.0
        return self._phase(t - self.T_app)

    def _phase(self, t):
        """Returns phi, dphi/dt for t in [0, T_8]."""
        w, Tr = self.omega, self.T_ramp
        sstep = lambda x: 3 * x**2 - 2 * x**3
        sint = lambda x: x**3 - 0.5 * x**4  # integral of smoothstep on [0, x]
        if t < Tr:
            x = t / Tr
            return w * Tr * sint(x), w * sstep(x)
        if t < Tr + self.T_cruise:
            return w * Tr / 2.0 + w * (t - Tr), w
        x = max(0.0, (self.T_8 - t) / Tr)
        return self.phi_total - w * Tr * sint(x), w * sstep(x)

    def __call__(self, t):
        R_goal = exp_so3(self.rv) @ self.R0
        if t < self.T_app:
            tau = t / self.T_app
            s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
            ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T_app
            p = self.p0 + s * (self.center - self.p0)
            v = ds * (self.center - self.p0)
            R = exp_so3(s * self.rv) @ self.R0
            return Reference(p, v, R, ds * self.rv, "approach")

        t8 = t - self.T_app
        if t8 >= self.T_8:
            return Reference(self.center.copy(), np.zeros(3), R_goal, np.zeros(3), "done", True)
        phi, dphi = self._phase(t8)
        s, c = np.sin(phi), np.cos(phi)
        p = self.center + self.A_a * s * c * self.a_hat + self.A_b * s * self.b_hat
        v = dphi * (self.A_a * np.cos(2 * phi) * self.a_hat + self.A_b * c * self.b_hat)
        return Reference(p, v, R_goal, np.zeros(3), "figure8")

    def sample_path(self, n=240):
        phis = np.linspace(0, 2 * np.pi, n)
        return [self.center + self.A_a * np.sin(f) * np.cos(f) * self.a_hat + self.A_b * np.sin(f) * self.b_hat
                for f in phis]


# ══════════════════════════════════════════════════════════════════════════════
# Whole-body QP controller
# ══════════════════════════════════════════════════════════════════════════════
class GotoReference:
    """Point-to-point EE reference through waypoints [(p, R), ...]: min-jerk position and
    orientation per segment (speed-limited), then hold the last one for `hold` s. Used for
    --goto_file (VGN grasp: pre-grasp above the object, then the grasp pose)."""

    def __init__(self, p0, R0, waypoints, v_max=0.05, w_max=0.2, hold=3.0, names=None):
        self.segs, p, R, t = [], np.asarray(p0, float), np.asarray(R0, float), 0.0
        for i, (pg, Rg) in enumerate(waypoints):
            pg, Rg = np.asarray(pg, float), np.asarray(Rg, float)
            rv = log_so3(Rg @ R.T)
            T = max(2.0, np.linalg.norm(pg - p) / v_max, np.linalg.norm(rv) / w_max)
            self.segs.append((t, T, p, R, pg, rv, (names or [])[i] if names and i < len(names) else f"wp{i}"))
            p, R, t = pg, Rg, t + T
        self.p_end, self.R_end = p, R
        self.T_total = t + hold
        self.center = p

    def __call__(self, t):
        for t0, T, pa, Ra, pb, rv, name in self.segs:
            if t < t0 + T:
                tau = (t - t0) / T
                s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
                ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
                return Reference(pa + s * (pb - pa), ds * (pb - pa), exp_so3(s * rv) @ Ra, ds * rv, name)
        done = t >= self.T_total
        return Reference(self.p_end.copy(), np.zeros(3), self.R_end.copy(), np.zeros(3), "hold", done)

    def sample_path(self, n=60):
        return [self(t).p for t in np.linspace(0, self.T_total, n)]


def goto_from_file(path, p0, R0, args):
    """GotoReference from vgn_grasp_real's JSON (frame must be the WBC world = odom)."""
    import json
    f = Path(path).expanduser()
    if not f.is_file():
        raise SystemExit(f"[ABORT] grasp file {f} not found — run vgn_grasp_real (on the PC) first "
                         f"and copy its ~/ros_logs/vgn_grasp.json to the robot")
    g = json.loads(f.read_text())
    w, x, y, z = g["quat_wxyz"]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pinch = np.asarray(g["pinch"], float) + np.array([0.0, 0.0, args.grasp_dz])
    pre = pinch + np.array([0.0, 0.0, args.pregrasp])
    print(f"  goto: {g.get('time', '')} score {g.get('score', 0):.2f}  pre-grasp {np.round(pre, 3)} -> "
          f"grasp {np.round(pinch, 3)} ({g.get('frame', '?')})")
    wps, names = [(pre, R)], ["pregrasp"]
    if not args.pregrasp_only:
        wps.append((pinch, R))
        names.append("grasp")
    return GotoReference(p0, R0, wps, v_max=args.goto_speed, hold=args.goto_hold, names=names)


TASK_NAMES = ("ee", "pos", "ori", "base", "posture")


def parse_priorities(spec):
    """'ee,base,posture' -> [['ee'], ['base'], ['posture']]; '+' joins tasks on one level.
    ee = pos + ori (6-D); posture is appended as the last level if it is missing."""
    levels = [[n.strip() for n in lvl.split("+") if n.strip()] for lvl in spec.split(",") if lvl.strip()]
    names = [n for lvl in levels for n in lvl]
    bad = [n for n in names if n not in TASK_NAMES]
    if bad:
        raise SystemExit(f"[ERROR] --priorities: unknown task(s) {bad}; use {', '.join(TASK_NAMES)}")
    if len(set(names)) != len(names):
        raise SystemExit(f"[ERROR] --priorities: a task appears twice in '{spec}'")
    if "ee" in names and ({"pos", "ori"} & set(names)):
        raise SystemExit("[ERROR] --priorities: 'ee' already contains 'pos' and 'ori'")
    if not ({"ee", "pos"} & set(names)):
        raise SystemExit("[ERROR] --priorities: the EE position ('ee' or 'pos') must be a task")
    if "posture" not in names:
        levels.append(["posture"])   # without it the redundancy would be unresolved
    return levels


class WholeBodyController:
    def __init__(self, kin: WholeBodyKinematics, args, q_nom, yaw_nom):
        self.kin = kin
        self.a = args
        self.q_nom = np.asarray(q_nom, float).copy()
        self.yaw_nom = float(yaw_nom)

        self.W_t = np.diag([args.w_pos] * 3 + [args.w_ori] * 3)
        wx = args.w_base_x if args.w_base_x is not None else args.w_base_lin
        wy = args.w_base_y if args.w_base_y is not None else args.w_base_lin
        self.W_u = np.diag([wx, wy, args.w_base_yaw] + [args.w_arm] * 6)
        self.sqrt_Wt = np.sqrt(self.W_t)
        self.sqrt_Wu = np.sqrt(self.W_u)
        self.W_u_inv = np.linalg.inv(self.W_u)

        self.v_max = np.array([args.max_base_v] * 2 + [args.max_base_w] + [args.max_arm_vel] * 6)
        self.a_max = np.array([args.max_base_acc] * 2 + [args.max_base_wacc] + [args.max_arm_acc] * 6)
        if args.lock_yaw:
            self.v_max[2] = 0.0

        self.coll = CollisionAvoidance(kin, args) if args.collision else None
        if self.coll and not SCIPY_AVAILABLE:
            print("[WARN] collision avoidance needs scipy — disabled")
            self.coll = None
        self.levels = parse_priorities(args.priorities)
        self.qp_failures = 0
        self.level_skips = 0   # strict hierarchy: lower levels dropped (higher ones kept)

        if not SCIPY_AVAILABLE:
            print("[WARN] scipy not found -> falling back to damped pseudoinverse + clipping "
                  "(constraints are no longer handled optimally). `pip install scipy`.")

    def _bounds(self, st: RobotState, u_prev, dt, acc_limits=True):
        lb, ub = -self.v_max.copy(), self.v_max.copy()
        # Joint-position limits -> velocity bounds
        m, k = self.a.joint_margin, self.a.k_joint_limit
        lb[3:] = np.maximum(lb[3:], k * (self.kin.q_min + m - st.q))
        ub[3:] = np.minimum(ub[3:], k * (self.kin.q_max - m - st.q))
        # Acceleration limits
        if acc_limits:
            lb = np.maximum(lb, u_prev - self.a_max * dt)
            ub = np.minimum(ub, u_prev + self.a_max * dt)
        # Keep the box feasible (BVLS needs lb < ub)
        bad = ub - lb < 1e-9
        mid = 0.5 * (lb + ub)
        lb[bad], ub[bad] = mid[bad] - 1e-9, mid[bad] + 1e-9
        return lb, ub

    def step(self, st: RobotState, ref, u_prev, dt, base_ref=None):
        """
        ref      : Reference for the EE task, or None (EE free, e.g. posture moves)
        base_ref : BaseReference for the soft base-pose task, or None
        """
        self.kin.set_state(st)
        p, R = self.kin.ee_pose()
        rows_A, rows_b = [], []

        # ── Primary task: EE pose ──────────────────────────────────────────
        if ref is not None:
            J = self.kin.jacobian(st.base[2])
            e_p = ref.p - p
            e_o = orientation_error(ref.R, R)
            e_p_fb = e_p
            n = np.linalg.norm(e_p)
            if n > self.a.max_fb_err:  # saturate feedback, never the feed-forward
                e_p_fb = e_p * (self.a.max_fb_err / n)
            v_des = np.concatenate([ref.v + self.a.kp_pos * e_p_fb, ref.w + self.a.kp_ori * e_o])
            JW = J @ self.W_u_inv
            J_pinv_W = self.W_u_inv @ J.T @ np.linalg.inv(JW @ J.T + self.a.damping * np.eye(6))
            N = np.eye(NU) - J_pinv_W @ J
            rows_A.append(self.sqrt_Wt @ J)
            rows_b.append(self.sqrt_Wt @ v_des)
            A_ee, b_ee = self.sqrt_Wt @ J, self.sqrt_Wt @ v_des   # strict hierarchy tasks
            T = {"ee": (A_ee, b_ee), "pos": (A_ee[:3], b_ee[:3]), "ori": (A_ee[3:], b_ee[3:])}
        else:
            e_p, e_o, N = np.zeros(3), np.zeros(3), np.eye(NU)
            T = {}

        # ── Soft base-pose task (e.g. orbit around a table) ─────────────────
        if base_ref is not None:
            yaw = st.base[2]
            v_w = base_ref.vel[:2] + self.a.k_base * (base_ref.pose[:2] - st.base[:2])
            u_b = np.concatenate([rot_z(yaw)[:2, :2].T @ v_w,
                                  [base_ref.vel[2] + self.a.k_base * wrap_angle(base_ref.pose[2] - yaw)]])
            S = np.hstack([np.eye(3), np.zeros((3, 6))])
            w = np.sqrt(self.a.w_base_task)
            rows_A.append(w * S)
            rows_b.append(w * u_b)
            T["base"] = (w * S, w * u_b)

        # ── Posture, projected into the null space of the EE task ───────────
        u_post = np.zeros(NU)
        if base_ref is None:  # base heading is otherwise owned by the base task
            u_post[2] = self.a.k_posture * wrap_angle(self.yaw_nom - st.base[2])
        u_post[3:] = self.a.k_posture * (self.q_nom - st.q)
        u_sec = N @ u_post
        rows_A.append(self.sqrt_Wu)
        rows_b.append(self.sqrt_Wu @ u_sec)
        # strict hierarchy: posture is the last level, the equality constraints of the levels
        # above replace the (unconstrained) null-space projector N
        T["posture"] = (self.sqrt_Wu, self.sqrt_Wu @ u_post)
        # levels in --priorities order; tasks joined with '+' share a level, absent tasks are skipped
        tasks = []
        for level in self.levels:
            rows = [T[n] for n in level if n in T]
            if rows:
                tasks.append((np.vstack([r[0] for r in rows]), np.concatenate([r[1] for r in rows])))

        # ── Constrained QP: weighted (one solve) or strict hierarchy (one per level) ──
        A, b = np.vstack(rows_A), np.concatenate(rows_b)
        lb, ub = self._bounds(st, u_prev, dt)
        C, c, d_min, closest = (self.coll.constraints(st.base[2]) if self.coll
                                else (np.zeros((0, NU)), np.zeros(0), np.inf, ""))
        strict = self.a.hierarchy == "strict" and SCIPY_AVAILABLE
        solve = ((lambda lo, hi: self._solve_hierarchy(tasks, lo, hi, C, c)) if strict
                 else (lambda lo, hi: self._solve(A, b, lo, hi, C, c)))
        if not SCIPY_AVAILABLE:
            u = np.clip(np.linalg.lstsq(A, b, rcond=None)[0], lb, ub)
        else:
            u = solve(lb, ub)
            if u is None:  # acceleration limits can make the collision constraints infeasible
                u = solve(*self._bounds(st, u_prev, dt, acc_limits=False))
            if u is None:  # u = 0 satisfies every constraint with d >= d_safe -> stop
                self.qp_failures += 1
                u = np.zeros(NU)

        return u, dict(p=p, R=R, e_p=e_p, e_o=e_o, d_min=d_min, closest=closest, n_coll=len(c))

    def _solve_hierarchy(self, tasks, lb, ub, C, c):
        """Strict task hierarchy (lexicographic least squares): level k minimises its own
        residual subject to all constraints AND to every higher level keeping what it achieved
        (|A_i u - A_i u_i| <= hqp_tol). A level that cannot be solved is skipped — the result
        of the levels above it is kept, so a lower task can never degrade a higher one.
        A tiny damping term (--damping) keeps each level unique near singularities."""
        tol, lam = self.a.hqp_tol, np.sqrt(self.a.damping)
        E, e, u = np.zeros((0, NU)), np.zeros(0), None
        for k, (Ak, bk) in enumerate(tasks):
            A_ = np.vstack([Ak, lam * np.eye(NU)])
            b_ = np.concatenate([bk, np.zeros(NU)])
            uk = (self._solve(A_, b_, lb, ub, C, c) if len(e) == 0
                  else self._solve_eq(A_, b_, lb, ub, C, c, E, e, tol, u))
            if uk is None:
                if u is None:
                    return None
                self.level_skips += 1
                break
            u = uk
            if k < len(tasks) - 1:  # fix what this level achieved for the levels below
                E, e = np.vstack([E, Ak]), np.concatenate([e, Ak @ u])
        return u

    @staticmethod
    def _solve_eq(A, b, lb, ub, C, c, E, e, tol, u0):
        """min ||A u - b||^2  s.t.  lb <= u <= ub,  C u >= c,  |E u - e| <= tol  (SLSQP,
        warm-started with the higher level's solution, which is feasible)."""
        H, g = A.T @ A, A.T @ b
        G = np.vstack([C, E, -E])
        h = np.concatenate([c, e - tol, -e - tol])
        res = minimize(lambda x: 0.5 * x @ H @ x - g @ x, u0, jac=lambda x: H @ x - g,
                       bounds=list(zip(lb, ub)), method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda x: G @ x - h, "jac": lambda x: G}],
                       options={"ftol": 1e-10, "maxiter": 50})
        if res.success and np.all(res.x >= lb - 1e-6) and np.all(res.x <= ub + 1e-6) \
                and np.all(G @ res.x >= h - 1e-5):
            return res.x
        return None

    @staticmethod
    def _solve(A, b, lb, ub, C, c):
        """min ||A u - b||^2  s.t.  lb <= u <= ub,  C u >= c.
        Box-only problems are solved exactly by BVLS; with active collision rows
        the BVLS solution warm-starts SLSQP."""
        u = lsq_linear(A, b, bounds=(lb, ub), method="bvls").x
        if len(c) == 0 or np.all(C @ u >= c - 1e-7):
            return u
        H, g = A.T @ A, A.T @ b
        res = minimize(lambda x: 0.5 * x @ H @ x - g @ x, u, jac=lambda x: H @ x - g,
                       bounds=list(zip(lb, ub)), method="SLSQP",
                       constraints=[{"type": "ineq", "fun": lambda x: C @ x - c, "jac": lambda x: C}],
                       options={"ftol": 1e-10, "maxiter": 50})  # bounded solve time
        if res.success and np.all(res.x >= lb - 1e-6) and np.all(res.x <= ub + 1e-6) \
                and np.all(C @ res.x >= c - 1e-5):
            return res.x
        return None


@dataclass
class BaseReference:
    pose: np.ndarray  # desired [x, y, yaw] (world)
    vel: np.ndarray   # feed-forward [vx, vy, wz] (world)


# ══════════════════════════════════════════════════════════════════════════════
# Backends — identical interface for simulation and hardware
# ══════════════════════════════════════════════════════════════════════════════
class Backend:
    has_lidar = False           # can this backend see the environment with the LiDARs?
    em_stop = None              # EmStopMonitor if the robot's EM-stop state is subscribed (ROS)

    def lidar_points(self):     # Nx2 points in base_link, None = no (fresh) data
        return None

    def wait_for_state(self, timeout): ...
    def read_state(self): ...
    def send(self, u): ...
    def tick(self): ...        # advance one control period (step physics / sleep)
    def now(self): ...
    def ok(self): return True
    def stop(self): ...
    def shutdown(self): ...


class KinematicBackend(Backend):
    """Perfect-tracking integrator. Used by --mode kin and by --dry_run."""

    def __init__(self, st0: RobotState, dt):
        self.st = RobotState(st0.base.copy(), st0.q.copy())
        self.dt, self.t, self.u = dt, 0.0, np.zeros(NU)

    def wait_for_state(self, timeout): return self.read_state()
    def read_state(self): return RobotState(self.st.base.copy(), self.st.q.copy())
    def send(self, u): self.u = u.copy()
    def now(self): return self.t

    def tick(self):
        yaw = self.st.base[2]
        self.st.base[:2] += rot_z(yaw)[:2, :2] @ self.u[:2] * self.dt
        self.st.base[2] += self.u[2] * self.dt
        self.st.q += self.u[3:] * self.dt
        self.t += self.dt

    def stop(self): self.u = np.zeros(NU)
    def shutdown(self): pass


class MujocoSimBackend(Backend):
    """
    Physics simulation of Steve. The base is driven exactly like the real one
    (body-frame twist, first-order lag emulating the MMO-700 caster steering),
    the arm by joint-velocity actuators with gravity + damping compensation.
    """

    def __init__(self, model, args, dt, ref_path_fn=None):
        self.model, self.args, self.dt = model, args, dt
        self.data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, self.data, 0)

        jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        joints = [jid(n) for n in BASE_JOINTS + ARM_JOINTS]
        self.qpos_ids = [model.jnt_qposadr[j] for j in joints]
        self.dof_ids = [model.jnt_dofadr[j] for j in joints]
        base_names = ["base_x", "base_y", "base_yaw"]
        arm_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        self.vel_act = [aid(f"{n}_vel") for n in base_names + arm_names]
        self.motor_act = [aid(f"{n}_motor") for n in base_names + arm_names]
        if min(self.vel_act + self.motor_act) < 0:
            raise RuntimeError("Model is missing the *_vel / *_motor actuators")

        # The planar base joints have no z/tilt DOF, so floor contact only adds
        # ~1.4 kN of dry friction that the velocity servo cannot overcome (the base
        # would reach ~20% of the command). The real MMO-700 tracks cmd_vel well;
        # its lag is modelled explicitly by --sim_base_tau instead.
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        if floor >= 0:
            model.geom_contype[floor] = 0
            model.geom_conaffinity[floor] = 0

        self.data.qpos[self.qpos_ids[:3]] = 0.0
        self.data.qpos[self.qpos_ids[3:]] = args.sim_q0
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(model, self.data)

        self.n_sub = max(1, int(round(dt / model.opt.timestep)))
        self.cmd_world = np.zeros(NU)       # joint-space command actually applied
        self.base_act = np.zeros(3)         # lagged base velocity (world frame)
        self.delay = deque([np.zeros(NU)] * args.sim_latency_ticks, maxlen=args.sim_latency_ticks + 1)

        self.on_step = None  # optional callback after every physics step (e.g. contact checks)

        # Simulated LiDARs: horizontal ray fans from both scanner sites; hits on the robot
        # itself are dropped (like the real robot's scan_filtered)
        self.has_lidar = True
        sid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        self.lidar_sites = [s for s in (sid("lidar_front_site"), sid("lidar_rear_site")) if s >= 0]
        self.base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        root = lambda b: b if model.body_parentid[b] in (0, b) else root(model.body_parentid[b])
        self.robot_body = np.array([root(b) == self.base_body for b in range(model.nbody)])
        ang = np.radians(np.arange(0.0, 360.0, 1.0))
        self.lidar_dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
        self._ray_id = np.zeros(len(ang), np.int32)
        self._ray_d = np.zeros(len(ang))
        self._ray_groups = np.array([1, 1, 0, 1, 1, 1], np.uint8)  # skip visual-only geoms (group 2)
        ib = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, INTRUDER)
        self.intruder_mocap = model.body_mocapid[ib] if ib >= 0 else -1
        self.viewer = None
        self.trail = deque(maxlen=400)
        self.n_static = 0
        if not args.headless:
            from mujoco import viewer
            self.viewer = viewer.launch_passive(model, self.data)
            self.viewer.cam.lookat[:] = [0.6, 0.0, 0.9]
            self.viewer.cam.distance, self.viewer.cam.elevation, self.viewer.cam.azimuth = 3.2, -15, 150
            self.viewer.opt.sitegroup[:] = 1
        self._wall0 = None

    # Viewer decorations ────────────────────────────────────────────────────
    def _sphere(self, pos, r, rgba):
        scn = self.viewer.user_scn
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([r, 0, 0]), np.asarray(pos, float), np.eye(3).flatten(),
                            np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def draw_reference(self, pts, rgba=(0.2, 0.6, 1.0, 0.5), radius=0.006, append=False):
        if not self.viewer:
            return
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = self.n_static if append else 0
            for p in pts:
                self._sphere(p, radius, rgba)
            self.n_static = self.viewer.user_scn.ngeom
        self.trail.clear()

    def draw_live(self, p_ref, p_ee):
        if not self.viewer:
            return
        self.trail.append(p_ee.copy())
        with self.viewer.lock():
            self.viewer.user_scn.ngeom = self.n_static
            for p in list(self.trail)[::3]:
                self._sphere(p, 0.005, [1.0, 0.3, 0.1, 0.9])
            self._sphere(p_ref, 0.012, [1.0, 1.0, 0.0, 0.8])

    # Backend interface ─────────────────────────────────────────────────────
    def wait_for_state(self, timeout): return self.read_state()

    def read_state(self):
        q = self.data.qpos[self.qpos_ids]
        return RobotState(q[:3].copy(), q[3:].copy())

    def send(self, u):
        self.delay.append(u.copy())
        u = self.delay.popleft() if len(self.delay) > self.args.sim_latency_ticks else np.zeros(NU)
        yaw = self.data.qpos[self.qpos_ids[2]]
        v_world = np.array([*rot_z(yaw)[:2, :2] @ u[:2], u[2]])
        alpha = self.dt / (self.args.sim_base_tau + self.dt)
        self.base_act += alpha * (v_world - self.base_act)
        self.cmd_world = np.concatenate([self.base_act, u[3:]])

    def lidar_points(self):
        m, d = self.model, self.data
        bp, bR = d.xpos[self.base_body], d.xmat[self.base_body].reshape(3, 3)
        pts = []
        for s in self.lidar_sites:
            o = d.site_xpos[s].copy()
            # exclude the scanner's own body: the rays start inside its mesh
            mujoco.mj_multiRay(m, d, o, self.lidar_dirs.flatten(), self._ray_groups, 1, m.site_bodyid[s],
                               self._ray_id, self._ray_d, None, len(self.lidar_dirs), 10.0)
            hit = self._ray_id >= 0
            hit[hit] &= ~self.robot_body[m.geom_bodyid[self._ray_id[hit]]]
            pw = o + self._ray_d[hit, None] * self.lidar_dirs[hit]
            pts.append(((pw - bp) @ bR)[:, :2])
        return np.vstack(pts) if pts else np.zeros((0, 2))

    def _move_intruder(self):
        a = self.args.sim_intruder
        if self.intruder_mocap < 0 or a is None:
            return
        t_in, t_out, x, y = a
        here = t_in <= self.data.time < t_out
        self.data.mocap_pos[self.intruder_mocap] = [x, y, 0.85] if here else [*INTRUDER_PARK, 0.85]

    def tick(self):
        m, d = self.model, self.data
        self._move_intruder()
        if self._wall0 is None:
            self._wall0 = time.perf_counter() - d.time
        for _ in range(self.n_sub):
            for i in range(NU):
                dof = self.dof_ids[i]
                d.ctrl[self.vel_act[i]] = self.cmd_world[i]
                # gravity/Coriolis + joint-damping feed-forward, plus an inertia-scaled
                # velocity loop (~20 ms) => joint velocities track the command like
                # the real drives; remaining lag comes only from --sim_base_tau/latency
                k_v = d.qM[m.dof_Madr[dof]] / 0.02
                d.ctrl[self.motor_act[i]] = (d.qfrc_bias[dof] + m.dof_damping[dof] * self.cmd_world[i]
                                             + k_v * (self.cmd_world[i] - d.qvel[dof]))
            mujoco.mj_step(m, d)
            if self.on_step:
                self.on_step()
        if self.viewer:
            self.viewer.sync()
            lag = d.time / self.args.sim_speed - (time.perf_counter() - self._wall0)
            if lag > 0:
                time.sleep(lag)

    def now(self): return self.data.time
    def ok(self): return self.viewer.is_running() if self.viewer else True

    def stop(self):
        self.delay.clear()
        self.delay.extend([np.zeros(NU)] * self.args.sim_latency_ticks)
        self.base_act[:] = 0.0
        self.cmd_world[:] = 0.0

    def shutdown(self):
        if self.viewer:
            # keep the window open until the user closes it
            print("Simulation finished — close the viewer window to exit.")
            while self.viewer.is_running():
                self.tick()
            self.viewer.close()


class RosBackend(Backend):
    """
    Real robot / Gazebo.
      in : /joint_states (sensor_msgs/JointState), /odom (nav_msgs/Odometry)
      out: /cmd_vel (geometry_msgs/Twist, base frame)
           arm: forward_velocity_controller (Float64MultiArray)   [--arm_interface velocity]
                or scaled_joint_trajectory_controller (JointTrajectory) [--arm_interface trajectory]
    """

    def __init__(self, args, dt):
        import rclpy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        self.rclpy, self.Twist = rclpy, Twist
        self.Float64MultiArray = Float64MultiArray
        self.JointTrajectory, self.JointTrajectoryPoint = JointTrajectory, JointTrajectoryPoint

        # Handle Ctrl+C ourselves so the context is still valid for stop()
        ros_args = ["steve_figure8_wbc", *getattr(args, "ros_args", [])]
        try:
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
        except (ImportError, TypeError):
            rclpy.init(args=ros_args)

        self.args, self.dt = args, dt
        self.node = rclpy.create_node("steve_figure8_wbc")
        self.log = self.node.get_logger()

        self.q = None
        self.q_ros = None   # raw /joint_states (ROS convention)
        self.q_cmd = None   # integrated arm setpoint (ROS convention), trajectory interface
        self.dq_f = None    # low-pass filtered arm velocity command (--arm_cmd_tau)
        self.q_stamp = 0.0
        self.arm_resolved = False
        self.arm_names = list(ARM_JOINTS)
        self.base = None
        self.base_stamp = 0.0

        self.node.create_subscription(JointState, args.joint_states_topic, self._js_cb, 10)
        if args.base_feedback == "odom":
            self.node.create_subscription(Odometry, args.odom_topic, self._odom_cb, 10)
        else:
            self.base, self.base_stamp = np.zeros(3), float("inf")

        # Safety stop: the robot's EM-stop state (relayboard), LiDAR scans only as the fallback
        # where no EM-stop system runs (Gazebo)
        self.em_stop = EmStopMonitor(self.node, args.em_stop_topic) if args.safety in ("auto", "em_stop") else None
        self.has_lidar = args.safety in ("auto", "lidar")
        self.scan = ScanPoints(self.node, args.lidar_topics) if self.has_lidar else None

        self.pub_twist = self.node.create_publisher(Twist, args.cmd_vel_topic, 10)
        if args.arm_interface == "velocity":
            self.pub_arm = self.node.create_publisher(Float64MultiArray, args.arm_vel_topic, 10)
        else:
            self.pub_arm = self.node.create_publisher(JointTrajectory, args.arm_traj_topic, 10)

        # Velocity interface on the real robot: switch trajectory -> velocity controller on the
        # first command (not in dry runs) and back on shutdown, so MoveIt / send_goal keep working
        self.switch_cli = None
        self.switched = False
        if args.arm_interface == "velocity" and args.switch_controllers:
            from controller_manager_msgs.srv import ListControllers, SwitchController
            self.SwitchController, self.ListControllers = SwitchController, ListControllers
            self.switch_cli = self.node.create_client(
                SwitchController, f"{args.controller_manager}/switch_controller")
            self.list_cli = self.node.create_client(
                ListControllers, f"{args.controller_manager}/list_controllers")

        self._t_next = None

    def _controller_states(self):
        """{name: state} from the controller manager, or None if it does not answer."""
        if not self.list_cli.wait_for_service(timeout_sec=3.0):
            return None
        fut = self.list_cli.call_async(self.ListControllers.Request())
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        if not fut.done() or fut.result() is None:
            return None
        return {c.name: c.state for c in fut.result().controller}

    def _switch(self, activate, deactivate):
        """Synchronous controller switch (STRICT); True on success. Only what is needed is requested:
        a STRICT request to deactivate a controller that is not active fails as a whole, so a
        state left over from an earlier run (e.g. both arm controllers inactive after a failed
        switch-back) would otherwise block every start."""
        if not self.switch_cli.wait_for_service(timeout_sec=3.0):
            self.log.error("controller_manager/switch_controller not available")
            return False
        states = self._controller_states() or {}  # unknown: request both, as before
        act = [] if states.get(activate) == "active" else [activate]
        deact = [deactivate] if states.get(deactivate, "active") == "active" else []
        if not act and not deact:
            self.log.info(f"controllers: {activate} already active")
            return True
        req = self.SwitchController.Request(activate_controllers=act, deactivate_controllers=deact,
                                            strictness=2)
        fut = self.switch_cli.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
        ok = fut.done() and fut.result() is not None and fut.result().ok
        was = ", ".join(f"{n} {states.get(n, '?')}" for n in (deactivate, activate))
        self.log.info(f"controllers: {deactivate} -> {activate}: {'ok' if ok else 'FAILED'}"
                      + ("" if ok or not states else f" (before: {was})"))
        return ok

    # Callbacks ─────────────────────────────────────────────────────────────
    def _js_cb(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if not self.arm_resolved:
            # match with or without tf_prefix ("ur5e", "ur5e_", ...)
            names = [next((n for n in msg.name if n.endswith(j)), None) for j in ARM_JOINTS]
            if None in names:
                return  # message from another publisher (gripper, pan-tilt, ...)
            self.arm_names, self.arm_resolved = names, True
            self.log.info(f"Arm joints: {names}")
        if not all(n in pos for n in self.arm_names):
            return
        self.q_ros = np.array([pos[n] for n in self.arm_names])
        # ROS (URDF) -> MuJoCo model convention; wrap so the offset cannot push pan past +-2*pi
        self.q = self.q_ros.copy()
        self.q[0] = wrap_angle(self.q[0] + self.args.pan_offset)
        self.q_stamp = time.monotonic()

    def _odom_cb(self, msg):
        o = msg.pose.pose.orientation
        yaw = np.arctan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y**2 + o.z**2))
        self.base = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        self.base_stamp = time.monotonic()

    # Backend interface ─────────────────────────────────────────────────────
    def lidar_points(self):
        return self.scan.points() if self.scan else None

    def _spin(self):
        for _ in range(10):  # drain queued callbacks
            self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def wait_for_state(self, timeout=10.0):
        t0 = time.monotonic()
        while self.rclpy.ok() and time.monotonic() - t0 < timeout:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            if self.q is not None and self.base is not None:
                return self.read_state()
        missing = [n for n, v in [("/joint_states (arm)", self.q), ("odom", self.base)] if v is None]
        self.log.error(f"Timeout waiting for: {missing}")
        return None

    def read_state(self):
        self._spin()
        now = time.monotonic()
        if now - self.q_stamp > self.args.state_timeout:
            self.log.error(f"/joint_states stale ({now - self.q_stamp:.2f}s)")
            return None
        if self.args.base_feedback == "odom" and now - self.base_stamp > self.args.state_timeout:
            self.log.error(f"odom stale ({now - self.base_stamp:.2f}s)")
            return None
        return RobotState(self.base.copy(), self.q.copy())

    def send(self, u, filtered=True):
        tw = self.Twist()
        tw.linear.x, tw.linear.y, tw.angular.z = float(u[0]), float(u[1]), float(u[2])
        self.pub_twist.publish(tw)

        dq = np.asarray(u[3:], float)
        # First-order low-pass on the arm command: the QP re-solves on measured joint states
        # every tick, so its output flips sign by ~1% at the control rate — heard as a buzz
        # from the joints. Trajectory interface: only the streamed velocity and look-ahead
        # are filtered; the position setpoint integrates the raw command, so the filter adds
        # no lag to the tracking loop (filtering it all made fast figure-8s unstable).
        # Off with --arm_cmd_tau 0; bypassed when stopping.
        tau = self.args.arm_cmd_tau
        if tau > 0 and filtered:
            if self.dq_f is None:
                self.dq_f = dq.copy()
            self.dq_f += self.dt / (tau + self.dt) * (dq - self.dq_f)
        else:
            self.dq_f = dq.copy()
        dq_s = self.dq_f.copy()
        if self.args.arm_interface == "velocity":
            if self.switch_cli is not None and not self.switched:
                if not self._switch(self.args.arm_vel_controller, self.args.arm_traj_controller):
                    raise RuntimeError(f"could not activate {self.args.arm_vel_controller} "
                                       "(loaded? ros2 control list_controllers)")
                self.switched = True
            self.pub_arm.publish(self.Float64MultiArray(data=[float(v) for v in dq_s]))
        else:
            h = self.args.traj_horizon
            traj = self.JointTrajectory()
            traj.joint_names = self.arm_names
            pt = self.JointTrajectoryPoint()
            # Integrate a continuous setpoint instead of re-anchoring on the measured q every
            # tick: a compliant arm (Gazebo) sags under gravity and would never catch up.
            # The lead over the measurement is bounded, so a stalled arm cannot wind it up.
            if self.q_cmd is None:
                self.q_cmd = self.q_ros.copy()
            L = self.args.max_cmd_lead
            self.q_cmd = self.q_ros + np.clip(self.q_cmd + dq * self.dt - self.q_ros, -L, L)
            pt.positions = [float(v) for v in self.q_cmd + dq_s * h]  # ROS convention
            pt.velocities = [float(v) for v in dq_s]
            pt.time_from_start.sec = int(h)
            pt.time_from_start.nanosec = int((h % 1.0) * 1e9)
            traj.points.append(pt)
            self.pub_arm.publish(traj)

        if self.args.base_feedback == "open_loop":
            self.base[:2] += rot_z(self.base[2])[:2, :2] @ u[:2] * self.dt
            self.base[2] += u[2] * self.dt

    def tick(self):
        if self._t_next is None:
            self._t_next = time.monotonic()
        self._t_next += self.dt
        sleep = self._t_next - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        elif sleep < -self.dt:
            self.log.warn(f"Control loop overrun by {-sleep * 1e3:.1f} ms")
            self._t_next = time.monotonic()

    def now(self): return time.monotonic()
    def ok(self): return self.rclpy.ok()

    def stop(self):
        if not self.rclpy.ok():
            return
        for _ in range(5):
            self.pub_twist.publish(self.Twist())
            if self.args.arm_interface == "velocity":
                self.pub_arm.publish(self.Float64MultiArray(data=[0.0] * 6))
            elif self.q is not None:
                self.send(np.zeros(NU), filtered=False)  # hold current position
            time.sleep(0.02)
        self.log.info("Zero commands sent to base and arm.")

    def shutdown(self):
        if self.switched and self.rclpy.ok():
            for _ in range(5):  # standstill before the trajectory controller takes over
                self.pub_arm.publish(self.Float64MultiArray(data=[0.0] * 6))
                time.sleep(0.02)
            if self._switch(self.args.arm_traj_controller, self.args.arm_vel_controller):
                self.switched = False
            else:
                self.log.error(f"{self.args.arm_vel_controller} still active — the watchdog holds "
                               f"zero velocity; switch back: ros2 control switch_controllers "
                               f"--activate {self.args.arm_traj_controller} "
                               f"--deactivate {self.args.arm_vel_controller}")
        self.node.destroy_node()
        if self.rclpy.ok():
            self.rclpy.shutdown()


class DryRunBackend(KinematicBackend):
    """Reads the real initial state over ROS, then integrates commands offline."""

    def __init__(self, ros: RosBackend, dt):
        self.ros = ros
        st0 = ros.wait_for_state(ros.args.wait_timeout)
        if st0 is None:
            raise RuntimeError("No robot state received — is the robot / bringup running?")
        super().__init__(st0, dt)

    def read_state(self):
        self.ros._spin()  # keep sensor callbacks (LiDAR, perception, ...) flowing
        return super().read_state()

    def tick(self):
        super().tick()   # integrate the commands offline ...
        self.ros.tick()  # ... at the real control rate, like the real run (live sensors)

    def ok(self): return self.ros.ok()
    def shutdown(self): self.ros.shutdown()

    @property
    def has_lidar(self): return self.ros.has_lidar
    @property
    def em_stop(self): return self.ros.em_stop
    def lidar_points(self): return self.ros.lidar_points()  # real scans; nothing is commanded anyway


# ══════════════════════════════════════════════════════════════════════════════
# LiDAR safety stop (all WBC scripts)
# ══════════════════════════════════════════════════════════════════════════════
class SafetyStop:
    """
    Stops base and arm while the safety source is red, see steve_manipulation/safety_stop.py:
      em_stop  the robot's EM-stop system (/emergency_stop_state: EM stop, scanner field)
      lidar    fallback without an EM-stop system (MuJoCo, Gazebo): LiDAR points closer than
               --d_safe_base to the base footprint
    Motion resumes after --safety_clear_time without red; callers subtract `paused` from their
    task time, so trajectories continue where they stopped.
    check(now) -> "run" | "stop" | "abort" (red longer than --safety_stop_timeout).
    """

    def __init__(self, backend, args, source):
        self.b, self.a, self.kind = backend, args, source
        self.src = (backend.em_stop if source == "em_stop"
                    else LidarSafety(backend.lidar_points, args.d_safe_base))
        self.timer = ClearTimer(args.safety_clear_time)
        self.paused, self.n_stops, self.t_stop, self._last = 0.0, 0, None, None
        print(f"[SAFETY] stop source: {'EM-stop system (' + args.em_stop_topic + ')' if source == 'em_stop' else f'LiDAR (red below {args.d_safe_base * 100:.0f} cm from the base; no EM-stop system)'}")

    def check(self, now):
        red, why = self.src.status()
        stopped = self.timer.update(red, now)
        dt = 0.0 if self._last is None else now - self._last
        self._last = now
        if not stopped:
            if self.t_stop is not None:
                print(f"[SAFETY] clear for {self.a.safety_clear_time:.1f} s -> resuming "
                      f"(stopped {now - self.t_stop:.1f} s)")
                self.t_stop = None
            return "run"
        self.b.send(np.zeros(NU))
        if self.t_stop is None:
            self.t_stop = now
            self.n_stops += 1
            print(f"[SAFETY STOP] {why or 'clearing'} -> base and arm stopped")
        else:
            self.paused += dt
        if now - self.t_stop > self.a.safety_stop_timeout:
            print(f"[ABORT] safety stop for more than {self.a.safety_stop_timeout:.0f} s ({why})")
            return "abort"
        return "stop"

    def summary(self):
        extra = ""
        if self.kind == "lidar" and np.isfinite(self.src.min_distance):
            extra = f", closest {self.src.min_distance * 100:.1f} cm (red below {self.a.d_safe_base * 100:.0f} cm)"
        return (f"  Safety stop [{self.kind}]: {self.n_stops} stop(s), {self.paused:.1f} s stopped{extra}")


def make_safety(backend, args, wait=3.0):
    """Safety source for this backend: the robot's EM-stop system if it runs (required with
    --safety em_stop), the LiDAR fallback only without one; None in the kinematic mode."""
    if args.safety == "off":
        print("[WARN] safety stop DISABLED (--safety off)")
        return None
    em = backend.em_stop
    if em is not None and args.safety == "auto":
        t0 = time.monotonic()
        while not em.available and time.monotonic() - t0 < wait:
            backend.read_state()  # spins the node
            time.sleep(0.05)
    if args.safety == "em_stop" or (em is not None and em.available):
        if em is None:
            sys.exit("[ERROR] --safety em_stop needs the ROS backend")
        return SafetyStop(backend, args, "em_stop")
    if backend.has_lidar:
        return SafetyStop(backend, args, "lidar")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Control loop (backend-agnostic)
# ══════════════════════════════════════════════════════════════════════════════
def table_setup(args, st0):
    """Figure-8 over the table: height from the table top, centre over its near half (horizontal
    plane), EE never below the top. Returns False if the robot does not face the table."""
    t = args.table_model
    if args.goto_file or args.goto_table is not None:
        args.min_ee_z = max(args.min_ee_z, t.top_z)   # the grasp / reach pose is near the top
        return True
    if args.cz is None:
        args.cz = t.top_z + args.table_clearance
    args.min_ee_z = max(args.min_ee_z, t.top_z + 0.02)
    if args.plane != "horizontal":
        return True
    Rt = t.R()
    hx, hy = 0.5 * t.size[0], 0.5 * t.size[1]
    if args.table_position == "centre":
        c = np.zeros(2)                                           # the table centre
    else:                                                         # near half, facing the robot
        b = Rt.T @ (st0.base[:2] - np.asarray(t.center))          # base in the table frame
        if abs(b[0]) - hx < abs(b[1]) - hy:                       # robot at a long (y) side
            s = np.sign(b[1]) or 1.0
            c = np.array([np.clip(b[0], -hx + args.table_margin + args.amplitude,
                                  hx - args.table_margin - args.amplitude), s * (hy - args.table_inset)])
        else:                                                     # robot at an x side
            s = np.sign(b[0]) or 1.0
            c = np.array([s * (hx - args.table_inset),
                          np.clip(b[1], -hy + args.table_margin + args.amplitude,
                                  hy - args.table_margin - args.amplitude)])
    args.center_xy = np.asarray(t.center) + Rt @ c
    d = np.hypot(*(args.center_xy - st0.base[:2]))
    print(f"  table: 8 centred {d:.2f} m from the base at z {args.cz:.2f} m "
          f"(top {t.top_z:.2f} + {args.table_clearance:.2f})")
    return True


def table_fits(args, ref_gen):
    """The whole 8 lies over the table top (minus --table_margin)."""
    t = args.table_model
    if args.plane != "horizontal":
        return True
    pts = np.array(ref_gen.sample_path(120))[:, :2]
    loc = (pts - np.asarray(t.center)) @ t.R()
    lim = np.array([0.5 * t.size[0], 0.5 * t.size[1]]) - args.table_margin
    over = np.max(np.abs(loc) - lim)
    if over > 0:
        print(f"[ABORT] the figure-8 leaves the table top by {over * 100:.1f} cm — reduce "
              f"amplitude:= or table_inset / table_margin")
        return False
    return True


class RosViz:
    """RViz markers while the WBC runs on ROS: the table model (/table_probe/markers, as table_probe)
    and the planned EE path + current target (/wbc/markers), in the odom frame."""

    def __init__(self, node, table, path, frame="odom"):
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from visualization_msgs.msg import Marker, MarkerArray
        self.M, self.MA, self.frame, self.table = Marker, MarkerArray, frame, table
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_t = node.create_publisher(MarkerArray, "/table_probe/markers", q) if table is not None else None
        self.pub_w = node.create_publisher(MarkerArray, "/wbc/markers", q)
        self.path = [np.asarray(p, float) for p in path]

    def _m(self, i, ns, kind, rgba):
        m = self.M()
        m.header.frame_id, m.ns, m.id, m.type = self.frame, ns, i, kind
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = (float(c) for c in rgba)
        return m

    def publish(self, target=None):
        from geometry_msgs.msg import Point
        t = self.table
        if self.pub_t is not None:
            ma = self.MA()
            m = self._m(0, "table", self.M.CUBE, (0.6, 0.4, 0.2, 0.8))
            m.pose.position.x, m.pose.position.y = (float(v) for v in t.center)
            m.pose.position.z = float(t.top_z - t.thickness / 2)
            m.pose.orientation.z, m.pose.orientation.w = float(np.sin(t.yaw / 2)), float(np.cos(t.yaw / 2))
            m.scale.x, m.scale.y, m.scale.z = float(t.size[0]), float(t.size[1]), float(t.thickness)
            ma.markers.append(m)
            for i, (lx, ly, r) in enumerate(t.legs):
                w = t.to_world([lx, ly])
                m = self._m(1 + i, "table", self.M.CYLINDER, (0.8, 0.2, 0.2, 0.8))
                m.pose.position.x, m.pose.position.y, m.pose.position.z = float(w[0]), float(w[1]), float(t.top_z / 2)
                m.scale.x = m.scale.y = float(2 * r)
                m.scale.z = float(t.top_z)
                ma.markers.append(m)
            self.pub_t.publish(ma)
        ma = self.MA()
        m = self._m(0, "wbc_path", self.M.LINE_STRIP, (0.1, 0.8, 0.1, 0.9))
        m.scale.x = 0.006
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in self.path]
        ma.markers.append(m)
        if target is not None:
            m = self._m(1, "wbc_target", self.M.SPHERE, (1.0, 0.5, 0.0, 0.9))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in target)
            m.scale.x = m.scale.y = m.scale.z = 0.025
            ma.markers.append(m)
        self.pub_w.publish(ma)


def run(backend: Backend, kin: WholeBodyKinematics, args):
    dt = 1.0 / args.rate
    st0 = backend.wait_for_state(args.wait_timeout)
    if st0 is None:
        return 1

    kin.set_state(st0)
    p0, R0 = kin.ee_pose()
    if getattr(args, "table_model", None) is not None and not table_setup(args, st0):
        return 1
    if args.goto_table is not None:
        t = args.table_model
        if t is None:
            print("[ABORT] --goto_table needs --table")
            return 1
        goal = np.array([*t.center, t.top_z + args.goto_table])
        R = rot_z(st0.base[2]) @ R_TOPDOWN
        print(f"  goto: gripper point {args.goto_table:.2f} m above the table centre {np.round(goal, 3)}")
        ref_gen = GotoReference(p0, R0, [(goal, R)], v_max=args.goto_speed, hold=args.goto_hold, names=["table"])
    elif args.goto_file:
        ref_gen = goto_from_file(args.goto_file, p0, R0, args)
    else:
        ref_gen = Figure8Reference.from_args(p0, R0, st0.base[2], args)
        if getattr(args, "table_model", None) is not None and not table_fits(args, ref_gen):
            return 1
    ctrl = WholeBodyController(kin, args, q_nom=st0.q, yaw_nom=st0.base[2])
    if isinstance(backend, MujocoSimBackend):
        backend.draw_reference(ref_gen.sample_path())

    print(f"\n{'═' * 74}\n  Steve Figure-8 — unified velocity-level WBC   [{args.mode}"
          f"{' / dry-run' if args.dry_run else ''}]\n{'═' * 74}")
    print(f"  base (x,y,yaw) : {np.round(st0.base, 3)}")
    print(f"  arm q          : {np.round(st0.q, 3)}")
    print(f"  EE start       : {np.round(p0, 3)}   centre: {np.round(ref_gen.center, 3)}")
    print(f"  plane={args.plane}  A={args.amplitude:.3f} m  omega={args.frequency:.3f} rad/s  "
          f"loops={args.loops}  duration={ref_gen.T_total:.1f} s")
    print(f"  limits: base {args.max_base_v} m/s, {args.max_base_w} rad/s | arm {args.max_arm_vel} rad/s"
          f" | solver: {('strict hierarchy (' + ' > '.join('+'.join(l) for l in parse_priorities(args.priorities)) + ')' if args.hierarchy == 'strict' else 'weighted BVLS box-QP') if SCIPY_AVAILABLE else 'pinv+clip'}\n")

    u_prev = np.zeros(NU)
    rows = []
    safety = make_safety(backend, args)
    ros_node = getattr(backend, "node", None) or getattr(getattr(backend, "ros", None), "node", None)
    viz = None
    if ros_node is not None:   # table + planned path in RViz (Fixed Frame odom)
        try:
            viz = RosViz(ros_node, getattr(args, "table_model", None), ref_gen.sample_path(120))
            viz.publish()
        except Exception as e:  # visualisation must never stop the controller
            print(f"  [viz] RViz markers disabled: {e}")
            viz = None
    next_viz = 0.0
    t0 = backend.now()
    next_print = 0.0
    status = 0
    try:
        while backend.ok():
            st = backend.read_state()
            if st is None:
                print("[ABORT] no valid robot state")
                status = 2
                break
            if safety is not None:
                s = safety.check(backend.now())
                if s == "abort":
                    status = 6
                    break
                if s == "stop":  # zero command sent; the figure-8 clock is paused
                    u_prev = np.zeros(NU)
                    backend.tick()
                    continue
            t = backend.now() - t0 - (safety.paused if safety else 0.0)

            ref = ref_gen(t)
            if ref.done:
                break

            tic = time.perf_counter()
            u, info = ctrl.step(st, ref, u_prev, dt)
            solve_ms = (time.perf_counter() - tic) * 1e3

            err = np.linalg.norm(info["e_p"])
            if not np.all(np.isfinite(u)):
                print("[ABORT] non-finite command")
                status = 3
                break
            if err > args.max_track_err:
                print(f"[ABORT] tracking error {err * 100:.1f} cm > {args.max_track_err * 100:.1f} cm")
                status = 4
                break
            if info["p"][2] < args.min_ee_z:
                print(f"[ABORT] EE below safety height ({info['p'][2]:.3f} m)")
                status = 5
                break

            backend.send(u)
            if isinstance(backend, MujocoSimBackend):
                backend.draw_live(ref.p, info["p"])

            rows.append([t, *ref.p, *info["p"], err, np.linalg.norm(info["e_o"]),
                         *u, *st.base, *st.q, solve_ms, info["d_min"]])
            if viz is not None and t >= next_viz:
                next_viz = t + 0.5
                try:
                    viz.publish(ref.p)
                except Exception:
                    viz = None
            if t >= next_print:
                next_print += 0.5
                clr = f"clr={info['d_min'] * 100:4.1f}cm " if np.isfinite(info["d_min"]) else ""
                print(f"t={t:5.1f}s [{ref.phase:8s}] err={err * 100:5.2f}cm {clr}"
                      f"base v=({u[0]:+.3f},{u[1]:+.3f}) w={u[2]:+.3f} | "
                      f"|dq|max={np.abs(u[3:]).max():.3f} | QP {solve_ms:.1f}ms")
            u_prev = u
            backend.tick()
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        status = 130
    finally:
        backend.stop()

    summarize(rows, args)
    if safety is not None:
        print(safety.summary())
    return status


LOG_HEADER = ("t,px_ref,py_ref,pz_ref,px,py,pz,err_pos,err_ori,"
              "u_vx,u_vy,u_wz,u_dq1,u_dq2,u_dq3,u_dq4,u_dq5,u_dq6,"
              "base_x,base_y,base_yaw,q1,q2,q3,q4,q5,q6,solve_ms,min_clearance")


def summarize(rows, args):
    if not rows:
        return
    L = np.array(rows)
    err = L[:, 7]
    base_xy = L[:, 18:20]
    q = L[:, 21:27]
    ee = L[:, 4:7]
    print(f"\n{'─' * 74}\n  Summary ({len(L)} control steps)")
    print(f"  EE position error : RMS {np.sqrt(np.mean(err**2)) * 100:.2f} cm   max {err.max() * 100:.2f} cm")
    print(f"  EE orient. error  : max {np.degrees(L[:, 8].max()):.2f} deg")
    print(f"  EE range  (x,y,z) : {np.round(np.ptp(ee, axis=0), 3)} m")
    print(f"  Base range (x,y)  : {np.round(np.ptp(base_xy, axis=0), 3)} m   "
          f"path {np.sum(np.linalg.norm(np.diff(base_xy, axis=0), axis=1)):.3f} m   "
          f"yaw range {np.degrees(np.ptp(L[:, 20])):.1f} deg")
    print(f"  Arm joint range   : {np.round(np.ptp(q, axis=0), 3)} rad")
    print(f"  QP solve time     : mean {L[:, 27].mean():.2f} ms   max {L[:, 27].max():.2f} ms")
    clr = L[:, 28][np.isfinite(L[:, 28])]
    if len(clr):
        print(f"  Min clearance     : {clr.min() * 100:.1f} cm   (d_safe {args.d_safe * 100:.1f} cm)")
    if args.log:
        np.savetxt(args.log, L, delimiter=",", header=LOG_HEADER, comments="", fmt="%.6f")
        print(f"  Log written to    : {args.log}")
    print(f"{'─' * 74}")


# ══════════════════════════════════════════════════════════════════════════════
def build_arg_parser(description="Steve figure-8 unified velocity-level WBC (MuJoCo / ROS 2)"):
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("mode")
    g.add_argument("--mode", choices=["sim", "kin", "ros"], default="sim")
    g.add_argument("--dry_run", action="store_true", help="ros mode: read real state, publish nothing")
    g.add_argument("--model", type=Path, default=None, help="path to mmo_700_wbc.xml")
    g.add_argument("--log", type=str, default=None, help="write CSV log to this path")

    g = p.add_argument_group("trajectory")
    g.add_argument("--plane", choices=["vertical", "horizontal"], default="vertical")
    g.add_argument("--amplitude", type=float, default=0.15, help="figure-8 amplitude [m]")
    g.add_argument("--frequency", type=float, default=0.45, help="figure-8 angular rate [rad/s]")
    g.add_argument("--loops", type=int, default=3)
    g.add_argument("--ramp_time", type=float, default=3.0, help="speed ramp in/out [s]")
    g.add_argument("--cz", type=float, default=None, help="figure-8 centre height [m] (default: start EE)")
    g.add_argument("--orientation", choices=["hold", "forward", "down"], default="hold",
                   help="hold start EE orientation, turn the wrist to point along base +X, or "
                        "point the gripper down (drawing on a table)")
    g.add_argument("--table", default=None,
                   help="table model (config/tables/<name>.yaml or a path): collision model for arm "
                        "and base; with --plane horizontal the 8 is centred over the table's near half "
                        "at top_z + --table_clearance and must fit on the top")
    g.add_argument("--table_clearance", type=float, default=0.10,
                   help="height of the 8 (gripper tip / EE point) above the table top [m]")
    g.add_argument("--table_position", choices=["centre", "near"], default="centre",
                   help="horizontal 8 over the table centre (default), or over its near half "
                        "(--table_inset from the edge facing the robot; less reach needed)")
    g.add_argument("--table_inset", type=float, default=0.25,
                   help="figure-8 centre this far inside the table's near edge [m]")
    g.add_argument("--goto_table", type=float, default=None, metavar="HEIGHT",
                   help="instead of a figure-8: move the gripper point HEIGHT m above the table centre, "
                        "pointing down (needs --table); the base moves as needed")
    g.add_argument("--goto_file", default=None,
                   help="instead of a figure-8: go to the grasp in this JSON (vgn_grasp_real), first "
                        "the pre-grasp above it, then the grasp pose, then hold")
    g.add_argument("--pregrasp", type=float, default=0.10, help="pre-grasp height above the grasp [m]")
    g.add_argument("--pregrasp_only", action="store_true", help="stop at the pre-grasp")
    g.add_argument("--grasp_dz", type=float, default=0.0,
                   help="shift the grasp point up (+) / down (-) [m], e.g. without a gripper mounted")
    g.add_argument("--goto_speed", type=float, default=0.05, help="max EE speed for --goto_file [m/s]")
    g.add_argument("--goto_hold", type=float, default=3.0, help="hold the grasp pose this long [s]")
    g.add_argument("--table_margin", type=float, default=0.05,
                   help="the 8 must stay this far inside the table top outline [m]")

    g = p.add_argument_group("controller")
    g.add_argument("--rate", type=float, default=50.0, help="control rate [Hz]")
    g.add_argument("--kp_pos", type=float, default=4.0)
    g.add_argument("--kp_ori", type=float, default=3.0)
    g.add_argument("--max_fb_err", type=float, default=0.05, help="saturation of position feedback [m]")
    g.add_argument("--w_pos", type=float, default=1000.0, help="task weight, EE position")
    g.add_argument("--w_ori", type=float, default=300.0, help="task weight, EE orientation")
    g.add_argument("--w_base_lin", type=float, default=1.0, help="effort weight, base translation")
    g.add_argument("--w_base_yaw", type=float, default=5.0, help="effort weight, base rotation")
    g.add_argument("--w_base_x", type=float, default=None,
                   help="effort weight, base forward (base frame); default --w_base_lin")
    g.add_argument("--w_base_y", type=float, default=None,
                   help="effort weight, base sideways (base frame); default --w_base_lin")
    g.add_argument("--w_arm", type=float, default=1.0, help="effort weight, arm joints")
    g.add_argument("--k_posture", type=float, default=1.0, help="null-space posture gain [1/s]")
    g.add_argument("--damping", type=float, default=1e-3, help="damping of null-space projector")
    g.add_argument("--hierarchy", choices=["weighted", "strict"], default="strict",
                   help="weighted: one QP, tasks traded off by their weights | strict: one QP per "
                        "priority level (EE pose > base pose > posture), lower levels act only in "
                        "the constrained null space of the higher ones")
    g.add_argument("--priorities", default="ee,base,posture",
                   help="strict hierarchy: task order, highest first; ',' separates levels, '+' puts "
                        "tasks on the same level (traded off by their weights). Tasks: ee (= pos+ori), "
                        "pos, ori, base (only when a base path is set), posture. Examples: "
                        "'pos,ori,posture'  'base,ee,posture'  'pos,ori+posture'")
    g.add_argument("--hqp_tol", type=float, default=1e-3,
                   help="strict hierarchy: slack with which a lower level must keep a higher "
                        "level's (weighted) task value")
    g.add_argument("--w_base_task", type=float, default=50.0, help="weight of soft base-pose task")
    g.add_argument("--k_base", type=float, default=2.0, help="base-pose task gain [1/s]")
    g.add_argument("--lock_yaw", action="store_true", help="forbid base rotation")

    g = p.add_argument_group("collision avoidance")
    g.add_argument("--no_collision", dest="collision", action="store_false",
                   help="disable collision-avoidance constraints")
    g.add_argument("--collision_env", choices=["model", "none"], default=None,
                   help="static obstacles from the MuJoCo model (table, ...) or none; self + floor "
                        "are always checked. Default: model in sim/kin, none in ros (the model's "
                        "table does not exist in the real robot's odom frame)")
    g.add_argument("--d_safe", type=float, default=0.02, help="minimum allowed distance [m]")
    g.add_argument("--d_influence", type=float, default=0.10, help="constraints active below [m]")
    g.add_argument("--xi", type=float, default=0.30, help="max approach speed at d_influence [m/s]")
    g.add_argument("--d_safe_top", type=float, default=0.03,
                   help="cabinet / arm mount vs obstacles above the base (a table top) [m]; the base "
                        "body keeps --d_safe_base from everything at its height (legs, walls)")
    g.add_argument("--d_safe_base", type=float, default=0.15,
                   help="minimum distance base footprint <-> environment [m] (LiDAR safety field; never relaxed)")
    g.add_argument("--d_influence_base", type=float, default=0.45,
                   help="base constraints active below [m]")
    g.add_argument("--d_safe_obstacle", type=float, default=0.02,
                   help="minimum distance arm <-> obstacles named obst_* [m], kept during the grasp")

    g = p.add_argument_group("safety stop (base + arm)")
    g.add_argument("--safety", choices=["auto", "em_stop", "lidar", "off"], default="auto",
                   help="em_stop: the robot's EM-stop system (buttons + scanner protective field), "
                        "required on the real robot | lidar: fallback without one (MuJoCo, Gazebo) | "
                        "auto: em_stop if its state is published, else lidar")
    g.add_argument("--em_stop_topic", default=EM_STOP_TOPIC, help="neo_msgs2/EmergencyStopState")
    g.add_argument("--lidar_topics", nargs="+", default=list(SCAN_TOPICS["real"]),
                   help="ROS, lidar fallback: LaserScan topics of both scanners")
    g.add_argument("--safety_clear_time", type=float, default=1.0, help="clear this long before resuming [s]")
    g.add_argument("--safety_stop_timeout", type=float, default=60.0, help="abort if stopped this long [s]")
    g.add_argument("--sim_intruder", type=float, nargs=4, default=None, metavar=("T_IN", "T_OUT", "X", "Y"),
                   help="MuJoCo test: a person stands at world (X, Y) from T_IN to T_OUT [s]")

    g = p.add_argument_group("limits / safety")
    g.add_argument("--max_base_v", type=float, default=0.20, help="[m/s] per axis")
    g.add_argument("--max_base_w", type=float, default=0.30, help="[rad/s]")
    g.add_argument("--max_arm_vel", type=float, default=0.50, help="[rad/s] per joint")
    g.add_argument("--max_base_acc", type=float, default=0.50, help="[m/s^2]")
    g.add_argument("--max_base_wacc", type=float, default=1.00, help="[rad/s^2]")
    g.add_argument("--max_arm_acc", type=float, default=2.00, help="[rad/s^2]")
    g.add_argument("--joint_margin", type=float, default=0.10, help="margin to joint limits [rad]")
    g.add_argument("--k_joint_limit", type=float, default=2.0, help="joint-limit velocity gain [1/s]")
    g.add_argument("--max_track_err", type=float, default=0.10, help="abort above this EE error [m]")
    g.add_argument("--min_ee_z", type=float, default=0.30, help="abort if EE goes below [m]")

    g = p.add_argument_group("simulation")
    g.add_argument("--headless", action="store_true")
    g.add_argument("--sim_speed", type=float, default=1.0, help="real-time factor with viewer")
    g.add_argument("--sim_base_tau", type=float, default=0.15, help="base velocity lag [s]")
    g.add_argument("--sim_latency_ticks", type=int, default=1, help="command delay [control ticks]")
    g.add_argument("--sim_q0", type=float, nargs=6, default=SIM_ARM_Q0, help="initial arm config")

    g = p.add_argument_group("ROS 2")
    g.add_argument("--joint_states_topic", default="/joint_states")
    g.add_argument("--odom_topic", default="/odom")
    g.add_argument("--cmd_vel_topic", default="/cmd_vel")
    g.add_argument("--arm_interface", choices=["trajectory", "velocity"], default="trajectory")
    g.add_argument("--arm_traj_topic", default="/scaled_joint_trajectory_controller/joint_trajectory")
    g.add_argument("--arm_vel_topic", default="/forward_velocity_controller/commands",
                   help="real robot: /arm_velocity_cmd (arm_velocity_watchdog), set by the launch file")
    g.add_argument("--switch_controllers", action="store_true",
                   help="velocity interface: activate --arm_vel_controller instead of "
                        "--arm_traj_controller for the run and switch back afterwards")
    g.add_argument("--arm_vel_controller", default="forward_velocity_controller")
    g.add_argument("--arm_traj_controller", default="scaled_joint_trajectory_controller")
    g.add_argument("--controller_manager", default="/controller_manager")
    g.add_argument("--traj_horizon", type=float, default=0.1, help="look-ahead of streamed traj point [s]")
    g.add_argument("--arm_cmd_tau", type=float, default=0.0,
                   help="low-pass time constant of the arm command [s] (0 = off; ~0.06 against joint buzzing)")
    g.add_argument("--max_cmd_lead", type=float, default=0.2,
                   help="max lead of the streamed arm setpoint over the measured joints [rad]")
    g.add_argument("--pan_offset", type=float, default=np.pi / 2,
                   help="q_model = q_ros + offset on shoulder_pan [rad]: the URDF mounts the UR5e "
                        "with yaw +90 deg on the cabinet (Gazebo + real robot), the MJCF with 0")
    g.add_argument("--base_feedback", choices=["odom", "open_loop"], default="odom")
    g.add_argument("--state_timeout", type=float, default=0.25, help="max age of feedback [s]")
    g.add_argument("--wait_timeout", type=float, default=10.0,
                   help="max wait for the first /joint_states + odom [s] (Gazebo needs longer)")
    return p


def split_ros_args(argv=None):
    """Separate our CLI args from the `--ros-args ...` that ros2 run/launch append."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in argv:
        i = argv.index("--ros-args")
        return argv[:i], argv[i:]
    return argv, []


def parse_args(argv=None, parser=None):
    own, ros = split_ros_args(argv)
    args = (parser or build_arg_parser()).parse_args(own)
    args.ros_args = ros
    if args.collision_env is None:
        args.collision_env = "none" if args.mode == "ros" else "model"
    return args


def resolve_model(path):
    cands = [path] if path else MODEL_CANDIDATES
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    sys.exit(f"[ERROR] mmo_700_wbc.xml not found. Tried: {[str(c) for c in cands]}")


def main(argv=None):
    args = parse_args(argv)
    dt = 1.0 / args.rate
    model_path = resolve_model(args.model)
    print(f"Model: {model_path}")
    args.table_model = TableModel.load(args.table) if args.table else None
    if args.table_model is not None:
        t = args.table_model
        if t.frame not in ("odom", "world"):
            sys.exit(f"[ERROR] table '{args.table}' is in frame '{t.frame}'; the WBC needs odom")
        args.collision_env = "model"   # the table is the (only) static obstacle of the model
        print(f"Table: {t.name}  centre {np.round(t.center, 3)} ({t.frame})  size {np.round(t.size, 2)} m  "
              f"top {t.top_z:.2f} m  {len(t.legs)} legs")
    model = load_model(model_path, intruder=args.sim_intruder is not None,
                       spec_edit=args.table_model.apply_to_spec if args.table_model is not None else None)
    kin = WholeBodyKinematics(model)

    if args.mode == "sim":
        backend = MujocoSimBackend(model, args, dt)
    elif args.mode == "kin":
        backend = KinematicBackend(RobotState(np.zeros(3), np.array(args.sim_q0, float)), dt)
    else:
        try:
            ros = RosBackend(args, dt)
        except ImportError:
            sys.exit("[ERROR] rclpy not found — source ROS 2 first: source /opt/ros/humble/setup.bash")
        backend = DryRunBackend(ros, dt) if args.dry_run else ros

    try:
        status = run(backend, kin, args)
    finally:
        backend.shutdown()
    return status


if __name__ == "__main__":
    sys.exit(main())
