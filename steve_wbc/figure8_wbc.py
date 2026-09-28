#!/usr/bin/env python3
"""
figure8_wbc.py
==============
MMO-700 + UR5e  •  mmo_700_wbc.xml

Whole-Body Control demonstration: drawing figure-8s (lemniscates) with a
UNIFIED velocity-level WBC. Every motion — driving, arm reconfiguration and
figure-8 tracking — is produced by one constrained QP over all 9 DOF
(base vx, vy, wz + 6 arm joints); nothing is commanded to the base or arm
outside of that solve. The controller core is shared with the real-robot
script (steve_figure8_wbc.py), see its docstring for the full formulation:

    min_u  || J_wb u - v_ee ||^2_{W_t}                 EE pose task      (primary)
         + w_b || u_base - u_base_des ||^2             base-pose task    (soft, optional)
         + || u - N (K (q_nom - q)) ||^2_{W_u}          null-space posture + effort
    s.t.   velocity / acceleration / joint-position limits
           collision avoidance: arm/base vs table + floor, arm vs own body
           (velocity dampers, hard constraints — see CollisionAvoidance)

The original decoupled version (scripted base + arm-only IK) is kept in
porta_ros2_ws/scripts/figure8_wbc_decoupled.py for comparison.

PHASE 1 — HORIZONTAL figure-8 ON THE TABLE, base orbiting the table
    a) arm to drive pose              (posture task, base holds)
    b) base drives to orbit start     (base-pose task, EE free)
    c) arm to table-reach pose        (posture task, base holds)
    d) EE draws the lemniscate on the table top, wrist down (primary task),
       while the base follows a ±60° orbit around the table facing its centre
       (SOFT base-pose task). Conflicts are resolved by the QP in favour of
       the EE; the orbit is advanced by the figure-8 phase so both start and
       stop together.

PHASE 2 — VERTICAL figure-8 IN THE FRONTAL PLANE
    a) arm to forward-pointing pose   (posture task; base backs off the table)
    b) EE draws an upright lemniscate, wrist forward. No base trajectory is
       scripted: the QP decides how base and arm share the motion, and the
       null-space posture task hands sustained lateral motion to the base.

Run  (MuJoCo only; on macOS use `mjpython` for the viewer)
---
  ros2 launch steve_wbc figure8_wbc_demo.launch.py [headless:=true] [phase:=1|2|all] [loops:=N]
  ros2 run steve_wbc figure8_wbc_demo [--headless] [--phase1_only | --phase2_only]
                                      [--loops N] [--scale S] [--speed S] [--log out.csv]
  ros2 run steve_wbc figure8_wbc_demo --help      # all WBC weights / limits
  python3 figure8_wbc.py --headless --phase1_only    # without ROS (macOS: mjpython for the viewer)
"""

import copy
import sys
import time

import mujoco
import numpy as np

try:
    from steve_wbc.steve_figure8_wbc import (
        NU, BaseReference, Figure8Reference, KinematicBackend, MujocoSimBackend,
        RobotState, WholeBodyController, WholeBodyKinematics, build_arg_parser, load_model,
        make_safety, parse_args, resolve_model, rot_z, summarize,
    )
except ImportError:  # run as a plain script next to steve_figure8_wbc.py
    from steve_figure8_wbc import (
        NU, BaseReference, Figure8Reference, KinematicBackend, MujocoSimBackend,
        RobotState, WholeBodyController, WholeBodyKinematics, build_arg_parser, load_model,
        make_safety, parse_args, resolve_model, rot_z, summarize,
    )

# ── Table geometry (from mmo_700_wbc.xml) ────────────────────────────────────
TABLE_X, TABLE_Y = 2.0, 0.0
TABLE_Z_TOP = 0.80
# Pinch point 6 cm above the table top: the finger pads reach ~2.5 cm below the
# pinch point (pads touched the table at 2.5 cm), leaving > d_safe of clearance.
EE_TABLE_Z = TABLE_Z_TOP + 0.06

# ── Phase 1: table figure-8 + orbit ─────────────────────────────────────────
H8_AMP_X, H8_AMP_Y = 0.10, 0.18           # depth / lateral amplitude [m]
H8_OMEGA = 0.40                           # [rad/s]
ORBIT_R = 0.85                            # base distance from table centre [m]
ORBIT_PHI_MAX = np.pi / 3.0               # ±60° arc
ORBIT_PHASE_FAC = 0.5                     # orbit phase = FAC * figure-8 phase

# ── Phase 2: vertical frontal figure-8 ──────────────────────────────────────
V8_AMPLITUDE = 0.22
V8_OMEGA = 0.45
V8_CZ = 1.05

# ── Arm poses ────────────────────────────────────────────────────────────────
DRIVE_POSE = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
ARM_HOME_Q = [0.0, -1.0, 1.5, -2.05, -1.57, 0.0]      # table reach, wrist down
# Raised, gripper pointing along base +X (R_FORWARD) at z ≈ 1.05 m, wrist_2 = +pi/2
# (well clear of the wrist singularity wrist_2 = 0). Found by IK on the model;
# the old ARM_REACH_Q [0,-1.2,1.2,-1.6,-1.57,0] actually points the gripper DOWN,
# so demanding R_FORWARD from it drove the wrist through the singularity.
ARM_FORWARD_Q = [0.0, -1.558, 2.075, -0.516, 1.571, 0.0]
BASE_CLEAR_X = TABLE_X - 0.40 - 0.95      # base x that keeps the forward EE clear of the table

R_TOPDOWN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
R_FORWARD = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)

COL_P1 = (0.15, 0.85, 0.15, 0.55)
COL_P2 = (0.20, 0.60, 1.00, 0.55)
COL_ORBIT = (0.9, 0.9, 0.9, 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# Base references
# ══════════════════════════════════════════════════════════════════════════════
def hold_base(pose):
    pose = np.asarray(pose, float).copy()
    return lambda t: BaseReference(pose, np.zeros(3))


def min_jerk_base(start, goal, T):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    d = goal - start

    def ref(t):
        tau = min(t / T, 1.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T
        return BaseReference(start + s * d, ds * d)
    return ref


def orbit_pose(psi, dpsi):
    """Base on a ±ORBIT_PHI_MAX arc of radius ORBIT_R around the table, facing it."""
    a = ORBIT_PHI_MAX * np.sin(psi)
    da = ORBIT_PHI_MAX * np.cos(psi) * dpsi
    pose = np.array([TABLE_X - ORBIT_R * np.cos(a), TABLE_Y - ORBIT_R * np.sin(a), a])
    vel = np.array([ORBIT_R * np.sin(a) * da, -ORBIT_R * np.cos(a) * da, da])
    return BaseReference(pose, vel)


# ══════════════════════════════════════════════════════════════════════════════
# Segment runner — one WBC solve per control tick for every segment
# ══════════════════════════════════════════════════════════════════════════════
class Runner:
    def __init__(self, backend, kin, args):
        self.backend, self.kin, self.args = backend, kin, args
        self.dt = 1.0 / args.rate
        st0 = backend.read_state()
        self.ctrl = WholeBodyController(kin, args, q_nom=st0.q, yaw_nom=st0.base[2])
        self.u_prev = np.zeros(NU)
        self.sim = isinstance(backend, MujocoSimBackend)
        self.safety = make_safety(backend, args)

    def state(self):
        return self.backend.read_state()

    def ee_pose(self):
        self.kin.set_state(self.state())
        return self.kin.ee_pose()

    def segment(self, name, q_nom, base_ref=None, ee_ref=None, timeout=60.0, settle_tol=None,
                duration=None):
        """
        Run one segment. Ends when the EE reference is done, after `duration`,
        when the arm has settled to q_nom (settle_tol [rad], not before
        `duration`), or on timeout. Returns (ok, rows).
        """
        self.ctrl.q_nom = np.asarray(q_nom, float)
        self.ctrl.yaw_nom = self.state().base[2]
        t0 = self.backend.now()
        paused0 = self.safety.paused if self.safety else 0.0
        rows, next_print = [], 0.0
        print(f"\n  ▶ {name}")
        while self.backend.ok():
            st = self.state()
            if self.safety is not None:
                s = self.safety.check(self.backend.now())
                if s == "abort":
                    return False, rows
                if s == "stop":  # zero command sent; the segment clock is paused
                    self.u_prev = np.zeros(NU)
                    self.backend.tick()
                    continue
            t = self.backend.now() - t0 - ((self.safety.paused if self.safety else 0.0) - paused0)
            ref = ee_ref(t) if ee_ref else None
            if ref is not None and ref.done:
                break
            if duration is not None and t >= duration and settle_tol is None:
                break
            if settle_tol is not None and t > max(0.5, duration or 0.0) and np.max(np.abs(self.ctrl.q_nom - st.q)) < settle_tol \
                    and np.max(np.abs(self.u_prev)) < 0.02:
                break
            if t > timeout:
                if ee_ref is None:
                    print(f"    [WARN] {name}: timeout after {timeout:.0f}s")
                    break
            bref = base_ref(t) if base_ref else None

            tic = time.perf_counter()
            u, info = self.ctrl.step(st, ref, self.u_prev, self.dt, bref)
            solve_ms = (time.perf_counter() - tic) * 1e3

            if not np.all(np.isfinite(u)):
                print("    [ABORT] non-finite command")
                return False, rows
            if ref is not None:
                err = np.linalg.norm(info["e_p"])
                if err > self.args.max_track_err:
                    print(f"    [ABORT] tracking error {err * 100:.1f} cm")
                    return False, rows
                rows.append([t, *ref.p, *info["p"], err, np.linalg.norm(info["e_o"]),
                             *u, *st.base, *st.q, solve_ms, info["d_min"]])
                if self.sim:
                    self.backend.draw_live(ref.p, info["p"])

            if t >= next_print:
                next_print += 1.0
                ee = info["p"]
                err_s = f"err={np.linalg.norm(info['e_p']) * 100:5.2f}cm " if ref is not None else ""
                if np.isfinite(info["d_min"]):
                    err_s += f"clr={info['d_min'] * 100:4.1f}cm ({info['closest']}) "
                print(f"    t={t:5.1f}s {err_s}base=({st.base[0]:+.3f},{st.base[1]:+.3f},"
                      f"{np.degrees(st.base[2]):+5.1f}°) EE=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
                      f"u_b=({u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f}) |dq|={np.abs(u[3:]).max():.2f}")

            self.backend.send(u)
            self.u_prev = u
            self.backend.tick()
        return self.backend.ok(), rows

    def stop(self):
        self.backend.stop()
        self.u_prev = np.zeros(NU)


def report(name, rows, args):
    if not rows:
        return
    a = copy.copy(args)
    if args.log:
        stem, dot, ext = args.log.rpartition(".")
        a.log = f"{stem}_{name}.{ext}" if dot else f"{args.log}_{name}"
    print(f"\n  ✓ {name} complete")
    summarize(rows, a)


# ══════════════════════════════════════════════════════════════════════════════
def phase1(run: Runner, args):
    s, sp = args.scale, args.speed
    print(f"\n{'─' * 74}\n  PHASE 1 — table figure-8, base orbiting ±{np.degrees(ORBIT_PHI_MAX):.0f}°"
          f" (soft base task)\n{'─' * 74}")
    base0 = run.state().base

    ok, _ = run.segment("arm to drive pose", DRIVE_POSE, hold_base(base0), settle_tol=0.01, timeout=20)
    if not ok:
        return False

    orbit_start = orbit_pose(0.0, 0.0).pose
    dist = np.linalg.norm(orbit_start[:2] - base0[:2])
    T_drive = max(3.0, 1.875 * dist / (0.8 * args.max_base_v))  # min-jerk peak = 1.875 * mean
    ok, _ = run.segment(f"drive to orbit start {np.round(orbit_start, 2)} ({T_drive:.1f}s)",
                        DRIVE_POSE, min_jerk_base(base0, orbit_start, T_drive), duration=T_drive)
    if not ok:
        return False

    ok, _ = run.segment("arm to table-reach pose", ARM_HOME_Q, hold_base(run.state().base),
                        settle_tol=0.01, timeout=20)
    if not ok:
        return False

    p0, R0 = run.ee_pose()
    ref = Figure8Reference(p0, R0, center=[TABLE_X, TABLE_Y, EE_TABLE_Z],
                           a_hat=[1, 0, 0], b_hat=[0, 1, 0],
                           amp_a=H8_AMP_X * s, amp_b=H8_AMP_Y * s,
                           omega=H8_OMEGA * sp, loops=args.loops,
                           ramp_time=args.ramp_time, R_goal=R_TOPDOWN)

    def orbit(t):
        phi, dphi = ref.phase_at(t)
        return orbit_pose(ORBIT_PHASE_FAC * phi, ORBIT_PHASE_FAC * dphi)

    if run.sim:
        run.backend.draw_reference(ref.sample_path(), COL_P1)
        arc = [np.append(orbit_pose(np.arcsin(x), 0).pose[:2], 0.03) for x in np.linspace(-1, 1, 61)]
        run.backend.draw_reference(arc, COL_ORBIT, radius=0.015, append=True)
        if run.backend.viewer:
            run.backend.viewer.cam.lookat[:] = [1.6, 0.0, 0.8]
            run.backend.viewer.cam.distance, run.backend.viewer.cam.elevation = 4.5, -35
            run.backend.viewer.cam.azimuth = 200

    ok, rows = run.segment(f"table figure-8 ({args.loops} loops, {ref.T_total:.1f}s)",
                           ARM_HOME_Q, base_ref=orbit, ee_ref=ref)
    run.stop()
    report("phase1", rows, args)
    return ok


def phase2(run: Runner, args):
    s, sp = args.scale, args.speed
    print(f"\n{'─' * 74}\n  PHASE 2 — vertical frontal figure-8 (base/arm split decided by the QP)"
          f"\n{'─' * 74}")
    base = run.state().base
    goal = np.array([min(base[0], BASE_CLEAR_X), base[1], 0.0])
    T_move = max(3.0, 1.875 * np.linalg.norm(goal[:2] - base[:2]) / (0.8 * args.max_base_v))
    ok, _ = run.segment("arm to forward pose, base clear of the table", ARM_FORWARD_Q,
                        min_jerk_base(base, goal, T_move), settle_tol=0.01,
                        duration=T_move, timeout=T_move + 20)
    if not ok:
        return False

    st = run.state()
    yaw = st.base[2]
    p0, R0 = run.ee_pose()
    lat = rot_z(yaw)[:, 1]
    ref = Figure8Reference(p0, R0, center=[p0[0], p0[1], args.cz],
                           a_hat=lat, b_hat=[0, 0, 1],
                           amp_a=args.amplitude * s, amp_b=args.amplitude * s,
                           omega=args.frequency * sp, loops=args.loops,
                           ramp_time=args.ramp_time, R_goal=rot_z(yaw) @ R_FORWARD)
    if run.sim:
        run.backend.draw_reference(ref.sample_path(), COL_P2)
        if run.backend.viewer:
            run.backend.viewer.cam.lookat[:] = ref.center
            run.backend.viewer.cam.distance, run.backend.viewer.cam.elevation = 3.5, -5
            run.backend.viewer.cam.azimuth = 150

    ok, rows = run.segment(f"vertical figure-8 ({args.loops} loops, {ref.T_total:.1f}s)",
                           ARM_FORWARD_Q, ee_ref=ref)
    run.stop()
    report("phase2", rows, args)
    return ok


def main(argv=None):
    p = build_arg_parser("MMO-700 figure-8 demo — unified velocity-level WBC (MuJoCo)")
    g = p.add_argument_group("demo")
    g.add_argument("--phase1_only", action="store_true", help="only the table figure-8 + orbit")
    g.add_argument("--phase2_only", action="store_true", help="only the vertical figure-8")
    g.add_argument("--scale", type=float, default=1.0, help="amplitude multiplier")
    g.add_argument("--speed", type=float, default=1.0, help="frequency multiplier")
    p.set_defaults(amplitude=V8_AMPLITUDE, frequency=V8_OMEGA, cz=V8_CZ,
                   max_base_v=0.30, max_base_w=0.50, sim_q0=DRIVE_POSE)
    args = parse_args(argv, p)
    if args.mode == "ros":
        sys.exit("figure8_wbc.py is the MuJoCo demo — use steve_figure8_wbc.py --mode ros on the robot")

    model_path = resolve_model(args.model)
    model = load_model(model_path, intruder=args.sim_intruder is not None)
    kin = WholeBodyKinematics(model)
    dt = 1.0 / args.rate
    if args.mode == "sim":
        backend = MujocoSimBackend(model, args, dt)
        if backend.viewer:
            backend.viewer.cam.lookat[:] = [1.0, 0.0, 0.9]
            backend.viewer.cam.distance, backend.viewer.cam.elevation = 5.5, -20
            backend.viewer.cam.azimuth = 30
    else:
        backend = KinematicBackend(RobotState(np.zeros(3), np.array(args.sim_q0, float)), dt)

    print(f"\n{'═' * 74}\n  MMO-700 + UR5e — Figure-8 demo, unified velocity-level WBC  [{args.mode}]"
          f"\n{'═' * 74}\n  Model: {model_path}\n  Loops per phase: {args.loops}   "
          f"scale {args.scale}   speed {args.speed}")

    run = Runner(backend, kin, args)
    ok = True
    try:
        if not args.phase2_only:
            ok = phase1(run, args)
        if ok and not args.phase1_only:
            ok = phase2(run, args)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C")
        ok = False
    finally:
        run.stop()

    if run.safety is not None:
        print(run.safety.summary())
    print(f"\n{'═' * 74}\n  {'✅ Demo complete' if ok else '⚠ Demo stopped early'}\n{'═' * 74}\n")
    backend.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
