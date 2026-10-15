# steve_wbc

Unified velocity-level (kinematic) **whole-body controller** for Steve
(Neobotix MMO-700 omnidirectional base + UR5e) over all 9 DOF —
`u = [vx_b, vy_b, wz_b, dq1..dq6]` — as a **strict task hierarchy** (default,
`--hierarchy strict`): one constrained least-squares problem per priority level, each level
restricted to the solutions that keep what the levels above achieved:

```
1. EE pose     min || J_wb u - v_ee ||²_Wt
2. base pose   min || u_base - u_base_des ||²           (only when a base task is set)
3. posture     min || u - K(q_nom - q) ||²_Wu          (+ effort; base/arm split via W_u)
every level s.t.  velocity / acceleration / joint-position limits
                  collision avoidance (arm vs table, obstacles, floor, own body; base vs environment)
                  | A_i u - A_i u_i | <= hqp_tol   for all higher levels i
```

A lower task can never degrade a higher one, also when limits or collision constraints are
active (a level that cannot be solved is dropped, the higher levels' result kept). Weights act
only within a level. `--hierarchy weighted` keeps the previous single QP
(`|| J u - v_ee ||²_Wt + w_b || u_base - u_base_des ||² + || u - N·K(q_nom - q) ||²_Wu`).

| Kinematic check (`figure8_wbc_demo --mode kin`) | weighted | strict |
|---|---|---|
| Phase 1 — table figure-8 while the base orbits (EE vs base task compete) | 0.55 / 1.90 cm, 1.30° | **0.01 / 0.03 cm, 0.00°** |
| Phase 2 — vertical figure-8 (nothing competes) | 0.03 / 0.04 cm | 0.02 / 0.04 cm |

(EE position RMS / max, orientation max; mean QP time ~10–13 ms for both.)

`J_wb` is the exact whole-body Jacobian from the MuJoCo model (`models/mmo_700_wbc.xml`),
including the lever arm of base yaw. The same controller runs in MuJoCo, Gazebo and on
the real robot.

| Executable | Purpose |
|---|---|
| `steve_figure8_wbc` | Figure-8 with the EE — MuJoCo (`--mode sim`), kinematic check (`--mode kin`), Gazebo / real robot over ROS 2 (`--mode ros`) |
| `figure8_wbc_demo` | MuJoCo-only demo: figure-8 on the table while the base orbits it, then a vertical figure-8 |
| `active_perception_wbc` | Drive (LiDAR) → perceive the object from t = 0 → (VGN scan) → **MPC interception** of its predicted path → grasp → lift → hold; every phase solved by the same 9-DOF joint-velocity WBC |
| `red_ball_detector` | ROS 2 target detector (RGB-D cameras, parameter `cameras`: head + wrist, or `[gripper_camera]`) → `/steve_perception/target_point`, uses a target profile |
| `target_calibrator` | record a target object's colour profile with the real (or simulated) camera |
| `table_probe` | measure the real table (LiDAR + head depth, or `--lidar_only --top_z`) into a table model |
| `gripper_camera_calibrator` | eye-in-hand calibration of the gripper camera's mount (`ur5etool0` → camera) with the ball on the table, no marker → `config/gripper_camera.yaml` |
| `vgn_grasp_real` | one VGN grasp from the current camera view → `grasp.json` for `steve_figure8_wbc --goto_file` (table measured in the view, or `--table`) |

## Active perception + MPC interception (`active_perception_wbc`)

* **Perception from the first tick**: head (pan-tilt) camera while driving, wrist + head camera
  afterwards, detecting the object described by a **target profile** (`--target`, see
  [Sim2real](#sim2real-real-robot-setup)). The object centre is taken from the 3-D points of its
  **top** (top face of a box, cap of a sphere) — symmetric about the centre from any viewpoint,
  unlike the blob centroid. Detections feed a gated least-squares constant-velocity tracker: the
  position uses every detection, the velocity only detections taken while the arm holds the
  camera still (a swinging camera must not create fake motion); a velocity must be significant
  and move the object ≥ 4 cm over the window, otherwise the object counts as static.
* **`InterceptionMPC`**: receding-horizon (1.5 s, 0.1 s steps) double-integrator plan per axis,
  minimising distance to the *predicted* object path + velocity mismatch + acceleration,
  |a| ≤ a_max, solved exactly as bounded least squares. The gripper arrives above the object
  moving with it, descends and grasps; during descent, grasp and lift the base holds still (soft).
* **Hold check**: after the lift the object is held for `--hold_time` (3 s); success only if it
  is still in the gripper (catches slipping grasps).
* **Approach side by the object's place on the table** (seen from the side the robot arrives on,
  `QUADRANT_SIDE`): **near (bottom) half → no repositioning**, grasp from where it is; far (top)
  half split in thirds across: **left → the robot's left side, right → its right side, centre →
  the opposite (back) side** — for a static object; a moving one is intercepted from where the robot
  is if reachable (it would roll on while the robot drives around). The choice is verified by IK +
  collision checks; only if that side
  cannot reach the grasp are the others tried (here first, then by path length). A blocked or
  stalled grasp, or VGN finding no grasp, replans to another side. Of the two equivalent top-down
  grasps of the parallel gripper (180° apart) the one that suits the base heading on that side is
  used (from the back side the wrist would otherwise turn 180° and the WBC slid the base).
* **Driving around the table** (`TableOrbit`): one smooth path, not corner-to-corner — a ring at a
  distance on which the base clears the table at any heading, with straight sides joined by
  circular arcs, blended smoothly from the start's distance onto the ring and down to the goal's
  (never closer to the table than start or goal); the base faces the table all the way (the omni
  base moves sideways and turns through the corners), min-jerk speed profile (peak 0.25 m/s or
  `--max_base_v`), tracked by the WBC's base task as pose + velocity feed-forward.
* **Obstacles** (`--scene clutter`): only the target is picked; the gripper flies over obstacles in
  its path (transit height) and the arm keeps ≥ 2 cm from them (hard constraints, not relaxed in
  the final descent). Any robot–obstacle contact fails the run.
* **VGN** (MuJoCo and robot): TSDF views + inference in the `vgn_env` conda env (run isolated from
  `~/.local` and the ROS `PYTHONPATH`); VGN itself (bridge, worker, weights) lives in
  `steve_perception.vgn` — see its README for the `vgn_env` setup. Only grasps **on the target** are used (within its size
  of the tracked position). On the robot the views come from the camera with aligned depth (found
  automatically, gripper camera first, `--vgn_camera`) with its pose from TF (`odom` ← the depth
  image's frame, e.g. the calibrated `gripper_camera_color_optical_frame`); the image topic is subscribed only
  while views are taken, a view is a frame that arrives after the arm stopped, and with `--table` each
  view's pose is **levelled on the table plane** (tilt + height of the camera mount; `--vgn_no_level`,
  `--vgn_max_level` 8°). VGN runs on the PC that has `vgn_env` (see [Real robot](#real-robot-gripper-camera-on-the-pc)).
  Three timing modes, `--vgn_mode` / separate launch files:

  | Mode | Launch file | Views | Inference | Robot waits |
  |---|---|---|---|---|
  | `scan` (original) | `active_perception_wbc.launch.py` | stopped 4-view pan scan | blocking | scan + inference |
  | `async` | `active_perception_wbc_vgn_async.launch.py` | stopped 4-view pan scan | background | scan only |
  | `drive` | `active_perception_wbc_vgn_drive.launch.py` | while driving up (motion-compensated) | background, starts while driving | ~0 |

  Inference time is charged 1:1 to simulation time, so the modes compare fairly
  (`python3 test/compare_vgn_modes.py`). A result arriving after the descent started is ignored;
  `--vgn_max_wait` bounds a hover at the pre-grasp. On the robot the scan's inference always runs in
  the background (the robot holds in `VGN_WAIT`; a blocking call stalls the state feedback), and VGN
  runs in a **persistent worker** started with the WBC (~3 s while it drives; then ~0.1 s per inference
  instead of ~4–7 s for a new process).

  **Fastest** (2026-10-15, sim with a static ball: robot waits 10.3 s with 4 views → 3.6 s with 1 →
  1.2 s with `drive` + 1 view): `vgn_mode:=drive extra_args:="... --vgn_views 1 --vgn_drive_range 1.2"`
  — inference starts at the first good view while the robot still drives (`--vgn_views`: views that
  start it, default 6 in `drive`, 4 in `scan`; `--vgn_drive_range`: base–object distance for drive views,
  default 2.3 m, ~1.2 m for the D415's depth). No drive view (arm not yet in scan pose): a stopped
  1-view scan instead.
* **Go to the grasp pose only** (`--goto_grasp_only`, launch `goto_grasp_only:=true`; no gripper on
  the arm): after the descent the EE stays at the grasp pose with the object (static or moving) for
  `--hold_time`, never closes a gripper or lifts, then backs up to the pre-grasp height
  (`AT_GRASP → RETREAT → DONE`, result `reached grasp pose`, with the mean / max EE distance from the
  grasp point). `--grasp_dz 0.05` hovers 5 cm higher.
* **Camera tilt** (`--camera_tilt DEG`): a camera looking along the flange axis sees only the
  floor / table straight below in the drive and scan poses (the MJCF wrist camera is mounted tilted);
  the tilt turns wrist 1 so it looks forward-down. With the gripper camera ~25° covers the table from
  ~0.65 to ~1.3 m ahead of the base centre (0°: 0.38–0.80 m). Keep 0 in MuJoCo (the tilted arm
  blocks the simulated head camera).
* **Exploration when VGN finds no grasp** (`--vgn_explore`, default `fuse`, static objects): the
  side is marked as tried, the base drives around the table (the same ring path as a replan) to the
  next reachable side, the arm scans again (`--vgn_views` per side) and VGN runs on **all sides' views
  fused in one TSDF** (`fresh`: a new TSDF per side). Views of later sides are shifted by the tracked
  object's offset from the TSDF centre, which compensates the odometry drift of the drive around the
  table (camera pose and detections share it). The robot never descends without a VGN grasp (no
  `--vgn_max_wait` give-up). After `--vgn_explore_max_sides` (4) sides, or when no untried side is
  reachable: **`Result: no grasp candidate found`** — the robot stops, nothing is grasped.
  `off`, or a moving object: the old behaviour (top-down at the tracked position).

  ```
  [VGN_WAIT → REPOSITION] VGN: no grasp candidate on the target from the front side; ... -> left side (2.2 m)
  [SETTLE_ARM → VGN_SCAN] ... 4 TSDF views around pan=0.01 rad (+4 fused from the front side(s))
    VGN grasp ... score=1.00 ...
    Result: success   Explored: VGN found no grasp from the front side(s), grasped from the left side
  ```

```bash
ros2 launch steve_wbc active_perception_wbc.launch.py                          # MuJoCo, rolling ball
ros2 launch steve_wbc active_perception_wbc.launch.py scene:=clutter           # cube among obstacles
ros2 launch steve_wbc active_perception_wbc.launch.py vgn:=false               # no VGN
ros2 launch steve_wbc active_perception_wbc_vgn_drive.launch.py scene:=clutter
ros2 run steve_wbc active_perception_wbc --mode sim --object_type box --object_speed 0 --target_xy 2.2 0.2   # top-left quadrant
ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false dry_run:=true table:=lab_table target:=my_ball
```

### Real robot: gripper camera on the PC

Setup as tested (2026-10-14): **no gripper**; a D415 hangs ~5 cm below the flange on a short post,
looking along the flange axis, its USB cable to the **PC**. The robot runs its bringup
(`ROS_AUTOSTART.sh`: base, UR driver, LiDARs — no pan-tilt camera); the PC runs the camera driver,
the detector and `active_perception_wbc` (+ VGN in `vgn_env`). The arm is commanded with velocity
commands through the bringup's `arm_velocity_watchdog` (launch default on the robot; streamed
trajectories made it jerky). Over the lab Wi-Fi the robot state arrives at 544 Hz (joint states) /
102 Hz (odom) with gaps < 40 ms; Ethernet is still the better link for the control loop.

Every PC terminal: `export ROS_DOMAIN_ID=74 RMW_IMPLEMENTATION=rmw_fastrtps_cpp
FASTRTPS_DEFAULT_PROFILES_FILE=~/.ros/fastdds_steve.xml` — the PC profile must also list
`127.0.0.1` as discovery peer, else the PC's own nodes (camera, RViz, WBC) never find each other
(see [`steve_hardware_bringup`](../steve_hardware_bringup/README.md#remote-pc-rviz-etc-over-wi-fi)).

| Step | Command (PC) | Result |
|---|---|---|
| Camera | `ros2 launch steve_wbc gripper_camera.launch.py` | `/gripper_camera/{color,aligned_depth_to_color}/...` (848x480 @ 30, D415 min. range ~0.31 m), + mount TF once calibrated |
| Ball profile (once) | `ros2 run steve_wbc target_calibrator --name my_ball --size 0.065 --topic /gripper_camera/color/image_raw` | `config/targets/my_ball.yaml` |
| Camera mount (once per mounting) | `ros2 run steve_wbc gripper_camera_calibrator --target my_ball`; ball on the table, base still, 8 arm poses (`manipulator --direct joints ...`), Enter at each, `s` | `config/gripper_camera.yaml`: `ur5etool0` → `gripper_camera_color_optical_frame` |
| Table (after every bringup restart: `odom` resets) | robot 0.3–1.5 m in front of the table: `ros2 run steve_wbc table_probe --name lab_table --lidar_only --top_z 0.827` | `config/tables/lab_table.yaml` (odom) |
| Detector check | `ros2 run steve_wbc red_ball_detector --ros-args -p target:=my_ball -p topic_style:=realsense -p "cameras:=[gripper_camera]" -p "head_topic:=''"` | `/steve_perception/target_point` (odom) |
| Dry run (robot does not move) | `ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false launch_bringup:=false dry_run:=true table:=lab_table target:=my_ball detector_cameras:=gripper_camera vgn:=false goto_grasp_only:=true extra_args:="--camera_tilt 25"` | `... → AT_GRASP → RETREAT → DONE`, `reached grasp pose` |
| Real run (robot ~1 m back, arm `home`) | the same without `dry_run:=true`, `extra_args:="--camera_tilt 25 --max_base_v 0.10 --max_arm_vel 0.3 --grasp_dz 0.05 --state_timeout 0.5"` (`--state_timeout 0.5`: Wi-Fi) | drives up, finds the ball, hovers 5 cm above the grasp pose, backs up |

Then `vgn:=true` (VGN views from the gripper camera, levelled on the table plane; no grasp → other
sides, see above) and a slowly rolling ball. Without a gripper keep `--grasp_dz` ≥ 0.05: at the true
grasp pose (pinch point 16 cm beyond the flange) the camera would end ~1 cm above the ball.

Calibration of 2026-10-14: ball RMS 0.17 cm over 8 poses, table level within 0.3°; mount xyz
(0.037, −0.005, 0.048) m, rpy (0.6, 0.2, 179.3)° (±4 mm, ±0.6°); table top 0.827 m.
The 8 poses look at a ball 27 cm inside the table's near edge from 0.36–0.56 m and keep the flange
≥ 1.14 m high over the table — `table_reach` puts it only 0.20 m above the table: not with a tool
mounted.

**Clutter: Grounded SAM detector and Contact-GraspNet** (`steve_perception`, model servers in their
own conda envs, see [`model_server/README.md`](../steve_perception/steve_perception/model_server/README.md)):

| Launch argument | Values |
|---|---|
| `detector` | `hsv` (default; colour threshold of the target profile) \| `gsam` (GroundingDINO + SAM with `gsam_prompt`, sphere fit with the profile's size, ~4.5 Hz) \| `false` |
| `grasp_backend` | `vgn` (default; one TSDF of the scan views) \| `cgn` (Contact-GraspNet per view on the target's Grounded-SAM mask; the rest of the view is collision context) |
| `gsam_prompt` | text naming the target, default `red ball` |
| `save_images` | `true` (default): detections → `detector/`, every grasp scan's views with mask + grasps → `grasp_scan/` in the run's log folder (`~/ros_logs/latest`) \| `false` |
| `model_servers` | `auto` (start the servers `gsam` / `cgn` need) \| `false` (already running: `ros2 run steve_perception model_server gsam\|cgn`) |

```bash
ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false launch_bringup:=false \
    table:=lab_table target:=my_ball detector_cameras:=gripper_camera vgn:=true goto_grasp_only:=true \
    detector:=gsam grasp_backend:=cgn gsam_prompt:="red ball" \
    extra_args:="--camera_tilt 25 --max_base_v 0.10 --max_arm_vel 0.3 --grasp_dz 0.05 --state_timeout 0.5"
```

As with VGN, the scan's grasp decides go / explore another side (`--vgn_explore`) and, with
`--vgn_orientation yaw`, the gripper yaw; the approach stays top-down at the tracked position.

**`vgn_grasp_real`** (one grasp for `steve_figure8_wbc --goto_file`): camera found automatically
(gripper first), the table top measured in the view (dominant horizontal plane, saved as table model
`seen_table` for the WBC's collision checks; `--table` for a model), VGN's 30 cm TSDF placed as in its
training scenes (table plane 5 cm above the volume floor).

Without ROS (e.g. macOS): all scripts are also in `steve_ros2_ws/scripts/`
(`mjpython active_perception_wbc.py`, `mjpython figure8_wbc.py`, `mjpython steve_figure8_wbc.py --mode sim`).

## Build

```bash
pip install mujoco          # not available via rosdep; numpy/scipy come from rosdep
colcon build --symlink-install --packages-select steve_manipulation steve_wbc
source install/setup.bash
```

## Launch — simulation vs. physical robot

Same convention as `steve_navigation`: `use_sim_time:=true` → simulation, `false` → robot.

```bash
# 1. MuJoCo (no ROS robot needed)
ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=true simulator:=mujoco

# 2. Gazebo (starts steve_simulation, WBC talks to /cmd_vel, /odom, joint_trajectory_controller)
ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=true

# 3. Physical robot — dry run: reads the real state in real time, publishes nothing
ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false dry_run:=true

# 4. Physical robot — small and slow first
ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false \
    amplitude:=0.05 frequency:=0.2 loops:=1 \
    extra_args:="--max_base_v 0.05 --max_base_w 0.1 --max_arm_vel 0.3 --lock_yaw"
```

`launch_bringup:=auto` (default): on the robot the drivers are started only if no bringup is
running — `ROS_AUTOSTART.sh` already runs `steve_hardware_bringup`, and a second UR driver /
`/controller_manager` would conflict. `false` never starts them. Other controller options go
through `extra_args` (see `ros2 run steve_wbc steve_figure8_wbc --help`).

On the real robot, start the *External Control* program on the UR pendant; the figure-8 is
centred on the gripper's position at launch (start with the arm raised and free).
In Gazebo the figure-8 centre defaults to 1.1 m (the arm spawns sagged against the cabinet).

MuJoCo demo (simulation only):

```bash
ros2 launch steve_wbc figure8_wbc_demo.launch.py                 # both phases
ros2 launch steve_wbc figure8_wbc_demo.launch.py phase:=1 loops:=2
```

## Sim2real: real-robot setup

| Step | Tool | Result |
|---|---|---|
| 1. Network, USB devices, camera serials | `ros2 run steve_hardware_bringup steve_doctor ` | checks `steve_hardware_bringup/config/robot_config.yaml` against the hardware ([details](../steve_hardware_bringup/README.md)) |
| 2. Target object | `ros2 run steve_wbc target_calibrator --name my_ball --size 0.06` | `config/targets/my_ball.yaml` |
| 3. Table | `ros2 run steve_wbc table_probe --name lab_table` | `config/tables/lab_table.yaml` + debug plot |
| 4. Try it in MuJoCo with the same table | `ros2 launch steve_wbc active_perception_wbc.launch.py table:=lab_table` | |
| 5. Dry run, then real | `... use_sim_time:=false dry_run:=true table:=lab_table target:=my_ball` | |

**Target profiles** (`target_profile.py`, `config/targets/*.yaml`): HSV ranges + size + shape of
the object to pick. `target_calibrator` shows the live camera image; press `c` and box the object
(GrabCut separates it from the background), repeat from several angles / lighting, `s` saves.
The ranges are fitted **discriminatively**: as many object pixels as possible while excluding the
background of the same frames (e.g. a wooden table with a similar hue). The live overlay shows
detections (green) and background false positives (magenta); with depth, the measured object
size is shown next to the profile's. `--check <name>` verifies a stored profile, `--sim` /
`--image` work without a robot. The default `red` profile was recorded from the MuJoCo camera.
*Better long term*: for objects without a distinct colour, a text-prompted segmenter (the SAM3
detector in `steve_perception`) can replace the colour detector behind the same interface.

**Table models** (`table_model.py`, `config/tables/*.yaml`): the simulation's table is a closed
box (top + skirt to the floor); real tables usually stand on legs, and the LiDARs (at 0.22 m)
never see the top plate. `table_probe` fits the top plate from the head camera's depth (RANSAC
plane + minimum-area rectangle) and the legs from the LiDAR, saves the model in `odom` and a
top-view debug plot (LiDAR points, fitted table, robot footprint, 15 cm safety field);
`--check <name>` compares a stored model with the live LiDAR. `--table <name>` then replaces the
MuJoCo table and gives the WBC on the robot its table collision model and quadrant rectangle
(without it, the real-robot WBC knows no table). `sim_default.yaml` is the MJCF table,
`sim_legs_only.yaml` an open-leg example to try in MuJoCo.

## Interfaces (`--mode ros`)

| | Topic | Type |
|---|---|---|
| in | `/joint_states` | `sensor_msgs/JointState` (the `ur5e` joint prefix is detected automatically) |
| in | `/odom` | `nav_msgs/Odometry` |
| in | `/emergency_stop_state` | `neo_msgs2/EmergencyStopState` (safety stop, see below) |
| in | `/lidar_1/scan_filtered`, `/lidar_2/scan_filtered` | `sensor_msgs/LaserScan` (front distance; safety fallback) |
| in | `/steve_perception/target_point` | `geometry_msgs/PointStamped` from `red_ball_detector` (active perception) |
| out | `/cmd_vel` | `geometry_msgs/Twist` (base frame) |
| out | `/arm_velocity_cmd` → `arm_velocity_watchdog` → `/forward_velocity_controller/commands` (robot, default) | `std_msgs/Float64MultiArray` |
| out | `/joint_trajectory_controller/joint_trajectory` (Gazebo, default) or `/scaled_joint_trajectory_controller/joint_trajectory` (robot, `arm_interface:=trajectory`) | `trajectory_msgs/JointTrajectory` |

The UR5e is mounted with yaw +90° in the URDF (checked on the real robot; was −90°) but 0 in the MJCF; `--pan_offset` (default +π/2)
maps shoulder_pan between the two.

**Arm interface on the real robot: velocity (default).** Streaming a new one-point trajectory
every 20 ms makes the trajectory controller restart its spline from the measured state each
time — the joints buzz audibly, more at speed. Velocity commands avoid that (quiet, and the
small figure-8 tracked better: 0.12 cm vs 0.17 cm RMS). The WBC switches
`scaled_joint_trajectory_controller` → `forward_velocity_controller` on its first command and
back on exit, so MoveIt / `send_goal` work as before. Commands go through the bringup's
`arm_velocity_watchdog`, which sends zero velocity when no command arrives for 0.1 s (a velocity
controller would otherwise keep the last one). `arm_interface:=trajectory` still works (setpoint
integrated, lead bounded by `--max_cmd_lead`; `--arm_cmd_tau 0.06` smooths the streamed
velocity if it buzzes).

## Safety

**Safety stop** (base **and** arm, every script; `steve_manipulation/safety_stop.py`):

| `--safety` | Source | Used |
|---|---|---|
| `em_stop` | the robot's EM-stop system `/emergency_stop_state`: EM stop, not yet released, or the S300 **protective field (red)** | real robot (required — missing / stale state = stop) |
| `lidar` | fallback without an EM-stop system: a LiDAR point closer than `--d_safe_base` (15 cm) to the base footprint | MuJoCo, Gazebo |
| `auto` | `em_stop` if its state is published, else `lidar` | default |

While stopped, zero commands go out and the task clock pauses (tracking keeps world time);
motion resumes after 1 s clear (`--safety_clear_time`), and the run aborts after
`--safety_stop_timeout` (60 s). The relayboard no longer latches the EM stop on a scanner stop
(so the drives are not re-initialised), therefore the controllers stop themselves on
`scanner_stop`. Test in MuJoCo with a person walking in: `--sim_intruder T_IN T_OUT X Y`.

The controller also aborts if feedback is older than 0.25 s, the EE tracking error exceeds
10 cm, the EE drops below 0.3 m, or on Ctrl+C. The MMO-700 driver stops the base when
`/cmd_vel` is silent for 0.1 s.

## Collision avoidance

Hard constraints inside the same QP (velocity dampers): for every pair closer than
`d_influence` (10 cm) the approach speed is limited, reaching zero at `d_safe` (2 cm), so
the controller gives up EE tracking before it gives up clearance.

| Pair | Always | With an environment model (`sim`, or `--table` on the robot) |
|---|---|---|
| arm + gripper vs floor | ✓ | |
| arm + gripper vs own body (base, cabinet, pan-tilt tower) | ✓ | |
| arm + gripper vs table (2 cm; 0.5 cm in the final descent) | | ✓ |
| arm + gripper vs obstacles `obst_*` (2 cm, never relaxed) | | ✓ |
| base column (up to the arm mount, 0.87 m) vs environment: **≥ `--d_safe_base` (15 cm), never relaxed** | | ✓ |

The pan-tilt tower stands 0.15 m **behind** the base centre (arm mount 0.205 m in front, 0.355 m
between them); the upper arm can reach it, so all arm links — including shoulder and upper arm,
which otherwise skip the cabinet they sit on — are checked against it.

The arm is covered by spheres, obstacles by oriented boxes, and distances are computed in
closed form (MuJoCo's `mj_geomDistance` proved unreliable for these mesh/primitive pairs).
Tune with `--d_safe`, `--d_influence`, `--xi`, `--d_safe_base`; disable with `--no_collision`.

## Tests

```bash
python3 -m pytest test/test_active_perception_scenes.py -v   # MuJoCo scenes (single, clutter, control; far paused)
python3 test/compare_vgn_modes.py                             # VGN timing modes (needs vgn_env)
# VGN exploration (needs vgn_env): VGN's verdict forced to "no grasp" for the first FAIL inferences
FAIL=1   python3 test/vgn_explore_runner.py --mode sim --headless --object_speed 0 --target_xy 1.85 -0.10  # -> other side, success
FAIL=all python3 test/vgn_explore_runner.py --mode sim --headless --object_speed 0 --target_xy 1.85 -0.10  # -> no grasp candidate found
```
