# steve_wbc

Unified velocity-level (kinematic) **whole-body controller** for Steve
(Neobotix MMO-700 omnidirectional base + UR5e). Base and arm velocities come from
**one** constrained QP over all 9 DOF — `u = [vx_b, vy_b, wz_b, dq1..dq6]`:

```
min_u  || J_wb u - v_ee ||²_Wt                 EE pose task (primary)
     + w_b || u_base - u_base_des ||²          soft base-pose task (optional)
     + || u - N·K(q_nom - q) ||²_Wu             null-space posture + effort
s.t.   velocity / acceleration / joint-position limits
       collision avoidance (arm vs table, obstacles, floor, own body; base vs environment)
```

`J_wb` is the exact whole-body Jacobian from the MuJoCo model (`models/mmo_700_wbc.xml`),
including the lever arm of base yaw. The same controller runs in MuJoCo, Gazebo and on
the real robot.

| Executable | Purpose |
|---|---|
| `steve_figure8_wbc` | Figure-8 with the EE — MuJoCo (`--mode sim`), kinematic check (`--mode kin`), Gazebo / real robot over ROS 2 (`--mode ros`) |
| `figure8_wbc_demo` | MuJoCo-only demo: figure-8 on the table while the base orbits it, then a vertical figure-8 |
| `active_perception_wbc` | Drive (LiDAR) → perceive the object from t = 0 → (VGN scan) → **MPC interception** of its predicted path → grasp → lift → hold; every phase solved by the same 9-DOF joint-velocity WBC |
| `red_ball_detector` | ROS 2 target detector (head + wrist RGB-D) → `/steve_perception/target_point`, uses a target profile |
| `target_calibrator` | record a target object's colour profile with the real (or simulated) camera |
| `table_probe` | measure the real table (LiDAR + head depth) into a table model |

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
* **Approach side by table quadrant**: the table is split into four quadrants, seen from the side
  the robot arrives on — `bottom-left` / `bottom-right` (near half) → grasp from where it is;
  `top-left` / `top-right` (far half) → drive around the table to the left / right side
  (`QUADRANT_SIDE`). The choice is verified by IK + collision checks (other sides as fallback);
  a blocked or stalled grasp replans to another side. Driving around follows a ring on which the
  base clears the table at any heading.
* **Obstacles** (`--scene clutter`): only the target is picked; the gripper flies over obstacles in
  its path (transit height) and the arm keeps ≥ 2 cm from them (hard constraints, not relaxed in
  the final descent). Any robot–obstacle contact fails the run.
* **VGN** (MuJoCo): TSDF views + inference in the `vgn_env` conda env (run isolated from
  `~/.local` and the ROS `PYTHONPATH`); only grasps **on the target** are used (within its size
  of the tracked position). Three timing modes, `--vgn_mode` / separate launch files:

  | Mode | Launch file | Views | Inference | Robot waits |
  |---|---|---|---|---|
  | `scan` (original) | `active_perception_wbc.launch.py` | stopped 4-view pan scan | blocking | scan + inference |
  | `async` | `active_perception_wbc_vgn_async.launch.py` | stopped 4-view pan scan | background | scan only |
  | `drive` | `active_perception_wbc_vgn_drive.launch.py` | while driving up (motion-compensated) | background, starts while driving | ~0 |

  Inference time is charged 1:1 to simulation time, so the modes compare fairly
  (`python3 test/compare_vgn_modes.py`). A result arriving after the descent started is ignored;
  `--vgn_max_wait` bounds a hover at the pre-grasp.

```bash
ros2 launch steve_wbc active_perception_wbc.launch.py                          # MuJoCo, rolling ball
ros2 launch steve_wbc active_perception_wbc.launch.py scene:=clutter           # cube among obstacles
ros2 launch steve_wbc active_perception_wbc.launch.py vgn:=false               # no VGN
ros2 launch steve_wbc active_perception_wbc_vgn_drive.launch.py scene:=clutter
ros2 run steve_wbc active_perception_wbc --mode sim --object_type box --object_speed 0 --target_xy 2.2 0.2   # top-left quadrant
ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false dry_run:=true table:=lab_table target:=my_ball
```

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
| 1. Network, USB devices, camera serials | `ros2 run steve_hardware_bringup steve_doctor [--ros]` | checks `steve_hardware_bringup/config/robot_config.yaml` against the hardware ([details](../steve_hardware_bringup/README.md)) |
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
| out | `/scaled_joint_trajectory_controller/joint_trajectory` (robot) or `/joint_trajectory_controller/joint_trajectory` (Gazebo) | `trajectory_msgs/JointTrajectory` |
| out | `/forward_velocity_controller/commands` with `arm_interface:=velocity` | `std_msgs/Float64MultiArray` |

The UR5e is mounted with yaw −90° in the URDF but 0 in the MJCF; `--pan_offset` (default −π/2)
maps shoulder_pan between the two. Arm commands are streamed as a continuous integrated
setpoint (lead over the measurement bounded by `--max_cmd_lead`).

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
```
