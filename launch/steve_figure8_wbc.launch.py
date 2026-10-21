#!/usr/bin/env python3
"""
Figure-8 whole-body controller for Steve — simulation or physical robot.

    use_sim_time:=true  simulator:=gazebo  -> Gazebo (steve_simulation) + WBC over ROS 2
    use_sim_time:=true  simulator:=mujoco  -> standalone MuJoCo physics, no ROS robot needed
    use_sim_time:=false                     -> real robot (steve_hardware_bringup) + WBC

Usage:
    ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=true simulator:=mujoco
    ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=true world:=neo_workshop
    ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false launch_bringup:=false dry_run:=true
    ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false launch_bringup:=false \\
        amplitude:=0.08 frequency:=0.25 loops:=1

Any other controller option (weights, limits, ...) can be passed through
extra_args, e.g. extra_args:="--max_base_v 0.1 --kp_pos 3.0".
See `ros2 run steve_wbc steve_figure8_wbc --help`.
"""

import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from steve_manipulation.bringup_probe import bringup_running
from steve_manipulation.safety_stop import SCAN_TOPICS

# Arm trajectory controller that is active by default in each environment
ARM_TRAJ_TOPIC = {
    "gazebo": "/joint_trajectory_controller/joint_trajectory",
    "real": "/scaled_joint_trajectory_controller/joint_trajectory",
}
# Gazebo needs time to start, spawn the robot and activate controllers
WAIT_TIMEOUT = {"gazebo": "120", "real": "15"}
# motion_split -> WBC weights. equal: base and arm each ~50 % of the EE motion (measured on the
# table figure-8, kinematic check: base 47 % forward / 52-56 % sideways)
SPLITS = {
    "base": [],                                                              # default: base does most
    "equal": ["--w_base_x", "14", "--w_base_y", "6", "--k_posture", "0", "--lock_yaw"],
    "arm": ["--max_base_v", "0", "--lock_yaw", "--k_posture", "0"],          # base stands still
}
# Default figure-8 centre height in Gazebo [m]
GAZEBO_CZ = "1.1"


def launch_setup(context: LaunchContext):
    cfg = lambda name: LaunchConfiguration(name).perform(context)
    is_sim = cfg("use_sim_time").lower() == "true"
    simulator = cfg("simulator").lower()
    bringup_arg = cfg("launch_bringup").lower()
    launch_bringup = bringup_arg != "false"
    if simulator not in ("gazebo", "mujoco"):
        raise RuntimeError(f"simulator must be 'gazebo' or 'mujoco', got '{simulator}'")

    env = ("mujoco" if simulator == "mujoco" else "gazebo") if is_sim else "real"
    print(f"[INFO] steve_wbc: environment = {env}")
    actions = []

    # --- 1. ROBOT BRINGUP -------------------------------------------------------
    if launch_bringup and env == "gazebo":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("steve_simulation"), "launch", "simulation.launch.py")),
            launch_arguments={
                "my_robot": "mmo_700",
                "world": cfg("world"),
                "use_sim_time": "true",
                "arm_type": cfg("arm_type"),
                "include_pan_tilt": "true",
                "use_rviz": cfg("use_rviz"),
                "launch_map_server": "false",  # no navigation needed for the WBC
            }.items(),
        ))
    elif launch_bringup and env == "real" and bringup_running():
        # ROS_AUTOSTART.sh already runs steve_hardware_bringup: never start a second one
        print("[INFO] steve_wbc: robot bringup already running (/controller_manager found, e.g. "
              "ROS_AUTOSTART.sh) — using it instead of starting another")
    elif launch_bringup and env == "real" and bringup_arg != "true":
        # never start a (second) UR driver implicitly on the real robot: the bringup runs in its own
        # screen session (ROS_AUTOSTART.sh); a missed detection once started two drivers
        raise RuntimeError(
            "no robot bringup found (no UR driver process / /controller_manager). Start it first:\n"
            "  screen -S bringup -L -Logfile ~/ros_logs/bringup.log bash -c \"source "
            "~/steve_ros2_ws/install/setup.bash && ros2 launch steve_hardware_bringup "
            "hardware_bringup.launch.py enable_joystick:=false\"\n"
            "or let this launch start it with launch_bringup:=true")
    elif launch_bringup and env == "real":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("steve_navigation"), "launch", "robot_bringup.launch.py")),
            launch_arguments={
                "arm_type": cfg("arm_type"),
                "enable_camera": cfg("enable_camera"),
                "enable_joystick": cfg("enable_joystick"),
            }.items(),
        ))

    # --- 2. WHOLE-BODY CONTROLLER ----------------------------------------------
    args = [
        # what is on the flange (steve_grasping): the end effector `pinch` and the collision model
        "--tool", cfg("tool") or ("camera_only" if env == "real" else "robotiq_2f85"),
        "--amplitude", cfg("amplitude"),
        "--frequency", cfg("frequency"),
        "--loops", cfg("loops"),
        "--plane", cfg("plane"),
        "--orientation", cfg("orientation"),
    ]
    # Gazebo spawns the arm sagged and tucked against the cabinet (EE ~0.9 m); the lower
    # lobe of the 8 around that start would run into the robot's own body.
    cz = cfg("cz") or (GAZEBO_CZ if env == "gazebo" else "")
    if cz:
        args += ["--cz", cz]
    if cfg("log"):
        args += ["--log", cfg("log")]
    if cfg("table"):
        args += ["--table", cfg("table"), "--table_clearance", cfg("table_clearance")]
    if cfg("goto_table"):
        args += ["--goto_table", cfg("goto_table")]

    if env == "mujoco":
        args += ["--mode", "sim"]
        if cfg("headless").lower() == "true":
            args.append("--headless")
    else:
        # real robot: velocity (quiet joints, see arm_velocity_watchdog); Gazebo: trajectory
        arm_interface = cfg("arm_interface") or ("velocity" if env == "real" else "trajectory")
        args += [
            "--mode", "ros",
            "--arm_interface", arm_interface,
            "--arm_traj_topic", cfg("arm_traj_topic") or ARM_TRAJ_TOPIC[env],
            "--wait_timeout", WAIT_TIMEOUT[env],
        ]
        if arm_interface == "velocity" and env == "real":
            # through the bringup's arm_velocity_watchdog (zero velocity if this script stops)
            args += ["--arm_vel_topic", "/arm_velocity_cmd", "--switch_controllers"]
        if cfg("dry_run").lower() == "true":
            args.append("--dry_run")
    # Safety stop (base + arm): the robot's EM-stop system on the real robot (buttons + scanner
    # protective field via /emergency_stop_state; required), LiDAR fallback in Gazebo (no relayboard)
    if env == "real":
        args += ["--safety", "em_stop"]
    elif env == "gazebo":
        args += ["--safety", "lidar", "--lidar_topics", *SCAN_TOPICS["gazebo"]]
    # who does the motion (the posture level's weights; the EE path stays exact)
    split = cfg("motion_split")
    if split not in SPLITS:
        raise RuntimeError(f"motion_split must be one of {list(SPLITS)}, got '{split}'")
    args += SPLITS[split]
    args += shlex.split(cfg("extra_args"))   # extra_args last: they override

    actions.append(Node(
        package="steve_wbc",
        executable="steve_figure8_wbc",
        name="steve_figure8_wbc",
        output="screen",
        emulate_tty=True,
        arguments=args,
        parameters=[{"use_sim_time": is_sim and env == "gazebo"}],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        # --- environment (same convention as steve_navigation) ---
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="true: simulation, false: physical robot"),
        DeclareLaunchArgument("simulator", default_value="gazebo",
                              description="Simulator when use_sim_time:=true: gazebo or mujoco"),
        DeclareLaunchArgument("launch_bringup", default_value="auto",
                              description="auto/true: also start Gazebo / the robot hardware bringup "
                                          "(on the robot only if none is running yet, e.g. from "
                                          "ROS_AUTOSTART.sh) | false"),
        DeclareLaunchArgument("world", default_value="neo_workshop",
                              description="Gazebo world (simulation only)"),
        DeclareLaunchArgument("use_rviz", default_value="false",
                              description="RViz with the Gazebo simulation"),
        DeclareLaunchArgument("arm_type", default_value="ur5e", description="UR arm type"),
        DeclareLaunchArgument("tool", default_value="",
                              description="camera_only | robotiq_2f85 | dex5 (default: camera_only on the robot, "
                                          "robotiq_2f85 in simulation; same EE point 14.9 cm beyond the flange for both)"),
        DeclareLaunchArgument("enable_camera", default_value="false",
                              description="Real robot: start the L515 camera"),
        DeclareLaunchArgument("enable_joystick", default_value="true",
                              description="Real robot: start the joystick (dead-man / manual override)"),
        DeclareLaunchArgument("headless", default_value="false",
                              description="MuJoCo: run without the viewer"),
        # --- controller / safety ---
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="Read the robot state but publish no commands"),
        DeclareLaunchArgument("arm_interface", default_value="",
                              description="'' = velocity on the real robot, trajectory in Gazebo | "
                                          "trajectory (scaled_joint_trajectory_controller) or velocity "
                                          "(forward_velocity_controller via arm_velocity_watchdog; "
                                          "quieter joints on the real robot)"),
        DeclareLaunchArgument("arm_traj_topic", default_value="",
                              description="Override the arm JointTrajectory topic "
                                          "(default: per environment)"),
        # --- figure-8 ---
        DeclareLaunchArgument("amplitude", default_value="0.15", description="[m]"),
        DeclareLaunchArgument("frequency", default_value="0.45", description="[rad/s]"),
        DeclareLaunchArgument("loops", default_value="3"),
        DeclareLaunchArgument("plane", default_value="vertical", description="vertical or horizontal"),
        DeclareLaunchArgument("orientation", default_value="hold", description="hold, forward or down"),
        DeclareLaunchArgument("motion_split", default_value="base",
                              description="who does the figure-8: base (default, base does most), "
                                          "equal (base and arm ~50/50), arm (base stands still)"),
        DeclareLaunchArgument("table", default_value="",
                              description="table model (config/tables/<name>.yaml): collision model; with "
                                          "plane:=horizontal the 8 is drawn over the table"),
        DeclareLaunchArgument("goto_table", default_value="",
                              description="instead of a figure-8: gripper this many m above the table "
                                          "centre, pointing down (needs table:=)"),
        DeclareLaunchArgument("table_clearance", default_value="0.10",
                              description="height of the 8 above the table top [m]"),
        DeclareLaunchArgument("cz", default_value="", description="figure-8 centre height [m] (default: start EE; Gazebo: 1.1)"),
        DeclareLaunchArgument("log", default_value="", description="CSV log path"),
        DeclareLaunchArgument("extra_args", default_value="",
                              description="Additional steve_figure8_wbc CLI arguments"),
        OpaqueFunction(function=launch_setup),
    ])
