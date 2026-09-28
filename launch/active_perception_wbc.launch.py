#!/usr/bin/env python3
"""
Active perception + MPC interception + grasping with the unified joint-velocity WBC.

    use_sim_time:=true   simulator:=mujoco  -> MuJoCo (default): cameras, LiDAR, VGN, gripper simulated
    use_sim_time:=true   simulator:=gazebo  -> Gazebo (steve_simulation, gripper) + WBC over ROS 2
    use_sim_time:=false                      -> physical robot + WBC over ROS 2

Over ROS 2 the object comes from `target_topic` (geometry_msgs/PointStamped in odom or
base_link, e.g. from steve_perception), the LiDAR from `scan_topic`, and the gripper is
commanded through steve_manipulation's /gripper_controller/gripper_cmd.

Usage:
    ros2 launch steve_wbc active_perception_wbc.launch.py
    ros2 launch steve_wbc active_perception_wbc.launch.py object_type:=box vgn:=false
    ros2 launch steve_wbc active_perception_wbc.launch.py scene:=clutter   # cube among obstacles
    ros2 launch steve_wbc active_perception_wbc.launch.py scene:=far       # drive around the table
    ros2 launch steve_wbc active_perception_wbc.launch.py vgn_mode:=async  # = ..._vgn_async.launch.py
    ros2 launch steve_wbc active_perception_wbc.launch.py vgn_mode:=drive  # = ..._vgn_drive.launch.py
    ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false dry_run:=true
    ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false launch_bringup:=false \\
        target_topic:=/steve_perception/target_point
"""

import os
import shlex

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from steve_manipulation.safety_stop import SCAN_TOPICS

# Robot-specific defaults (IPs, devices, camera serials): steve_hardware_bringup/config/robot_config.yaml
import sys as _sys
_sys.path.insert(0, os.path.join(get_package_share_directory("steve_hardware_bringup"), "config"))
try:
    from steve_config import robot_config  # noqa: E402
    ROBOT_CFG = robot_config()
except Exception as _e:  # the config must never break the bringup: built-in defaults
    print(f"[WARN] robot_config.yaml not loaded ({_e}) - using built-in defaults. Rebuild: "
          "colcon build --symlink-install --packages-select steve_hardware_bringup")
    ROBOT_CFG = {"network": {"ur_robot_ip": "192.168.1.102", "ur_reverse_ip": ""},
                 "devices": {"relayboard": "/dev/neo-relayboard", "lidar_1": "/dev/neo-s300-1",
                             "lidar_2": "/dev/neo-s300-2"},
                 "cameras": {"pan_tilt_l515_serial": "", "wrist_d405_serial": ""}}

ARM_TRAJ_TOPIC = {
    "gazebo": "/joint_trajectory_controller/joint_trajectory",
    "real": "/scaled_joint_trajectory_controller/joint_trajectory",
}
WAIT_TIMEOUT = {"gazebo": "120", "real": "15"}


def launch_setup(context: LaunchContext):
    cfg = lambda name: LaunchConfiguration(name).perform(context)
    is_sim = cfg("use_sim_time").lower() == "true"
    simulator = cfg("simulator").lower()
    if simulator not in ("gazebo", "mujoco"):
        raise RuntimeError(f"simulator must be 'gazebo' or 'mujoco', got '{simulator}'")
    env = simulator if is_sim else "real"
    print(f"[INFO] active_perception_wbc: environment = {env}")
    actions = []

    # --- 1. BRINGUP: robot with gripper + MoveIt-independent gripper controller ---
    # steve_manipulation itself skips the drivers when a bringup is already running on the robot
    # (ROS_AUTOSTART.sh), so no second UR driver / controller_manager is started
    if cfg("launch_bringup").lower() != "false" and env != "mujoco":
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("steve_manipulation"), "launch", "steve_manipulation.launch.py")),
            launch_arguments={
                "use_sim_time": "true" if env == "gazebo" else "false",
                "world": cfg("world"),
                "robot_ip": cfg("robot_ip"),
                "real_bringup": "full",          # base, LiDARs and arm are all needed
                "launch_bringup": "auto",
                "gripper_hw": cfg("gripper_hw"),
                "use_rviz": cfg("use_rviz"),
            }.items(),
        ))

    # --- 2. CONTROLLER ------------------------------------------------------------
    args = []
    for name in ("object_type", "object_speed"):  # empty: the scene's default
        if cfg(name):
            args += [f"--{name}", cfg(name)]
    if cfg("vgn").lower() != "true":
        args.append("--no_vgn")
    args += ["--vgn_mode", cfg("vgn_mode"), "--vgn_max_wait", cfg("vgn_max_wait")]
    if cfg("log"):
        args += ["--log", cfg("log")]
    if env == "mujoco":
        args += ["--mode", "sim", "--scene", cfg("scene")]
        if cfg("headless").lower() == "true":
            args.append("--headless")
    else:
        args += [
            "--mode", "ros",
            "--arm_traj_topic", ARM_TRAJ_TOPIC[env],
            "--wait_timeout", WAIT_TIMEOUT[env],
            "--scan_topic", cfg("scan_topic"),
            "--target_topic", cfg("target_topic"),
            "--no_vgn",                          # VGN runs on the simulated depth camera only
        ]
        if cfg("dry_run").lower() == "true":
            args.append("--dry_run")
    # Safety stop (base + arm): the robot's EM-stop system on the real robot (buttons + scanner
    # protective field via /emergency_stop_state; required), LiDAR fallback in Gazebo (no relayboard)
    if env == "real":
        args += ["--safety", "em_stop"]
    elif env == "gazebo":
        args += ["--safety", "lidar", "--lidar_topics", *SCAN_TOPICS["gazebo"]]
    if cfg("table"):
        args += ["--table", cfg("table")]
    args += ["--target", cfg("target")]
    args += shlex.split(cfg("extra_args"))

    if env != "mujoco" and cfg("detector").lower() == "true":
        # object position from the head + wrist RGB-D cameras (target profile from target_calibrator)
        actions.append(Node(
            package="steve_wbc",
            executable="red_ball_detector",
            name="target_detector",
            output="screen",
            parameters=[{"use_sim_time": env == "gazebo", "target": cfg("target"),
                         "topic_style": "gazebo" if env == "gazebo" else "realsense",
                         "target_topic": cfg("target_topic")}],
        ))
    actions.append(Node(
        package="steve_wbc",
        executable="active_perception_wbc",
        name="active_perception_wbc",
        output="screen",
        emulate_tty=True,
        arguments=args,
        parameters=[{"use_sim_time": env == "gazebo"}],
    ))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="true: simulation, false: physical robot"),
        DeclareLaunchArgument("simulator", default_value="mujoco",
                              description="Simulator when use_sim_time:=true: mujoco or gazebo"),
        DeclareLaunchArgument("launch_bringup", default_value="auto",
                              description="Gazebo / robot: also start steve_manipulation's bringup "
                                          "(robot: skipped if one is already running) | false"),
        DeclareLaunchArgument("world", default_value="neo_workshop", description="Gazebo world"),
        DeclareLaunchArgument("robot_ip", default_value=str(ROBOT_CFG["network"]["ur_robot_ip"]), description="UR5e IP (robot_config.yaml)"),
        DeclareLaunchArgument("gripper_hw", default_value="none",
                              description="Physical robot gripper: none, mock or robotiq_driver"),
        DeclareLaunchArgument("use_rviz", default_value="false", description="RViz (Gazebo / robot)"),
        DeclareLaunchArgument("headless", default_value="false", description="MuJoCo without viewer"),
        DeclareLaunchArgument("dry_run", default_value="false",
                              description="Read the robot state but publish no commands"),
        DeclareLaunchArgument("scene", default_value="single",
                              description="MuJoCo: single (rolling ball) | clutter (cube among obstacles, "
                                          "only the cube is picked) | far (cube out of reach from the front)"),
        DeclareLaunchArgument("object_type", default_value="",
                              description="MuJoCo: sphere or box (default: per scene)"),
        DeclareLaunchArgument("object_speed", default_value="",
                              description="MuJoCo: object speed [m/s] (default: per scene)"),
        DeclareLaunchArgument("vgn", default_value="true",
                              description="MuJoCo: VGN grasp scan (needs the vgn_env conda env)"),
        DeclareLaunchArgument("vgn_mode", default_value="scan",
                              description="MuJoCo VGN timing: scan (stop, scan, wait for inference) | "
                                          "async (same scan, inference in the background) | "
                                          "drive (views while driving up, inference in the background)"),
        DeclareLaunchArgument("vgn_max_wait", default_value="5.0",
                              description="async/drive: max hover at the pre-grasp for a running VGN [s]"),
        DeclareLaunchArgument("scan_topic", default_value="/lidar_1/scan_filtered",
                              description="ROS: front LiDAR"),
        DeclareLaunchArgument("target_topic", default_value="/steve_perception/target_point",
                              description="ROS: object position (geometry_msgs/PointStamped)"),
        DeclareLaunchArgument("target", default_value="red",
                              description="target profile (steve_wbc/config/targets, recorded with target_calibrator)"),
        DeclareLaunchArgument("table", default_value="",
                              description="measured table model (steve_wbc/config/tables, table_probe); "
                                          "robot: WBC collision model, MuJoCo: replaces the sim table"),
        DeclareLaunchArgument("detector", default_value="true",
                              description="Gazebo / robot: start the target detector (red_ball_detector)"),
        DeclareLaunchArgument("log", default_value="", description="CSV log path"),
        DeclareLaunchArgument("extra_args", default_value="",
                              description="Additional active_perception_wbc CLI arguments"),
        OpaqueFunction(function=launch_setup),
    ])
