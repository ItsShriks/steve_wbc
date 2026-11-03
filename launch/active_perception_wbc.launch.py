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
    ros2 launch steve_wbc active_perception_wbc.launch.py vgn_mode:=scan   # original stopped scan
    ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false dry_run:=true
    ros2 launch steve_wbc active_perception_wbc.launch.py use_sim_time:=false launch_bringup:=false \\
        target_topic:=/steve_perception/target_point

Perception (robot / Gazebo):
    detector:=hsv          colour threshold of the target profile (steve_wbc red_ball_detector, default)
    detector:=gsam         Grounded SAM with gsam_prompt (steve_perception gsam_target_detector)
    grasp_backend:=vgn     VGN on the TSDF of the scan views (default)
    grasp_backend:=cgn     Contact-GraspNet per view on the target's Grounded-SAM mask
    grasp_backend:=dgn2    DexGraspNet 2.0 per view, retargeted to the hand (tool:=ability_hand)
gsam / cgn need the model servers (steve_perception/model_server/README.md); this launch file
starts them (model_servers:=auto), or keep them running in another terminal (model_servers:=false):
    ros2 run steve_perception model_server gsam
    ros2 run steve_perception model_server cgn
"""

import os
import shlex
import time

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


def warn_wifi_link():
    """Robot over Wi-Fi (DDS profile with the robot's Wi-Fi address 10.7.4.213, see ~/.ros/steve_dds.sh):
    gaps in /odom or /joint_states longer than --state_timeout abort the WBC ("odom stale")."""
    prof = os.path.expanduser(os.environ.get("FASTRTPS_DEFAULT_PROFILES_FILE", ""))
    try:
        txt = open(prof).read() if prof else ""
    except OSError:
        txt = ""
    if os.environ.get("STEVE_ROS_LINK") == "wifi" or "10.7.4.213" in txt:
        print("\033[1;33m[WARN] active_perception_wbc: ROS to the robot over Wi-Fi (10.7.4.213, "
              f"{prof or 'no profile'}).\n       State gaps > --state_timeout abort the run (\"odom stale\"): "
              "plug in Ethernet and open a new terminal (~/.ros/steve_dds.sh picks it).\033[0m")
    elif "192.168.60.90" not in txt:
        print(f"[WARN] active_perception_wbc: DDS profile '{prof}' lists neither the robot's Ethernet "
              "(192.168.60.90) nor Wi-Fi address — is the robot reachable?")


def launch_setup(context: LaunchContext):
    cfg = lambda name: LaunchConfiguration(name).perform(context)
    is_sim = cfg("use_sim_time").lower() == "true"
    simulator = cfg("simulator").lower()
    if simulator not in ("gazebo", "mujoco"):
        raise RuntimeError(f"simulator must be 'gazebo' or 'mujoco', got '{simulator}'")
    env = simulator if is_sim else "real"
    # what is on the flange (steve_grasping): the real robot has only the D415 since 2026-10-14
    tool = cfg("tool") or ("robotiq_2f85" if is_sim else "camera_only")
    camera_only = tool == "camera_only"
    # the real robot's only camera is the D415 on the flange — also with a hand there (ability_hand)
    flange_cam = camera_only or env == "real"
    print(f"[INFO] active_perception_wbc: environment = {env}, tool = {tool}, approach = {cfg('approach')}")
    if env == "real":
        warn_wifi_link()
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
    args += ["--tool", tool, "--approach", cfg("approach")]
    if cfg("dynamic").lower() == "true":  # moving objects (rolling ball); off: the static behaviour
        args.append("--dynamic")
    # wrist-camera detections closer than the depth range are wrong (the D415 on the real flange: ~0.3 m)
    min_rng = cfg("min_detect_range") or ("0.30" if env == "real" and flange_cam else "")
    if min_rng:
        args += ["--min_detect_range", min_rng]
    # base <-> table legs: the real robot needs more than the 15 cm of the model — the scanners' protective
    # field went red at 20 cm model clearance (2026-10-22: odometry drift around the table + table model error)
    d_base = cfg("d_safe_base") or ("0.25" if env == "real" else "")
    if d_base:
        args += ["--d_safe_base", d_base]
    # camera looking along the flange axis (D415 on the flange): tilt it forward in the drive / scan poses
    tilt = cfg("camera_tilt") or ("25" if flange_cam else "0")
    if float(tilt):
        args += ["--camera_tilt", tilt]
    if cfg("log"):
        args += ["--log", cfg("log")]
    if env == "mujoco":
        args += ["--mode", "sim", "--scene", cfg("scene")]
        # like the robot: with the D415 on the flange the detector has only that camera
        args += ["--sim_cameras", cfg("sim_cameras") or ("wrist" if camera_only else "wrist,head")]
        if cfg("use_rviz").lower() == "true":  # MuJoCo -> RViz (steve_wbc/mujoco_rviz.py): TF of every body + scene
            args.append("--rviz")
            actions.append(Node(package="rviz2", executable="rviz2", name="rviz_mujoco", output="log",
                                arguments=["-d", os.path.join(get_package_share_directory("steve_wbc"), "config",
                                                              "rviz", "mujoco.rviz")]))
        if cfg("headless").lower() == "true":
            args.append("--headless")
    else:
        # real robot: velocity commands (smooth; streaming one-point trajectories every 20 ms makes the
        # trajectory controller restart its spline each time -> jerky), as steve_figure8_wbc.launch.py
        arm_interface = cfg("arm_interface") or ("velocity" if env == "real" else "trajectory")
        args += [
            "--mode", "ros",
            "--arm_interface", arm_interface,
            "--arm_traj_topic", ARM_TRAJ_TOPIC[env],
            "--wait_timeout", WAIT_TIMEOUT[env],
            "--scan_topic", cfg("scan_topic"),
            "--target_topic", cfg("target_topic"),
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
    if cfg("table"):
        args += ["--table", cfg("table")]
    if cfg("goto_grasp_only").lower() == "true":
        args.append("--goto_grasp_only")
    if cfg("continuous").lower() == "true":
        args.append("--continuous")
    args += ["--target", cfg("target")]
    detector = {"true": "hsv"}.get(cfg("detector").lower(), cfg("detector").lower())
    if detector not in ("hsv", "gsam", "false"):
        raise RuntimeError(f"detector must be hsv, gsam or false, got '{cfg('detector')}'")
    backend = cfg("grasp_backend").lower()
    if backend not in ("vgn", "cgn", "dgn2"):
        raise RuntimeError(f"grasp_backend must be vgn, cgn or dgn2, got '{cfg('grasp_backend')}'")
    if backend == "dgn2":
        args += ["--grasp_backend", "dgn2"]
    if backend == "cgn":
        if env == "mujoco":
            raise RuntimeError("grasp_backend:=cgn needs real camera views (robot / Gazebo); MuJoCo uses VGN")
        args += ["--grasp_backend", "cgn", "--grasp_prompt", cfg("gsam_prompt")]
    # records: detections (detector/) and grasp scans (grasp_scan/) in this run's log folder
    save_dir = ""
    if cfg("save_images").lower() == "true" and env != "mujoco":
        save_dir = os.environ.get("STEVE_LAUNCH_LOG_DIR") or os.path.expanduser(
            f"~/ros_logs/active_perception_wbc_{time.strftime('%Y%m%d_%H%M%S')}")
        print(f"[INFO] active_perception_wbc: images -> {save_dir}/{{detector,grasp_scan}}")
    args += ["--save_dir", save_dir]
    args += shlex.split(cfg("extra_args"))

    # GPU model servers (each in its own conda env), shared by the detector and the grasp scan
    servers = []
    if env != "mujoco" and cfg("model_servers").lower() != "false":
        servers += ["gsam"] if detector == "gsam" or (backend == "cgn" and cfg("vgn").lower() == "true") else []
        servers += ["cgn"] if backend == "cgn" and cfg("vgn").lower() == "true" else []
        servers += ["dgn2"] if backend == "dgn2" and cfg("vgn").lower() == "true" else []
    if env == "real" and tool == "ability_hand" and cfg("launch_hand").lower() == "true":
        # the Ability Hand: ah_node (serial) + the GripperCommand bridge the WBC closes / opens it through
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("steve_grasping"), "launch",
                                                       "ability_hand.launch.py")),
            launch_arguments={"port": cfg("hand_port"), "simulated_hand": cfg("hand_simulated")}.items()))
    for m in servers:
        actions.append(Node(package="steve_perception", executable="model_server", name=f"model_server_{m}",
                            arguments=[m], output="screen", emulate_tty=True))

    det_cams = cfg("detector_cameras") or ("gripper_camera" if flange_cam else "pan_tilt_camera,wrist_camera")
    cameras = [c.strip() for c in det_cams.split(",") if c.strip()]
    if env != "mujoco" and detector == "gsam":
        # object named by gsam_prompt, size and shape from the target profile (sphere fit for balls)
        try:
            from steve_wbc.target_profile import TargetProfile
            prof = TargetProfile.load(cfg("target"))
            size, shape = float(prof.size), ("sphere" if prof.shape == "sphere" else "other")
        except Exception as e:
            print(f"[WARN] target profile '{cfg('target')}' not loaded ({e}): 6.5 cm sphere assumed")
            size, shape = 0.065, "sphere"
        actions.append(Node(
            package="steve_perception",
            executable="gsam_target_detector",
            name="target_detector",
            output="screen",
            parameters=[{"use_sim_time": env == "gazebo", "prompt": cfg("gsam_prompt"),
                         "object_size": size, "shape": shape, "cameras": cameras, "save_dir": save_dir,
                         "topic_style": "gazebo" if env == "gazebo" else "realsense",
                         "target_topic": cfg("target_topic")}],
        ))
    if env != "mujoco" and detector == "hsv":
        # object position from the head + wrist RGB-D cameras (target profile from target_calibrator)
        actions.append(Node(
            package="steve_wbc",
            executable="red_ball_detector",
            name="target_detector",
            output="screen",
            parameters=[{"use_sim_time": env == "gazebo", "target": cfg("target"),
                         "topic_style": "gazebo" if env == "gazebo" else "realsense",
                         "target_topic": cfg("target_topic"), "save_dir": save_dir,
                         # cameras the detector uses: <name> -> topics /<name>/... (gripper_camera:
                         # the D415 on the gripper via gripper_camera.launch.py on this PC)
                         "cameras": cameras,
                         # no pan-tilt head among them: do not command it
                         **({} if "pan_tilt_camera" in det_cams else {"head_topic": ""})}],
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
        DeclareLaunchArgument("use_rviz", default_value="false",
                              description="RViz: Gazebo / robot; MuJoCo: the simulated scene and the WBC's markers"),
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
                              description="VGN grasp scan (needs the vgn_env conda env on this PC; robot: wrist-camera "
                                          "views, no grasp -> explore other table sides)"),
        DeclareLaunchArgument("vgn_mode", default_value="drive",
                              description="VGN timing (MuJoCo and robot): drive (default; views while driving "
                                          "up, inference in the background) | scan (stop, scan, wait for "
                                          "inference) | async (same scan, inference in the background)"),
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
        DeclareLaunchArgument("detector_cameras", default_value="",
                              description="robot/Gazebo: cameras of the target detector (default: gripper_camera "
                                          "with tool camera_only, else pan_tilt_camera,wrist_camera)"),
        DeclareLaunchArgument("launch_hand", default_value="true",
                              description="robot with tool:=ability_hand: start ah_node + the GripperCommand bridge "
                                          "(steve_grasping ability_hand.launch.py); false: running elsewhere"),
        DeclareLaunchArgument("hand_port", default_value="", description="Ability Hand serial port (empty: auto)"),
        DeclareLaunchArgument("hand_simulated", default_value="false",
                              description="ah_node's virtual hand (no hardware): test the gripper path"),
        DeclareLaunchArgument("dynamic", default_value="false",
                              description="true: moving objects — camera kept on the object up to the descent, "
                                          "re-acquire, velocity while driving (active_perception_wbc --dynamic)"),
        DeclareLaunchArgument("min_detect_range", default_value="",
                              description="ignore wrist-camera detections closer than this [m] (default: 0.30 with the "
                                          "D415 of tool camera_only on the robot, else off)"),
        DeclareLaunchArgument("tool", default_value="",
                              description="what is on the flange (steve_grasping): camera_only | robotiq_2f85 | dex5 | ability_hand "
                                          "(default: camera_only on the robot, robotiq_2f85 in simulation); "
                                          "camera_only = go to the grasp pose only"),
        DeclareLaunchArgument("d_safe_base", default_value="",
                              description="min. distance base footprint <-> table legs / environment [m] (default: "
                                          "0.25 on the robot, 0.15 in simulation); the orbit around the table and the "
                                          "stand-off at each side grow with it"),
        DeclareLaunchArgument("approach", default_value="move",
                              description="move: plan the grasp as soon as the object is tracked and go for it without "
                                          "stopping (base + arm together) | stop: drive, stop, settle, look, reach"),
        DeclareLaunchArgument("camera_tilt", default_value="",
                              description="tilt of the drive / scan poses for a camera along the flange axis [deg] "
                                          "(default: 25 with camera_only, else 0)"),
        DeclareLaunchArgument("sim_cameras", default_value="",
                              description="MuJoCo: detector cameras wrist,head (default: wrist only with camera_only, "
                                          "like the robot)"),
        DeclareLaunchArgument("arm_interface", default_value="",
                              description="velocity | trajectory (default: velocity on the robot, trajectory in Gazebo)"),
        DeclareLaunchArgument("goto_grasp_only", default_value="false",
                              description="go to the grasp pose and stay there (--hold_time), no gripper / lift"),
        DeclareLaunchArgument("continuous", default_value="false",
                              description="after each episode watch for the object to be put somewhere else and "
                                          "go for it again (gripper: opens after --handover_time); Ctrl+C ends"),
        DeclareLaunchArgument("detector", default_value="hsv",
                              description="Gazebo / robot target detector: hsv (colour threshold, red_ball_detector; "
                                          "'true' = hsv) | gsam (Grounded SAM with gsam_prompt) | false (none)"),
        DeclareLaunchArgument("gsam_prompt", default_value="red ball",
                              description="detector:=gsam / grasp_backend:=cgn: text naming the target"),
        DeclareLaunchArgument("grasp_backend", default_value="vgn",
                              description="grasp scan (vgn:=true): vgn (TSDF of the views) | cgn (robot / Gazebo: "
                                          "Contact-GraspNet per view on the target's Grounded-SAM mask) | dgn2 "
                                          "(DexGraspNet 2.0, retargeted to the hand)"),
        DeclareLaunchArgument("model_servers", default_value="auto",
                              description="auto: start the GPU model servers detector:=gsam / grasp_backend:=cgn "
                                          "need | false: already running (ros2 run steve_perception model_server ...)"),
        DeclareLaunchArgument("save_images", default_value="true",
                              description="robot / Gazebo: save annotated detections (detector/) and every grasp "
                                          "scan's views with mask + grasps (grasp_scan/) in the run's log folder"),
        DeclareLaunchArgument("log", default_value="", description="CSV log path"),
        DeclareLaunchArgument("extra_args", default_value="",
                              description="Additional active_perception_wbc CLI arguments"),
        OpaqueFunction(function=launch_setup),
    ])
