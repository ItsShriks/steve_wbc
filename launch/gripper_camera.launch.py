"""Gripper camera: a RealSense (D415) mounted on the gripper, plugged into THIS PC.

Starts the RealSense driver (aligned depth on, no TF of its own: frames gripper_camera_*) — topics
/gripper_camera/{color,aligned_depth_to_color}/... with the workspace's realsense2_camera 4.54
(/gripper_camera/gripper_camera/... with newer drivers) — and the static TF of its mount

    ur5etool0 -> gripper_camera_color_optical_frame

per `mount` (steve_wbc.camera_mount): its calibration from gripper_camera_calibrator
(config/gripper_camera_ability_hand.yaml | config/gripper_camera.yaml for camera_only), else — not calibrated
yet — the mount's design pose from its steve_grasping tool config (ability_hand: the D415 on the wrist collar
beside the palm). Without either only the driver runs (for target_calibrator and the calibration itself).
The rest of the TF tree (odom -> ... -> ur5etool0) comes from the robot.

    ros2 launch steve_wbc gripper_camera.launch.py                      # first camera found, mount ability_hand
    ros2 launch steve_wbc gripper_camera.launch.py mount:=camera_only   # the D415 on its post (old setup)
    ros2 launch steve_wbc gripper_camera.launch.py mount_tf:=false      # driver only (e.g. while calibrating)
    ros2 launch steve_wbc gripper_camera.launch.py serial_no:=_909522061251 profile:=640x480x15   # USB 2
    ros2 launch steve_wbc gripper_camera.launch.py initial_reset:=true    # USB control transfers time out
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from steve_wbc.camera_mount import DEFAULT_MOUNT, MOUNTS, calibration_files, nominal

NAME = "gripper_camera"


def calibration_file(mount):
    """The mount's calibration in the package SOURCE folder (where the calibrator writes it, as table
    models), else the installed share folder, else None."""
    ws = Path(get_package_prefix("steve_wbc")).parent.parent
    for c in calibration_files(mount, ws / "src" / "steve_wbc", get_package_share_directory("steve_wbc")):
        if c.is_file():
            return c
    return None


def _setup(context):
    profile = LaunchConfiguration("profile").perform(context)
    actions = [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")),
        launch_arguments={
            "camera_namespace": NAME, "camera_name": NAME,
            "serial_no": LaunchConfiguration("serial_no").perform(context),
            "publish_tf": "false",                     # the mount TF below replaces the driver's
            "enable_color": "true", "enable_depth": "true", "align_depth.enable": "true",
            # parameter names: realsense2_camera 4.54 (workspace) | >= 4.55 (apt)
            "rgb_camera.profile": profile, "depth_module.profile": profile,
            "rgb_camera.color_profile": profile, "depth_module.depth_profile": profile,
            "initial_reset": LaunchConfiguration("initial_reset").perform(context), "wait_for_device_timeout": "10.0",
        }.items())]
    mount = LaunchConfiguration("mount").perform(context)
    if mount not in MOUNTS:
        raise RuntimeError(f"mount must be one of {MOUNTS}, got '{mount}'")
    if LaunchConfiguration("mount_tf").perform(context).lower() == "false":
        print("[gripper_camera] mount_tf:=false: camera only, no mount TF")
        return actions
    f = calibration_file(mount)
    if f is not None:
        c = yaml.safe_load(f.read_text())
        print(f"[gripper_camera] mount {mount}: TF {c['parent']} -> {c['child']} from {f} "
              f"(calibrated {c.get('time', '?')}, residual {c.get('rms_cm', '?')} cm)")
    else:
        c = nominal(mount)
        if c is None:
            print(f"[gripper_camera] mount {mount}: not calibrated and no design pose: camera only, no mount TF "
                  f"(run gripper_camera_calibrator --mount {mount})")
            return actions
        print(f"\033[1;33m[gripper_camera] mount {mount}: NOT CALIBRATED — TF {c['parent']} -> {c['child']} from the "
              f"mount's design pose (steve_grasping config/tools/{mount}.yaml camera:) xyz {c['xyz']}; object positions "
              f"can be off by a few cm. Calibrate: ros2 run steve_wbc gripper_camera_calibrator --mount {mount}\033[0m")
    x, y, z = c["xyz"]
    qx, qy, qz, qw = c["quat_xyzw"]
    actions.append(Node(package="tf2_ros", executable="static_transform_publisher", name=f"{NAME}_mount_tf",
                        arguments=["--x", str(x), "--y", str(y), "--z", str(z), "--qx", str(qx), "--qy", str(qy),
                                   "--qz", str(qz), "--qw", str(qw), "--frame-id", c["parent"],
                                   "--child-frame-id", c["child"]]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mount", default_value=DEFAULT_MOUNT,
                              description=f"where the camera is: {' | '.join(MOUNTS)} (calibration file / design pose "
                                          "of its mount TF, steve_wbc.camera_mount)"),
        DeclareLaunchArgument("mount_tf", default_value="true",
                              description="false: driver only, no mount TF"),
        DeclareLaunchArgument("serial_no", default_value="''",
                              description="RealSense serial ('' = first found; prefix '_' for numeric serials)"),
        DeclareLaunchArgument("profile", default_value="848x480x30",
                              description="colour + depth profile WxHxFPS (D415 min. range ~0.31 m at 848x480, "
                                          "~0.45 m at 1280x720; 640x480x15 on USB 2)"),
        DeclareLaunchArgument("initial_reset", default_value="false",
                              description="hardware-reset the camera on start (helps when USB control "
                                          "transfers time out, e.g. on macOS)"),
        OpaqueFunction(function=_setup),
    ])
