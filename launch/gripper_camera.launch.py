"""Gripper camera: a RealSense (D415) mounted on the gripper, plugged into THIS PC.

Starts the RealSense driver (aligned depth on, no TF of its own: frames gripper_camera_*) — topics
/gripper_camera/{color,aligned_depth_to_color}/... with the workspace's realsense2_camera 4.54
(/gripper_camera/gripper_camera/... with newer drivers) — and, once calibrated, the static TF of its mount

    ur5etool0 -> gripper_camera_color_optical_frame      (config/gripper_camera.yaml)

from gripper_camera_calibrator. Without that file only the driver runs (for target_calibrator and
the calibration itself). The rest of the TF tree (odom -> ... -> ur5etool0) comes from the robot.

    ros2 launch steve_wbc gripper_camera.launch.py                      # first camera found
    ros2 launch steve_wbc gripper_camera.launch.py serial_no:=_909522061251 profile:=640x480x15   # USB 2
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

NAME = "gripper_camera"


def calibration_file():
    """config/gripper_camera.yaml in the package SOURCE folder (where the calibrator writes it, as
    table models), else the installed share folder."""
    ws = Path(get_package_prefix("steve_wbc")).parent.parent
    for c in (ws / "src" / "steve_wbc" / "config" / f"{NAME}.yaml",
              Path(get_package_share_directory("steve_wbc")) / "config" / f"{NAME}.yaml"):
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
            "initial_reset": "false", "wait_for_device_timeout": "10.0",
        }.items())]
    f = calibration_file()
    if f is None:
        print(f"[gripper_camera] no {NAME}.yaml yet: camera only, no mount TF "
              "(run gripper_camera_calibrator)")
        return actions
    c = yaml.safe_load(f.read_text())
    x, y, z = c["xyz"]
    qx, qy, qz, qw = c["quat_xyzw"]
    print(f"[gripper_camera] mount TF {c['parent']} -> {c['child']} from {f} "
          f"(calibrated {c.get('time', '?')}, residual {c.get('rms_cm', '?')} cm)")
    actions.append(Node(package="tf2_ros", executable="static_transform_publisher", name=f"{NAME}_mount_tf",
                        arguments=["--x", str(x), "--y", str(y), "--z", str(z), "--qx", str(qx), "--qy", str(qy),
                                   "--qz", str(qz), "--qw", str(qw), "--frame-id", c["parent"],
                                   "--child-frame-id", c["child"]]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("serial_no", default_value="''",
                              description="RealSense serial ('' = first found; prefix '_' for numeric serials)"),
        DeclareLaunchArgument("profile", default_value="848x480x30",
                              description="colour + depth profile WxHxFPS (D415 min. range ~0.31 m at 848x480, "
                                          "~0.45 m at 1280x720; 640x480x15 on USB 2)"),
        OpaqueFunction(function=_setup),
    ])
