#!/usr/bin/env python3
"""
Active perception WBC in MuJoCo with VGN mode 'drive':
    VGN depth views are taken with the wrist camera while driving up to the table (arm
    in scan pose, object motion compensated); inference starts in the background as
    soon as 6 views are in. Falls back to a short stopped scan if the object was not
    visible during the drive (e.g. occluded in the clutter scene).
Same as `active_perception_wbc.launch.py vgn_mode:=drive`; every other argument of that launch
file can be passed as well, e.g.

    ros2 launch steve_wbc active_perception_wbc_vgn_drive.launch.py
    ros2 launch steve_wbc active_perception_wbc_vgn_drive.launch.py scene:=clutter
    ros2 launch steve_wbc active_perception_wbc_vgn_drive.launch.py scene:=far headless:=true

Compare with the original (vgn_mode:=scan):
    ros2 launch steve_wbc active_perception_wbc.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(
                get_package_share_directory("steve_wbc"), "launch", "active_perception_wbc.launch.py")),
            launch_arguments={"use_sim_time": "true", "simulator": "mujoco", "vgn_mode": "drive"}.items(),
        ),
    ])
