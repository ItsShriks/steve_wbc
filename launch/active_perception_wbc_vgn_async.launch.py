#!/usr/bin/env python3
"""
Active perception WBC in MuJoCo with VGN mode 'async':
    the same stopped 4-view scan as the original, but VGN inference runs in the
    background while the gripper already approaches (hovers up to vgn_max_wait at the
    pre-grasp if the result is not there yet).
Same as `active_perception_wbc.launch.py vgn_mode:=async`; every other argument of that launch
file can be passed as well, e.g.

    ros2 launch steve_wbc active_perception_wbc_vgn_async.launch.py
    ros2 launch steve_wbc active_perception_wbc_vgn_async.launch.py scene:=clutter
    ros2 launch steve_wbc active_perception_wbc_vgn_async.launch.py scene:=far headless:=true

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
            launch_arguments={"use_sim_time": "true", "simulator": "mujoco", "vgn_mode": "async"}.items(),
        ),
    ])
