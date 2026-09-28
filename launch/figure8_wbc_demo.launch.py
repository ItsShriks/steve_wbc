#!/usr/bin/env python3
"""
MuJoCo whole-body control demo (simulation only):
  Phase 1 — figure-8 on the table top while the base orbits the table
  Phase 2 — vertical figure-8 in front of the robot

Usage:
    ros2 launch steve_wbc figure8_wbc_demo.launch.py
    ros2 launch steve_wbc figure8_wbc_demo.launch.py phase:=1 loops:=2 log:=/tmp/f8.csv
    ros2 launch steve_wbc figure8_wbc_demo.launch.py headless:=true

For the physical robot use steve_figure8_wbc.launch.py use_sim_time:=false.
"""

import shlex

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context: LaunchContext):
    cfg = lambda name: LaunchConfiguration(name).perform(context)
    if cfg("use_sim_time").lower() != "true":
        raise RuntimeError("figure8_wbc_demo is simulation-only (MuJoCo). For the physical robot use:\n"
                           "  ros2 launch steve_wbc steve_figure8_wbc.launch.py use_sim_time:=false")

    phase = cfg("phase")
    args = ["--mode", "sim", "--loops", cfg("loops"), "--scale", cfg("scale"), "--speed", cfg("speed")]
    if phase == "1":
        args.append("--phase1_only")
    elif phase == "2":
        args.append("--phase2_only")
    elif phase != "all":
        raise RuntimeError(f"phase must be 1, 2 or all, got '{phase}'")
    if cfg("headless").lower() == "true":
        args.append("--headless")
    if cfg("log"):
        args += ["--log", cfg("log")]
    args += shlex.split(cfg("extra_args"))

    return [Node(
        package="steve_wbc",
        executable="figure8_wbc_demo",
        name="figure8_wbc_demo",
        output="screen",
        emulate_tty=True,
        arguments=args,
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true",
                              description="Must be true — this demo runs in MuJoCo"),
        DeclareLaunchArgument("headless", default_value="false", description="Run without the viewer"),
        DeclareLaunchArgument("phase", default_value="all", description="1, 2 or all"),
        DeclareLaunchArgument("loops", default_value="3", description="Figure-8 loops per phase"),
        DeclareLaunchArgument("scale", default_value="1.0", description="Amplitude multiplier"),
        DeclareLaunchArgument("speed", default_value="1.0", description="Frequency multiplier"),
        DeclareLaunchArgument("log", default_value="", description="CSV log path (per phase)"),
        DeclareLaunchArgument("extra_args", default_value="",
                              description="Additional WBC CLI arguments (weights, limits, ...)"),
        OpaqueFunction(function=launch_setup),
    ])
