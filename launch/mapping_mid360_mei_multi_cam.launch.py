#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file_dir = os.path.join(get_package_share_directory("fast_livo"), "config")
    rviz_config_file = os.path.join(
        get_package_share_directory("fast_livo"), "rviz_cfg", "fast_livo2.rviz"
    )

    mid360_config_cmd = os.path.join(config_file_dir, "mid360_mei_multi_cam.yaml")
    camera_config_cmd = os.path.join(config_file_dir, "camera_mei_multi_cam.yaml")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch Rviz2",
    )

    mid360_config_arg = DeclareLaunchArgument(
        "mid360_params_file",
        default_value=mid360_config_cmd,
        description="Full path to the ROS2 parameters file for MID360/MEI mapping",
    )

    camera_config_arg = DeclareLaunchArgument(
        "camera_params_file",
        default_value=camera_config_cmd,
        description="Full path to the ROS2 parameters file for the MEI camera model",
    )

    mid360_params_file = LaunchConfiguration("mid360_params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")

    return LaunchDescription(
        [
            use_rviz_arg,
            mid360_config_arg,
            camera_config_arg,
            Node(
                package="fast_livo",
                executable="fastlivo_mapping_multi_cam",
                name="laserMapping",
                parameters=[
                    mid360_params_file,
                    camera_params_file,
                ],
                output="screen",
            ),
            Node(
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config_file],
                output="screen",
            ),
        ]
    )
