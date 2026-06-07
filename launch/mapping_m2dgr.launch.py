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

    m2dgr_config_cmd = os.path.join(config_file_dir, "M2DGR.yaml")
    camera_config_cmd = os.path.join(config_file_dir, "camera_M2DGR.yaml")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch Rviz2",
    )

    m2dgr_config_arg = DeclareLaunchArgument(
        "m2dgr_params_file",
        default_value=m2dgr_config_cmd,
        description="Full path to the ROS2 parameters file for M2DGR mapping",
    )

    camera_config_arg = DeclareLaunchArgument(
        "camera_params_file",
        default_value=camera_config_cmd,
        description="Full path to the ROS2 parameters file for the M2DGR camera",
    )

    m2dgr_params_file = LaunchConfiguration("m2dgr_params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")

    return LaunchDescription(
        [
            use_rviz_arg,
            m2dgr_config_arg,
            camera_config_arg,
            Node(
                package="fast_livo",
                executable="fastlivo_mapping",
                name="laserMapping",
                parameters=[
                    m2dgr_params_file,
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
