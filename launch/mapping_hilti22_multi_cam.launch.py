#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_file_dir = os.path.join(get_package_share_directory("cake_slam"), "config")
    rviz_config_file = os.path.join(
        get_package_share_directory("cake_slam"), "rviz_cfg", "cake_slam2.rviz"
    )

    hilti22_config_cmd = os.path.join(config_file_dir, "HILTI22_multi_cam.yaml")
    camera_config_cmd = os.path.join(config_file_dir, "camera_fisheye_HILTI22_multi_cam.yaml")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch Rviz2",
    )

    hilti22_config_arg = DeclareLaunchArgument(
        "hilti22_params_file",
        default_value=hilti22_config_cmd,
        description="Full path to the ROS2 parameters file for HILTI22 multi-camera mapping",
    )

    camera_config_arg = DeclareLaunchArgument(
        "camera_params_file",
        default_value=camera_config_cmd,
        description="Full path to the ROS2 parameters file for the HILTI22 multi-camera fisheye models",
    )

    hilti22_params_file = LaunchConfiguration("hilti22_params_file")
    camera_params_file = LaunchConfiguration("camera_params_file")

    return LaunchDescription(
        [
            use_rviz_arg,
            hilti22_config_arg,
            camera_config_arg,
            Node(
                package="cake_slam",
                executable="cake_slam_mapping_multi_cam",
                name="laserMapping",
                parameters=[
                    hilti22_params_file,
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
