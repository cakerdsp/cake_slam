#!/usr/bin/python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config_file_dir = os.path.join(get_package_share_directory("cake_slam"), "config")
    rviz_config_file = os.path.join(
        get_package_share_directory("cake_slam"), "rviz_cfg", "cake_slam2.rviz"
    )

    mapping_config = os.path.join(config_file_dir, "M2DGR_multi_cam.yaml")
    camera_config = os.path.join(config_file_dir, "camera_M2DGR_multi_cam.yaml")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch Rviz2",
    )
    mapping_config_arg = DeclareLaunchArgument(
        "m2dgr_params_file",
        default_value=mapping_config,
        description="M2DGR multi-camera mapping parameter file",
    )
    camera_config_arg = DeclareLaunchArgument(
        "camera_params_file",
        default_value=camera_config,
        description="M2DGR fisheye camera-model parameter file",
    )

    mapping_params = LaunchConfiguration("m2dgr_params_file")
    camera_params = LaunchConfiguration("camera_params_file")

    return LaunchDescription(
        [
            use_rviz_arg,
            mapping_config_arg,
            camera_config_arg,
            Node(
                package="cake_slam",
                executable="cake_slam_mapping_multi_cam",
                name="laserMapping",
                parameters=[
                    mapping_params,
                    camera_params,
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
