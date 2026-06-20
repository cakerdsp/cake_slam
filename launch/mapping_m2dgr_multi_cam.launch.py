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
    config_file_dir = os.path.join(get_package_share_directory("fast_livo"), "config")
    rviz_config_file = os.path.join(
        get_package_share_directory("fast_livo"), "rviz_cfg", "fast_livo2.rviz"
    )

    mapping_config = os.path.join(config_file_dir, "M2DGR_multi_cam.yaml")
    camera_config = os.path.join(config_file_dir, "camera_M2DGR_multi_cam.yaml")

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="True",
        description="Whether to launch Rviz2",
    )
    num_cameras_arg = DeclareLaunchArgument(
        "num_cameras",
        default_value="6",
        description="Enable the first N M2DGR fisheye cameras (1..7)",
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

    num_cameras = LaunchConfiguration("num_cameras")
    mapping_params = LaunchConfiguration("m2dgr_params_file")
    camera_params = LaunchConfiguration("camera_params_file")

    return LaunchDescription(
        [
            use_rviz_arg,
            num_cameras_arg,
            mapping_config_arg,
            camera_config_arg,
            Node(
                package="fast_livo",
                executable="fastlivo_mapping_multi_cam",
                name="laserMapping",
                parameters=[
                    mapping_params,
                    camera_params,
                    {
                        "common.num_cameras": ParameterValue(
                            num_cameras, value_type=int
                        )
                    },
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
