from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("cake_slam"), "config", "cake_slam.yaml"]
    )

    config_file = LaunchConfiguration("config_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to the unified cake_slam OpenCV YAML config.",
            ),
            Node(
                package="cake_slam",
                executable="cake_slam_node",
                name="cake_slam",
                output="screen",
                parameters=[{"config_file": config_file}],
            ),
        ]
    )
