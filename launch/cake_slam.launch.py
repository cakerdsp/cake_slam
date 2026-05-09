from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("cake_slam"), "config", "cake_slam.yaml"]
    )

    config_file = LaunchConfiguration("config_file")
    rviz_config = LaunchConfiguration("rviz_config")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_value = ParameterValue(use_sim_time, value_type=bool)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="Path to the unified cake_slam OpenCV YAML config.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("cake_slam"), "config", "cake_slam.rviz"]
                ),
                description="RViz config file for cake_slam visualization.",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Whether to launch RViz.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use /clock from simulation or rosbag playback.",
            ),
            Node(
                package="cake_slam",
                executable="cake_slam_node",
                name="cake_slam",
                output="screen",
                parameters=[{"config_file": config_file, "use_sim_time": use_sim_time_value}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time_value}],
                output="screen",
                condition=IfCondition(use_rviz),
            ),
        ]
    )
