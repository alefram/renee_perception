from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([
                FindPackageShare("renee_perception"),
                "config",
                "rgbd_capture_sim.yaml",
            ]),
            description="Path to the RGB-D capture parameter YAML file.",
        ),
        Node(
            package="renee_perception",
            executable="rgbd_capture_node",
            name="rgbd_capture_node",
            output="screen",
            parameters=[params_file],
        ),
    ])
