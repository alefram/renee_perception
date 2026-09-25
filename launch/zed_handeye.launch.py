"""Publish validated ZED left optical frame relative to UR5 tool0.

ROS 2 Jazzy. Translation in metres, quaternion xyzw.
PARK result from 15 refined captures, validated on 5 held-out captures.
This adds a separate calibrated optical frame; it does not alter CAD frames
or change Image/PointCloud header.frame_id. Do not run a second broadcaster
for the same child frame. Recalibrate if camera mounting changes.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='zed_handeye_static_tf',
            output='screen',
            arguments=[
                '--x', '0.0702075204',
                '--y', '0.1125351639',
                '--z', '-0.0383535673',
                '--qx', '-0.00776924',
                '--qy', '-0.05934987',
                '--qz', '0.99810419',
                '--qw', '-0.01432717',
                '--frame-id', 'robot_arm_tool0',
                '--child-frame-id', 'zed_left_optical_calibrated',
            ],
        ),
    ])
