from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    launch_arguments = [
        DeclareLaunchArgument('camera_namespace', default_value='robot'),
        DeclareLaunchArgument('camera_name', default_value='arm_rgbd_camera'),
        DeclareLaunchArgument('serial_no', default_value="''"),
        DeclareLaunchArgument('publish_tf', default_value='false'),
        DeclareLaunchArgument('tf_prefix', default_value='robot_'),
        DeclareLaunchArgument('color_profile', default_value='640x480x30'),
        DeclareLaunchArgument('depth_profile', default_value='640x480x30'),
        DeclareLaunchArgument('use_rviz', default_value='false'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=PathJoinSubstitution([
                FindPackageShare('renee_perception'),
                'config',
                'realsense_wrist_preview.rviz',
            ]),
        ),
    ]

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py',
            ])
        ),
        launch_arguments={
            'camera_namespace': LaunchConfiguration('camera_namespace'),
            'camera_name': LaunchConfiguration('camera_name'),
            'serial_no': LaunchConfiguration('serial_no'),
            'device_type': 'd435i',
            'enable_color': 'true',
            'enable_depth': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'publish_tf': LaunchConfiguration('publish_tf'),
            'tf_prefix': LaunchConfiguration('tf_prefix'),
            'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
            'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
        }.items(),
    )

    preview = Node(
        package='rviz2',
        executable='rviz2',
        name='realsense_wrist_preview',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    overlay_preview = Node(
        package='renee_perception',
        executable='rgbd_preview.py',
        name='rgbd_preview',
        output='screen',
        parameters=[{
            'rgb_topic': '/robot/arm_rgbd_camera/color/image_raw',
            'depth_topic': '/robot/arm_rgbd_camera/aligned_depth_to_color/image_raw',
            'camera_frame': 'robot_arm_rgbd_camera_color_optical_frame',
            'reference_frame': 'robot_base_link',
        }],
        condition=UnlessCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription(launch_arguments + [
        camera_launch,
        preview,
        overlay_preview,
    ])
