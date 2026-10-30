from launch import LaunchDescription
from launch.substitutions import EnvironmentVariable
import launch.actions
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    use_sim_time = launch.substitutions.LaunchConfiguration('use_sim_time', default='false')
    linear_update = launch.substitutions.LaunchConfiguration('linear_update')
    angular_update = launch.substitutions.LaunchConfiguration('angular_update')
    temporal_update = launch.substitutions.LaunchConfiguration('temporal_update')
    map_update_interval = launch.substitutions.LaunchConfiguration('map_update_interval')
    delta = launch.substitutions.LaunchConfiguration('delta')
    base_frame = launch.substitutions.LaunchConfiguration('base_frame')
    odom_frame = launch.substitutions.LaunchConfiguration('odom_frame')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation time')
    declare_linear_update = DeclareLaunchArgument(
        'linear_update', default_value='0.25', description='Minimum translation before scan matching')
    declare_angular_update = DeclareLaunchArgument(
        'angular_update', default_value='0.15', description='Minimum rotation before scan matching')
    declare_temporal_update = DeclareLaunchArgument(
        'temporal_update', default_value='0.5', description='Maximum time between scan updates')
    declare_map_update_interval = DeclareLaunchArgument(
        'map_update_interval', default_value='0.2', description='Minimum interval for publishing the map')
    declare_delta = DeclareLaunchArgument(
        'delta', default_value='0.05', description='Map resolution in metres')
    declare_base_frame = DeclareLaunchArgument(
        'base_frame', default_value='base_link', description='Robot base frame')
    declare_odom_frame = DeclareLaunchArgument(
        'odom_frame', default_value='odom_combined', description='Odometry frame')

    # bringup_dir = get_package_share_directory('racecar')

    return LaunchDescription([
        # wheeltec_robot,
        # wheeltec_robot,wheeltec_lidar,
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '1'),
        declare_use_sim_time,
        declare_linear_update,
        declare_angular_update,
        declare_temporal_update,
        declare_map_update_interval,
        declare_delta,
        declare_base_frame,
        declare_odom_frame,
        launch_ros.actions.Node(
            package='slam_gmapping',
            executable='slam_gmapping',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'linear_update': linear_update,
                'angular_update': angular_update,
                'temporal_update': temporal_update,
                'map_update_interval': map_update_interval,
                'delta': delta,
                'base_frame': base_frame,
                'odom_frame': odom_frame,
            }]),
    ])
