#!/usr/bin/env python3
"""Launch the first-lap cone tracker, odometry, gmapping, and map saver."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def schedule_tracker(context, tracker, delay):
    """Resolve the launch argument to the float required by TimerAction."""
    delay_seconds = float(delay.perform(context))
    if delay_seconds < 0.0:
        raise ValueError('tracking_start_delay must not be negative')
    return [TimerAction(period=delay_seconds, actions=[tracker])]


def generate_launch_description():
    racecar_share = get_package_share_directory('racecar')
    lidar_share = get_package_share_directory('lidar_tracking')
    slam_share = get_package_share_directory('slam_gmapping')

    lap_duration = LaunchConfiguration('lap_duration')
    map_dir = LaunchConfiguration('map_dir')
    map_name = LaunchConfiguration('map_name')
    route_file = LaunchConfiguration('route_file')
    pose_file = LaunchConfiguration('pose_file')
    route_sample_distance = LaunchConfiguration('route_sample_distance')
    route_closure_warning_distance = LaunchConfiguration(
        'route_closure_warning_distance')
    stop_settle_time = LaunchConfiguration('stop_settle_time')
    tracking_topic = LaunchConfiguration('tracking_topic')
    tracking_start_delay = LaunchConfiguration('tracking_start_delay')
    startup_delay = LaunchConfiguration('startup_delay')
    save_timeout = LaunchConfiguration('save_timeout')
    linear_update = LaunchConfiguration('linear_update')
    angular_update = LaunchConfiguration('angular_update')
    temporal_update = LaunchConfiguration('temporal_update')
    map_update_interval = LaunchConfiguration('map_update_interval')

    arguments = [
        DeclareLaunchArgument(
            'lap_duration', default_value='20.0',
            description='First-lap duration in seconds; tune this on the track'),
        DeclareLaunchArgument(
            'map_dir', default_value='/home/davinci-mini/racecar/src/racecar/map',
            description='Directory for the saved map'),
        DeclareLaunchArgument(
            'map_name', default_value='first_lap_map',
            description='Map file prefix without .yaml or .pgm'),
        DeclareLaunchArgument(
            'route_file', default_value='',
            description='Optional route YAML path; empty derives it from map_name'),
        DeclareLaunchArgument(
            'pose_file', default_value='',
            description='Optional final-pose YAML path; empty derives it from map_name'),
        DeclareLaunchArgument(
            'route_sample_distance', default_value='0.25',
            description='Minimum distance in metres between recorded route points'),
        DeclareLaunchArgument(
            'route_closure_warning_distance', default_value='1.0',
            description='Warn when the recorded lap closure exceeds this distance'),
        DeclareLaunchArgument(
            'stop_settle_time', default_value='0.5',
            description='Stop settling time before saving final route pose and map'),
        DeclareLaunchArgument(
            'tracking_topic', default_value='/first_lap/tracking_cmd_vel',
            description='Private topic between tracker and supervisor'),
        DeclareLaunchArgument(
            'tracking_start_delay', default_value='12.0',
            description='Delay from launch/RViz startup before starting the tracker'),
        DeclareLaunchArgument(
            'startup_delay', default_value='0.0',
            description='Optional extra stop delay after the tracker sends its first command'),
        DeclareLaunchArgument(
            'save_timeout', default_value='10.0',
            description='Map saver timeout in seconds'),
        DeclareLaunchArgument(
            'linear_update', default_value='0.25',
            description='Gmapping translation update threshold in metres'),
        DeclareLaunchArgument(
            'angular_update', default_value='0.15',
            description='Gmapping rotation update threshold in radians'),
        DeclareLaunchArgument(
            'temporal_update', default_value='0.5',
            description='Gmapping maximum scan update period in seconds'),
        DeclareLaunchArgument(
            'map_update_interval', default_value='0.2',
            description='Minimum map publication interval in seconds'),
    ]

    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(racecar_share, 'launch', 'Run_car.launch.py')))

    gmapping = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_share, 'launch', 'slam_gmapping.launch.py')),
        launch_arguments={
            'linear_update': linear_update,
            'angular_update': angular_update,
            'temporal_update': temporal_update,
            'map_update_interval': map_update_interval,
        }.items())

    tracker = Node(
        package='lidar_tracking',
        executable='lidar_tracking',
        name='lidar_tracking_first_lap',
        output='screen',
        parameters=[
            os.path.join(lidar_share, 'config', 'params.yaml'),
            {'topic': tracking_topic},
        ])

    delayed_tracker = OpaqueFunction(
        function=schedule_tracker,
        args=[tracker, tracking_start_delay],
    )

    supervisor = Node(
        package='racecar',
        executable='first_lap_supervisor.py',
        name='first_lap_supervisor',
        output='screen',
        parameters=[{
            'source_topic': tracking_topic,
            'output_topic': '/teleop_cmd_vel',
            # Explicit types prevent Humble from interpreting command-line
            # values such as "20" and "12" as INTEGER parameters.
            'lap_duration': ParameterValue(lap_duration, value_type=float),
            'start_delay': ParameterValue(startup_delay, value_type=float),
            'map_dir': map_dir,
            'map_name': map_name,
            'route_file': route_file,
            'pose_file': pose_file,
            'route_sample_distance': ParameterValue(
                route_sample_distance, value_type=float),
            'route_closure_warning_distance': ParameterValue(
                route_closure_warning_distance, value_type=float),
            'stop_settle_time': ParameterValue(
                stop_settle_time, value_type=float),
            'save_timeout': ParameterValue(save_timeout, value_type=float),
        }])

    return LaunchDescription(arguments + [base, gmapping, supervisor, delayed_tracker])
