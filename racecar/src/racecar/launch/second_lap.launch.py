#!/usr/bin/env python3
"""Launch saved-map localization and low-speed first-lap route replay."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def launch_ab_detector(context, script, visible):
    """Run the board's existing A_B.py in detection-only mode."""
    command = ['python3', script.perform(context)]
    if visible.perform(context).strip().lower() in ('1', 'true', 'yes', 'on'):
        command.append('--visible')
    return [ExecuteProcess(cmd=command, output='screen')]


def generate_launch_description():
    racecar_share = get_package_share_directory('racecar')
    launch_dir = os.path.join(racecar_share, 'launch')
    config_dir = os.path.join(racecar_share, 'config')

    map_file = LaunchConfiguration('map')
    route_file = LaunchConfiguration('route_file')
    pose_file = LaunchConfiguration('pose_file')
    params_file = LaunchConfiguration('params_file')
    navigation_start_delay = LaunchConfiguration('navigation_start_delay')
    localization_settle_time = LaunchConfiguration('localization_settle_time')
    max_initial_pose_error = LaunchConfiguration('max_initial_pose_error')
    max_initial_yaw_error = LaunchConfiguration('max_initial_yaw_error')
    maximum_linear_speed = LaunchConfiguration('maximum_linear_speed')
    maximum_angular_speed = LaunchConfiguration('maximum_angular_speed')
    nav_angular_to_steering_scale = LaunchConfiguration(
        'nav_angular_to_steering_scale')
    steering_command_bias = LaunchConfiguration('steering_command_bias')
    steering_command_sign = LaunchConfiguration('steering_command_sign')
    maximum_steering_command = LaunchConfiguration(
        'maximum_steering_command')
    steering_command_smoothing = LaunchConfiguration(
        'steering_command_smoothing')
    maximum_steering_step = LaunchConfiguration('maximum_steering_step')
    derive_route_headings = LaunchConfiguration('derive_route_headings')
    sign_arm_progress = LaunchConfiguration('sign_arm_progress')
    sign_confirm_count = LaunchConfiguration('sign_confirm_count')
    start_ab_detector = LaunchConfiguration('start_ab_detector')
    require_detector_heartbeat = LaunchConfiguration(
        'require_detector_heartbeat')
    detector_heartbeat_timeout = LaunchConfiguration(
        'detector_heartbeat_timeout')
    closed_loop_distance = LaunchConfiguration('closed_loop_distance')
    maximum_route_point_gap = LaunchConfiguration(
        'maximum_route_point_gap')
    ab_script = LaunchConfiguration('ab_script')
    ab_visible = LaunchConfiguration('ab_visible')
    use_rviz = LaunchConfiguration('use_rviz')

    arguments = [
        DeclareLaunchArgument(
            'map',
            default_value='/home/davinci-mini/racecar/src/racecar/map/first_lap_map.yaml',
            description='Saved first-lap map YAML'),
        DeclareLaunchArgument(
            'route_file',
            default_value='/home/davinci-mini/racecar/src/racecar/map/first_lap_route.yaml',
            description='Saved first-lap route YAML'),
        DeclareLaunchArgument(
            'pose_file',
            default_value='/home/davinci-mini/racecar/src/racecar/map/first_lap_pose.yaml',
            description='Saved first-lap final-pose YAML'),
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(config_dir, 'nav.yaml'),
            description='Nav2 parameter YAML'),
        DeclareLaunchArgument(
            'navigation_start_delay', default_value='5.0',
            description='Minimum wait before sending the saved pose/path'),
        DeclareLaunchArgument(
            'localization_settle_time', default_value='5.0',
            description='Seconds for AMCL laser localization to settle after initialpose'),
        DeclareLaunchArgument(
            'max_initial_pose_error', default_value='0.30',
            description='Maximum AMCL XY error from saved starting pose in metres'),
        DeclareLaunchArgument(
            'max_initial_yaw_error', default_value='0.25',
            description='Maximum AMCL heading error from saved starting pose in radians'),
        DeclareLaunchArgument(
            'maximum_linear_speed', default_value='0.55',
            description='Second-lap speed cap; 0.55 is the driver minimum'),
        DeclareLaunchArgument(
            'maximum_angular_speed', default_value='0.45',
            description='Second-lap safety angular-speed cap in rad/s'),
        DeclareLaunchArgument(
            'nav_angular_to_steering_scale', default_value='0.80',
            description='Convert Nav2 angular velocity to driver steering command'),
        DeclareLaunchArgument(
            'steering_command_bias', default_value='-0.064',
            description='Normalized steering bias matching the first-lap servo neutral'),
        DeclareLaunchArgument(
            'steering_command_sign', default_value='1.0',
            description='Sign mapping from Nav2 left-positive angular velocity to steering'),
        DeclareLaunchArgument(
            'maximum_steering_command', default_value='0.36',
            description='Maximum normalized steering sent to the chassis'),
        DeclareLaunchArgument(
            'steering_command_smoothing', default_value='0.35',
            description='Low-pass smoothing for converted steering command [0,1)'),
        DeclareLaunchArgument(
            'maximum_steering_step', default_value='0.04',
            description='Maximum normalized steering change per 20 Hz command'),
        DeclareLaunchArgument(
            'derive_route_headings', default_value='true',
            description='Derive path headings from recorded route geometry'),
        DeclareLaunchArgument(
            'sign_arm_progress', default_value='0.70',
            description='Ignore A/B signs until this route progress [0,1]'),
        DeclareLaunchArgument(
            'sign_confirm_count', default_value='4',
            description='Consecutive A/B messages required to stop'),
        DeclareLaunchArgument(
            'start_ab_detector', default_value='true',
            description='Start the existing A_B.py without --drive'),
        DeclareLaunchArgument(
            'require_detector_heartbeat', default_value='true',
            description='Stop if A_B.py/camera heartbeat is unavailable'),
        DeclareLaunchArgument(
            'detector_heartbeat_timeout', default_value='10.0',
            description='Allowed seconds between slow A/B inference frames'),
        DeclareLaunchArgument(
            'closed_loop_distance', default_value='1.0',
            description='Maximum safe first-lap end-to-start gap in metres'),
        DeclareLaunchArgument(
            'maximum_route_point_gap', default_value='0.75',
            description='Maximum safe distance between adjacent route points'),
        DeclareLaunchArgument(
            'ab_script', default_value='/home/davinci-mini/racecar/A_B/A_B.py',
            description='Absolute path to the existing A/B recognition script'),
        DeclareLaunchArgument(
            'ab_visible', default_value='false',
            description='Show the A/B OpenCV window'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true',
            description='Start RViz with the navigation display'),
    ]

    base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'Run_car.launch.py')),
        launch_arguments={'use_rviz': use_rviz}.items())

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'bringup_launch.py')),
        launch_arguments={
            'map': map_file,
            'params_file': params_file,
            'use_sim_time': 'false',
            'slam': 'false',
            'autostart': 'true',
        }.items())

    manager = Node(
        package='racecar',
        executable='second_lap_manager.py',
        name='second_lap_manager',
        output='screen',
        parameters=[{
            'route_file': route_file,
            'pose_file': pose_file,
            'navigation_start_delay': ParameterValue(
                navigation_start_delay, value_type=float),
            'localization_settle_time': ParameterValue(
                localization_settle_time, value_type=float),
            'max_initial_pose_error': ParameterValue(
                max_initial_pose_error, value_type=float),
            'max_initial_yaw_error': ParameterValue(
                max_initial_yaw_error, value_type=float),
            'maximum_linear_speed': ParameterValue(
                maximum_linear_speed, value_type=float),
            'maximum_angular_speed': ParameterValue(
                maximum_angular_speed, value_type=float),
            'nav_angular_to_steering_scale': ParameterValue(
                nav_angular_to_steering_scale, value_type=float),
            'steering_command_bias': ParameterValue(
                steering_command_bias, value_type=float),
            'steering_command_sign': ParameterValue(
                steering_command_sign, value_type=float),
            'maximum_steering_command': ParameterValue(
                maximum_steering_command, value_type=float),
            'steering_command_smoothing': ParameterValue(
                steering_command_smoothing, value_type=float),
            'maximum_steering_step': ParameterValue(
                maximum_steering_step, value_type=float),
            'derive_route_headings': ParameterValue(
                derive_route_headings, value_type=bool),
            'sign_arm_progress': ParameterValue(
                sign_arm_progress, value_type=float),
            'sign_confirm_count': ParameterValue(
                sign_confirm_count, value_type=int),
            'require_detector_heartbeat': ParameterValue(
                require_detector_heartbeat, value_type=bool),
            'detector_heartbeat_timeout': ParameterValue(
                detector_heartbeat_timeout, value_type=float),
            'closed_loop_distance': ParameterValue(
                closed_loop_distance, value_type=float),
            'maximum_route_point_gap': ParameterValue(
                maximum_route_point_gap, value_type=float),
        }],
        # The manager owns localization initialization and the final command
        # arbitration.  Keeping Nav2 and the driver alive after it fails only
        # produces misleading AMCL/TF errors and can leave stale processes.
        on_exit=Shutdown(reason='second_lap_manager exited'),
    )

    detector = OpaqueFunction(
        function=launch_ab_detector,
        args=[ab_script, ab_visible],
        condition=IfCondition(start_ab_detector),
    )

    return LaunchDescription(arguments + [base, nav2, manager, detector])
