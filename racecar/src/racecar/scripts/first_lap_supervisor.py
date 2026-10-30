#!/usr/bin/env python3
"""Run a timed first lap, then save its map, route, and final map pose."""

import math
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
import yaml


class FirstLapSupervisor(Node):
    """A small command mux with a latched timed-stop state.

    The lidar tracker publishes PWM-style values on ``source_topic``.  This
    node is the only publisher used for the driver's teleop topic, so the
    tracker cannot overwrite the stop command after the lap timer expires.
    """

    def __init__(self) -> None:
        super().__init__('first_lap_supervisor')

        self.source_topic = str(self.declare_parameter(
            'source_topic', '/first_lap/tracking_cmd_vel').value)
        self.output_topic = str(self.declare_parameter(
            'output_topic', '/teleop_cmd_vel').value)
        self.map_topic = str(self.declare_parameter('map_topic', '/map').value)
        self.map_frame = str(self.declare_parameter('map_frame', 'map').value)
        self.odom_frame = str(self.declare_parameter(
            'odom_frame', 'odom_combined').value)
        self.base_frame = str(self.declare_parameter(
            'base_frame', 'base_footprint').value)
        self.lap_duration = float(self.declare_parameter('lap_duration', 20.0).value)
        self.start_delay = float(self.declare_parameter('start_delay', 0.0).value)
        self.stop_publish_rate = float(self.declare_parameter(
            'stop_publish_rate', 20.0).value)
        self.start_on_first_command = bool(self.declare_parameter(
            'start_on_first_command', True).value)
        self.map_dir = os.path.expanduser(str(self.declare_parameter(
            'map_dir', '~/racecar/src/racecar/map').value))
        self.map_name = str(self.declare_parameter(
            'map_name', 'first_lap_map').value)
        self.route_file = os.path.expanduser(str(self.declare_parameter(
            'route_file', '').value))
        self.pose_file = os.path.expanduser(str(self.declare_parameter(
            'pose_file', '').value))
        self.route_sample_distance = float(self.declare_parameter(
            'route_sample_distance', 0.25).value)
        self.route_closure_warning_distance = float(self.declare_parameter(
            'route_closure_warning_distance', 1.0).value)
        self.stop_settle_time = float(self.declare_parameter(
            'stop_settle_time', 0.5).value)
        self.save_timeout = float(self.declare_parameter(
            'save_timeout', 10.0).value)
        self.shutdown_after_save = bool(self.declare_parameter(
            'shutdown_after_save', False).value)
        self.stop_linear = float(self.declare_parameter('stop_linear', 1500.0).value)
        self.stop_steering = float(self.declare_parameter('stop_steering', 90.0).value)

        if self.lap_duration <= 0.0:
            raise ValueError('lap_duration must be greater than zero')
        if self.stop_publish_rate <= 0.0:
            raise ValueError('stop_publish_rate must be greater than zero')
        if self.route_sample_distance <= 0.0:
            raise ValueError('route_sample_distance must be greater than zero')
        if self.route_closure_warning_distance <= 0.0:
            raise ValueError(
                'route_closure_warning_distance must be greater than zero')
        if self.stop_settle_time < 0.0:
            raise ValueError('stop_settle_time must not be negative')

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(Twist, self.output_topic, qos)
        self.subscription = self.create_subscription(
            Twist, self.source_topic, self._tracking_callback, qos)
        self.timer = self.create_timer(
            1.0 / self.stop_publish_rate, self._timer_callback)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_command: Optional[Twist] = None
        self.started_at: Optional[float] = None
        self.finished = False
        self.finished_at: Optional[float] = None
        self.save_started = False
        self._last_status = 0.0
        self._last_tf_warning = 0.0
        # Record in odometry coordinates because gmapping continuously adjusts
        # map->odom while optimizing the map.  Sampling map->base directly
        # would mix poses produced by different map corrections and can leave
        # metre-scale jumps in the saved FollowPath route.  At save time the
        # complete odometry route is transformed once with the final
        # map->odom transform.
        self.odom_route = []

        self.get_logger().info(
            f'First lap: {self.lap_duration:.2f}s, '
            f'source={self.source_topic}, output={self.output_topic}, '
            f'map={self.map_dir}/{self.map_name}, '
            f'route spacing={self.route_sample_distance:.2f}m'
        )

    def _is_motion_command(self, message: Twist) -> bool:
        return abs(message.linear.x) > 1.0e-6 or abs(message.angular.z) > 1.0e-6

    def _tracking_callback(self, message: Twist) -> None:
        if self.finished:
            return

        self.last_command = message
        if self.started_at is None:
            if self.start_on_first_command and not self._is_motion_command(message):
                return
            self.started_at = time.monotonic() + self.start_delay
            self.get_logger().info(
                f'Lap timer started; start delay={self.start_delay:.2f}s, '
                f'duration={self.lap_duration:.2f}s'
            )

    def _stop_command(self) -> Twist:
        command = Twist()
        # The standard driver interprets teleop_cmd_vel as raw PWM/servo
        # values: 1500 is motor stop and 90 degrees is steering neutral.
        command.linear.x = self.stop_linear
        command.angular.z = self.stop_steering
        return command

    def _timer_callback(self) -> None:
        now = time.monotonic()

        # Keep the lap time adjustable with ``ros2 param set`` while the
        # vehicle is running.  This is useful when the first timing pass shows
        # that the configured duration is short or long.
        configured_duration = float(self.get_parameter('lap_duration').value)
        if configured_duration <= 0.0:
            self.get_logger().error('lap_duration must remain greater than zero')
            return
        if configured_duration != self.lap_duration:
            self.lap_duration = configured_duration
            self.get_logger().info(
                f'Lap duration updated to {self.lap_duration:.2f}s')

        if self.finished:
            self.publisher.publish(self._stop_command())
            if (
                not self.save_started
                and self.finished_at is not None
                and now - self.finished_at >= self.stop_settle_time
            ):
                self._record_odom_pose(force=True)
                self._start_map_save()
            return

        if self.started_at is None or now < self.started_at:
            self.publisher.publish(self._stop_command())
            return

        self._record_odom_pose()

        if now - self.started_at >= self.lap_duration:
            self.finished = True
            self.finished_at = now
            self.publisher.publish(self._stop_command())
            self.get_logger().info(
                'Lap duration reached; vehicle stopped. '
                f'Waiting {self.stop_settle_time:.2f}s before saving'
            )
            return

        if self.last_command is not None:
            self.publisher.publish(self.last_command)

        if now - self._last_status >= 1.0:
            remaining = max(0.0, self.lap_duration - (now - self.started_at))
            self.get_logger().info(
                f'First-lap time remaining: {remaining:.1f}s')
            self._last_status = now

    def _map_prefix(self) -> str:
        os.makedirs(self.map_dir, exist_ok=True)
        if os.path.isabs(self.map_name):
            return self.map_name
        return os.path.join(self.map_dir, self.map_name)

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        sin_yaw = 2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y
        )
        cos_yaw = 1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
        )
        return math.atan2(sin_yaw, cos_yaw)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def _record_odom_pose(self, force: bool = False) -> None:
        """Record the driven centerline in the continuous odometry frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame, self.base_frame, Time())
        except TransformException as error:
            now = time.monotonic()
            if now - self._last_tf_warning >= 1.0:
                self.get_logger().warn(
                    f'Waiting for {self.odom_frame}->{self.base_frame} TF: '
                    f'{error}'
                )
                self._last_tf_warning = now
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        pose = {
            'x': float(translation.x),
            'y': float(translation.y),
            'yaw': float(self._yaw_from_quaternion(rotation)),
        }
        if not self.odom_route:
            self.odom_route.append(pose)
            self.get_logger().info('First route point recorded')
            return

        previous = self.odom_route[-1]
        distance = math.hypot(
            pose['x'] - previous['x'], pose['y'] - previous['y'])
        if distance >= self.route_sample_distance:
            self.odom_route.append(pose)
        elif force and distance > 1.0e-3:
            self.odom_route.append(pose)

    def _artifact_path(self, configured_path: str, suffix: str) -> str:
        if configured_path:
            if os.path.isabs(configured_path):
                return configured_path
            return os.path.join(self.map_dir, configured_path)

        map_prefix = self._map_prefix()
        directory, basename = os.path.split(map_prefix)
        stem = basename[:-4] if basename.endswith('_map') else basename
        return os.path.join(directory, f'{stem}{suffix}')

    def _route_in_final_map(self):
        if not self.odom_route:
            raise RuntimeError('no odometry route points were recorded')

        # Use one final SLAM correction for every route point.  This preserves
        # route continuity while placing it in the same final map frame that
        # map_saver_cli writes to disk.
        transform = self.tf_buffer.lookup_transform(
            self.map_frame, self.odom_frame, Time())
        translation = transform.transform.translation
        map_to_odom_yaw = self._yaw_from_quaternion(
            transform.transform.rotation)
        cos_yaw = math.cos(map_to_odom_yaw)
        sin_yaw = math.sin(map_to_odom_yaw)

        route = []
        for index, pose in enumerate(self.odom_route):
            route.append({
                'index': index,
                'x': float(
                    translation.x
                    + cos_yaw * pose['x']
                    - sin_yaw * pose['y']),
                'y': float(
                    translation.y
                    + sin_yaw * pose['x']
                    + cos_yaw * pose['y']),
                'yaw': float(self._normalize_angle(
                    map_to_odom_yaw + pose['yaw'])),
            })
        return route

    @staticmethod
    def _route_length(route) -> float:
        return sum(
            math.hypot(
                current['x'] - previous['x'],
                current['y'] - previous['y'])
            for previous, current in zip(route, route[1:]))

    @staticmethod
    def _write_yaml(path: str, data) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f'{path}.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as output:
            yaml.safe_dump(
                data, output, allow_unicode=True, sort_keys=False)
        os.replace(temporary_path, path)

    def _save_route_artifacts(self) -> bool:
        try:
            route = self._route_in_final_map()
        except (TransformException, RuntimeError) as error:
            self.get_logger().error(f'Could not prepare first-lap route: {error}')
            return False

        route_path = self._artifact_path(self.route_file, '_route.yaml')
        pose_path = self._artifact_path(self.pose_file, '_pose.yaml')
        final_pose = dict(route[-1])
        final_pose.pop('index', None)
        route_length = self._route_length(route)
        closure_distance = math.hypot(
            route[-1]['x'] - route[0]['x'],
            route[-1]['y'] - route[0]['y'])
        generated_at = datetime.now(timezone.utc).isoformat()
        route_document = {
            'format_version': 1,
            'frame_id': self.map_frame,
            'source_frame': self.odom_frame,
            'odom_frame': self.odom_frame,
            'base_frame': self.base_frame,
            'map_yaml': f'{self._map_prefix()}.yaml',
            'generated_at': generated_at,
            'lap_duration': float(self.lap_duration),
            'sample_distance': float(self.route_sample_distance),
            'point_count': len(route),
            'route_length': float(route_length),
            'closure_distance': float(closure_distance),
            'poses': route,
        }
        pose_document = {
            'format_version': 1,
            'frame_id': self.map_frame,
            'generated_at': generated_at,
            'pose': final_pose,
        }

        try:
            self._write_yaml(route_path, route_document)
            self._write_yaml(pose_path, pose_document)
        except OSError as error:
            self.get_logger().error(f'Could not save route artifacts: {error}')
            return False

        self.get_logger().info(
            f'First-lap route saved: {route_path} ({len(route)} points)')
        self.get_logger().info(f'First-lap final pose saved: {pose_path}')
        if closure_distance > self.route_closure_warning_distance:
            self.get_logger().error(
                'First-lap route is not closed enough for second-lap replay: '
                f'end-to-start={closure_distance:.2f}m, '
                f'limit={self.route_closure_warning_distance:.2f}m. '
                'Adjust lap_duration and record the first lap again')
        else:
            self.get_logger().info(
                f'First-lap route closure check passed: '
                f'{closure_distance:.2f}m <= '
                f'{self.route_closure_warning_distance:.2f}m')
        return True

    def _start_map_save(self) -> None:
        if self.save_started:
            return
        self.save_started = True
        thread = threading.Thread(target=self._save_map, daemon=True)
        thread.start()

    def _save_map(self) -> None:
        prefix = self._map_prefix()
        self._save_route_artifacts()
        command = [
            'ros2', 'run', 'nav2_map_server', 'map_saver_cli',
            '-f', prefix,
            '--ros-args',
            '-p', f'map_topic:={self.map_topic}',
            '-p', 'map_subscribe_transient_local:=true',
            '-p', f'save_map_timeout:={self.save_timeout}',
        ]
        self.get_logger().info(f'Saving first-lap map to {prefix}')
        try:
            result = subprocess.run(command, check=False)
        except OSError as error:
            self.get_logger().error(f'Could not start map_saver_cli: {error}')
            return

        if result.returncode == 0:
            self.get_logger().info(
                f'First-lap map saved: {prefix}.yaml and {prefix}.pgm')
        else:
            self.get_logger().error(
                f'map_saver_cli failed with exit code {result.returncode}')

        if self.shutdown_after_save:
            self.get_logger().info('Map save finished; shutting down supervisor')
            rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FirstLapSupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
