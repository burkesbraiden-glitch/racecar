#!/usr/bin/env python3
"""Initialize AMCL, replay the first-lap path, and latch a safe stop."""

import math
import os
import time
import fcntl
from enum import Enum, auto
from typing import Dict, List, Optional

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import FollowPath
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, Float32, String
from tf2_ros import Buffer, TransformException, TransformListener


class StageState(Enum):
    WAITING_FOR_MAP = auto()
    INITIALIZING_AMCL = auto()
    WAITING_FOR_NAV2 = auto()
    NAVIGATING = auto()
    SIGN_STOP = auto()
    COMPLETE = auto()
    EMERGENCY_STOP = auto()
    FAULT = auto()


class SecondLapManager(Node):
    """The only second-stage node allowed to publish to ``/car_cmd_vel``."""

    def __init__(self) -> None:
        super().__init__('second_lap_manager')

        # Keep a completed/old second-stage launch from silently fighting a
        # newly started manager on /car_cmd_vel.  flock is released
        # automatically when the process exits, including Ctrl+C.
        self._instance_lock = open(
            '/tmp/racecar_second_lap_manager.lock', 'w', encoding='utf-8')
        try:
            fcntl.flock(
                self._instance_lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                'another second_lap_manager is still running; stop the old '
                'second_lap.sh with Ctrl+C before starting a new lap') from error
        self._instance_lock.write(str(os.getpid()))
        self._instance_lock.flush()

        self.route_file = os.path.expanduser(str(self.declare_parameter(
            'route_file',
            '~/racecar/src/racecar/map/first_lap_route.yaml').value))
        self.pose_file = os.path.expanduser(str(self.declare_parameter(
            'pose_file',
            '~/racecar/src/racecar/map/first_lap_pose.yaml').value))
        self.map_frame = str(self.declare_parameter('map_frame', 'map').value)
        self.base_frame = str(self.declare_parameter(
            'base_frame', 'base_footprint').value)
        self.odom_topic = str(self.declare_parameter(
            'odom_topic', '/odom_combined').value)
        self.nav_command_topic = str(self.declare_parameter(
            'nav_command_topic', '/cmd_vel').value)
        self.output_command_topic = str(self.declare_parameter(
            'output_command_topic', '/car_cmd_vel').value)
        self.sign_topic = str(self.declare_parameter(
            'sign_topic', '/ab_sign').value)
        self.emergency_stop_topic = str(self.declare_parameter(
            'emergency_stop_topic', '/stage2/emergency_stop').value)
        self.detector_heartbeat_topic = str(self.declare_parameter(
            'detector_heartbeat_topic', '/ab_detector/heartbeat').value)

        self.navigation_start_delay = float(self.declare_parameter(
            'navigation_start_delay', 5.0).value)
        self.startup_timeout = float(self.declare_parameter(
            'startup_timeout', 90.0).value)
        self.localization_settle_time = float(self.declare_parameter(
            'localization_settle_time', 5.0).value)
        self.initial_pose_publish_period = float(self.declare_parameter(
            'initial_pose_publish_period', 1.0).value)
        self.max_xy_covariance = float(self.declare_parameter(
            'max_xy_covariance', 0.50).value)
        self.max_yaw_covariance = float(self.declare_parameter(
            'max_yaw_covariance', 0.30).value)
        self.max_initial_pose_error = float(self.declare_parameter(
            'max_initial_pose_error', 0.30).value)
        self.max_initial_yaw_error = float(self.declare_parameter(
            'max_initial_yaw_error', 0.25).value)
        self.initial_xy_covariance = float(self.declare_parameter(
            'initial_xy_covariance', 0.25).value)
        self.initial_yaw_covariance = float(self.declare_parameter(
            'initial_yaw_covariance', 0.0685).value)
        self.maximum_linear_speed = float(self.declare_parameter(
            'maximum_linear_speed', 0.55).value)
        self.maximum_angular_speed = float(self.declare_parameter(
            'maximum_angular_speed', 0.45).value)
        # Nav2 publishes angular.z as a yaw rate in rad/s, whereas this
        # chassis expects a normalized steering command on /car_cmd_vel.
        # Keep the converted command within the range used by the proven
        # camera/cone followers, then limit its rate before it reaches PWM.
        self.nav_angular_to_steering_scale = float(self.declare_parameter(
            'nav_angular_to_steering_scale', 0.80).value)
        self.steering_command_bias = float(self.declare_parameter(
            'steering_command_bias', -0.064).value)
        self.steering_command_sign = float(self.declare_parameter(
            'steering_command_sign', 1.0).value)
        self.maximum_steering_command = float(self.declare_parameter(
            'maximum_steering_command', 0.36).value)
        self.steering_command_smoothing = float(self.declare_parameter(
            'steering_command_smoothing', 0.35).value)
        self.maximum_steering_step = float(self.declare_parameter(
            'maximum_steering_step', 0.04).value)
        self.derive_route_headings = bool(self.declare_parameter(
            'derive_route_headings', True).value)
        self.command_timeout = float(self.declare_parameter(
            'command_timeout', 1.0).value)
        self.require_detector_heartbeat = bool(self.declare_parameter(
            'require_detector_heartbeat', True).value)
        self.detector_startup_timeout = float(self.declare_parameter(
            'detector_startup_timeout', 60.0).value)
        self.detector_heartbeat_timeout = float(self.declare_parameter(
            'detector_heartbeat_timeout', 10.0).value)
        self.sign_confirm_count = int(self.declare_parameter(
            'sign_confirm_count', 4).value)
        self.sign_confirm_window = float(self.declare_parameter(
            'sign_confirm_window', 0.5).value)
        self.sign_arm_progress = float(self.declare_parameter(
            'sign_arm_progress', 0.70).value)
        self.closed_loop_distance = float(self.declare_parameter(
            'closed_loop_distance', 1.0).value)
        self.maximum_route_point_gap = float(self.declare_parameter(
            'maximum_route_point_gap', 0.75).value)
        self.stop_publish_rate = float(self.declare_parameter(
            'stop_publish_rate', 20.0).value)

        if self.navigation_start_delay < 0.0:
            raise ValueError('navigation_start_delay must not be negative')
        if self.startup_timeout <= 0.0:
            raise ValueError('startup_timeout must be greater than zero')
        if self.initial_pose_publish_period <= 0.0:
            raise ValueError('initial_pose_publish_period must be positive')
        if self.max_initial_pose_error <= 0.0:
            raise ValueError('max_initial_pose_error must be positive')
        if self.max_initial_yaw_error <= 0.0:
            raise ValueError('max_initial_yaw_error must be positive')
        if self.command_timeout <= 0.0:
            raise ValueError('command_timeout must be greater than zero')
        if self.detector_startup_timeout <= 0.0:
            raise ValueError('detector_startup_timeout must be positive')
        if self.detector_heartbeat_timeout <= 0.0:
            raise ValueError('detector_heartbeat_timeout must be positive')
        if self.sign_confirm_count < 1:
            raise ValueError('sign_confirm_count must be at least one')
        if self.sign_confirm_window <= 0.0:
            raise ValueError('sign_confirm_window must be positive')
        if not 0.0 <= self.sign_arm_progress <= 1.0:
            raise ValueError('sign_arm_progress must be in [0, 1]')
        if self.closed_loop_distance < 0.0:
            raise ValueError('closed_loop_distance must not be negative')
        if self.maximum_route_point_gap <= 0.0:
            raise ValueError('maximum_route_point_gap must be positive')
        if self.nav_angular_to_steering_scale <= 0.0:
            raise ValueError(
                'nav_angular_to_steering_scale must be positive')
        if abs(self.steering_command_sign) < 0.5:
            raise ValueError('steering_command_sign must be 1 or -1')
        if self.maximum_steering_command <= 0.0:
            raise ValueError('maximum_steering_command must be positive')
        if not 0.0 <= self.steering_command_smoothing < 1.0:
            raise ValueError(
                'steering_command_smoothing must be in [0, 1)')
        if self.maximum_steering_step <= 0.0:
            raise ValueError('maximum_steering_step must be positive')
        if self.stop_publish_rate <= 0.0:
            raise ValueError('stop_publish_rate must be positive')

        self.route, route_frame = self._load_route(self.route_file)
        self.initial_pose, pose_frame = self._load_pose(self.pose_file)
        if route_frame != pose_frame:
            raise ValueError(
                f'route frame {route_frame!r} and pose frame '
                f'{pose_frame!r} do not match')
        self.map_frame = route_frame
        self.route = self._prepare_replay_route(
            self.route, self.initial_pose)
        if self.derive_route_headings:
            self.route = self._route_with_geometry_headings(self.route)
        self.route_length = self._route_length(self.route)
        self.route_segments = self._split_closed_route(self.route)
        largest_gap, gap_index = max(
            (
                math.hypot(
                    current['x'] - previous['x'],
                    current['y'] - previous['y']),
                index,
            )
            for index, (previous, current) in enumerate(
                zip(self.route, self.route[1:]))
        )

        normal_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.command_publisher = self.create_publisher(
            Twist, self.output_command_topic, normal_qos)
        self.initial_pose_publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', normal_qos)
        self.route_publisher = self.create_publisher(
            Path, '/stage2/route', latched_qos)
        # The supplied RViz profile already displays /plan. FollowPath does
        # not invoke the planner, so mirroring the recorded route here makes
        # it visible without changing the user's existing RViz layout.
        self.rviz_route_publisher = self.create_publisher(
            Path, '/plan', latched_qos)
        self.state_publisher = self.create_publisher(
            String, '/stage2/state', latched_qos)
        self.progress_publisher = self.create_publisher(
            Float32, '/stage2/progress', normal_qos)
        self.detected_sign_publisher = self.create_publisher(
            String, '/stage2/detected_sign', latched_qos)

        self.create_subscription(
            Twist, self.nav_command_topic, self._nav_command_callback,
            normal_qos)
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_callback,
            normal_qos)
        self.create_subscription(
            OccupancyGrid, '/map', self._map_callback, latched_qos)
        self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, normal_qos)
        self.create_subscription(
            String, self.sign_topic, self._sign_callback, normal_qos)
        self.create_subscription(
            Bool, self.emergency_stop_topic,
            self._emergency_stop_callback, normal_qos)
        self.create_subscription(
            Bool, self.detector_heartbeat_topic,
            self._detector_heartbeat_callback, normal_qos)

        self.follow_path_client = ActionClient(self, FollowPath, 'follow_path')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(
            1.0 / self.stop_publish_rate, self._timer_callback)

        self.state = StageState.WAITING_FOR_MAP
        self.started_at = time.monotonic()
        self.map_received = False
        self.initial_pose_first_sent_at: Optional[float] = None
        self.initial_pose_last_sent_at = 0.0
        self.initial_pose_acknowledged_at: Optional[float] = None
        self.amcl_pose_received = False
        self.amcl_covariance_ok = False
        self.amcl_pose_ok = False
        self.goal_sent = False
        self.goal_handle = None
        self.segment_index = 0
        self.last_nav_command_at = 0.0
        self.last_steering_command = 0.0
        self.last_command_log_at = 0.0
        self.last_status_log_at = 0.0
        self.last_odom_xy = None
        self.distance_travelled = 0.0
        self.progress = 0.0
        self.sign_candidate = ''
        self.sign_candidate_count = 0
        self.last_sign_at = 0.0
        self.detected_sign = ''
        self.detector_heartbeat_received = False
        self.last_detector_heartbeat_at = 0.0

        self._publish_route()
        self._publish_state()
        self._publish_stop()
        self.get_logger().info(
            f'Second lap loaded {len(self.route)} route points, '
            f'length={self.route_length:.2f}m, '
            f'segments={len(self.route_segments)}, '
            f'initial pose=({self.initial_pose["x"]:.2f}, '
            f'{self.initial_pose["y"]:.2f}, '
            f'{self.initial_pose["yaw"]:.2f}), '
            f'sign topic={self.sign_topic}, '
            f'heading_mode={"geometry" if self.derive_route_headings else "saved"}, '
            f'max_gap={largest_gap:.2f}m at {gap_index}->{gap_index + 1}, '
            f'steering_bias={self.steering_command_bias:.3f}, '
            f'steering_scale={self.nav_angular_to_steering_scale:.3f}, '
            f'steering_limit={self.maximum_steering_command:.3f}'
        )

    @staticmethod
    def _read_yaml(path: str) -> Dict:
        with open(path, 'r', encoding='utf-8') as source:
            document = yaml.safe_load(source)
        if not isinstance(document, dict):
            raise ValueError(f'YAML root must be a mapping: {path}')
        return document

    @classmethod
    def _load_route(cls, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'first-lap route not found: {path}')
        document = cls._read_yaml(path)
        frame_id = str(document.get('frame_id', 'map'))
        raw_poses = document.get('poses')
        if not isinstance(raw_poses, list) or len(raw_poses) < 2:
            raise ValueError('route YAML must contain at least two poses')
        route: List[Dict[str, float]] = []
        for index, item in enumerate(raw_poses):
            if not isinstance(item, dict):
                raise ValueError(f'route pose {index} is not a mapping')
            try:
                route.append({
                    'x': float(item['x']),
                    'y': float(item['y']),
                    'yaw': float(item['yaw']),
                })
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'invalid route pose {index}: {error}') from error
        return route, frame_id

    @classmethod
    def _load_pose(cls, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'first-lap pose not found: {path}')
        document = cls._read_yaml(path)
        frame_id = str(document.get('frame_id', 'map'))
        item = document.get('pose')
        if not isinstance(item, dict):
            raise ValueError('pose YAML must contain a pose mapping')
        try:
            pose = {
                'x': float(item['x']),
                'y': float(item['y']),
                'yaw': float(item['yaw']),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'invalid initial pose: {error}') from error
        return pose, frame_id

    @staticmethod
    def _route_length(route: List[Dict[str, float]]) -> float:
        return sum(
            math.hypot(
                current['x'] - previous['x'],
                current['y'] - previous['y'])
            for previous, current in zip(route, route[1:])
        )

    @staticmethod
    def _route_with_geometry_headings(
        route: List[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        """Use the recorded centerline tangent as the path heading.

        Odom yaw can contain short spikes and +/-pi wrapping.  FollowPath
        needs a continuous heading matching the x/y centerline, especially at
        the first sharp bend; derive it from neighboring route points while
        retaining the saved coordinates.
        """
        if len(route) < 2:
            return route
        result = [dict(item) for item in route]
        for index, item in enumerate(result):
            if index == 0:
                previous = result[0]
                following = result[1]
            elif index == len(result) - 1:
                previous = result[-2]
                following = result[-1]
            else:
                previous = result[index - 1]
                following = result[index + 1]
            dx = following['x'] - previous['x']
            dy = following['y'] - previous['y']
            if math.hypot(dx, dy) > 1.0e-6:
                item['yaw'] = math.atan2(dy, dx)
        return result

    def _prepare_replay_route(
        self,
        route: List[Dict[str, float]],
        initial_pose: Dict[str, float],
    ) -> List[Dict[str, float]]:
        """Validate the lap closure and start replay at the saved final pose."""
        final_pose_error = math.hypot(
            route[-1]['x'] - initial_pose['x'],
            route[-1]['y'] - initial_pose['y'])
        if final_pose_error > 0.20:
            raise ValueError(
                'saved pose is not the final route point '
                f'(difference={final_pose_error:.2f}m)')

        largest_gap, gap_index = max(
            (
                math.hypot(
                    current['x'] - previous['x'],
                    current['y'] - previous['y']),
                index,
            )
            for index, (previous, current) in enumerate(
                zip(route, route[1:]))
        )
        if largest_gap > self.maximum_route_point_gap:
            raise ValueError(
                'first-lap route contains a discontinuity: '
                f'point {gap_index}->{gap_index + 1} is '
                f'{largest_gap:.2f}m apart, '
                f'limit={self.maximum_route_point_gap:.2f}m. '
                'Rebuild and record the first lap again')

        closure = math.hypot(
            route[-1]['x'] - route[0]['x'],
            route[-1]['y'] - route[0]['y'])
        if closure > self.closed_loop_distance:
            raise ValueError(
                f'first-lap route is not closed: end-to-start={closure:.2f}m, '
                f'limit={self.closed_loop_distance:.2f}m. Adjust lap_duration '
                'and record the first lap again')

        # The car begins the second stage at route[-1].  Put that exact pose
        # first, bridge the small lap-closure gap to route[0], and finish back
        # at route[-1].
        return [dict(route[-1])] + [dict(item) for item in route]

    def _split_closed_route(self, route: List[Dict[str, float]]):
        """Split a loop so its goal is not already satisfied at startup.

        A closed FollowPath goal has its final pose beside its initial pose.
        Nav2's goal checker can therefore report success immediately.  Two
        sequential FollowPath goals keep each segment's endpoint far from the
        vehicle when that segment starts.
        """
        closure = math.hypot(
            route[-1]['x'] - route[0]['x'],
            route[-1]['y'] - route[0]['y'])
        if closure > 0.20 or len(route) < 6:
            return [route]

        halfway = self.route_length * 0.5
        accumulated = 0.0
        split_index = len(route) // 2
        for index in range(1, len(route)):
            accumulated += math.hypot(
                route[index]['x'] - route[index - 1]['x'],
                route[index]['y'] - route[index - 1]['y'])
            if accumulated >= halfway:
                split_index = index
                break
        split_index = min(max(2, split_index), len(route) - 3)
        self.get_logger().info(
            f'Closed route detected (end-to-start={closure:.2f}m); '
            f'splitting FollowPath at point {split_index}')
        return [route[:split_index + 1], route[split_index:]]

    @staticmethod
    def _set_yaw(pose, yaw: float) -> None:
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(yaw * 0.5)
        pose.orientation.w = math.cos(yaw * 0.5)

    @staticmethod
    def _yaw_from_quaternion(quaternion) -> float:
        sin_yaw = 2.0 * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y)
        cos_yaw = 1.0 - 2.0 * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z)
        return math.atan2(sin_yaw, cos_yaw)

    def _path_message(self, route=None) -> Path:
        if route is None:
            route = self.route
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for item in route:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = item['x']
            pose.pose.position.y = item['y']
            self._set_yaw(pose.pose, item['yaw'])
            path.poses.append(pose)
        return path

    def _publish_route(self) -> None:
        route = self._path_message()
        self.route_publisher.publish(route)
        self.rviz_route_publisher.publish(route)

    def _publish_state(self) -> None:
        message = String()
        message.data = self.state.name
        self.state_publisher.publish(message)

    def _set_state(self, state: StageState, reason: str) -> None:
        if state == self.state:
            return
        old_state = self.state
        self.state = state
        self._publish_state()
        self.get_logger().info(
            f'{old_state.name} -> {state.name}: {reason}')

    def _publish_stop(self) -> None:
        self.last_steering_command = 0.0
        self.command_publisher.publish(Twist())

    def _initial_pose_message(self) -> PoseWithCovarianceStamped:
        message = PoseWithCovarianceStamped()
        message.header.frame_id = self.map_frame
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.pose.position.x = self.initial_pose['x']
        message.pose.pose.position.y = self.initial_pose['y']
        self._set_yaw(message.pose.pose, self.initial_pose['yaw'])
        message.pose.covariance[0] = self.initial_xy_covariance
        message.pose.covariance[7] = self.initial_xy_covariance
        message.pose.covariance[35] = self.initial_yaw_covariance
        return message

    def _publish_initial_pose(self, now: float) -> None:
        self.initial_pose_publisher.publish(self._initial_pose_message())
        self.initial_pose_last_sent_at = now
        if self.initial_pose_first_sent_at is None:
            self.initial_pose_first_sent_at = now
            self.get_logger().info('Published saved pose to AMCL')

    def _map_callback(self, _message: OccupancyGrid) -> None:
        self.map_received = True

    def _amcl_callback(self, message: PoseWithCovarianceStamped) -> None:
        self.amcl_pose_received = True
        now = time.monotonic()
        if (
            self.initial_pose_first_sent_at is not None
            and self.initial_pose_acknowledged_at is None
        ):
            # AMCL has received a pose update after /initialpose was sent.
            # Stop resetting its particle cloud so laser scans can converge.
            self.initial_pose_acknowledged_at = now
            self.get_logger().info(
                'AMCL acknowledged saved initial pose; allowing laser '
                'localization to settle')
        covariance = message.pose.covariance
        xy_covariance = max(abs(covariance[0]), abs(covariance[7]))
        yaw_covariance = abs(covariance[35])
        self.amcl_covariance_ok = (
            math.isfinite(xy_covariance)
            and math.isfinite(yaw_covariance)
            and xy_covariance <= self.max_xy_covariance
            and yaw_covariance <= self.max_yaw_covariance
        )
        pose = message.pose.pose
        position_error = math.hypot(
            pose.position.x - self.initial_pose['x'],
            pose.position.y - self.initial_pose['y'])
        yaw = self._yaw_from_quaternion(pose.orientation)
        yaw_error = abs(math.atan2(
            math.sin(yaw - self.initial_pose['yaw']),
            math.cos(yaw - self.initial_pose['yaw'])))
        self.amcl_pose_ok = (
            position_error <= self.max_initial_pose_error
            and yaw_error <= self.max_initial_yaw_error)

    def _odom_callback(self, message: Odometry) -> None:
        point = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
        )
        if self.last_odom_xy is not None and self.state == StageState.NAVIGATING:
            step = math.hypot(
                point[0] - self.last_odom_xy[0],
                point[1] - self.last_odom_xy[1])
            if step <= 1.0:
                self.distance_travelled += step
                if self.route_length > 1.0e-6:
                    self.progress = min(
                        1.0, self.distance_travelled / self.route_length)
        self.last_odom_xy = point

        progress_message = Float32()
        progress_message.data = float(self.progress)
        self.progress_publisher.publish(progress_message)

    def _nav_command_callback(self, message: Twist) -> None:
        if self.state != StageState.NAVIGATING:
            return
        self.maximum_linear_speed = float(
            self.get_parameter('maximum_linear_speed').value)
        self.maximum_angular_speed = float(
            self.get_parameter('maximum_angular_speed').value)
        if self.maximum_linear_speed <= 0.0:
            self._enter_fault('maximum_linear_speed must remain positive')
            return
        if self.maximum_angular_speed <= 0.0:
            self._enter_fault('maximum_angular_speed must remain positive')
            return

        safe_command = Twist()
        # This Ackermann car must not receive reverse recovery commands.
        # The current driver maps m/s with 1500 + 100*v and clamps every
        # positive request below 0.55 m/s to PWM 1555.  A lower cap would be
        # misleading rather than slower, so validation starts at 0.55 m/s.
        safe_command.linear.x = min(
            max(0.0, float(message.linear.x)), self.maximum_linear_speed)
        nav_angular = min(
            max(float(message.angular.z), -self.maximum_angular_speed),
            self.maximum_angular_speed)
        target_steering = min(
            max(
                self.steering_command_bias
                + self.steering_command_sign
                * nav_angular * self.nav_angular_to_steering_scale,
                -self.maximum_steering_command),
            self.maximum_steering_command)
        alpha = self.steering_command_smoothing
        filtered_steering = (
            alpha * self.last_steering_command
            + (1.0 - alpha) * target_steering)
        steering_delta = min(
            max(
                filtered_steering - self.last_steering_command,
                -self.maximum_steering_step),
            self.maximum_steering_step)
        self.last_steering_command += steering_delta
        safe_command.angular.z = self.last_steering_command
        self.command_publisher.publish(safe_command)
        self.last_nav_command_at = time.monotonic()
        if self.last_nav_command_at - self.last_command_log_at >= 1.0:
            self.get_logger().info(
                'Nav2 command: '
                f'linear={message.linear.x:.3f}m/s, '
                f'angular={nav_angular:.3f}rad/s, '
                f'target={target_steering:.3f}, '
                f'steering={safe_command.angular.z:.3f}'
            )
            self.last_command_log_at = self.last_nav_command_at

    def _sign_callback(self, message: String) -> None:
        label = message.data.strip().upper()
        if label not in ('A', 'B'):
            return
        if self.progress < self.sign_arm_progress:
            self.sign_candidate = ''
            self.sign_candidate_count = 0
            return
        now = time.monotonic()
        if (
            label != self.sign_candidate
            or now - self.last_sign_at > self.sign_confirm_window
        ):
            self.sign_candidate = label
            self.sign_candidate_count = 1
        else:
            self.sign_candidate_count += 1
        self.last_sign_at = now

        if self.sign_candidate_count < self.sign_confirm_count:
            return
        if self.state in (
            StageState.SIGN_STOP,
            StageState.COMPLETE,
            StageState.EMERGENCY_STOP,
            StageState.FAULT,
        ):
            return

        self.detected_sign = label
        detected = String()
        detected.data = label
        self.detected_sign_publisher.publish(detected)
        self._set_state(
            StageState.SIGN_STOP,
            f'sign {label} confirmed {self.sign_candidate_count} times')
        self._cancel_goal()
        self._publish_stop()

    def _emergency_stop_callback(self, message: Bool) -> None:
        if not message.data:
            return
        if self.state == StageState.EMERGENCY_STOP:
            return
        self._set_state(StageState.EMERGENCY_STOP, 'emergency stop requested')
        self._cancel_goal()
        self._publish_stop()

    def _detector_heartbeat_callback(self, message: Bool) -> None:
        if message.data:
            self.detector_heartbeat_received = True
            self.last_detector_heartbeat_at = time.monotonic()

    def _localization_ready(self, now: float) -> bool:
        if self.initial_pose_acknowledged_at is None:
            return False
        if (
            now - self.initial_pose_acknowledged_at
            < self.localization_settle_time
        ):
            return False
        if (
            not self.amcl_pose_received
            or not self.amcl_covariance_ok
            or not self.amcl_pose_ok
        ):
            return False
        if self.require_detector_heartbeat and not self.detector_heartbeat_received:
            return False
        try:
            self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time())
        except TransformException:
            return False
        return True

    def _send_goal(self) -> None:
        if self.goal_sent:
            return
        goal = FollowPath.Goal()
        segment = self.route_segments[self.segment_index]
        goal.path = self._path_message(segment)
        goal.controller_id = 'FollowPath'
        goal.goal_checker_id = 'goal_checker'
        self.goal_sent = True
        future = self.follow_path_client.send_goal_async(
            goal, feedback_callback=self._feedback_callback)
        future.add_done_callback(self._goal_response_callback)
        self.get_logger().info(
            f'First-lap route segment {self.segment_index + 1}/'
            f'{len(self.route_segments)} sent to Nav2 FollowPath '
            f'({len(segment)} points)')

    def _goal_response_callback(self, future) -> None:
        try:
            self.goal_handle = future.result()
        except Exception as error:  # rclpy action errors vary by distro
            self._enter_fault(f'FollowPath goal response failed: {error}')
            return
        if not self.goal_handle.accepted:
            self._enter_fault('Nav2 rejected the FollowPath goal')
            return
        self.last_nav_command_at = time.monotonic()
        self._set_state(StageState.NAVIGATING, 'FollowPath goal accepted')
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _feedback_callback(self, _feedback) -> None:
        # Odometry distance supplies progress; receiving feedback proves that
        # the controller action is alive.
        pass

    def _result_callback(self, future) -> None:
        if self.state in (
            StageState.SIGN_STOP,
            StageState.EMERGENCY_STOP,
            StageState.FAULT,
        ):
            self._publish_stop()
            return
        try:
            wrapped_result = future.result()
        except Exception as error:
            self._enter_fault(f'FollowPath result failed: {error}')
            return
        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED:
            if self.segment_index + 1 < len(self.route_segments):
                self.get_logger().info(
                    f'Route segment {self.segment_index + 1} completed')
                self.segment_index += 1
                self.goal_handle = None
                self.goal_sent = False
                self._set_state(
                    StageState.WAITING_FOR_NAV2,
                    'starting the next closed-route segment')
                self._publish_stop()
                self._send_goal()
                return
            self.progress = 1.0
            self._set_state(StageState.COMPLETE, 'second-lap route completed')
        else:
            self._enter_fault(
                f'FollowPath ended with status {wrapped_result.status}')
        self._publish_stop()

    def _cancel_goal(self) -> None:
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    def _enter_fault(self, reason: str) -> None:
        if self.state == StageState.FAULT:
            return
        self._set_state(StageState.FAULT, reason)
        self._cancel_goal()
        self._publish_stop()

    def _timer_callback(self) -> None:
        now = time.monotonic()
        terminal_states = (
            StageState.SIGN_STOP,
            StageState.COMPLETE,
            StageState.EMERGENCY_STOP,
            StageState.FAULT,
        )
        if self.state in terminal_states:
            self._publish_stop()
            return

        if self.require_detector_heartbeat:
            if (
                not self.detector_heartbeat_received
                and now - self.started_at >= self.detector_startup_timeout
            ):
                self._enter_fault('A/B detector heartbeat startup timeout')
                return
            if (
                self.detector_heartbeat_received
                and now - self.last_detector_heartbeat_at
                >= self.detector_heartbeat_timeout
            ):
                self._enter_fault('A/B detector or camera heartbeat lost')
                return

        if now - self.started_at >= self.startup_timeout and not self.goal_sent:
            self._enter_fault('startup timeout while waiting for map/AMCL/Nav2')
            return

        if self.state == StageState.NAVIGATING:
            if (
                self.last_nav_command_at > 0.0
                and now - self.last_nav_command_at > self.command_timeout
            ):
                self._enter_fault('Nav2 velocity command timeout')
            return

        self._publish_stop()
        if not self.map_received:
            self._set_state(StageState.WAITING_FOR_MAP, 'waiting for saved map')
        else:
            # Initialize localization immediately after the map arrives.  The
            # configured delay controls only when motion may start; it must
            # not postpone /initialpose or the map->odom transform.
            if (
                self.initial_pose_acknowledged_at is None
                and (
                    self.initial_pose_first_sent_at is None
                    or now - self.initial_pose_last_sent_at
                    >= self.initial_pose_publish_period
                )
            ):
                self._publish_initial_pose(now)

            if self._localization_ready(now):
                self._set_state(
                    StageState.WAITING_FOR_NAV2,
                    'AMCL covariance and map TF are ready; waiting for start')
                delay_elapsed = (
                    now - self.started_at >= self.navigation_start_delay)
                if (
                    delay_elapsed
                    and self.follow_path_client.wait_for_server(timeout_sec=0.0)
                ):
                    self._send_goal()
            else:
                self._set_state(
                    StageState.INITIALIZING_AMCL,
                    'waiting for AMCL, map TF, and detector heartbeat')

        if now - self.last_status_log_at >= 1.0:
            self.get_logger().info(
                f'state={self.state.name}, map={self.map_received}, '
                f'amcl={self.amcl_pose_received}, '
                f'initialpose_ack={self.initial_pose_acknowledged_at is not None}, '
                f'covariance_ok={self.amcl_covariance_ok}, '
                f'pose_ok={self.amcl_pose_ok}, '
                f'nav2={self.follow_path_client.server_is_ready()}, '
                f'detector={self.detector_heartbeat_received}, '
                f'progress={self.progress * 100.0:.1f}%'
            )
            self.last_status_log_at = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SecondLapManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(f'Second-lap manager failed: {error}')
        else:
            print(f'Second-lap manager failed: {error}')
        raise
    finally:
        if node is not None:
            for _ in range(10):
                node._publish_stop()
                time.sleep(0.02)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
