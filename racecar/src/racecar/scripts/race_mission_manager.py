#!/usr/bin/env python3
"""Central command arbiter for the competition mission.

V1 step 1 deliberately keeps the existing first-lap and second-lap algorithms.
Its responsibility is narrower and safety-critical: every actuator command
must pass through this node before reaching the chassis driver.

Command semantics
-----------------
FIRST_LAP uses the existing raw teleop protocol:
  linear.x = motor PWM, angular.z = steering angle in degrees.

SECOND_LAP and PARKING use the normalized /car_cmd_vel protocol:
  linear.x = m/s, angular.z = normalized steering command.

Only the source belonging to the active mission phase is forwarded.
A stale command or a second publisher bypassing this arbiter causes a stop.
"""

import fcntl
import os
import time
from enum import Enum, auto

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


class MissionPhase(Enum):
    WAITING = auto()
    FIRST_LAP = auto()
    TRANSITION = auto()
    SECOND_LAP = auto()
    PARKING = auto()
    COMPLETE = auto()
    EMERGENCY_STOP = auto()
    FAULT = auto()


class RaceMissionManager(Node):
    """Own the final command topics that reach the chassis driver."""

    RAW_PHASES = {MissionPhase.FIRST_LAP}
    METRIC_PHASES = {MissionPhase.SECOND_LAP, MissionPhase.PARKING}
    TERMINAL_PHASES = {
        MissionPhase.COMPLETE,
        MissionPhase.EMERGENCY_STOP,
        MissionPhase.FAULT,
    }

    def __init__(self) -> None:
        super().__init__('race_mission_manager')

        # Prevent two mission managers from fighting over the chassis.
        self._instance_lock = open(
            '/tmp/racecar_mission_manager.lock', 'w', encoding='utf-8')
        try:
            fcntl.flock(
                self._instance_lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                'another race_mission_manager is already running') from error
        self._instance_lock.write(str(os.getpid()))
        self._instance_lock.flush()

        self.first_lap_source_topic = str(self.declare_parameter(
            'first_lap_source_topic', '/mission/first_lap_cmd').value)
        self.second_lap_source_topic = str(self.declare_parameter(
            'second_lap_source_topic', '/mission/second_lap_cmd').value)
        self.parking_source_topic = str(self.declare_parameter(
            'parking_source_topic', '/mission/parking_cmd').value)

        self.raw_output_topic = str(self.declare_parameter(
            'raw_output_topic', '/teleop_cmd_vel').value)
        self.metric_output_topic = str(self.declare_parameter(
            'metric_output_topic', '/car_cmd_vel').value)

        self.emergency_stop_topic = str(self.declare_parameter(
            'emergency_stop_topic', '/mission/emergency_stop').value)
        self.phase_request_topic = str(self.declare_parameter(
            'phase_request_topic', '/mission/phase_request').value)
        self.state_topic = str(self.declare_parameter(
            'state_topic', '/mission/state').value)

        initial_phase_name = str(self.declare_parameter(
            'initial_phase', 'WAITING').value).strip().upper()
        self.command_timeout = float(self.declare_parameter(
            'command_timeout', 0.50).value)
        self.watchdog_rate = float(self.declare_parameter(
            'watchdog_rate', 20.0).value)
        self.conflict_check_period = float(self.declare_parameter(
            'conflict_check_period', 1.0).value)
        self.allow_external_phase_request = bool(self.declare_parameter(
            'allow_external_phase_request', False).value)

        # The raw teleop driver uses 1500 PWM for motor stop and 90 degrees
        # for straight steering.
        self.raw_stop_linear = float(self.declare_parameter(
            'raw_stop_linear', 1500.0).value)
        self.raw_stop_steering = float(self.declare_parameter(
            'raw_stop_steering', 90.0).value)

        if self.command_timeout <= 0.0:
            raise ValueError('command_timeout must be positive')
        if self.watchdog_rate <= 0.0:
            raise ValueError('watchdog_rate must be positive')
        if self.conflict_check_period <= 0.0:
            raise ValueError('conflict_check_period must be positive')

        try:
            self.phase = MissionPhase[initial_phase_name]
        except KeyError as error:
            valid = ', '.join(item.name for item in MissionPhase)
            raise ValueError(
                f'unknown initial_phase {initial_phase_name!r}; '
                f'expected one of: {valid}') from error

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

        self.raw_output = self.create_publisher(
            Twist, self.raw_output_topic, normal_qos)
        self.metric_output = self.create_publisher(
            Twist, self.metric_output_topic, normal_qos)
        self.state_publisher = self.create_publisher(
            String, self.state_topic, latched_qos)

        self.create_subscription(
            Twist, self.first_lap_source_topic,
            self._first_lap_callback, normal_qos)
        self.create_subscription(
            Twist, self.second_lap_source_topic,
            self._second_lap_callback, normal_qos)
        self.create_subscription(
            Twist, self.parking_source_topic,
            self._parking_callback, normal_qos)
        self.create_subscription(
            Bool, self.emergency_stop_topic,
            self._emergency_stop_callback, normal_qos)
        self.create_subscription(
            String, self.phase_request_topic,
            self._phase_request_callback, normal_qos)

        self.last_source_command_at = 0.0
        self.last_conflict_check_at = 0.0
        self.watchdog_stopped = False

        self.timer = self.create_timer(
            1.0 / self.watchdog_rate, self._watchdog_callback)

        self._publish_state()
        self._publish_stop_for_phase(self.phase, repeat=5)
        self.get_logger().info(
            'Mission command arbiter ready: '
            f'phase={self.phase.name}, '
            f'first={self.first_lap_source_topic}, '
            f'second={self.second_lap_source_topic}, '
            f'parking={self.parking_source_topic}, '
            f'outputs=({self.raw_output_topic}, {self.metric_output_topic})'
        )

    def _publish_state(self) -> None:
        message = String()
        message.data = self.phase.name
        self.state_publisher.publish(message)

    def _raw_stop(self) -> Twist:
        command = Twist()
        command.linear.x = self.raw_stop_linear
        command.angular.z = self.raw_stop_steering
        return command

    @staticmethod
    def _metric_stop() -> Twist:
        return Twist()

    def _publish_stop_for_phase(
        self, phase: MissionPhase, repeat: int = 1
    ) -> None:
        if phase in self.RAW_PHASES:
            for _ in range(repeat):
                self.raw_output.publish(self._raw_stop())
        elif phase in self.METRIC_PHASES:
            for _ in range(repeat):
                self.metric_output.publish(self._metric_stop())
        else:
            # Non-driving states stop both command protocols.
            for _ in range(repeat):
                self.raw_output.publish(self._raw_stop())
                self.metric_output.publish(self._metric_stop())

    def _set_phase(self, phase: MissionPhase, reason: str) -> None:
        if phase == self.phase:
            return

        old_phase = self.phase
        self._publish_stop_for_phase(old_phase, repeat=5)
        self.phase = phase
        self.last_source_command_at = 0.0
        self.watchdog_stopped = False
        self._publish_state()
        self._publish_stop_for_phase(self.phase, repeat=5)
        self.get_logger().info(
            f'{old_phase.name} -> {phase.name}: {reason}')

    def _mark_source_command(self) -> None:
        self.last_source_command_at = time.monotonic()
        self.watchdog_stopped = False

    def _first_lap_callback(self, message: Twist) -> None:
        if self.phase != MissionPhase.FIRST_LAP:
            return
        self.raw_output.publish(message)
        self._mark_source_command()

    def _second_lap_callback(self, message: Twist) -> None:
        if self.phase != MissionPhase.SECOND_LAP:
            return
        self.metric_output.publish(message)
        self._mark_source_command()

    def _parking_callback(self, message: Twist) -> None:
        if self.phase != MissionPhase.PARKING:
            return
        self.metric_output.publish(message)
        self._mark_source_command()

    def _emergency_stop_callback(self, message: Bool) -> None:
        if message.data:
            self._set_phase(
                MissionPhase.EMERGENCY_STOP,
                'emergency stop requested')

    def _phase_request_callback(self, message: String) -> None:
        # Disabled in competition mode. It exists only for raised-wheel and
        # bench tests until automatic phase transitions are added in V1 step 2.
        if not self.allow_external_phase_request:
            return

        requested = message.data.strip().upper()
        if requested == 'RESET':
            self._set_phase(MissionPhase.WAITING, 'external reset request')
            return

        try:
            target = MissionPhase[requested]
        except KeyError:
            self.get_logger().warn(
                f'Ignoring unknown phase request: {message.data!r}')
            return

        if target == MissionPhase.FAULT:
            self.get_logger().warn(
                'FAULT cannot be requested externally; use emergency stop')
            return

        self._set_phase(target, 'external phase request')

    def _publisher_conflict_detected(self) -> bool:
        # count_publishers includes this node. More than one means a second
        # node is bypassing the arbiter and can fight for the actuator.
        raw_publishers = self.count_publishers(self.raw_output_topic)
        metric_publishers = self.count_publishers(self.metric_output_topic)
        if raw_publishers > 1 or metric_publishers > 1:
            self.get_logger().error(
                'Command publisher conflict: '
                f'{self.raw_output_topic}={raw_publishers}, '
                f'{self.metric_output_topic}={metric_publishers}. '
                'All actuator commands must pass through race_mission_manager.')
            return True
        return False

    def _watchdog_callback(self) -> None:
        now = time.monotonic()

        if now - self.last_conflict_check_at >= self.conflict_check_period:
            self.last_conflict_check_at = now
            if (
                self.phase != MissionPhase.FAULT
                and self._publisher_conflict_detected()
            ):
                self._set_phase(
                    MissionPhase.FAULT,
                    'another node is publishing directly to the chassis')
                return

        if self.phase in self.TERMINAL_PHASES:
            self._publish_stop_for_phase(self.phase)
            return

        if self.phase not in self.RAW_PHASES | self.METRIC_PHASES:
            self._publish_stop_for_phase(self.phase)
            return

        if self.last_source_command_at <= 0.0:
            if not self.watchdog_stopped:
                self._publish_stop_for_phase(self.phase, repeat=3)
                self.watchdog_stopped = True
            return

        command_age = now - self.last_source_command_at
        if command_age > self.command_timeout and not self.watchdog_stopped:
            self._publish_stop_for_phase(self.phase, repeat=3)
            self.watchdog_stopped = True
            self.get_logger().error(
                f'{self.phase.name} source command timeout '
                f'({command_age:.2f}s); vehicle stopped')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = RaceMissionManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node._publish_stop_for_phase(node.phase, repeat=10)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
