#!/usr/bin/env python3

import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args


class DriveDistance(Node):
    """Drive straight until the odometry displacement reaches the target."""

    def __init__(self, target_distance):
        super().__init__('drive_distance')

        self.declare_parameter('distance', target_distance)
        self.declare_parameter('speed', 1570.0)
        self.declare_parameter('arm_duration', 1.0)
        self.declare_parameter('stop_offset', 0.12)
        self.declare_parameter('steering', 90.0)
        self.declare_parameter('odom_timeout', 1.0)

        self.target_distance = float(self.get_parameter('distance').value)
        self.speed = float(self.get_parameter('speed').value)
        self.arm_duration = float(self.get_parameter('arm_duration').value)
        self.stop_offset = float(self.get_parameter('stop_offset').value)
        self.steering = float(self.get_parameter('steering').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)

        if self.target_distance <= 0.0:
            raise ValueError('distance must be greater than 0 metres')
        if self.speed <= 1500.0:
            raise ValueError('forward speed must be greater than the stop value 1500')
        if self.arm_duration < 0.0:
            raise ValueError('arm_duration cannot be negative')
        if self.stop_offset < 0.0:
            raise ValueError('stop_offset cannot be negative')
        if self.stop_offset >= self.target_distance:
            raise ValueError('stop_offset must be smaller than distance')
        if self.odom_timeout <= 0.0:
            raise ValueError('odom_timeout must be greater than 0 seconds')

        self.stop_distance = self.target_distance - self.stop_offset

        self.cmd_pub = self.create_publisher(Twist, '/teleop_cmd_vel', 5)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom_combined', self.odom_callback, 10)
        self.control_timer = self.create_timer(0.05, self.control_callback)

        self.start_x = None
        self.start_y = None
        self.travelled = 0.0
        self.last_odom_time = None
        self.start_time = None
        self.moving = False
        self.finished = False
        self.stop_publish_count = 0
        self.last_status_time = None
        self.last_motion_state = None

        self.get_logger().info(
            f'Waiting for /odom_combined; target distance: '
            f'{self.target_distance:.3f} m, stop trigger: '
            f'{self.stop_distance:.3f} m, forward speed: {self.speed:.0f}')

    def odom_callback(self, msg):
        if self.finished:
            return

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_odom_time = self.get_clock().now()

        if self.start_x is None:
            self.start_x = x
            self.start_y = y
            self.start_time = self.get_clock().now()
            self.last_status_time = self.start_time
            self.moving = True
            self.get_logger().info(
                f'Odometry received. Start: x={x:.4f}, y={y:.4f}. '
                f'Arming at neutral for {self.arm_duration:.1f} s')
            return

        self.travelled = math.hypot(x - self.start_x, y - self.start_y)
        if self.travelled >= self.stop_distance:
            self.finish(
                f'Stop triggered at {self.travelled:.3f} m; '
                f'expected to coast toward {self.target_distance:.3f} m '
                f'(stop offset {self.stop_offset:.3f} m)')

    def control_callback(self):
        if self.finished:
            self.publish_stop()
            self.stop_publish_count += 1
            if self.stop_publish_count >= 10:
                rclpy.shutdown()
            return

        if not self.moving:
            self.publish_stop()
            return

        odom_age = (
            self.get_clock().now() - self.last_odom_time
        ).nanoseconds / 1e9
        if odom_age > self.odom_timeout:
            self.finish(
                f'Odometry timeout ({odom_age:.2f} s); stopping for safety',
                error=True)
            return

        elapsed = (
            self.get_clock().now() - self.start_time
        ).nanoseconds / 1e9

        if elapsed < self.arm_duration:
            self.publish_stop()
            self.log_motion_state('arming')
            return

        command_speed = self.speed
        self.log_motion_state('cruising')

        command = Twist()
        command.linear.x = command_speed
        command.angular.z = self.steering
        self.cmd_pub.publish(command)

        now = self.get_clock().now()
        if (now - self.last_status_time).nanoseconds >= 1_000_000_000:
            self.get_logger().info(
                f'Travelled {self.travelled:.3f} / '
                f'{self.target_distance:.3f} m')
            self.last_status_time = now

    def log_motion_state(self, state):
        if state == self.last_motion_state:
            return
        self.last_motion_state = state
        if state == 'cruising':
            self.get_logger().info(f'Cruising: speed {self.speed:.0f}')

    def publish_stop(self):
        command = Twist()
        command.linear.x = 1500.0
        command.angular.z = 90.0
        self.cmd_pub.publish(command)

    def finish(self, message, error=False):
        if self.finished:
            return
        self.finished = True
        self.moving = False
        self.publish_stop()
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().info(message)


def parse_target_distance(args):
    parser = argparse.ArgumentParser(
        description='Drive the racecar forward by a specified distance.')
    parser.add_argument(
        'distance', nargs='?', type=float, default=1.0,
        help='target distance in metres (default: 1.0)')
    return parser.parse_args(remove_ros_args(args=args)[1:]).distance


def main(args=None):
    cli_args = sys.argv if args is None else args
    target_distance = parse_target_distance(cli_args)
    rclpy.init(args=args)
    node = None

    try:
        node = DriveDistance(target_distance)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except ValueError as exc:
        print(f'drive_distance: {exc}', file=sys.stderr)
    finally:
        if node is not None:
            if rclpy.ok():
                for _ in range(5):
                    node.publish_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
