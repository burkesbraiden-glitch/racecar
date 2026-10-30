#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from action_msgs.msg import GoalStatus
from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped, Twist
import csv

class FollowWaypointsClient(Node):
    def __init__(self):
        super().__init__('follow_waypoints_client')
        self._action_client = ActionClient(self, FollowWaypoints, 'follow_waypoints')
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.car_cmd_vel_pub = self.create_publisher(Twist, '/car_cmd_vel', 10)
        self.stop_timer = self.create_timer(0.1, self.stop_timer_callback)
        self.navigation_finished = False
        self.current_waypoint = None
        self.total_waypoints = 0

    def send_goal(self, poses):
        if not poses:
            self.get_logger().error('No waypoints were loaded; navigation was not started.')
            self.navigation_finished = True
            return

        self.total_waypoints = len(poses)
        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = poses

        self.get_logger().info(
            f'Waiting for FollowWaypoints server; loaded {self.total_waypoints} waypoints.'
        )
        if not self._action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('FollowWaypoints action server is unavailable.')
            self.navigation_finished = True
            return

        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Waypoint navigation goal was rejected.')
            self.navigation_finished = True
            return

        self.get_logger().info('Waypoint navigation goal accepted; starting waypoint 1.')

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        wrapped_result = future.result()
        missed_waypoints = list(wrapped_result.result.missed_waypoints)
        self.navigation_finished = True

        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED and not missed_waypoints:
            self.get_logger().info(
                f'Waypoint {self.total_waypoints} confirmed reached.'
            )
            self.get_logger().info(
                f'All {self.total_waypoints} waypoints were reached in order.'
            )
        else:
            missed = ', '.join(str(index + 1) for index in missed_waypoints) or 'unknown'
            self.get_logger().error(
                f'Waypoint navigation failed or was interrupted; missed waypoint(s): {missed}.'
            )

    def feedback_callback(self, feedback_msg):
        waypoint = feedback_msg.feedback.current_waypoint
        if waypoint == self.current_waypoint:
            return

        if self.current_waypoint is not None:
            self.get_logger().info(
                f'Waypoint {self.current_waypoint + 1} confirmed reached.'
            )
        self.current_waypoint = waypoint
        self.get_logger().info(
            f'Navigating to waypoint {waypoint + 1}/{self.total_waypoints}.'
        )

    def cmd_vel_callback(self, msg):
        if not self.navigation_finished:
            self.car_cmd_vel_pub.publish(msg)

    def stop_timer_callback(self):
        if self.navigation_finished:
            self.car_cmd_vel_pub.publish(Twist())

def read_waypoints_from_csv(filename):
    waypoints = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 4:  # Ensure each row has x, y, z, w
                waypoints.append({
                    "x": float(row[0]),
                    "y": float(row[1]),
                    "z": float(row[2]),
                    "w": float(row[3])
                })
    return waypoints

def main(args=None):
    rclpy.init(args=args)

    client = FollowWaypointsClient()

    # 从CSV文件读取导航点
    # waypoints = read_waypoints_from_csv('/home/davinci-mini/racecar/src/racecar/scripts/ai_test.csv')
    waypoints = read_waypoints_from_csv('/home/davinci-mini/racecar/src/racecar/scripts/out_test.csv')


    # 将导航点转换为 PoseStamped 消息
    poses = []
    for point in waypoints:
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = client.get_clock().now().to_msg()
        pose.pose.position.x = point["x"]
        pose.pose.position.y = point["y"]
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = point["z"]
        pose.pose.orientation.w = point["w"]
        poses.append(pose)

    client.send_goal(poses)

    rclpy.spin(client)

if __name__ == '__main__':
    main()
