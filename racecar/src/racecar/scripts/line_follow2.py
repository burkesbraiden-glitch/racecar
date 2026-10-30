#!/usr/bin/env python3

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


SCRIPT_VERSION = 'dual_color_centerline_v2'
WINDOW_NAME = 'Adjust_hsv'
COLOR_SWITCH = '0:Stop\n1:Red\n2:Green\n3:Blue\n4:Yellow\n5:Black'
COLOR_RANGES = {
    1: ('Red', (((0, 100, 80), (10, 255, 255)),
                ((170, 100, 80), (180, 255, 255)))),
    2: ('Green', (((35, 43, 46), (77, 255, 255)),)),
    3: ('Blue', (((100, 43, 46), (124, 255, 255)),)),
    # The low saturation limit tolerates yellow washed out by floor glare.
    4: ('Yellow', (((18, 45, 80), (45, 255, 255)),)),
    5: ('Black', (((0, 0, 0), (180, 255, 46)),)),
}

LOOKAHEAD_BANDS = (
    (0.42, 0.52, 0.55),
    (0.52, 0.62, 0.30),
    (0.62, 0.72, 0.15),
)
MIN_COMPONENT_PIXELS = 60.0
MIN_VALID_BANDS = 2
MIN_LANE_WIDTH_RATIO = 0.18
MAX_LANE_WIDTH_RATIO = 0.90
ERROR_FILTER_ALPHA = 0.35
ERROR_DEADBAND_PIXELS = 3.0
MAX_STEERING_STEP = 0.025
INTEGRAL_ACTIVE_ERROR_PIXELS = 120.0
MAX_INTEGRAL_STEERING = 0.035
IMAGE_TIMEOUT_SECONDS = 0.35


class Follower(Node):
    def __init__(self):
        super().__init__('color_lane_follower')

        self.declare_parameter('cruise_speed', 0.55)
        self.declare_parameter('turn_speed', 0.45)
        self.declare_parameter('steering_kp', 0.0012)
        self.declare_parameter('steering_ki', 0.00012)
        # Image x grows to the right, while positive angular.z turns this
        # chassis left. Negating the pixel error gives negative feedback.
        self.declare_parameter('steering_sign', -1.0)
        self.declare_parameter('steering_bias', 0.0)
        self.declare_parameter('max_angular_z', 0.18)
        self.declare_parameter('default_lane_width_ratio', 0.58)

        self.cruise_speed = float(self.get_parameter('cruise_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.steering_kp = float(self.get_parameter('steering_kp').value)
        self.steering_ki = float(self.get_parameter('steering_ki').value)
        self.steering_sign = float(self.get_parameter('steering_sign').value)
        self.steering_bias = float(self.get_parameter('steering_bias').value)
        self.max_angular_z = float(self.get_parameter('max_angular_z').value)
        self.default_lane_width_ratio = float(
            self.get_parameter('default_lane_width_ratio').value
        )

        self.bridge = CvBridge()
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        command_qos = QoSProfile(depth=10)
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, image_qos
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/car_cmd_vel', command_qos)
        self.watchdog = self.create_timer(0.1, self.watchdog_callback)

        self.twist = Twist()
        self.last_image_time = time.monotonic()
        self.watchdog_stopped = False
        self.filtered_error = 0.0
        self.error_integral = 0.0
        self.last_control_time = time.monotonic()
        self.last_steering = 0.0
        self.band_lane_widths = [None] * len(LOOKAHEAD_BANDS)
        self.band_centers = [None] * len(LOOKAHEAD_BANDS)
        self.selected_color = 0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(COLOR_SWITCH, WINDOW_NAME, 0, 5, lambda _value: None)

        self.get_logger().info(
            f'{SCRIPT_VERSION}: select a color to start; 0 keeps the vehicle stopped'
        )

    def publish_command(self, linear_x, angular_z):
        self.twist.linear.x = float(linear_x)
        self.twist.angular.z = float(angular_z)
        self.cmd_vel_pub.publish(self.twist)

    def publish_stop(self, repeat=1):
        self.last_steering = 0.0
        for _ in range(repeat):
            self.publish_command(0.0, 0.0)
            if repeat > 1:
                time.sleep(0.02)

    def watchdog_callback(self):
        image_age = time.monotonic() - self.last_image_time
        if image_age > IMAGE_TIMEOUT_SECONDS and not self.watchdog_stopped:
            self.publish_stop(repeat=3)
            self.watchdog_stopped = True
            self.get_logger().warn('camera image timeout; vehicle stopped')

    def reset_tracking(self):
        self.filtered_error = 0.0
        self.error_integral = 0.0
        self.last_control_time = time.monotonic()
        self.last_steering = 0.0
        self.band_lane_widths = [None] * len(LOOKAHEAD_BANDS)
        self.band_centers = [None] * len(LOOKAHEAD_BANDS)

    @staticmethod
    def component_centers(mask, y0, y1):
        band = mask[y0:y1, :]
        contours, _ = cv2.findContours(
            band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        centers = []
        band_height = max(1, y1 - y0)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_COMPONENT_PIXELS:
                continue

            _, _, width, height = cv2.boundingRect(contour)
            if height < band_height * 0.25 or width > mask.shape[1] * 0.35:
                continue

            moments = cv2.moments(contour)
            if moments['m00'] > 0:
                centers.append(float(moments['m10'] / moments['m00']))

        return sorted(centers)

    def estimate_band_center(self, candidates, band_index, image_width, y_ratio):
        previous_center = self.band_centers[band_index]
        if previous_center is None:
            previous_center = image_width / 2.0

        lane_width = self.band_lane_widths[band_index]
        if lane_width is None:
            # Lane separation grows toward the bottom because of perspective.
            perspective_scale = 0.70 + 0.55 * y_ratio
            lane_width = (
                image_width * self.default_lane_width_ratio * perspective_scale
            )

        best_pair = None
        best_score = float('inf')
        min_width = image_width * MIN_LANE_WIDTH_RATIO
        max_width = image_width * MAX_LANE_WIDTH_RATIO
        for left_index, left_x in enumerate(candidates):
            for right_x in candidates[left_index + 1:]:
                pair_width = right_x - left_x
                if not min_width <= pair_width <= max_width:
                    continue
                pair_center = (left_x + right_x) / 2.0
                score = (
                    abs(pair_center - previous_center)
                    + 0.35 * abs(pair_width - lane_width)
                )
                if score < best_score:
                    best_score = score
                    best_pair = (left_x, right_x)

        if best_pair is not None:
            left_x, right_x = best_pair
            measured_width = right_x - left_x
            old_width = self.band_lane_widths[band_index]
            if old_width is None:
                self.band_lane_widths[band_index] = measured_width
            else:
                self.band_lane_widths[band_index] = (
                    0.85 * old_width + 0.15 * measured_width
                )
            center = (left_x + right_x) / 2.0
            self.band_centers[band_index] = center
            return center, (left_x, right_x), 'both'

        if not candidates:
            return None, (), 'none'

        expected_left = previous_center - lane_width / 2.0
        expected_right = previous_center + lane_width / 2.0
        best_x = min(
            candidates,
            key=lambda x: min(abs(x - expected_left), abs(x - expected_right)),
        )
        if abs(best_x - expected_left) <= abs(best_x - expected_right):
            center = best_x + lane_width / 2.0
            side = 'left'
        else:
            center = best_x - lane_width / 2.0
            side = 'right'

        # Limit a single-line estimate so one bad component cannot command a
        # sudden full-lock turn. The temporal steering limiter handles the rest.
        max_center_step = image_width * 0.12
        center = max(
            previous_center - max_center_step,
            min(previous_center + max_center_step, center),
        )
        self.band_centers[band_index] = center
        return center, (best_x,), side

    def detect_lane_center(self, mask):
        height, width = mask.shape
        weighted_center_sum = 0.0
        total_weight = 0.0
        debug_points = []

        for index, (y0_ratio, y1_ratio, weight) in enumerate(LOOKAHEAD_BANDS):
            y0 = int(height * y0_ratio)
            y1 = int(height * y1_ratio)
            y_mid = (y0 + y1) // 2
            candidates = self.component_centers(mask, y0, y1)
            center, used_points, mode = self.estimate_band_center(
                candidates, index, width, (y0_ratio + y1_ratio) / 2.0
            )
            if center is None:
                continue

            weighted_center_sum += center * weight
            total_weight += weight
            debug_points.append((y_mid, center, used_points, mode))

        if len(debug_points) < MIN_VALID_BANDS or total_weight <= 0:
            return None, debug_points
        return weighted_center_sum / total_weight, debug_points

    @staticmethod
    def color_mask(image, selection):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        color_name = 'Stop'

        color_config = COLOR_RANGES.get(selection)
        if color_config is not None:
            color_name, ranges = color_config
            for lower, upper in ranges:
                range_mask = cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                )
                mask = cv2.bitwise_or(mask, range_mask)

        height, _ = mask.shape
        mask[:int(height * 0.35), :] = 0
        mask[int(height * 0.88):, :] = 0

        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        return mask, color_name

    def image_callback(self, msg):
        self.last_image_time = time.monotonic()
        self.watchdog_stopped = False

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        selection = cv2.getTrackbarPos(COLOR_SWITCH, WINDOW_NAME)
        if selection != self.selected_color:
            self.selected_color = selection
            self.reset_tracking()
            self.publish_stop()

        mask, color_name = self.color_mask(image, selection)
        if selection == 0:
            lane_center = None
            debug_points = []
        else:
            lane_center, debug_points = self.detect_lane_center(mask)
        image_center = image.shape[1] / 2.0
        raw_error = 0.0
        proportional_steering = 0.0
        integral_steering = 0.0

        if lane_center is None:
            self.filtered_error = 0.0
            self.error_integral = 0.0
            self.last_control_time = time.monotonic()
            self.publish_stop()
            detected = False
            steering = 0.0
            speed = 0.0
        else:
            raw_error = lane_center - image_center
            self.filtered_error = (
                ERROR_FILTER_ALPHA * raw_error
                + (1.0 - ERROR_FILTER_ALPHA) * self.filtered_error
            )
            if abs(self.filtered_error) < ERROR_DEADBAND_PIXELS:
                self.filtered_error = 0.0

            now = time.monotonic()
            control_dt = min(0.2, max(0.0, now - self.last_control_time))
            self.last_control_time = now
            self.error_integral *= 0.995
            if abs(self.filtered_error) <= INTEGRAL_ACTIVE_ERROR_PIXELS:
                self.error_integral += self.filtered_error * control_dt
            else:
                self.error_integral *= 0.9

            if abs(self.steering_ki) > 1e-9:
                integral_limit = MAX_INTEGRAL_STEERING / abs(self.steering_ki)
                self.error_integral = max(
                    -integral_limit,
                    min(integral_limit, self.error_integral),
                )

            proportional_steering = (
                self.steering_sign * self.steering_kp * self.filtered_error
            )
            integral_steering = (
                self.steering_sign * self.steering_ki * self.error_integral
            )
            integral_steering = max(
                -MAX_INTEGRAL_STEERING,
                min(MAX_INTEGRAL_STEERING, integral_steering),
            )
            desired_steering = (
                self.steering_bias
                + proportional_steering
                + integral_steering
            )
            desired_steering = max(
                -self.max_angular_z,
                min(self.max_angular_z, desired_steering),
            )
            steering_delta = max(
                -MAX_STEERING_STEP,
                min(MAX_STEERING_STEP, desired_steering - self.last_steering),
            )
            steering = self.last_steering + steering_delta
            self.last_steering = steering
            speed = (
                self.turn_speed
                if abs(steering) > self.max_angular_z * 0.55
                else self.cruise_speed
            )
            self.publish_command(speed, steering)
            detected = True

        debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.line(
            debug_image,
            (int(image_center), 0),
            (int(image_center), debug_image.shape[0] - 1),
            (255, 0, 0),
            1,
        )
        for y, center, used_points, mode in debug_points:
            color = (0, 255, 0) if mode == 'both' else (0, 165, 255)
            for point in used_points:
                cv2.circle(debug_image, (int(point), y), 4, color, -1)
            cv2.circle(debug_image, (int(center), y), 5, (255, 0, 255), -1)

        cv2.putText(
            debug_image,
            f'Selected: {color_name}',
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, debug_image)
        cv2.waitKey(1)
        modes = ','.join(point[3] for point in debug_points)
        print(
            f'selected={color_name}, detected={detected}, '
            f'bands={len(debug_points)}, modes={modes}, '
            f'raw_error={raw_error:.1f}, filtered_error={self.filtered_error:.1f}, '
            f'p={proportional_steering:.3f}, i={integral_steering:.3f}, '
            f'angular_z={steering:.3f}, '
            f'linear_x={speed:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    follower = Follower()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        follower.publish_stop(repeat=10)
        cv2.destroyAllWindows()
        follower.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
