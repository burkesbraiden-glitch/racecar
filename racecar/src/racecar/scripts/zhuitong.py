#!/usr/bin/env python3

import time
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


WINDOW_NAME = 'zhuitong_debug'
SCRIPT_VERSION = 'zhuitong_fusion_v14_fixed_turn_stop_test'
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

RED_RANGES = (
    (np.array([0, 90, 60]), np.array([12, 255, 255])),
    (np.array([168, 90, 60]), np.array([180, 255, 255])),
)
BLUE_RANGES = (
    (np.array([90, 80, 35]), np.array([135, 255, 255])),
)
YELLOW_HUE_MIN = 16
YELLOW_HUE_MAX = 42
YELLOW_VALUE_MIN = 65


class DriveState(Enum):
    FIRST_CONE_LANE = auto()
    FIXED_TURN = auto()
    SECOND_LANE_ACQUIRE = auto()
    SECOND_CONE_LANE = auto()
    STOPPED = auto()


class RacecarNode(Node):
    def __init__(self):
        super().__init__('zhuitong_fusion')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('flip_180', True)
        self.declare_parameter('cruise_speed', 0.55)
        self.declare_parameter('turn_speed', 0.20)
        self.declare_parameter('fixed_turn_steering', 0.25)
        self.declare_parameter('fixed_turn_seconds', 10.0)
        self.declare_parameter('stop_after_fixed_turn', True)
        self.declare_parameter('cone_steering_kp', 0.0040)
        self.declare_parameter('yellow_steering_kp', 0.0045)
        self.declare_parameter('yellow_base_left_steering', 0.20)
        self.declare_parameter('cone_max_steering', 0.22)
        self.declare_parameter('yellow_max_steering', 0.30)
        self.declare_parameter('yellow_minimum_left_steering', 0.20)
        self.declare_parameter('yellow_inside_margin_pixels', 18.0)
        self.declare_parameter('yellow_min_saturation', 80)
        self.declare_parameter('yellow_lab_b_min', 145)
        self.declare_parameter('yellow_excess_min', 30)
        self.declare_parameter('left_search_steering', 0.20)
        self.declare_parameter('cone_missing_frames', 8)
        self.declare_parameter('finish_missing_frames', 10)
        self.declare_parameter('second_lane_confirm_frames', 5)
        self.declare_parameter('yellow_confirm_frames', 3)
        self.declare_parameter('minimum_first_lane_seconds', 3.0)
        self.declare_parameter('minimum_yellow_turn_seconds', 5.0)
        self.declare_parameter('yellow_turn_timeout_seconds', 22.0)
        self.declare_parameter('yellow_lost_stop_seconds', 2.0)
        self.declare_parameter('blue_exit_speed', 0.12)
        self.declare_parameter('blue_exit_desired_x_ratio', 0.72)
        self.declare_parameter('blue_exit_steering_kp', 0.0040)
        self.declare_parameter('blue_exit_timeout_seconds', 5.0)
        self.declare_parameter('maximum_run_seconds', 50.0)

        self.camera_index = int(self.get_parameter('camera_index').value)
        self.flip_180 = bool(self.get_parameter('flip_180').value)
        self.cruise_speed = float(self.get_parameter('cruise_speed').value)
        self.turn_speed = float(self.get_parameter('turn_speed').value)
        self.fixed_turn_steering = max(
            0.0,
            min(0.50, float(self.get_parameter('fixed_turn_steering').value)),
        )
        self.fixed_turn_seconds = max(
            0.1, float(self.get_parameter('fixed_turn_seconds').value)
        )
        self.stop_after_fixed_turn = bool(
            self.get_parameter('stop_after_fixed_turn').value
        )
        self.cone_steering_kp = float(
            self.get_parameter('cone_steering_kp').value
        )
        self.yellow_steering_kp = float(
            self.get_parameter('yellow_steering_kp').value
        )
        self.yellow_base_left_steering = max(
            0.0,
            float(self.get_parameter('yellow_base_left_steering').value),
        )
        self.cone_max_steering = abs(
            float(self.get_parameter('cone_max_steering').value)
        )
        self.yellow_max_steering = abs(
            float(self.get_parameter('yellow_max_steering').value)
        )
        self.yellow_minimum_left_steering = max(
            0.0,
            float(
                self.get_parameter('yellow_minimum_left_steering').value
            ),
        )
        self.yellow_inside_margin_pixels = max(
            0.0,
            float(self.get_parameter('yellow_inside_margin_pixels').value),
        )
        self.yellow_min_saturation = max(
            0,
            min(255, int(self.get_parameter('yellow_min_saturation').value)),
        )
        self.yellow_lab_b_min = max(
            0,
            min(255, int(self.get_parameter('yellow_lab_b_min').value)),
        )
        self.yellow_excess_min = max(
            0,
            min(255, int(self.get_parameter('yellow_excess_min').value)),
        )
        self.left_search_steering = abs(
            float(self.get_parameter('left_search_steering').value)
        )
        self.cone_missing_limit = int(
            self.get_parameter('cone_missing_frames').value
        )
        self.finish_missing_limit = int(
            self.get_parameter('finish_missing_frames').value
        )
        self.second_lane_confirm_limit = int(
            self.get_parameter('second_lane_confirm_frames').value
        )
        self.yellow_confirm_limit = int(
            self.get_parameter('yellow_confirm_frames').value
        )
        self.minimum_first_lane_seconds = float(
            self.get_parameter('minimum_first_lane_seconds').value
        )
        self.minimum_yellow_turn_seconds = float(
            self.get_parameter('minimum_yellow_turn_seconds').value
        )
        self.yellow_turn_timeout_seconds = float(
            self.get_parameter('yellow_turn_timeout_seconds').value
        )
        self.yellow_lost_stop_seconds = max(
            0.2,
            float(self.get_parameter('yellow_lost_stop_seconds').value),
        )
        self.blue_exit_speed = max(
            0.0, float(self.get_parameter('blue_exit_speed').value)
        )
        self.blue_exit_desired_x_ratio = max(
            0.55,
            min(
                0.90,
                float(self.get_parameter('blue_exit_desired_x_ratio').value),
            ),
        )
        self.blue_exit_steering_kp = max(
            0.0, float(self.get_parameter('blue_exit_steering_kp').value)
        )
        self.blue_exit_timeout_seconds = max(
            1.0,
            float(self.get_parameter('blue_exit_timeout_seconds').value),
        )
        self.maximum_run_seconds = float(
            self.get_parameter('maximum_run_seconds').value
        )

        self.publisher_ = self.create_publisher(Twist, '/car_cmd_vel', 10)
        self.twist = Twist()
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        time.sleep(1.0)

        self.state = DriveState.FIRST_CONE_LANE
        self.start_time = time.monotonic()
        self.state_start_time = self.start_time
        self.last_frame_time = self.start_time
        self.last_log_time = 0.0
        self.frame_count = 0
        self.camera_failures = 0

        self.first_lane_seen = False
        self.second_lane_seen = False
        self.cone_missing_count = 0
        self.second_lane_confirm_count = 0
        self.yellow_confirm_count = 0
        self.yellow_missing_count = 0

        self.last_steering = 0.0
        self.last_cone_center = FRAME_WIDTH / 2.0
        self.cone_lane_width = None
        self.cone_red_offset = None
        self.cone_blue_offset = None
        self.yellow_lane_widths = [None, None, None]
        self.yellow_band_centers = [None, None, None]
        self.filtered_yellow_heading = 0.0
        self.filtered_yellow_center = None
        self.yellow_seen_in_turn = False
        self.last_yellow_time = None
        self.yellow_double_line_seen = False
        self.yellow_single_line_only = False

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f'{SCRIPT_VERSION}: first cone lane -> fixed timed left turn '
            '-> second cone lane -> stop; '
            f'turn speed={self.turn_speed:.2f}, '
            f'steering={self.fixed_turn_steering:.3f}, '
            f'duration={self.fixed_turn_seconds:.1f}s, '
            f'stop_after_turn={self.stop_after_fixed_turn}'
        )

    @staticmethod
    def color_mask(hsv, ranges):
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    def detect_yellow_mask(self, bgr, hsv):
        """Reject warm white glare while retaining genuinely yellow tape."""
        hsv_mask = cv2.inRange(
            hsv,
            np.array(
                [YELLOW_HUE_MIN, self.yellow_min_saturation, YELLOW_VALUE_MIN],
                dtype=np.uint8,
            ),
            np.array([YELLOW_HUE_MAX, 255, 255], dtype=np.uint8),
        )

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_b = lab[:, :, 2]
        lab_mask = np.where(
            lab_b >= self.yellow_lab_b_min, 255, 0
        ).astype(np.uint8)

        blue, green, red = cv2.split(bgr)
        yellow_excess = (
            (red.astype(np.int16) + green.astype(np.int16)) // 2
            - blue.astype(np.int16)
        )
        excess_mask = np.where(
            yellow_excess >= self.yellow_excess_min, 255, 0
        ).astype(np.uint8)

        mask = cv2.bitwise_and(hsv_mask, lab_mask)
        mask = cv2.bitwise_and(mask, excess_mask)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    @staticmethod
    def cone_points(mask):
        height, width = mask.shape
        roi = mask.copy()
        roi[:int(height * 0.08), :] = 0
        roi[int(height * 0.86):, :] = 0
        contours, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        points = []
        min_area = max(45.0, width * height * 0.0006)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_height < 7 or box_width > width * 0.35:
                continue
            points.append((x + box_width / 2.0, float(y + box_height), contour))
        return sorted(points, key=lambda point: point[1], reverse=True)

    def detect_cone_lane(self, image, red_mask, blue_mask):
        height, width = red_mask.shape
        red_points = self.cone_points(red_mask)
        blue_points = self.cone_points(blue_mask)
        y_reference = height * 0.72
        measured_center = None
        mode = 'none'

        max_center_distance = width * (
            0.48
            if self.state in (
                DriveState.FIXED_TURN,
                DriveState.SECOND_LANE_ACQUIRE,
            )
            else 0.28
        )
        best_pair = None
        best_score = float('inf')
        for red_x, red_y, _ in red_points:
            for blue_x, blue_y, _ in blue_points:
                # Both straight lanes are red on the camera's left and blue on
                # its right, including after the yellow U-turn.
                if red_x >= blue_x:
                    continue
                pair_width = abs(blue_x - red_x)
                if not width * 0.12 <= pair_width <= width * 0.92:
                    continue
                vertical_difference = abs(red_y - blue_y)
                if vertical_difference > max(18.0, 0.16 * max(red_y, blue_y)):
                    continue
                center = (red_x + blue_x) / 2.0
                if abs(center - self.last_cone_center) > max_center_distance:
                    continue
                width_penalty = 0.0
                if self.cone_lane_width is not None:
                    width_penalty = abs(pair_width - self.cone_lane_width) * 0.35
                depth_reward = 0.10 * (red_y + blue_y)
                score = (
                    abs(center - self.last_cone_center)
                    + 1.4 * vertical_difference
                    + width_penalty
                    - depth_reward
                )
                if score < best_score:
                    best_score = score
                    best_pair = (red_x, blue_x, center, pair_width)

        if best_pair is not None:
            red_x, blue_x, measured_center, measured_width = best_pair
            if self.cone_lane_width is None:
                self.cone_lane_width = measured_width
            else:
                self.cone_lane_width = (
                    0.85 * self.cone_lane_width + 0.15 * measured_width
                )
            self.cone_red_offset = red_x - measured_center
            self.cone_blue_offset = blue_x - measured_center
            mode = 'red+blue'
        elif self.cone_red_offset is not None and red_points:
            estimates = [point[0] - self.cone_red_offset for point in red_points]
            candidate = min(estimates, key=lambda x: abs(x - self.last_cone_center))
            if abs(candidate - self.last_cone_center) <= max_center_distance:
                measured_center = candidate
                mode = 'red'
        elif self.cone_blue_offset is not None and blue_points:
            estimates = [point[0] - self.cone_blue_offset for point in blue_points]
            candidate = min(estimates, key=lambda x: abs(x - self.last_cone_center))
            if abs(candidate - self.last_cone_center) <= max_center_distance:
                measured_center = candidate
                mode = 'blue'

        if measured_center is not None:
            max_step = width * 0.16
            measured_center = max(
                self.last_cone_center - max_step,
                min(self.last_cone_center + max_step, measured_center),
            )
            self.last_cone_center = (
                0.45 * measured_center + 0.55 * self.last_cone_center
            )

        for _, _, contour in red_points:
            cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
        for _, _, contour in blue_points:
            cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)
        if measured_center is not None:
            cv2.circle(
                image,
                (int(self.last_cone_center), int(y_reference)),
                5,
                (0, 255, 0),
                -1,
            )

        return {
            'center': self.last_cone_center if measured_center is not None else None,
            'red_count': len(red_points),
            'blue_count': len(blue_points),
            'nearest_red_x': red_points[0][0] if red_points else None,
            'nearest_blue_x': blue_points[0][0] if blue_points else None,
            'both_visible': mode == 'red+blue',
            'any_visible': measured_center is not None,
            'mode': mode,
        }

    @staticmethod
    def yellow_candidates(mask, y0, y1):
        band = mask[y0:y1, :]
        column_counts = np.count_nonzero(band, axis=0)
        active_columns = column_counts >= max(2, int((y1 - y0) * 0.08))
        candidates = []
        run_start = None
        for index, active in enumerate(np.append(active_columns, False)):
            if active and run_start is None:
                run_start = index
            elif not active and run_start is not None:
                if index - run_start >= 2:
                    weights = column_counts[run_start:index].astype(np.float32)
                    columns = np.arange(run_start, index, dtype=np.float32)
                    candidates.append(float(np.average(columns, weights=weights)))
                run_start = None
        return candidates

    def estimate_yellow_center(self, candidates, band_index, image_width, y_ratio):
        previous_center = self.yellow_band_centers[band_index]
        if previous_center is None:
            previous_center = image_width / 2.0
        lane_width = self.yellow_lane_widths[band_index]
        if lane_width is None:
            lane_width = image_width * (0.48 + 0.16 * y_ratio)

        best_pair = None
        best_score = float('inf')
        for left_index, left_x in enumerate(candidates):
            for right_x in candidates[left_index + 1:]:
                pair_width = right_x - left_x
                if not image_width * 0.14 <= pair_width <= image_width * 0.92:
                    continue
                pair_center = (left_x + right_x) / 2.0
                score = abs(pair_center - previous_center) + 0.30 * abs(
                    pair_width - lane_width
                )
                if score < best_score:
                    best_score = score
                    best_pair = (left_x, right_x)

        if best_pair is not None:
            left_x, right_x = best_pair
            measured_width = right_x - left_x
            old_width = self.yellow_lane_widths[band_index]
            if old_width is None:
                self.yellow_lane_widths[band_index] = measured_width
            else:
                self.yellow_lane_widths[band_index] = (
                    0.88 * old_width + 0.12 * measured_width
                )
            center = (left_x + right_x) / 2.0
            mode = 'both'
            used = best_pair
        elif candidates:
            expected_left = previous_center - lane_width / 2.0
            expected_right = previous_center + lane_width / 2.0
            point = min(
                candidates,
                key=lambda x: min(abs(x - expected_left), abs(x - expected_right)),
            )
            if abs(point - expected_left) <= abs(point - expected_right):
                center = point + lane_width / 2.0
                mode = 'left'
            else:
                center = point - lane_width / 2.0
                mode = 'right'
            used = (point,)
        else:
            return None, (), 'none'

        max_step = image_width * 0.18
        center = max(previous_center - max_step, min(previous_center + max_step, center))
        center = max(0.0, min(image_width - 1.0, center))
        self.yellow_band_centers[band_index] = center
        return center, used, mode

    def detect_yellow_lane(self, mask, debug_image):
        height, width = mask.shape
        roi_mask = mask.copy()
        roi_mask[:int(height * 0.28), :] = 0
        roi_mask[int(height * 0.94):, :] = 0
        bands = (
            # y0, y1, desired right-boundary x
            (0.36, 0.50, 0.68),
            (0.50, 0.65, 0.75),
            (0.65, 0.82, 0.82),
        )
        observations = []

        for index, (y0_ratio, y1_ratio, desired_right_ratio) in enumerate(bands):
            y0 = int(height * y0_ratio)
            y1 = int(height * y1_ratio)
            candidates = self.yellow_candidates(roi_mask, y0, y1)
            center, used, mode = self.estimate_yellow_center(
                candidates, index, width, (y0_ratio + y1_ratio) / 2.0
            )
            if center is None:
                continue
            y_mid = (y0 + y1) // 2
            if mode == 'both':
                left_x, right_x = used
                target_x = (left_x + right_x) / 2.0
                self.yellow_double_line_seen = True
            elif not self.yellow_double_line_seen or not used:
                continue
            elif mode in ('left', 'right'):
                # Course-specific rule: through the left U-turn the inner/left
                # tape often leaves the camera, while the remaining tape is the
                # physical right boundary even after it curves across the image.
                # Do not relabel it as a left boundary from its screen x value.
                line_x = used[0]
                desired_x = width * desired_right_ratio
                target_x = width / 2.0 + line_x - desired_x
                mode = 'right'
            else:
                continue
            observations.append((y_mid, target_x, used, mode))

        double_observations = [
            observation for observation in observations if observation[3] == 'both'
        ]
        if double_observations:
            # Directly follow the midpoint of the two yellow points in the
            # farthest band. Do not connect unrelated centers across bands.
            selected = min(double_observations, key=lambda observation: observation[0])
            self.yellow_single_line_only = False
        elif observations and self.yellow_double_line_seen:
            # In the U-turn the left boundary may leave the camera. Follow the
            # remaining identified boundary at a fixed screen-side reference.
            selected = min(observations, key=lambda observation: observation[0])
            self.yellow_single_line_only = True
        else:
            self.yellow_single_line_only = False
            return None, 0, 0.0

        y_target, raw_lane_center, _, _ = selected
        for y, _, points, mode in observations:
            color = (0, 255, 0) if mode == 'both' else (0, 165, 255)
            for point in points:
                cv2.circle(debug_image, (int(point), y), 5, color, -1)

        if self.filtered_yellow_center is None:
            self.filtered_yellow_center = raw_lane_center
        else:
            max_center_step = width * (
                0.10 if self.yellow_single_line_only else 0.18
            )
            raw_lane_center = max(
                self.filtered_yellow_center - max_center_step,
                min(self.filtered_yellow_center + max_center_step, raw_lane_center),
            )
            measurement_weight = 0.45 if self.yellow_single_line_only else 0.60
            self.filtered_yellow_center = (
                measurement_weight * raw_lane_center
                + (1.0 - measurement_weight) * self.filtered_yellow_center
            )

        target_for_display = max(0.0, min(width - 1.0, self.filtered_yellow_center))
        cv2.circle(
            debug_image,
            (int(target_for_display), y_target),
            6,
            (255, 0, 255),
            -1,
        )
        cv2.line(
            debug_image,
            (width // 2, int(height * 0.84)),
            (int(target_for_display), y_target),
            (255, 0, 255),
            2,
        )
        self.filtered_yellow_heading = 0.0
        return (
            self.filtered_yellow_center,
            len(observations),
            0.0,
        )

    def publish_command(self, speed, steering):
        self.twist.linear.x = float(speed)
        self.twist.angular.z = float(steering)
        self.publisher_.publish(self.twist)

    def publish_stop(self, repeat=1):
        self.last_steering = 0.0
        for _ in range(repeat):
            self.publish_command(0.0, 0.0)
            if repeat > 1:
                time.sleep(0.02)

    def smooth_steering(self, target, max_value, alpha=0.55):
        target = max(-max_value, min(max_value, target))
        steering = alpha * self.last_steering + (1.0 - alpha) * target
        max_step = 0.07
        steering = max(
            self.last_steering - max_step,
            min(self.last_steering + max_step, steering),
        )
        self.last_steering = steering
        return steering

    def change_state(self, new_state, reason):
        if new_state == self.state:
            return
        previous = self.state
        self.state = new_state
        self.state_start_time = time.monotonic()
        self.cone_missing_count = 0
        self.second_lane_confirm_count = 0
        self.yellow_confirm_count = 0
        self.yellow_missing_count = 0
        self.last_steering = 0.0
        if new_state in (
            DriveState.SECOND_LANE_ACQUIRE,
            DriveState.SECOND_CONE_LANE,
        ):
            self.last_cone_center = FRAME_WIDTH / 2.0
            self.cone_lane_width = None
            self.cone_red_offset = None
            self.cone_blue_offset = None
        self.get_logger().info(f'{previous.name} -> {new_state.name}: {reason}')

    def control_first_lane(self, cone, image_width, elapsed):
        if cone['both_visible']:
            self.first_lane_seen = True
        if cone['both_visible']:
            self.cone_missing_count = 0
        else:
            self.cone_missing_count += 1

        if not self.first_lane_seen:
            self.publish_stop()
            return

        ready_to_turn = (
            self.first_lane_seen
            and elapsed >= self.minimum_first_lane_seconds
            and self.cone_missing_count >= self.cone_missing_limit * 2
        )
        if ready_to_turn:
            self.change_state(
                DriveState.FIXED_TURN,
                'end of first cone lane; starting fixed timed turn',
            )
            self.last_steering = self.fixed_turn_steering
            self.publish_command(self.turn_speed, self.fixed_turn_steering)
            return

        if cone['center'] is None:
            # A brief cone gap is common between cone groups. Keep moving slowly
            # with the last valid steering instead of stopping on a single frame.
            self.publish_command(self.cruise_speed, self.last_steering)
            return
        error = image_width / 2.0 - cone['center']
        steering = self.smooth_steering(
            self.cone_steering_kp * error, self.cone_max_steering
        )
        self.publish_command(self.cruise_speed, steering)

    def control_fixed_turn(self, state_elapsed):
        if state_elapsed < self.fixed_turn_seconds:
            self.last_steering = self.fixed_turn_steering
            self.publish_command(self.turn_speed, self.fixed_turn_steering)
            return

        if self.stop_after_fixed_turn:
            self.change_state(
                DriveState.STOPPED,
                'fixed turn time reached; test stop position',
            )
            self.publish_stop(repeat=10)
            return

        self.change_state(
            DriveState.SECOND_LANE_ACQUIRE,
            'fixed turn time reached; acquiring left-red/right-blue lane',
        )
        self.publish_command(self.blue_exit_speed, 0.0)

    def control_second_lane_acquire(self, cone, image_width, state_elapsed):
        centered_second_lane = (
            cone['center'] is not None
            and abs(cone['center'] - image_width / 2.0) <= image_width * 0.22
        )
        if cone['both_visible']:
            cone_error = image_width / 2.0 - cone['center']
            target = self.cone_steering_kp * cone_error
            steering = self.smooth_steering(
                target, self.cone_max_steering, 0.35
            )
            self.publish_command(self.blue_exit_speed, steering)
            if centered_second_lane:
                self.second_lane_confirm_count += 1
            else:
                self.second_lane_confirm_count = 0
            if self.second_lane_confirm_count >= self.second_lane_confirm_limit:
                self.second_lane_seen = True
                self.change_state(
                    DriveState.SECOND_CONE_LANE,
                    'left-red/right-blue pair aligned after fixed turn',
                )
            return

        blue_x = cone['nearest_blue_x']
        if blue_x is not None:
            desired_blue_x = image_width * self.blue_exit_desired_x_ratio
            blue_error = desired_blue_x - blue_x
            target = self.blue_exit_steering_kp * blue_error
            target = max(-0.12, min(self.yellow_max_steering, target))
            steering = self.smooth_steering(
                target, self.yellow_max_steering, 0.35
            )
            self.publish_command(self.blue_exit_speed, steering)
        elif cone['nearest_red_x'] is not None:
            desired_red_x = image_width * (1.0 - self.blue_exit_desired_x_ratio)
            red_error = desired_red_x - cone['nearest_red_x']
            target = self.blue_exit_steering_kp * red_error
            target = max(-0.12, min(self.yellow_max_steering, target))
            steering = self.smooth_steering(
                target, self.yellow_max_steering, 0.35
            )
            self.publish_command(self.blue_exit_speed, steering)
        else:
            steering = self.smooth_steering(
                0.0, self.yellow_max_steering, 0.45
            )
            self.publish_command(min(self.blue_exit_speed, 0.08), steering)

        self.second_lane_confirm_count = 0
        if state_elapsed >= self.blue_exit_timeout_seconds:
            self.change_state(DriveState.STOPPED, 'second lane acquisition timeout')
            self.publish_stop(repeat=5)

    def control_second_lane(self, cone, image_width):
        if cone['both_visible']:
            self.second_lane_seen = True
        if cone['both_visible']:
            self.cone_missing_count = 0
        elif self.second_lane_seen:
            self.cone_missing_count += 1

        if (
            self.second_lane_seen
            and self.cone_missing_count >= self.finish_missing_limit
        ):
            self.change_state(DriveState.STOPPED, 'end of second cone lane')
            self.publish_stop(repeat=5)
            return

        if cone['center'] is None:
            self.publish_command(self.cruise_speed, self.last_steering)
            return
        error = image_width / 2.0 - cone['center']
        steering = self.smooth_steering(
            self.cone_steering_kp * error, self.cone_max_steering
        )
        self.publish_command(self.cruise_speed, steering)

    def process_frame(self, frame):
        now = time.monotonic()
        elapsed = now - self.start_time
        state_elapsed = now - self.state_start_time
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        if self.flip_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        red_mask = self.color_mask(hsv, RED_RANGES)
        blue_mask = self.color_mask(hsv, BLUE_RANGES)

        debug_image = frame.copy()
        cone = self.detect_cone_lane(debug_image, red_mask, blue_mask)

        if elapsed >= self.maximum_run_seconds and self.state != DriveState.STOPPED:
            self.change_state(DriveState.STOPPED, 'maximum run time reached')
            self.publish_stop(repeat=5)

        if self.state == DriveState.FIRST_CONE_LANE:
            self.control_first_lane(cone, frame.shape[1], elapsed)
        elif self.state == DriveState.FIXED_TURN:
            self.control_fixed_turn(state_elapsed)
        elif self.state == DriveState.SECOND_LANE_ACQUIRE:
            self.control_second_lane_acquire(
                cone,
                frame.shape[1],
                state_elapsed,
            )
        elif self.state == DriveState.SECOND_CONE_LANE:
            self.control_second_lane(cone, frame.shape[1])
        else:
            self.publish_stop()

        cv2.line(
            debug_image,
            (frame.shape[1] // 2, 0),
            (frame.shape[1] // 2, frame.shape[0] - 1),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            debug_image,
            f'{self.state.name}  t={elapsed:.1f}s',
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f"R={cone['red_count']} B={cone['blue_count']} mode={cone['mode']}",
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if self.state == DriveState.FIXED_TURN:
            phase_text = (
                f'fixed={min(state_elapsed, self.fixed_turn_seconds):.1f}/'
                f'{self.fixed_turn_seconds:.1f}s '
            )
        elif self.state == DriveState.SECOND_LANE_ACQUIRE:
            phase_text = 'acquiring second lane '
        else:
            phase_text = ''
        cv2.putText(
            debug_image,
            f'{phase_text}speed={self.twist.linear.x:.2f} '
            f'steer={self.twist.angular.z:.2f}',
            (8, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, debug_image)

        if now - self.last_log_time >= 0.5:
            self.last_log_time = now
            turn_text = (
                f' fixed_turn={state_elapsed:.1f}/{self.fixed_turn_seconds:.1f}s'
                if self.state == DriveState.FIXED_TURN
                else ''
            )
            self.get_logger().info(
                f"state={self.state.name} R={cone['red_count']} "
                f"B={cone['blue_count']} cone_mode={cone['mode']} "
                f'speed={self.twist.linear.x:.2f} '
                f'steering={self.twist.angular.z:.3f}{turn_text}'
            )

    def camera_loop(self):
        if not self.cap.isOpened():
            self.get_logger().error(f'cannot open camera index {self.camera_index}')
            self.publish_stop(repeat=10)
            return

        while rclpy.ok() and self.state != DriveState.STOPPED:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_failures += 1
                self.publish_stop()
                if self.camera_failures >= 5:
                    self.get_logger().error('camera failed for 5 consecutive frames')
                    self.change_state(DriveState.STOPPED, 'camera failure')
                    break
                time.sleep(0.05)
                continue

            self.camera_failures = 0
            self.frame_count += 1
            self.last_frame_time = time.monotonic()
            self.process_frame(frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self.change_state(DriveState.STOPPED, 'operator requested stop')
                break

        self.publish_stop(repeat=10)

    def close(self):
        self.publish_stop(repeat=10)
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = RacecarNode()
    try:
        node.camera_loop()
    except KeyboardInterrupt:
        node.get_logger().info('interrupted by operator')
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
