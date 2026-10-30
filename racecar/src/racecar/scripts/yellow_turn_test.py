#!/usr/bin/env python3

import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


WINDOW_NAME = 'yellow_turn_test'
MASK_WINDOW_NAME = 'yellow_turn_mask'
SCRIPT_VERSION = 'yellow_turn_test_v4_glare_rejection'
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

YELLOW_HUE_MIN = 16
YELLOW_HUE_MAX = 42
YELLOW_VALUE_MIN = 65


def steering_target(
    path_center,
    heading_error,
    image_width,
    base_left_steering,
    boundary_kp,
    heading_kp,
    max_steering,
):
    """Follow right-boundary position and its forward curve direction."""
    position_error = image_width / 2.0 - path_center
    heading_error = max(-image_width * 0.45, min(image_width * 0.45, heading_error))

    # During this left U-turn, a negative heading is usually a perspective
    # change of the visible right boundary, not a command to steer right.
    # Only positive heading contributes to the left-turn curve bias.
    left_curve_evidence = max(
        position_error / 40.0,
        max(0.0, heading_error) / 50.0,
        0.0,
    )
    curve_strength = min(1.0, left_curve_evidence)
    target = (
        base_left_steering * curve_strength
        + boundary_kp * position_error
        + heading_kp * max(0.0, heading_error)
    )
    return max(-max_steering, min(max_steering, target))


class YellowTurnTestNode(Node):
    def __init__(self):
        super().__init__('yellow_turn_test')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('flip_180', True)
        self.declare_parameter('turn_speed', 0.24)
        self.declare_parameter('base_left_steering', 0.20)
        self.declare_parameter('boundary_kp', 0.0040)
        self.declare_parameter('heading_kp', 0.0020)
        self.declare_parameter('max_steering', 0.32)
        self.declare_parameter('minimum_left_steering', 0.10)
        self.declare_parameter('inside_margin_pixels', 18.0)
        self.declare_parameter('yellow_min_saturation', 80)
        self.declare_parameter('yellow_lab_b_min', 145)
        self.declare_parameter('yellow_excess_min', 30)
        self.declare_parameter('run_seconds', 10.0)
        self.declare_parameter('yellow_lost_stop_seconds', 0.6)
        self.declare_parameter('camera_failure_limit', 5)

        self.camera_index = int(self.get_parameter('camera_index').value)
        self.flip_180 = bool(self.get_parameter('flip_180').value)
        self.turn_speed = max(
            0.0, float(self.get_parameter('turn_speed').value)
        )
        self.base_left_steering = max(
            0.0, float(self.get_parameter('base_left_steering').value)
        )
        self.boundary_kp = max(
            0.0, float(self.get_parameter('boundary_kp').value)
        )
        self.heading_kp = max(
            0.0, float(self.get_parameter('heading_kp').value)
        )
        self.max_steering = abs(
            float(self.get_parameter('max_steering').value)
        )
        self.minimum_left_steering = max(
            0.0, float(self.get_parameter('minimum_left_steering').value)
        )
        if self.base_left_steering > self.max_steering:
            self.get_logger().warn(
                f'base_left_steering={self.base_left_steering:.3f} exceeds '
                f'max_steering={self.max_steering:.3f}; clamping it'
            )
            self.base_left_steering = self.max_steering
        self.minimum_left_steering = min(
            self.minimum_left_steering, self.max_steering
        )
        self.inside_margin_pixels = max(
            0.0, float(self.get_parameter('inside_margin_pixels').value)
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
        self.run_seconds = max(
            0.1, float(self.get_parameter('run_seconds').value)
        )
        self.yellow_lost_stop_seconds = max(
            0.1,
            float(self.get_parameter('yellow_lost_stop_seconds').value),
        )
        self.camera_failure_limit = max(
            1, int(self.get_parameter('camera_failure_limit').value)
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

        self.start_time = time.monotonic()
        self.last_log_time = 0.0
        self.last_steering = min(
            self.base_left_steering, self.max_steering
        )
        self.camera_failures = 0
        self.stop_requested = False
        self.yellow_seen = False
        self.yellow_missing_frames = 0
        self.last_yellow_time = None

        self.right_boundary_centers = [None, None, None]
        self.band_path_centers = [None, None, None]
        self.filtered_path_center = None
        self.filtered_heading_error = 0.0

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MASK_WINDOW_NAME, cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f'{SCRIPT_VERSION}: speed={self.turn_speed:.2f}, '
            f'base_left={self.base_left_steering:.3f}, '
            f'position_kp={self.boundary_kp:.4f}, '
            f'heading_kp={self.heading_kp:.4f}, max={self.max_steering:.3f}, '
            f'inside_margin={self.inside_margin_pixels:.0f}px, '
            f'yellow S>={self.yellow_min_saturation}, '
            f'LAB-b>={self.yellow_lab_b_min}, '
            f'excess>={self.yellow_excess_min}, '
            f'run={self.run_seconds:.1f}s'
        )

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
    def yellow_components(mask, y0, y1):
        band = mask[y0:y1, :]
        contours, _ = cv2.findContours(
            band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        band_height = max(1, y1 - y0)
        image_width = mask.shape[1]
        min_area = max(35.0, image_width * band_height * 0.0035)
        components = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            _, _, box_width, box_height = cv2.boundingRect(contour)
            if box_height < band_height * 0.20 or box_width > image_width * 0.85:
                continue
            moments = cv2.moments(contour)
            if moments['m00'] <= 0.0:
                continue
            center_x = float(moments['m10'] / moments['m00'])
            image_contour = contour.copy()
            image_contour[:, :, 1] += y0
            components.append((center_x, area, image_contour))

        return sorted(components, key=lambda item: item[0])

    def select_right_boundary(self, components, band_index, image_width):
        if not components:
            return None

        previous = self.right_boundary_centers[band_index]
        if previous is None:
            # At the turn entrance the physical right tape is the rightmost
            # substantial yellow component. Area prevents tiny glare patches
            # at the image edge from winning this first selection.
            largest_area = max(item[1] for item in components)
            strong_components = [
                item for item in components if item[1] >= largest_area * 0.35
            ]
            selected = max(strong_components, key=lambda item: item[0])
        else:
            selected = min(
                components,
                key=lambda item: abs(item[0] - previous)
                - 0.02 * min(item[1], 500.0),
            )

        measured_x = selected[0]
        if previous is None:
            filtered_x = measured_x
        else:
            max_step = image_width * 0.22
            measured_x = max(
                previous - max_step, min(previous + max_step, measured_x)
            )
            filtered_x = 0.60 * measured_x + 0.40 * previous
        self.right_boundary_centers[band_index] = filtered_x
        return filtered_x, selected[2]

    def select_lane_pair(self, components, band_index, image_width):
        """Return a genuine left/right pair, filtering small glare fragments."""
        if len(components) < 2:
            return None

        largest_area = max(item[1] for item in components)
        strong_components = [
            item for item in components
            if item[1] >= max(35.0, largest_area * 0.20)
        ]
        if len(strong_components) < 2:
            return None

        previous = self.band_path_centers[band_index]
        if previous is None:
            previous = image_width / 2.0
        expected_width = image_width * 0.58
        best_pair = None
        best_score = float('inf')
        for left_index, left in enumerate(strong_components):
            for right in strong_components[left_index + 1:]:
                pair_width = right[0] - left[0]
                if not image_width * 0.22 <= pair_width <= image_width * 0.90:
                    continue
                pair_center = (left[0] + right[0]) / 2.0
                score = (
                    abs(pair_center - previous)
                    + 0.30 * abs(pair_width - expected_width)
                )
                if score < best_score:
                    best_score = score
                    best_pair = (left, right)
        return best_pair

    def detect_yellow_lane(self, mask, debug_image):
        height, width = mask.shape
        roi_mask = mask.copy()
        roi_mask[:int(height * 0.28), :] = 0
        roi_mask[int(height * 0.94):, :] = 0
        bands = (
            # y0, y1, desired right-boundary x, lookahead weight
            (0.38, 0.50, 0.68, 0.55),
            (0.50, 0.64, 0.75, 0.30),
            (0.64, 0.80, 0.82, 0.15),
        )
        observations = []
        has_double_line = False

        for index, band_config in enumerate(bands):
            y0_ratio, y1_ratio, desired_right_ratio, weight = band_config
            y0 = int(height * y0_ratio)
            y1 = int(height * y1_ratio)
            components = self.yellow_components(roi_mask, y0, y1)
            y_mid = (y0 + y1) // 2
            for component_x, _, _ in components:
                cv2.circle(
                    debug_image,
                    (int(component_x), y_mid),
                    4,
                    (0, 255, 0),
                    -1,
                )

            pair = self.select_lane_pair(components, index, width)
            if pair is not None:
                left, right = pair
                path_x = (
                    (left[0] + right[0]) / 2.0
                    - self.inside_margin_pixels
                )
                path_x = max(0.0, min(width - 1.0, path_x))
                self.band_path_centers[index] = path_x
                self.right_boundary_centers[index] = right[0]
                cv2.drawContours(debug_image, [left[2]], -1, (0, 255, 0), 2)
                cv2.drawContours(debug_image, [right[2]], -1, (0, 255, 0), 2)
                cv2.circle(
                    debug_image, (int(path_x), y_mid), 5, (0, 255, 0), -1
                )
                observation_mode = 'both'
                has_double_line = True
            else:
                selected = self.select_right_boundary(components, index, width)
                if selected is None:
                    continue
                boundary_x, contour = selected
                boundary_offset = width * (desired_right_ratio - 0.50)
                path_x = (
                    boundary_x
                    - boundary_offset
                    - self.inside_margin_pixels
                )
                path_x = max(0.0, min(width - 1.0, path_x))
                self.band_path_centers[index] = path_x
                cv2.drawContours(debug_image, [contour], -1, (0, 165, 255), 2)
                cv2.circle(
                    debug_image,
                    (int(boundary_x), y_mid),
                    5,
                    (0, 165, 255),
                    -1,
                )
                observation_mode = 'right-contour'
            observations.append((y_mid, path_x, weight, observation_mode))

        if not observations:
            return None, 0, 'lost', 0.0

        total_weight = sum(item[2] for item in observations)
        raw_path_center = sum(
            item[1] * item[2] for item in observations
        ) / total_weight

        ordered = sorted(observations, key=lambda item: item[0])
        if len(ordered) >= 2:
            raw_heading_error = ordered[-1][1] - ordered[0][1]
        else:
            raw_heading_error = 0.0

        if self.filtered_path_center is None:
            self.filtered_path_center = raw_path_center
        else:
            max_step = width * 0.18
            raw_path_center = max(
                self.filtered_path_center - max_step,
                min(self.filtered_path_center + max_step, raw_path_center),
            )
            self.filtered_path_center = (
                0.60 * raw_path_center + 0.40 * self.filtered_path_center
            )
        self.filtered_heading_error = (
            0.55 * raw_heading_error + 0.45 * self.filtered_heading_error
        )

        for y_mid, path_x, _, _ in ordered:
            cv2.circle(debug_image, (int(path_x), y_mid), 5, (255, 0, 255), -1)
        for first, second in zip(ordered, ordered[1:]):
            cv2.line(
                debug_image,
                (int(first[1]), first[0]),
                (int(second[1]), second[0]),
                (255, 0, 255),
                2,
            )
        display_x = max(0.0, min(width - 1.0, self.filtered_path_center))
        cv2.line(
            debug_image,
            (width // 2, int(height * 0.84)),
            (int(display_x), ordered[0][0]),
            (255, 0, 255),
            2,
        )
        return (
            self.filtered_path_center,
            len(observations),
            'both' if has_double_line else 'right-contour',
            self.filtered_heading_error,
        )

    def smooth_steering(self, target):
        target = max(-self.max_steering, min(self.max_steering, target))
        steering = 0.35 * self.last_steering + 0.65 * target
        max_step = 0.07
        steering = max(
            self.last_steering - max_step,
            min(self.last_steering + max_step, steering),
        )
        self.last_steering = steering
        return steering

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

    def process_frame(self, frame):
        now = time.monotonic()
        elapsed = now - self.start_time
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        if self.flip_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        yellow_mask = self.detect_yellow_mask(blurred, hsv)
        debug_image = frame.copy()
        path_center, yellow_bands, mode, heading_error = self.detect_yellow_lane(
            yellow_mask, debug_image
        )

        if path_center is None:
            self.yellow_missing_frames += 1
            target = self.last_steering
            heading_error = 0.0
            lost_seconds = (
                0.0
                if self.last_yellow_time is None
                else now - self.last_yellow_time
            )
            if not self.yellow_seen:
                # Do not move until a real yellow contour has been acquired.
                speed = 0.0
                steering = 0.0
                self.last_steering = 0.0
            elif lost_seconds >= self.yellow_lost_stop_seconds:
                speed = 0.0
                steering = 0.0
                self.last_steering = 0.0
                self.stop_requested = True
                self.get_logger().warn(
                    f'yellow contour lost for {lost_seconds:.2f}s; '
                    'stopping test'
                )
            else:
                # Bridge only a short mask gap; never continue a blind turn.
                speed = min(self.turn_speed, 0.20)
                steering = self.last_steering
        else:
            self.yellow_seen = True
            self.yellow_missing_frames = 0
            self.last_yellow_time = now
            target = steering_target(
                path_center,
                heading_error,
                frame.shape[1],
                self.base_left_steering,
                self.boundary_kp,
                self.heading_kp,
                self.max_steering,
            )
            # Do not let a transient contour/heading change turn the car right
            # before it has acquired the second straight cone lane.
            target = max(self.minimum_left_steering, target)
            steering = self.smooth_steering(target)
            speed = self.turn_speed
        self.publish_command(speed, steering)

        cv2.line(
            debug_image,
            (frame.shape[1] // 2, 0),
            (frame.shape[1] // 2, frame.shape[0] - 1),
            (255, 255, 255),
            1,
        )
        center_text = 'none' if path_center is None else f'{path_center:.0f}'
        cv2.putText(
            debug_image,
            f'YELLOW TEST  t={elapsed:.1f}/{self.run_seconds:.0f}s',
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f'Pc={center_text} mode={mode} bands={yellow_bands}',
            (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_image,
            f'heading={heading_error:.0f} target={target:.3f} steer={steering:.3f}',
            (8, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, debug_image)
        cv2.imshow(MASK_WINDOW_NAME, yellow_mask)

        if now - self.last_log_time >= 0.5:
            self.last_log_time = now
            self.get_logger().info(
                f't={elapsed:.1f}s path_center={center_text} mode={mode} '
                f'bands={yellow_bands} heading={heading_error:.1f} '
                f'target={target:.3f} speed={speed:.2f} '
                f'steering={steering:.3f}'
            )

        if elapsed >= self.run_seconds:
            self.get_logger().info('test run time reached; stopping car')
            self.stop_requested = True

    def camera_loop(self):
        if not self.cap.isOpened():
            self.get_logger().error(f'cannot open camera index {self.camera_index}')
            self.publish_stop(repeat=10)
            return

        while rclpy.ok() and not self.stop_requested:
            ret, frame = self.cap.read()
            if not ret:
                self.camera_failures += 1
                self.publish_stop()
                if self.camera_failures >= self.camera_failure_limit:
                    self.get_logger().error(
                        f'camera failed for {self.camera_failures} consecutive frames'
                    )
                    break
                time.sleep(0.05)
                continue

            self.camera_failures = 0
            self.process_frame(frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self.get_logger().info('operator requested stop')
                break

        self.publish_stop(repeat=10)

    def close(self):
        self.publish_stop(repeat=10)
        self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = YellowTurnTestNode()
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
