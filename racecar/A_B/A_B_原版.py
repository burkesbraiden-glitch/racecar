#!/usr/bin/env python3
"""Detect A/B signs with the trained OM model and optionally park the car."""

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
import torch
from ais_bench.infer.interface import InferSession
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


SCRIPT_DIR = Path(__file__).resolve().parent
INFER_PROJECT_DIR = SCRIPT_DIR / 'infer_project'
if str(INFER_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(INFER_PROJECT_DIR))

from edge_infer.det_utils import letterbox, nms, scale_coords  # noqa: E402


CLASS_NAMES = {0: 'A', 1: 'B'}
COLORS = {'A': (0, 210, 0), 'B': (255, 120, 0)}
INPUT_SHAPE = (640, 640)
WINDOW_NAME = 'A_B_detection'


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Tuple[float, float, float, float]

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.box[0] + self.box[2]) * 0.5

    @property
    def bottom(self) -> float:
        return self.box[3]


class DriveState(Enum):
    DETECT_DECISION = auto()
    PASS_DECISION = auto()
    APPROACH_TARGET = auto()
    PARKED = auto()
    FAILED = auto()


class ABDetector:
    def __init__(
        self,
        model_path: Path,
        device_id: int,
        confidence_threshold: float,
        iou_threshold: float,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f'OM model not found: {model_path}')
        self.session = InferSession(device_id, str(model_path))
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def infer(self, frame: np.ndarray) -> List[Detection]:
        padded, scale_ratio, pad_size = letterbox(
            frame, new_shape=INPUT_SHAPE
        )
        input_tensor = padded[:, :, ::-1].transpose(2, 0, 1)
        input_tensor = np.ascontiguousarray(
            input_tensor[None, ...], dtype=np.float16
        ) / 255.0

        output = self.session.infer([input_tensor])[0]
        predictions = nms(
            torch.as_tensor(output),
            conf_thres=self.confidence_threshold,
            iou_thres=self.iou_threshold,
        )[0]
        if predictions.numel() == 0:
            return []

        boxes = predictions.detach().cpu().numpy()
        scale_coords(
            INPUT_SHAPE,
            boxes[:, :4],
            frame.shape,
            ratio_pad=(scale_ratio, pad_size),
        )

        detections = []
        for x1, y1, x2, y2, confidence, class_id in boxes:
            label = CLASS_NAMES.get(int(class_id))
            if label is None:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=float(confidence),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return detections


class ABParkingNode(Node):
    def __init__(self, detector: ABDetector, args: argparse.Namespace) -> None:
        super().__init__('ab_sign_parking')
        self.detector = detector
        self.args = args
        self.bridge = CvBridge()

        self.sign_publisher = self.create_publisher(String, '/ab_sign', 10)
        self.target_publisher = self.create_publisher(
            String, '/ab_parking/target', 10
        )
        self.state_publisher = self.create_publisher(
            String, '/ab_parking/state', 10
        )
        self.image_publisher = self.create_publisher(
            Image, '/camera/image_result', 10
        )
        self.command_publisher = None
        if args.drive:
            self.command_publisher = self.create_publisher(
                Twist, args.command_topic, 10
            )

        self.state = DriveState.DETECT_DECISION
        self.state_started_at = time.monotonic()
        self.run_started_at = self.state_started_at
        self.decision_label = ''
        self.target_label = ''
        self.candidate_label = ''
        self.candidate_count = 0
        self.decision_max_area_ratio = 0.0
        self.decision_near_seen = False
        self.decision_lost_count = 0
        self.close_count = 0
        self.target_seen = False
        self.last_target_seen_at = 0.0
        self.last_steering = 0.0
        self.last_log_at = 0.0
        self.last_conflict_check_at = 0.0
        self.camera_failures = 0
        self._publish_state()

    @staticmethod
    def _best_detection(
        detections: Sequence[Detection], label: Optional[str] = None
    ) -> Optional[Detection]:
        eligible = [
            detection
            for detection in detections
            if label is None or detection.label == label
        ]
        if not eligible:
            return None
        # The nearest sign has the largest box and is the decision-area sign.
        return max(eligible, key=lambda item: (item.area, item.confidence))

    def _publish_text(self, publisher, text: str) -> None:
        message = String()
        message.data = text
        publisher.publish(message)

    def _publish_state(self) -> None:
        self._publish_text(self.state_publisher, self.state.name)

    def _change_state(self, state: DriveState, reason: str) -> None:
        if self.state == state:
            return
        old_state = self.state
        self.state = state
        self.state_started_at = time.monotonic()
        self._publish_state()
        self.get_logger().info(f'{old_state.name} -> {state.name}: {reason}')

    def _publish_command(self, speed: float, steering: float) -> None:
        if self.command_publisher is None:
            return
        command = Twist()
        command.linear.x = float(speed)
        command.angular.z = float(steering)
        self.command_publisher.publish(command)

    def stop(self, repeat: int = 10) -> None:
        if self.command_publisher is None:
            return
        for _ in range(repeat):
            self._publish_command(0.0, 0.0)
            if repeat > 1:
                time.sleep(0.02)

    def _check_command_conflict(self, now: float) -> None:
        if self.command_publisher is None:
            return
        if now - self.last_conflict_check_at < 5.0:
            return
        self.last_conflict_check_at = now
        publisher_count = self.count_publishers(self.args.command_topic)
        if publisher_count > 1:
            self.get_logger().error(
                f'{self.args.command_topic} has {publisher_count} publishers; '
                'stop other driving nodes before using --drive'
            )

    def _confirm_decision(self, detection: Optional[Detection]) -> None:
        if detection is None:
            self.candidate_label = ''
            self.candidate_count = 0
            return
        if detection.label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = detection.label
            self.candidate_count = 1

        if self.candidate_count < self.args.confirm_frames:
            return
        self.decision_label = self.candidate_label
        if self.args.opposite_target:
            self.target_label = 'B' if self.decision_label == 'A' else 'A'
        else:
            self.target_label = self.decision_label
        self.decision_max_area_ratio = 0.0
        self.decision_near_seen = False
        self.decision_lost_count = 0
        self._publish_text(self.target_publisher, self.target_label)
        self._change_state(
            DriveState.PASS_DECISION,
            f'decision sign {self.decision_label} confirmed in '
            f'{self.candidate_count} frames; parking target '
            f'{self.target_label}',
        )

    def _pass_decision_sign(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> None:
        detection = self._best_detection(detections, self.decision_label)
        if detection is not None:
            area_ratio = detection.area / float(frame_width * frame_height)
            width_ratio = detection.width / float(frame_width)
            bottom_ratio = detection.bottom / float(frame_height)
            self.decision_max_area_ratio = max(
                self.decision_max_area_ratio, area_ratio
            )
            if (
                width_ratio >= self.args.pass_near_width_ratio
                or bottom_ratio >= self.args.pass_near_bottom_ratio
            ):
                self.decision_near_seen = True

            area_dropped = (
                self.decision_near_seen
                and self.decision_max_area_ratio > 0.0
                and area_ratio
                < self.decision_max_area_ratio * self.args.pass_area_drop_ratio
            )
            self.decision_lost_count = (
                self.decision_lost_count + 1 if area_dropped else 0
            )
        elif self.decision_near_seen:
            self.decision_lost_count += 1

        elapsed = now - self.state_started_at
        passed_by_vision = (
            elapsed >= self.args.pass_min_seconds
            and self.decision_lost_count >= self.args.pass_lost_frames
        )
        timed_out = elapsed >= self.args.pass_max_seconds
        if passed_by_vision or timed_out:
            reason = (
                'decision sign passed'
                if passed_by_vision
                else 'decision pass timeout reached'
            )
            self._change_state(DriveState.APPROACH_TARGET, reason)

    def _approach_target(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> Tuple[float, float]:
        detection = self._best_detection(detections, self.target_label)
        if detection is None:
            self.close_count = 0
            self.last_steering *= 0.75
            search_elapsed = now - self.state_started_at
            target_lost = (
                self.target_seen
                and now - self.last_target_seen_at
                >= self.args.target_lost_timeout
            )
            search_timeout = search_elapsed >= self.args.target_search_timeout
            if target_lost or search_timeout:
                reason = (
                    f'target {self.target_label} lost'
                    if target_lost
                    else f'target {self.target_label} search timeout'
                )
                self._change_state(DriveState.FAILED, reason)
                return 0.0, 0.0
            return self.args.approach_speed, self.last_steering

        self.target_seen = True
        self.last_target_seen_at = now
        width_ratio = detection.width / float(frame_width)
        bottom_ratio = detection.bottom / float(frame_height)
        close_enough = (
            width_ratio >= self.args.stop_width_ratio
            or bottom_ratio >= self.args.stop_bottom_ratio
        )
        self.close_count = self.close_count + 1 if close_enough else 0
        if self.close_count >= self.args.stop_confirm_frames:
            self._change_state(
                DriveState.PARKED,
                f'target {self.target_label} reached '
                f'(width={width_ratio:.2f}, bottom={bottom_ratio:.2f})',
            )
            return 0.0, 0.0

        normalized_error = (
            detection.center_x - frame_width * 0.5
        ) / (frame_width * 0.5)
        desired_steering = (
            self.args.steering_sign
            * self.args.steering_kp
            * normalized_error
        )
        desired_steering = float(
            np.clip(
                desired_steering,
                -self.args.steering_limit,
                self.args.steering_limit,
            )
        )
        alpha = self.args.steering_smoothing
        self.last_steering = (
            alpha * self.last_steering + (1.0 - alpha) * desired_steering
        )
        speed = (
            self.args.slow_speed
            if width_ratio >= self.args.slow_width_ratio
            else self.args.approach_speed
        )
        return speed, self.last_steering

    def _drive_control(
        self,
        detections: Sequence[Detection],
        frame_width: int,
        frame_height: int,
        now: float,
    ) -> None:
        if now - self.run_started_at >= self.args.maximum_run_seconds:
            self._change_state(DriveState.FAILED, 'maximum run time reached')

        if self.state == DriveState.DETECT_DECISION:
            if (
                now - self.state_started_at
                >= self.args.decision_search_timeout
            ):
                self._change_state(
                    DriveState.FAILED, 'decision sign search timeout'
                )
                self._publish_command(0.0, 0.0)
                return
            self._confirm_decision(self._best_detection(detections))
            self._publish_command(self.args.decision_speed, 0.0)
        elif self.state == DriveState.PASS_DECISION:
            self._pass_decision_sign(
                detections, frame_width, frame_height, now
            )
            self._publish_command(self.args.decision_speed, 0.0)
        elif self.state == DriveState.APPROACH_TARGET:
            speed, steering = self._approach_target(
                detections, frame_width, frame_height, now
            )
            self._publish_command(speed, steering)
        else:
            self._publish_command(0.0, 0.0)

    def _annotate(
        self, frame: np.ndarray, detections: Sequence[Detection]
    ) -> np.ndarray:
        output = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = (int(value) for value in detection.box)
            color = COLORS[detection.label]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f'{detection.label} {detection.confidence:.2f}',
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
        mode = self.state.name if self.args.drive else 'DETECT_ONLY'
        decision = self.decision_label or '-'
        target = self.target_label or '-'
        cv2.putText(
            output,
            f'{mode}  decision={decision}  target={target}',
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.detector.infer(frame)
        now = time.monotonic()
        best = self._best_detection(detections)
        if best is not None:
            self._publish_text(self.sign_publisher, best.label)

        if self.args.drive:
            self._check_command_conflict(now)
            self._drive_control(
                detections, frame.shape[1], frame.shape[0], now
            )

        if now - self.last_log_at >= 1.0:
            self.last_log_at = now
            summary = ', '.join(
                f'{item.label}:{item.confidence:.2f}' for item in detections
            ) or 'none'
            self.get_logger().info(
                f'state={self.state.name} '
                f'decision={self.decision_label or "-"} '
                f'target={self.target_label or "-"} '
                f'detections={summary}'
            )
        return self._annotate(frame, detections)

    def run(self) -> None:
        camera_index = self.args.camera
        if camera_index < 0:
            camera_index = find_camera_index()
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f'cannot open camera index {camera_index}')

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.camera_height)
        self.get_logger().info(
            f'camera={camera_index}, drive={self.args.drive}, '
            f'model={self.args.model}'
        )
        if not self.args.drive:
            self.get_logger().warn(
                'detection-only mode: add --drive only after a raised-wheel test'
            )

        now = time.monotonic()
        self.run_started_at = now
        self.state_started_at = now

        try:
            while rclpy.ok():
                ok, frame = capture.read()
                if not ok:
                    self.camera_failures += 1
                    self.stop(repeat=3)
                    if self.camera_failures >= 5:
                        raise RuntimeError('camera failed for 5 consecutive frames')
                    continue
                self.camera_failures = 0
                if self.args.flip_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                annotated = self.process_frame(frame)
                self.image_publisher.publish(
                    self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                )
                rclpy.spin_once(self, timeout_sec=0.001)

                if self.args.visible:
                    cv2.imshow(WINDOW_NAME, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
        finally:
            self.stop()
            capture.release()
            if self.args.visible:
                cv2.destroyAllWindows()


def find_camera_index() -> int:
    for index in range(10):
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        ok = capture.isOpened() and capture.read()[0]
        capture.release()
        if ok:
            return index
    raise RuntimeError('no usable camera found at index 0..9')


def bounded_ratio(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError('value must be in [0, 1]')
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise argparse.ArgumentTypeError('value must be greater than zero')
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='A/B OM detection and optional autonomous parking'
    )
    parser.add_argument(
        '--model',
        default='infer_project/output/yolov5s_bs1.om',
        help='OM model path; relative paths are resolved from the A_B directory',
    )
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument(
        '--camera', type=int, default=-1, help='camera index; -1 scans 0..9'
    )
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--no-flip', dest='flip_180', action='store_false')
    parser.set_defaults(flip_180=True)
    parser.add_argument('--visible', action='store_true')
    parser.add_argument(
        '--drive',
        action='store_true',
        help='publish autonomous commands; omitted means detection only',
    )
    parser.add_argument(
        '--opposite-target',
        action='store_true',
        help='temporary test mode: decision A parks at B, and B parks at A',
    )
    parser.add_argument('--command-topic', default='/car_cmd_vel')
    parser.add_argument('--conf-thres', type=bounded_ratio, default=0.45)
    parser.add_argument('--iou-thres', type=bounded_ratio, default=0.50)
    parser.add_argument('--confirm-frames', type=int, default=4)
    parser.add_argument('--decision-speed', type=positive_float, default=0.55)
    parser.add_argument('--decision-search-timeout', type=float, default=8.0)
    parser.add_argument('--approach-speed', type=positive_float, default=0.55)
    parser.add_argument('--slow-speed', type=positive_float, default=0.55)
    parser.add_argument('--pass-min-seconds', type=float, default=0.8)
    parser.add_argument('--pass-max-seconds', type=float, default=3.0)
    parser.add_argument('--pass-near-width-ratio', type=bounded_ratio, default=0.25)
    parser.add_argument('--pass-near-bottom-ratio', type=bounded_ratio, default=0.82)
    parser.add_argument('--pass-area-drop-ratio', type=bounded_ratio, default=0.40)
    parser.add_argument('--pass-lost-frames', type=int, default=3)
    parser.add_argument('--steering-kp', type=positive_float, default=0.28)
    parser.add_argument('--steering-limit', type=positive_float, default=0.22)
    parser.add_argument(
        '--steering-sign',
        type=float,
        default=-1.0,
        help='-1 when positive angular.z turns the car left',
    )
    parser.add_argument('--steering-smoothing', type=bounded_ratio, default=0.55)
    parser.add_argument('--slow-width-ratio', type=bounded_ratio, default=0.16)
    parser.add_argument('--stop-width-ratio', type=bounded_ratio, default=0.30)
    parser.add_argument('--stop-bottom-ratio', type=bounded_ratio, default=0.88)
    parser.add_argument('--stop-confirm-frames', type=int, default=3)
    parser.add_argument('--target-lost-timeout', type=float, default=0.8)
    parser.add_argument('--target-search-timeout', type=float, default=5.0)
    parser.add_argument('--maximum-run-seconds', type=float, default=30.0)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    args.model = model_path.resolve()
    args.confirm_frames = max(1, args.confirm_frames)
    args.pass_lost_frames = max(1, args.pass_lost_frames)
    args.stop_confirm_frames = max(1, args.stop_confirm_frames)
    args.pass_min_seconds = max(0.0, args.pass_min_seconds)
    args.pass_max_seconds = max(args.pass_min_seconds, args.pass_max_seconds)
    args.decision_search_timeout = max(0.1, args.decision_search_timeout)
    args.target_lost_timeout = max(0.1, args.target_lost_timeout)
    args.target_search_timeout = max(0.1, args.target_search_timeout)
    args.maximum_run_seconds = max(1.0, args.maximum_run_seconds)
    return args


def main() -> None:
    args = parse_args()
    detector = ABDetector(
        args.model,
        args.device_id,
        args.conf_thres,
        args.iou_thres,
    )
    rclpy.init()
    node = ABParkingNode(detector, args)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
