#!/usr/bin/env python3
"""Three-OAK autonomous route following with parallel perception, BEV obstacle avoidance, and pure pursuit.

Starts disarmed. A arms autonomous steering; S, E, or Space stops; Q exits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

# A root-launched OpenCV/Qt window must not attempt to join the desktop user's
# ICE session manager; its authentication cookie belongs to the user session.
# DISPLAY/XAUTHORITY are still preserved by sudo -E for the actual window.
os.environ.pop("SESSION_MANAGER", None)

import cv2
import depthai as dai
import numpy as np


RC_ROOT = Path(__file__).resolve().parent

from bev import (  # noqa: E402
    PANO_WIDTH,
    PurePursuitConfig,
    PurePursuitState,
    StereoDepthProvider,
    bearing_deg_from_x,
    detections_to_obstacles,
    plan_pure_pursuit,
)
from image_processing import create_stitch_state, stitch_depth_sift, stitch_sift  # noqa: E402
from inputs import KeyPoller, host_time_us  # noqa: E402
from models import Detection  # noqa: E402
from oak_live_inputs import get_camera_inputs  # noqa: E402
from gps import VectorNavReader  # noqa: E402
from parallel_perception import FusedStereoDepthProvider, ParallelPerception, semantic_array  # noqa: E402
from route_following import RouteFollower, bearing_deg, haversine_m  # noqa: E402
from trike_geometry import MAX_WHEEL_ANGLE_DEG, WHEELBASE_M  # noqa: E402
from live_dashboard import LiveDashboard  # noqa: E402
from road_bev import (  # noqa: E402
    corridor_bounds,
    detection_in_valid_panorama,
    mask_semantic_to_valid_frame,
    overlay_road,
    road_boundaries,
    valid_panorama_mask,
)

try:  # noqa: E402
    import voice
except Exception:
    voice = None


PCA9685_ADDRESS = 0x40
STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1

THROTTLE_NEUTRAL = 90.0
THROTTLE_FIRST_FORWARD = 99.0
THROTTLE_BRAKE = 80.0
STEERING_CENTER = 90.0
STEERING_MIN = 20.0
STEERING_MAX = 160.0
STEERING_MAX_OFFSET = 60.0

SPEED_MODES = {
    1: (1.0, 5.0, 99.0, 105.0),
    2: (5.0, 10.0, 105.0, 115.0),
    3: (10.0, 15.0, 115.0, 125.0),
    4: (15.0, 20.0, 125.0, 135.0),
    5: (20.0, 25.0, 135.0, 145.0),
}

DEFAULT_MODEL = RC_ROOT / "models" / "yolo26n.pt"
DEFAULT_DETECTION_ENGINE = RC_ROOT / "models" / "yolo26s_q32.engine"
DEFAULT_SEGMENTATION_ENGINE = RC_ROOT / "models" / "yolo26s-sem_q32.engine"
DEFAULT_DEPTH_ENGINE = RC_ROOT / "models" / "yolo26s-depth_q32.engine"
DEFAULT_ROUTE_FILE = RC_ROOT / "route_waypoints_dense_1m.json"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 400
DEFAULT_CLASS_ID = 0
BBOX_SLOW_WIDTH_PX = 300


def normalized_steering_to_servo_angle(steering: float, invert: bool = False) -> float:
    """Map normalized trike steering (-1 left, +1 right) to current servo angle."""
    steering = clamp(steering, -1.0, 1.0)
    if invert:
        steering *= -1.0
    return clamp(
        STEERING_CENTER + steering * STEERING_MAX_OFFSET,
        STEERING_MIN,
        STEERING_MAX,
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def route_goal_local(fix, goal: dict) -> tuple[float, float]:
    """VLA route target -> trike-local (forward, left) metres."""
    current = (float(fix.lat), float(fix.lon))
    target = (float(goal["lat"]), float(goal["lon"]))
    distance = haversine_m(current, target)
    target_bearing = bearing_deg(current, target)
    heading = float(fix.yaw_deg) if fix.yaw_deg is not None else float(goal["yaw_deg"])
    relative = math.radians((target_bearing - heading + 180.0) % 360.0 - 180.0)
    return max(0.25, distance * math.cos(relative)), -distance * math.sin(relative)


@dataclass
class ObjectTrack:
    track_id: int
    confidence: float
    bbox: tuple[int, int, int, int]
    class_id: int = DEFAULT_CLASS_ID
    label: str = "object"

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) * 0.5

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) * 0.5

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def as_detection(self, inset: bool = False) -> Detection:
        x1, y1, x2, y2 = self.bbox
        if inset:
            width = x2 - x1
            height = y2 - y1
            x1 += int(width * 0.20)
            x2 -= int(width * 0.20)
            y1 += int(height * 0.18)
            y2 -= int(height * 0.12)
        return Detection(self.label, self.confidence, (x1, y1, x2, y2))


class Announcer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and voice is not None
        self.last_text = ""
        self.last_time = 0.0

    def warmup(self) -> None:
        if not self.enabled:
            return
        try:
            voice.warmup()
            print("[voice] ready")
        except Exception as exc:
            print(f"[voice] disabled: {exc}")
            self.enabled = False

    def say(self, text: str, *, force: bool = False) -> None:
        print(f"[voice] {text}")
        now = time.monotonic()
        if not self.enabled:
            return
        if not force and text == self.last_text and now - self.last_time < 3.0:
            return
        self.last_text = text
        self.last_time = now
        try:
            voice.speak(text, blocking=False)
        except Exception as exc:
            print(f"[voice] failed: {exc}")


class CarController:
    """50 Hz output worker with command timeout and neutral-by-default state."""

    def __init__(self, dry_run: bool, timeout_s: float = 0.75) -> None:
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.armed = False
        self.target_steering = STEERING_CENTER
        self.target_throttle = THROTTLE_NEUTRAL
        self.current_steering = STEERING_CENTER
        self.current_throttle = THROTTLE_NEUTRAL
        self.brake_until = 0.0
        self.last_command_time = time.monotonic()
        self.error = ""

        self.i2c = None
        self.steering = None
        self.throttle = None
        self.last_sent_steering = None
        self.last_sent_throttle = None

        if not dry_run:
            try:
                import board
                import busio
                from adafruit_servokit import ServoKit

                self.i2c = busio.I2C(board.SCL, board.SDA)
                kit = ServoKit(channels=16, i2c=self.i2c, address=PCA9685_ADDRESS)
                self.steering = kit.servo[STEERING_CHANNEL]
                self.throttle = kit.servo[THROTTLE_CHANNEL]
                self.steering.set_pulse_width_range(1200, 2200)
                self.throttle.set_pulse_width_range(1000, 2000)
                self.steering.angle = STEERING_CENTER
                self.throttle.angle = THROTTLE_NEUTRAL
            except Exception as exc:
                raise RuntimeError(
                    "PCA9685 startup failed. Install ServoKit/Blinka for this Jetson, "
                    "or run with --dry-run."
                ) from exc

        self.thread = threading.Thread(target=self._loop, name="pca-control", daemon=True)
        self.thread.start()

    def arm(self) -> None:
        with self.lock:
            self.armed = True
            self.brake_until = 0.0
            self.last_command_time = time.monotonic()

    def disarm(self) -> None:
        with self.lock:
            self.armed = False
            self.target_steering = STEERING_CENTER
            self.target_throttle = THROTTLE_NEUTRAL
            self.brake_until = 0.0
            self.last_command_time = time.monotonic()

    def command(self, steering: float, throttle: float) -> None:
        with self.lock:
            self.target_steering = clamp(steering, STEERING_MIN, STEERING_MAX)
            self.target_throttle = clamp(throttle, THROTTLE_NEUTRAL, 180.0)
            self.last_command_time = time.monotonic()

    def brake_then_stop(self, duration_s: float) -> None:
        with self.lock:
            if self.armed and self.current_throttle > THROTTLE_NEUTRAL + 1.0:
                self.brake_until = max(self.brake_until, time.monotonic() + duration_s)
            self.target_throttle = THROTTLE_NEUTRAL
            self.last_command_time = time.monotonic()

    def snapshot(self) -> tuple[bool, float, float, str]:
        with self.lock:
            return self.armed, self.current_steering, self.current_throttle, self.error

    def _write_outputs(self, steering: float, throttle: float) -> None:
        if self.dry_run:
            return
        if self.last_sent_steering is None or abs(steering - self.last_sent_steering) >= 0.12:
            self.steering.angle = steering
            self.last_sent_steering = steering
        if self.last_sent_throttle is None or abs(throttle - self.last_sent_throttle) >= 0.12:
            self.throttle.angle = throttle
            self.last_sent_throttle = throttle

    def _loop(self) -> None:
        period = 0.02
        next_tick = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic()
            with self.lock:
                stale = now - self.last_command_time > self.timeout_s
                active = self.armed and not stale and not self.error
                target_steering = self.target_steering if active else STEERING_CENTER

                steer_delta = clamp(
                    (target_steering - self.current_steering) * 0.25,
                    -3.0,
                    3.0,
                )
                self.current_steering = clamp(
                    self.current_steering + steer_delta,
                    STEERING_MIN,
                    STEERING_MAX,
                )

                if not active:
                    self.current_throttle = THROTTLE_NEUTRAL
                    self.brake_until = 0.0
                elif now < self.brake_until:
                    self.current_throttle = THROTTLE_BRAKE
                elif self.target_throttle <= THROTTLE_NEUTRAL:
                    self.current_throttle = THROTTLE_NEUTRAL
                elif self.current_throttle < THROTTLE_FIRST_FORWARD:
                    self.current_throttle = THROTTLE_FIRST_FORWARD
                elif self.current_throttle < self.target_throttle:
                    self.current_throttle = min(self.target_throttle, self.current_throttle + 0.20)
                else:
                    self.current_throttle = max(self.target_throttle, self.current_throttle - 0.65)

                steering_out = self.current_steering
                throttle_out = self.current_throttle

            try:
                self._write_outputs(steering_out, throttle_out)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.armed = False
                    self.current_throttle = THROTTLE_NEUTRAL
                print(f"[control] PCA output failed: {exc}")

            next_tick += period
            self.stop_event.wait(max(0.0, next_tick - time.monotonic()))
            if time.monotonic() > next_tick + 0.25:
                next_tick = time.monotonic()

    def close(self) -> None:
        self.disarm()
        time.sleep(0.25)
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        if not self.dry_run:
            try:
                self.throttle.angle = THROTTLE_NEUTRAL
                self.steering.angle = STEERING_CENTER
                time.sleep(0.25)
                self.throttle.angle = None
                self.steering.angle = None
            finally:
                if self.i2c is not None:
                    self.i2c.deinit()


class DistancePolicy:
    def __init__(self, stop_m: float, stop_release_m: float, slow_m: float, full_m: float) -> None:
        self.stop_m = stop_m
        self.stop_release_m = max(stop_release_m, stop_m)
        self.slow_m = max(slow_m, self.stop_release_m)
        self.full_m = max(full_m, self.slow_m + 0.1)
        self.state = "IDLE"

    def update(
        self,
        distance_m: float | None,
        bearing_deg: float,
        steering: float,
        speed_mode: int,
    ) -> tuple[str, float, float]:
        """Return state, throttle PWM, and simulated target mph."""
        if distance_m is None or not math.isfinite(distance_m):
            self.state = "NO_DEPTH"
            return self.state, THROTTLE_NEUTRAL, 0.0

        if distance_m <= self.stop_m or (
            self.state == "STOP" and distance_m <= self.stop_release_m
        ):
            self.state = "STOP"
            return self.state, THROTTLE_NEUTRAL, 0.0

        if abs(bearing_deg) >= 55.0:
            self.state = "TURN_LEFT" if bearing_deg < 0 else "TURN_RIGHT"
            return self.state, THROTTLE_NEUTRAL, 0.0

        low_mph, high_mph, low_pwm, high_pwm = SPEED_MODES[speed_mode]
        if distance_m <= self.slow_m:
            ratio = clamp(
                (distance_m - self.stop_release_m) / max(0.01, self.slow_m - self.stop_release_m),
                0.0,
                1.0,
            )
            throttle = THROTTLE_FIRST_FORWARD + ratio * (min(105.0, high_pwm) - THROTTLE_FIRST_FORWARD)
            simulated_mph = 1.0 + ratio * min(4.0, high_mph - 1.0)
            self.state = "SLOW"
        else:
            ratio = clamp((distance_m - self.slow_m) / (self.full_m - self.slow_m), 0.0, 1.0)
            follow_start_pwm = max(low_pwm, min(105.0, high_pwm))
            follow_start_mph = max(low_mph, min(5.0, high_mph))
            throttle = follow_start_pwm + ratio * (high_pwm - follow_start_pwm)
            simulated_mph = follow_start_mph + ratio * (high_mph - follow_start_mph)
            self.state = "FOLLOW"

        turn_ratio = clamp(abs(steering - STEERING_CENTER) / STEERING_MAX_OFFSET, 0.0, 1.0)
        throttle = THROTTLE_NEUTRAL + (throttle - THROTTLE_NEUTRAL) * (1.0 - 0.35 * turn_ratio)
        throttle = max(THROTTLE_FIRST_FORWARD, throttle)
        return self.state, throttle, simulated_mph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-OAK autonomous route follower")
    parser.add_argument("--model", "--detect-engine", dest="detect_engine", type=Path, default=DEFAULT_DETECTION_ENGINE, help="Detection TensorRT engine (--model is a compatibility alias)")
    parser.add_argument("--segmentation-engine", type=Path, default=DEFAULT_SEGMENTATION_ENGINE)
    parser.add_argument("--depth-engine", type=Path, default=DEFAULT_DEPTH_ENGINE)
    parser.add_argument("--route-file", type=Path, default=DEFAULT_ROUTE_FILE)
    parser.add_argument("--route-name")
    parser.add_argument("--route-lookahead-m", type=float, default=5.0)
    parser.add_argument("--gps-max-age", type=float, default=1.0)
    parser.add_argument("--vn-port", default="")
    parser.add_argument("--vn-baud", type=int, default=115200)
    parser.add_argument("--no-route", action="store_true", help="Use local obstacle avoidance without GPS route guidance")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--preview-fps", type=float, default=10.0)
    parser.add_argument("--warmup-runs", type=int, default=3, help="Parallel three-engine warmup passes before cameras open")
    parser.add_argument("--detect-interval", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--speed-mode", type=int, choices=tuple(SPEED_MODES), default=1)
    parser.add_argument(
        "--planner-mode",
        choices=("pure-pursuit",),
        default="pure-pursuit",
        help="pure-pursuit emits normalized steering + haptic advisories; pure-pursuit is the autonomous planner",
    )
    parser.add_argument("--ego-speed-mps", type=float, default=0.8, help="Estimated current speed for haptic TTC and steering caps")
    parser.add_argument("--pure-lookahead-m", type=float, default=4.5)
    parser.add_argument("--gps-lateral-weight", type=float, default=0.15, help="Bounded GPS influence on the vision road center; 0 is fully vision-only")
    parser.add_argument("--gps-lateral-limit-m", type=float, default=0.50, help="Maximum GPS lateral error considered by the local planner")
    parser.add_argument("--pure-normal-limit", type=float, default=0.42)
    parser.add_argument("--pure-low-speed-limit", type=float, default=0.70)
    parser.add_argument("--pure-high-speed-limit", type=float, default=0.30)
    parser.add_argument("--road-half-width-m", type=float, default=1.75, help="Half-width of the drivable road/corridor used to clamp pure-pursuit goals")
    parser.add_argument("--road-edge-margin-m", type=float, default=0.30, help="Keep pure-pursuit goals this far inside the road/corridor edge")
    parser.add_argument("--command-log", type=Path, help="Optional JSONL command log for ESP/haptic integration testing")
    parser.add_argument("--stop-m", type=float, default=1.0)
    parser.add_argument("--stop-release-m", type=float, default=1.2)
    parser.add_argument("--slow-m", type=float, default=2.0)
    parser.add_argument("--full-m", type=float, default=4.0)
    parser.add_argument("--brake-seconds", type=float, default=0.35)
    parser.add_argument("--lost-seconds", type=float, default=0.50)
    parser.add_argument("--invert-steering", action="store_true")
    parser.add_argument("--recompute-sift", action="store_true")
    parser.add_argument(
        "--single-oak",
        action="store_true",
        help="Open only the configured center OAK and bypass panorama stitching",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not open PCA9685 or move the car")
    parser.add_argument("--auto-arm", action="store_true", help="Arm automatically while route, depth, and road safety inputs are valid")
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-imshow", action="store_true")
    parser.add_argument("--web-dashboard", action="store_true", help="Serve a read-only live perception and planner dashboard")
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--web-jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    if not 0.0 <= args.gps_lateral_weight <= 1.0:
        parser.error("--gps-lateral-weight must be in [0,1]")
    if args.gps_lateral_limit_m < 0.0:
        parser.error("--gps-lateral-limit-m must be nonnegative")
    return args


def parse_object_tracks(result) -> list[ObjectTrack]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return []
    ids = boxes.id
    tracks = []
    for index in range(len(boxes)):
        cls_id = int(boxes.cls[index].item()) if boxes.cls is not None else DEFAULT_CLASS_ID
        x1, y1, x2, y2 = [int(value) for value in boxes.xyxy[index].tolist()]
        track_id = int(ids[index].item()) if ids is not None else -(index + 1)
        confidence = float(boxes.conf[index].item()) if boxes.conf is not None else 0.0
        names = getattr(result, "names", {})
        if isinstance(names, dict):
            label = str(names.get(cls_id, f"class_{cls_id}"))
        elif 0 <= cls_id < len(names):
            label = str(names[cls_id])
        else:
            label = f"class_{cls_id}"
        tracks.append(ObjectTrack(track_id, confidence, (x1, y1, x2, y2), cls_id, label))
    return tracks


def valid_camera_frames(camera_inputs) -> bool:
    now_us = host_time_us()
    for runtime in camera_inputs.runtimes:
        with runtime.lock:
            if runtime.failed or runtime.last_frame_host_us <= 0:
                return False
            if now_us - runtime.last_frame_host_us > 2_500_000:
                return False
    return True


def camera_health(camera_inputs) -> tuple[bool, list[str]]:
    """Return readiness and detailed per-camera stream/thread diagnostics."""
    now_us = host_time_us()
    details = []
    all_ready = True
    for index, runtime in enumerate(camera_inputs.runtimes):
        with runtime.lock:
            streams = sorted(runtime.latest)
            last_frame_host_us = runtime.last_frame_host_us
            failed = runtime.failed
            error = runtime.error
            ready = runtime.ready.is_set()
        age_s = None if last_frame_host_us <= 0 else max(0.0, (now_us - last_frame_host_us) / 1e6)
        thread_alive = index < len(camera_inputs.threads) and camera_inputs.threads[index].is_alive()
        required = {"rgb", "depth"}
        camera_ok = (
            ready
            and not failed
            and thread_alive
            and required.issubset(streams)
            and age_s is not None
            and age_s <= 2.5
        )
        all_ready = all_ready and camera_ok
        age_text = "never" if age_s is None else f"{age_s:.2f}s"
        detail = (
            f"cam{runtime.cid} mxid={runtime.mxid} ready={ready} "
            f"thread={'alive' if thread_alive else 'EXITED'} streams={streams or 'none'} "
            f"age={age_text}"
        )
        if failed or error:
            detail += f" error={error or 'unknown'}"
        details.append(detail)
    return all_ready, details
def colorize_depth(depth_mm: np.ndarray | None, min_mm: float = 250.0, max_mm: float = 8000.0) -> np.ndarray:
    if depth_mm is None:
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    depth = depth_mm.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    clipped = np.clip(depth, min_mm, max_mm)
    visual = ((max_mm - clipped) * (255.0 / max(1.0, max_mm - min_mm))).astype(np.uint8)
    visual[~valid] = 0
    colored = cv2.applyColorMap(visual, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def direct_aligned_detection_depth_m(depth_mm: np.ndarray | None, detection: Detection, rgb_shape) -> float | None:
    """Median aligned depth inside a detection bbox for single-camera mode."""
    if depth_mm is None:
        return None
    rgb_h, rgb_w = rgb_shape[:2]
    depth_h, depth_w = depth_mm.shape[:2]
    x1, y1, x2, y2 = detection.bbox
    x1 = int(clamp(x1 * depth_w / max(1, rgb_w), 0, depth_w - 1))
    x2 = int(clamp(x2 * depth_w / max(1, rgb_w), 0, depth_w))
    y1 = int(clamp(y1 * depth_h / max(1, rgb_h), 0, depth_h - 1))
    y2 = int(clamp(y2 * depth_h / max(1, rgb_h), 0, depth_h))
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_mm[y1:y2, x1:x2].astype(np.float32)
    values = region[np.isfinite(region) & (region > 0)]
    if values.size == 0:
        return None
    return float(np.median(values)) / 1000.0


def labeled_tile(image: np.ndarray, label: str, size: tuple[int, int] = (520, 220)) -> np.ndarray:
    tile = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def compose_camera_rows(
    rgb_images: tuple[np.ndarray, np.ndarray, np.ndarray],
    depths: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    *,
    live_suffix: str = "",
) -> np.ndarray:
    """Build the same per-camera RGB | depth rows as oak_collage.py."""
    rows = []
    tile_size = (576, 360)
    for role, rgb, depth in zip(("LEFT", "CENTER", "RIGHT"), rgb_images, depths):
        suffix = f" - {live_suffix}" if live_suffix else ""
        rows.append(
            cv2.hconcat(
                [
                    labeled_tile(rgb, f"{role} - RGB{suffix}", tile_size),
                    labeled_tile(
                        colorize_depth(depth),
                        f"{role} - DEPTH 0.25-8m{suffix}",
                        tile_size,
                    ),
                ]
            )
        )
    return cv2.vconcat(rows)


def annotate_panorama(
    panorama: np.ndarray,
    tracks: list[ObjectTrack],
    distance_m: float | None,
    state: str,
    auto_enabled: bool,
    speed_mode: int,
    steering: float,
    throttle: float,
) -> np.ndarray:
    vis = panorama.copy()
    height, width = vis.shape[:2]
    center_x = width // 2
    dead = int(width * 0.04)
    cv2.line(vis, (center_x, 0), (center_x, height - 1), (255, 255, 0), 2)
    cv2.line(vis, (center_x - dead, 0), (center_x - dead, height - 1), (0, 180, 255), 1)
    cv2.line(vis, (center_x + dead, 0), (center_x + dead, height - 1), (0, 180, 255), 1)

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        # Pure-pursuit consumes every valid detection as a BEV obstacle. There
        # is no person target, lock, or selected object in autonomous mode.
        color = (0, 220, 255)
        thickness = 2
        marker = "OBSTACLE"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            vis,
            f"{marker} {track.label} id={track.track_id} {track.confidence:.2f} "
            f"w={track.bbox[2] - track.bbox[0]}px",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    distance_text = "--" if distance_m is None else f"{distance_m:.2f}m"
    armed_text = "ARMED" if auto_enabled else "DISARMED"
    hud = (
        f"AUTONOMOUS {armed_text}  state={state}  nearest={distance_text}  "
        f"objects={len(tracks)}  steer={steering:.1f}  throttle={throttle:.1f}"
    )
    cv2.rectangle(vis, (0, 0), (width, 32), (0, 0, 0), -1)
    cv2.putText(vis, hud, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)
    return vis


def compose_preview(
    rgb_images: tuple[np.ndarray, np.ndarray, np.ndarray],
    depths: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    annotated_pano: np.ndarray,
) -> np.ndarray:
    camera_rows = compose_camera_rows(rgb_images, depths)
    pano_width = camera_rows.shape[1]
    pano_height = max(
        1, int(round(annotated_pano.shape[0] * pano_width / annotated_pano.shape[1]))
    )
    pano = labeled_tile(
        annotated_pano,
        "AUTONOMOUS PERCEPTION: DETECTIONS + ROAD CORRIDOR",
        (pano_width, pano_height),
    )
    preview = cv2.vconcat([camera_rows, pano])
    cv2.rectangle(preview, (0, preview.shape[0] - 28), (preview.shape[1], preview.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        preview,
        "PURE PURSUIT + BEV OBSTACLE AVOIDANCE | safety stop: S/E/Space | Q quit",
        (10, preview.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if preview.shape[0] > 900:
        scale = 900.0 / preview.shape[0]
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return preview


def compose_single_preview(
    rgb_image: np.ndarray,
    depth: np.ndarray | None,
    annotated: np.ndarray,
) -> np.ndarray:
    def letterbox(image: np.ndarray, width: int, height: int, label: str) -> np.ndarray:
        scale = min(width / image.shape[1], height / image.shape[0])
        resized_width = max(1, int(round(image.shape[1] * scale)))
        resized_height = max(1, int(round(image.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x = (width - resized_width) // 2
        y = (height - resized_height) // 2
        canvas[y : y + resized_height, x : x + resized_width] = resized
        cv2.rectangle(canvas, (0, 0), (width, 28), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            label,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    # 640x400 source aspect ratio is preserved in every panel. The annotated
    # image is centered with black side bars instead of stretched to 1280 wide.
    rgb_tile = letterbox(rgb_image, 640, 400, "CENTER OAK RGB (FORWARD VIEW)")
    depth_tile = letterbox(colorize_depth(depth), 640, 400, "ALIGNED DEPTH")
    top = cv2.hconcat([rgb_tile, depth_tile])
    pano = letterbox(annotated, top.shape[1], 400, "ANNOTATED RGB - ASPECT RATIO PRESERVED")
    preview = cv2.vconcat([top, pano])
    cv2.rectangle(preview, (0, preview.shape[0] - 28), (preview.shape[1], preview.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        preview,
        "SINGLE OAK AUTONOMOUS | PURE PURSUIT + BEV AVOIDANCE | S stop | Q quit",
        (10, preview.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if preview.shape[0] > 900:
        scale = 900.0 / preview.shape[0]
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return preview


def compose_stitching_preview(
    rgb_images: tuple[np.ndarray, np.ndarray, np.ndarray],
    depths: tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None],
    status: str,
) -> np.ndarray:
    """Show raw feeds in collage order while panorama calibration warms up."""
    camera_rows = compose_camera_rows(rgb_images, depths, live_suffix="LIVE")
    banner = np.full((96, camera_rows.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        banner,
        "CAMERAS READY - WAITING FOR PANORAMA; CONTROL DISARMED",
        (12, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        status or "SIFT status unavailable",
        (12, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        "Q/Esc quit",
        (12, 91),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    preview = cv2.vconcat([camera_rows, banner])
    if preview.shape[0] > 900:
        scale = 900.0 / preview.shape[0]
        preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return preview


def state_phrase(state: str, distance_m: float | None) -> str:
    if state == "STOP":
        return "Braking. Person is within one meter."
    if state == "SLOW":
        return "Person is within two meters. Slowing down."
    if state == "FOLLOW":
        return "Following person."
    if state == "TURN_LEFT":
        return "Target is far left. Turning left."
    if state == "TURN_RIGHT":
        return "Target is far right. Turning right."
    if state == "NO_DEPTH":
        return "Depth unavailable. Stopping."
    if state == "LOST":
        return "Target lost. Stopping."
    if state == "CAMERA_FAIL":
        return "Camera feed unavailable. Stopping."
    if state == "BBOX_SLOW":
        return "Person box is three hundred pixels or wider. Slowing down."
    if state == "BBOX_FOLLOW":
        return "Bounding box following active."
    return state.replace("_", " ").title()


def haptic_phrase(level: str, reason: str) -> str:
    if level == "low":
        return "Haptic low. Slow down."
    if level == "mid":
        return "Haptic mid. Brake slowly."
    if level == "high":
        return "Haptic high. Slow down and brake immediately."
    return f"Haptic clear. {reason}"


def write_command_record(handle, record: dict) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    handle.flush()


def main() -> int:
    args = parse_args()
    try:
        depthai_major = int(str(getattr(dai, "__version__", "0")).split(".", 1)[0])
    except ValueError:
        depthai_major = 0
    if depthai_major < 3:
        print(
            f"DepthAI 3.x is required by the three-camera reference pipeline; "
            f"this interpreter has DepthAI {getattr(dai, '__version__', 'unknown')}."
        )
        print("Run with ./run_autonomous.sh.")
        return 1
    engine_paths = (args.detect_engine, args.segmentation_engine, args.depth_engine)
    missing_engines = [path for path in engine_paths if not path.is_file()]
    if missing_engines:
        print(f"Model not found: {missing_engines[0]}")
        return 1
    if not args.no_route and not args.route_file.is_file():
        print(f"Route file not found: {args.route_file}")
        return 1
    gui_enabled = bool(os.environ.get("DISPLAY")) and not args.no_imshow
    announcer = Announcer(not args.no_tts)
    controller = None
    camera_inputs = None
    perception = None
    gps_reader = None
    route_follower = None
    latest_fix = None
    latest_route_goal = None
    latest_semantic = None
    auto_enabled = False
    auto_stop_latched = False
    speed_mode = args.speed_mode
    latest_tracks: list[ObjectTrack] = []
    filtered_distance = None
    depth_history: deque[float] = deque(maxlen=5)
    last_announced_state = ""
    last_policy_state = "IDLE"
    last_haptic_level = "none"
    last_control_error = ""
    latest_normalized_steering = 0.0
    latest_road_visible = False

    policy = DistancePolicy(args.stop_m, args.stop_release_m, args.slow_m, args.full_m)
    pure_config = PurePursuitConfig(
        wheelbase_m=WHEELBASE_M,
        max_wheel_angle_deg=MAX_WHEEL_ANGLE_DEG,
        lookahead_m=args.pure_lookahead_m,
        gps_lateral_weight=args.gps_lateral_weight,
        gps_lateral_limit_m=args.gps_lateral_limit_m,
        normal_steer_limit=args.pure_normal_limit,
        low_speed_steer_limit=args.pure_low_speed_limit,
        high_speed_steer_limit=args.pure_high_speed_limit,
        road_half_width_m=args.road_half_width_m,
        road_edge_margin_m=args.road_edge_margin_m,
    )
    pure_state = PurePursuitState()
    command_log = None
    dashboard = None
    latest_pure_command = None
    latest_model_timings = {}

    try:
        if args.command_log:
            args.command_log.parent.mkdir(parents=True, exist_ok=True)
            command_log = args.command_log.open("a", encoding="utf-8")
        controller = CarController(args.dry_run)
        print("[control] throttle neutral, steering centered")
        print("[control] auto-arm requested; waiting for safety inputs" if args.auto_arm else "[control] AUTO STARTS DISARMED; keep wheels raised")
        announcer.warmup()
        if args.web_dashboard:
            dashboard = LiveDashboard(args.web_host, args.web_port, args.web_jpeg_quality)
            dashboard.start()
            print(f"[dashboard] http://<jetson-ip>:{args.web_port}")

        print(f"[models] detection={args.detect_engine}")
        print(f"[models] segmentation={args.segmentation_engine}")
        print(f"[models] depth={args.depth_engine}")
        perception = ParallelPerception(args.detect_engine, args.segmentation_engine, args.depth_engine)
        warmup = np.zeros((416, 1120, 3), dtype=np.uint8)
        for warmup_index in range(max(0, args.warmup_runs)):
            perception.infer(warmup, args.conf, args.iou)
            print(f"[warmup] {warmup_index + 1}/{max(0, args.warmup_runs)}", flush=True)
        if not args.no_route:
            route_follower = RouteFollower(args.route_file, args.route_name, args.route_lookahead_m)
            gps_reader = VectorNavReader(args.vn_port, args.vn_baud)
            gps_reader.start()
            print(f"[route] {route_follower.name} lookahead={args.route_lookahead_m:.1f}m")
        if args.single_oak:
            print("[oak] opening one RGB + aligned-depth pipeline; stitching disabled")
        else:
            print("[oak] opening all three RGB + aligned-depth pipelines")
        camera_inputs = get_camera_inputs(
            fps=args.fps,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            auto_cleanup=False,
            single=args.single_oak,
        )

        print("[oak] waiting for first RGB + depth frame...")
        startup_deadline = time.monotonic() + 25.0
        next_health_log = 0.0
        startup_ready = False
        startup_details = []
        while time.monotonic() < startup_deadline:
            startup_ready, startup_details = camera_health(camera_inputs)
            if startup_ready:
                break
            if time.monotonic() >= next_health_log:
                for detail in startup_details:
                    print(f"[oak] {detail}")
                next_health_log = time.monotonic() + 2.0
            if any("thread=EXITED" in detail for detail in startup_details):
                break
            time.sleep(0.1)
        if not startup_ready:
            detail_text = "\n".join(f"  {detail}" for detail in startup_details)
            raise RuntimeError(
                "Timed out waiting for all OAK RGB/depth feeds. Camera diagnostics:\n"
                + detail_text
            )
        announcer.say(
            "Single camera connected. Object tracker ready."
            if args.single_oak
            else "Three cameras connected. Autonomous perception ready.",
            force=True,
        )

        stitch_state = create_stitch_state()
        depth_provider = FusedStereoDepthProvider(cam_hw=(CAMERA_HEIGHT, CAMERA_WIDTH))
        # The fixed rig uses final_data_capture/sift_stitch_config.json.
        recompute_sift = False
        if args.single_oak:
            print("[stitch] disabled in single-OAK mode")
        else:
            print(f"[stitch] using validated {stitch_state.config_path}")

        frame_index = 0
        last_detect_ts = -1
        last_pano = None
        last_preview_time = 0.0
        last_camera_health_log = 0.0
        last_stitch_status_log = 0.0
        last_bbox_log = 0.0
        latest_distance = None

        if gui_enabled:
            cv2.namedWindow("Trike Autonomous Navigation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Trike Autonomous Navigation", 1600, 900)
        else:
            print("[gui] no preview window; use terminal keys")

        with KeyPoller() as terminal_keys:
            while True:
                now = time.monotonic()
                c1 = camera_inputs.camera1_image()  # logical center
                c2 = camera_inputs.camera2_image()  # logical right
                c3 = camera_inputs.camera3_image()  # logical left
                d1 = camera_inputs.camera1_depth()
                d2 = camera_inputs.camera2_depth()
                d3 = camera_inputs.camera3_depth()

                if args.single_oak:
                    # Downstream preview variables remain defined, but only c1/d1
                    # are processed. No duplicate images enter stitching.
                    c2 = c1
                    c3 = c1
                    d2 = d1
                    d3 = d1

                frames_ready = c1 is not None if args.single_oak else (
                    c1 is not None and c2 is not None and c3 is not None
                )
                cameras_live = valid_camera_frames(camera_inputs)
                if not frames_ready or not cameras_live:
                    if auto_enabled:
                        auto_enabled = False
                        controller.disarm()
                    policy.state = "CAMERA_FAIL"
                    if last_announced_state != policy.state:
                        announcer.say(state_phrase(policy.state, None))
                        last_announced_state = policy.state
                    if now - last_camera_health_log >= 2.0:
                        _healthy, health_details = camera_health(camera_inputs)
                        for detail in health_details:
                            print(f"[oak] {detail}")
                        last_camera_health_log = now
                    time.sleep(0.01)
                    key = terminal_keys.poll()
                    if key in ("q", "Q", "\x1b"):
                        break
                    continue

                if args.single_oak:
                    panorama = c1.image
                else:
                    panorama = stitch_sift(
                        c3.image,
                        c1.image,
                        c2.image,
                        c1.ts_us,
                        stitch_state,
                        recompute=recompute_sift,
                    )
                    recompute_sift = False
                if panorama is None:
                    controller.disarm()
                    auto_enabled = False
                    stitch_status = stitch_state.pano_state.status
                    if now - last_stitch_status_log >= 2.0:
                        print(f"[stitch] {stitch_status}")
                        last_stitch_status_log = now
                    key_code = -1
                    if gui_enabled and now - last_preview_time >= 1.0 / max(1.0, args.preview_fps):
                        preview = compose_stitching_preview(
                            (c3.image, c1.image, c2.image),
                            (
                                d3.image if d3 is not None else None,
                                d1.image if d1 is not None else None,
                                d2.image if d2 is not None else None,
                            ),
                            stitch_status,
                        )
                        cv2.imshow("Trike Autonomous Navigation", preview)
                        last_preview_time = now
                    if gui_enabled:
                        key_code = cv2.waitKey(1) & 0xFF
                    terminal_key = terminal_keys.poll()
                    key = terminal_key if terminal_key is not None else (
                        chr(key_code) if 0 <= key_code < 128 else None
                    )
                    if key in ("q", "Q", "\x1b") or key_code == 27:
                        break
                    time.sleep(0.005)
                    continue
                last_pano = panorama

                if not args.single_oak:
                    stitched_oak_depth = stitch_depth_sift(
                        d3.image if d3 is not None else None,
                        d1.image if d1 is not None else None,
                        d2.image if d2 is not None else None,
                        stitch_state,
                    )
                    depth_provider.set_stitched_depth(stitched_oak_depth)

                should_detect = (
                    c1.ts_us != last_detect_ts
                    and (frame_index % max(1, args.detect_interval) == 0 or not latest_tracks)
                )
                if should_detect:
                    outputs = perception.infer(panorama, args.conf, args.iou)
                    latest_model_timings = dict(outputs.timings_s)
                    result = outputs.detection
                    panorama_valid = valid_panorama_mask(panorama)
                    depth_provider.set_valid_panorama_mask(panorama_valid)
                    depth_provider.set_model_depth(outputs.depth)
                    depth_provider.set_segmentation(outputs.segmentation)
                    latest_semantic = semantic_array(outputs.segmentation, panorama.shape[:2])
                    if latest_semantic is not None:
                        latest_semantic = mask_semantic_to_valid_frame(latest_semantic, panorama)
                    latest_tracks = parse_object_tracks(result)
                    latest_tracks = [
                        track for track in latest_tracks
                        if detection_in_valid_panorama(track.bbox, panorama_valid)
                    ]
                    last_detect_ts = c1.ts_us
                frame_index += 1

                route_goal_forward_m = None
                route_goal_left_m = 0.0
                route_ready = True
                if route_follower is not None and args.planner_mode == "pure-pursuit":
                    latest_fix = gps_reader.snapshot(args.gps_max_age) if gps_reader else None
                    latest_route_goal = (
                        route_follower.target(latest_fix.lat, latest_fix.lon)
                        if latest_fix is not None else None
                    )
                    if latest_route_goal is not None and latest_route_goal["complete"]:
                        route_ready = False
                        if auto_enabled:
                            auto_enabled = False
                            controller.disarm()
                    elif latest_route_goal is not None:
                        route_goal_forward_m, route_goal_left_m = route_goal_local(
                            latest_fix, latest_route_goal
                        )
                    # Missing/stale GPS leaves a zero lateral hint. Vision road
                    # centering and obstacle avoidance continue independently.

                if args.planner_mode == "pure-pursuit":
                    depth_callable = None
                    if args.single_oak:
                        depth_frame = d1.image if d1 is not None else None

                        def depth_callable(det, depth_frame=depth_frame, shape=panorama.shape):
                            return direct_aligned_detection_depth_m(depth_frame, det, shape)
                    elif depth_provider.ready:
                        depth_callable = depth_provider

                    obstacles = []
                    if depth_callable is not None:
                        obstacles = detections_to_obstacles(
                            [track.as_detection(inset=True) for track in latest_tracks],
                            depth_callable,
                            config=pure_config,
                            pano_width=panorama.shape[1],
                        )

                    semantic_bounds = None
                    dense_depth = depth_provider.dense_depth() if not args.single_oak else None
                    if latest_semantic is not None and dense_depth is not None:
                        road_features = road_boundaries(latest_semantic, dense_depth)
                        semantic_bounds = corridor_bounds(road_features, pure_config.lookahead_m)
                    latest_road_visible = semantic_bounds is not None
                    speed_mps = max(0.0, float(latest_fix.speed_mps if latest_fix is not None and latest_fix.speed_mps is not None else args.ego_speed_mps))
                    pure_command = plan_pure_pursuit(
                        obstacles,
                        speed_mps,
                        now,
                        pure_state,
                        pure_config,
                        route_goal_forward_m=route_goal_forward_m,
                        road_bounds_left_m=semantic_bounds,
                        route_goal_left_m=route_goal_left_m,
                    )
                    latest_pure_command = pure_command
                    latest_normalized_steering = pure_command.steering
                    latest_distance = pure_command.nearest_forward_m
                    steering_target = normalized_steering_to_servo_angle(
                        pure_command.steering,
                        args.invert_steering,
                    )
                    policy.state = (
                        "PURE_CLEAR"
                        if pure_command.haptic == "none"
                        else f"HAPTIC_{pure_command.haptic.upper()}"
                    )

                    if args.auto_arm:
                        depth_ready = args.single_oak or depth_provider.ready
                        auto_ready = route_ready and depth_ready and latest_road_visible
                        if auto_ready and not auto_stop_latched and not auto_enabled:
                            auto_enabled = True
                            controller.arm()
                            print("[control] AUTO ARMED: route/depth/road ready", flush=True)
                        elif (not auto_ready or auto_stop_latched) and auto_enabled:
                            auto_enabled = False
                            controller.disarm()
                            print("[control] AUTO CENTERED: safety input unavailable", flush=True)

                    if auto_enabled:
                        controller.command(steering_target, THROTTLE_NEUTRAL)
                    else:
                        controller.disarm()

                    if pure_command.haptic != last_haptic_level:
                        announcer.say(haptic_phrase(pure_command.haptic, pure_command.reason), force=True)
                        last_haptic_level = pure_command.haptic

                    if now - last_bbox_log >= 0.25:
                        nearest_text = (
                            "--"
                            if pure_command.nearest_forward_m is None
                            else f"{pure_command.nearest_forward_m:.2f}m"
                        )
                        print(
                            f"[pure] steer={pure_command.steering:+.3f} "
                            f"raw={pure_command.raw_steering:+.3f} "
                            f"haptic={pure_command.haptic} nearest={nearest_text} "
                            f"objects={pure_command.obstacle_count} "
                            f"road_clamped={pure_command.road_clamped} "
                            f"reason={pure_command.reason}"
                        )
                        if not args.single_oak:
                            print(
                                f"[fusion] a={depth_provider.scale:.3f} "
                                f"b={depth_provider.offset:+.2f} "
                                f"oak_pixels={depth_provider.calibration_pixels}"
                            )
                        if latest_route_goal is not None and route_goal_forward_m is not None:
                            print(
                                f"[route] name={latest_route_goal['route']} "
                                f"index={latest_route_goal['route_index']} "
                                f"goal=({route_goal_forward_m:.2f}m forward, "
                                f"{route_goal_left_m:+.2f}m left)"
                            )
                        last_bbox_log = now

                    write_command_record(
                        command_log,
                        {
                            "ts": time.time(),
                            "mode": "pure-pursuit",
                            "armed": bool(auto_enabled),
                            "steering": pure_command.steering,
                            "haptic": pure_command.haptic,
                            "reason": pure_command.reason,
                            "goal_forward_m": pure_command.goal_forward_m,
                            "goal_left_m": pure_command.goal_left_m,
                            "nearest_forward_m": pure_command.nearest_forward_m,
                            "road_clamped": pure_command.road_clamped,
                            "road_visible": pure_command.road_visible,
                            "vision_center_left_m": pure_command.vision_center_left_m,
                            "gps_hint_left_m": pure_command.gps_hint_left_m,
                            "path_blocked": pure_command.path_blocked,
                            "road_half_width_m": pure_config.road_half_width_m,
                            "road_edge_margin_m": pure_config.road_edge_margin_m,
                            "ego_speed_mps": speed_mps,
                            "objects": pure_command.obstacle_count,
                        },
                    )

                armed, current_steering, current_throttle, control_error = controller.snapshot()
                if control_error:
                    auto_enabled = False
                    auto_stop_latched = True
                    controller.disarm()
                    if control_error != last_control_error:
                        print(f"[esp] FAULT: {control_error}; centered and auto-arm latched", flush=True)
                        last_control_error = control_error

                key_code = -1
                if (gui_enabled or dashboard is not None) and now - last_preview_time >= 1.0 / max(1.0, args.preview_fps):
                    display_panorama = (
                        overlay_road(panorama, latest_semantic)
                        if latest_semantic is not None else panorama
                    )
                    annotated = annotate_panorama(
                        display_panorama,
                        latest_tracks,
                        latest_distance,
                        policy.state,
                        auto_enabled and armed,
                        speed_mode,
                        current_steering,
                        current_throttle,
                    )
                    if args.single_oak:
                        preview = compose_single_preview(
                            c1.image,
                            d1.image if d1 is not None else None,
                            annotated,
                        )
                    else:
                        preview = compose_preview(
                            (c3.image, c1.image, c2.image),
                            (
                                d3.image if d3 is not None else None,
                                d1.image if d1 is not None else None,
                                d2.image if d2 is not None else None,
                            ),
                            annotated,
                        )
                    if gui_enabled:
                        cv2.imshow("Trike Autonomous Navigation", preview)
                    if dashboard is not None:
                        command = latest_pure_command
                        direction = "CENTER" if abs(latest_normalized_steering) < 0.025 else ("RIGHT" if latest_normalized_steering > 0 else "LEFT")
                        dashboard.publish(preview, {
                            "state": policy.state, "planner": args.planner_mode,
                            "auto_enabled": bool(auto_enabled), "controller_armed": bool(armed),
                            "controller_error": control_error or None,
                            "steering_normalized": latest_normalized_steering,
                            "steering_direction": direction, "steering_target": current_steering,
                            "throttle": current_throttle,
                            "haptic": getattr(command, "haptic", "none"),
                            "reason": getattr(command, "reason", "waiting"),
                            "road_visible": bool(getattr(command, "road_visible", latest_road_visible)),
                            "path_blocked": bool(getattr(command, "path_blocked", False)),
                            "obstacle_count": int(getattr(command, "obstacle_count", len(latest_tracks))),
                            "nearest_forward_m": getattr(command, "nearest_forward_m", None),
                            "goal_forward_m": getattr(command, "goal_forward_m", None),
                            "goal_left_m": getattr(command, "goal_left_m", None),
                            "route": None if latest_route_goal is None else latest_route_goal.get("route"),
                            "gps_valid": latest_fix is not None,
                            "latitude": None if latest_fix is None else latest_fix.lat,
                            "longitude": None if latest_fix is None else latest_fix.lon,
                            "timings_s": latest_model_timings,
                            "objects": [{"label": track.label, "confidence": track.confidence, "track_id": track.track_id} for track in latest_tracks],
                        })
                    last_preview_time = now
                if gui_enabled:
                    key_code = cv2.waitKey(1) & 0xFF
                terminal_key = terminal_keys.poll()
                key = terminal_key if terminal_key is not None else (
                    chr(key_code) if 0 <= key_code < 128 else None
                )

                if key in ("q", "Q", "\x1b") or key_code == 27:
                    break
                if key in ("s", "S", "e", "E", " "):
                    auto_enabled = False
                    auto_stop_latched = True
                    controller.disarm()
                    announcer.say("Full stop. Automatic control disarmed. Throttle neutral.", force=True)
                elif key in ("a", "A"):
                    auto_stop_latched = False
                    depth_ready = args.single_oak or depth_provider.ready
                    if not route_ready:
                        auto_enabled = False
                        controller.disarm()
                        announcer.say("Cannot arm. Route is complete.", force=True)
                    elif not depth_ready:
                        auto_enabled = False
                        controller.disarm()
                        announcer.say("Cannot arm. Wait for BEV depth mapping.", force=True)
                    elif not latest_road_visible:
                        auto_enabled = False
                        controller.disarm()
                        announcer.say("Cannot arm. Wait for a visible road corridor.", force=True)
                    else:
                        auto_enabled = True
                        controller.arm()
                        announcer.say("Pure pursuit steering armed.", force=True)
                elif key in tuple(str(mode) for mode in SPEED_MODES):
                    speed_mode = int(key)
                    low_mph, high_mph, low_pwm, high_pwm = SPEED_MODES[speed_mode]
                    announcer.say(
                        f"Speed mode {speed_mode}. Simulated {low_mph:g} to {high_mph:g} miles per hour.",
                        force=True,
                    )
                    print(
                        f"[mode] {speed_mode}: simulated {low_mph:g}-{high_mph:g} mph, "
                        f"PWM {low_pwm:g}-{high_pwm:g}"
                    )

                if not gui_enabled:
                    distance_text = "--" if latest_distance is None else f"{latest_distance:.2f}m"
                    steer_text = (
                        f"norm={latest_normalized_steering:+.2f}"
                        if args.planner_mode == "pure-pursuit"
                        else f"steer={current_steering:5.1f}"
                    )
                    print(
                        f"\rstate={policy.state:16s} objects={len(latest_tracks):>3d} "
                        f"nearest={distance_text:>7s} mode={speed_mode} "
                        f"{steer_text} throttle={current_throttle:5.1f}",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[fatal] {type(exc).__name__}: {exc}")
        announcer.say("System fault. Throttle neutral.", force=True)
        return 1
    finally:
        if controller is not None:
            controller.disarm()
        if camera_inputs is not None:
            camera_inputs.stop()
        if controller is not None:
            controller.close()
        if gps_reader is not None:
            gps_reader.close()
        if perception is not None:
            perception.close()
        if command_log is not None:
            command_log.close()
        if dashboard is not None:
            dashboard.close()
        cv2.destroyAllWindows()
        print("\n[shutdown] throttle neutral, steering centered, cameras closed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
