#!/usr/bin/env python3
"""Offline three-model replay with a recorded-GPS point 5 m ahead."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from bev import (
    BevConfig,
    PurePursuitConfig,
    PurePursuitState,
    detections_to_obstacles,
    plan_pure_pursuit,
)
from image_processing import create_stitch_state, stitch_depth_sift, stitch_sift
from live_dashboard import LiveDashboard
from esp_controller import normalized_to_percent
from esp_trike_adapter import TrikeEspController
from models import Detection
from parallel_perception import FusedStereoDepthProvider, ParallelPerception, depth_array, semantic_array
from route_following import bearing_deg, haversine_m
from trike_geometry import MAX_WHEEL_ANGLE_DEG, WHEELBASE_M
from road_bev import (
    corridor_bounds,
    detection_in_valid_panorama,
    draw_boundaries,
    mask_semantic_to_valid_frame,
    overlay_road,
    road_boundaries,
    valid_frame_bounds,
    valid_panorama_mask,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("/home/h2x/Documents/data/jul17_r2")
DEFAULT_ENGINES = ROOT / "models"
DEFAULT_ESP_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58A6082118-if00"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "jul17_r2_modular_5m.mp4")
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Save per-frame semantic/depth arrays as compressed NPZ files",
    )
    parser.add_argument(
        "--waypoint-csv", type=Path,
        help="Plot-friendly waypoint/controller CSV (default: <output>_waypoints.csv)",
    )
    parser.add_argument("--lookahead-m", type=float, default=5.0)
    parser.add_argument("--gps-lateral-weight", type=float, default=0.15, help="Bounded GPS influence on the vision road center; 0 is fully vision-only")
    parser.add_argument("--gps-lateral-limit-m", type=float, default=0.50, help="Maximum GPS lateral error considered by the local planner")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--show", action="store_true", help="Display the stitched perception/BEV replay while it runs")
    parser.add_argument("--web-dashboard", action="store_true", help="Serve the replay visualization and telemetry over HTTP")
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8766)
    parser.add_argument("--web-jpeg-quality", type=int, default=80)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--esp-invert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Extra sign inversion after planner-to-curvature conversion (normally disabled)",
    )
    parser.add_argument("--enable-esp", action="store_true", help="Send replay steering to the physical ESP actuator")
    parser.add_argument("--esp-port", default=DEFAULT_ESP_PORT)
    parser.add_argument("--esp-baud", type=int, default=921600)
    parser.add_argument("--esp-rate-hz", type=float, default=50.0)
    parser.add_argument("--esp-speed-bound-mps", type=float, default=2.0)
    parser.add_argument("--esp-startup-seconds", type=float, default=2.5)
    parser.add_argument("--detect-engine", type=Path, default=DEFAULT_ENGINES / "yolo26s_q32.engine")
    parser.add_argument("--segmentation-engine", type=Path, default=DEFAULT_ENGINES / "yolo26s-seg_q32.engine")
    parser.add_argument("--depth-engine", type=Path, default=DEFAULT_ENGINES / "yolo26s-depth_q32.engine")
    args = parser.parse_args()
    if min(args.lookahead_m, args.stride, args.video_fps, args.esp_rate_hz) <= 0 or args.max_frames < 0:
        parser.error("lookahead, stride, video-fps, and ESP rate must be positive; max-frames must be nonnegative")
    if args.esp_startup_seconds < 0:
        parser.error("--esp-startup-seconds must be nonnegative")
    if not 0.0 <= args.gps_lateral_weight <= 1.0 or args.gps_lateral_limit_m < 0.0:
        parser.error("GPS lateral weight must be in [0,1] and limit must be nonnegative")
    return args


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise ValueError(f"No frame number in {path.name}")
    return int(match.group(1))


def raw_frame_indices(dataset: Path, gps: dict[int, dict[str, float]]) -> list[int]:
    camera_indices = []
    for camera in ("cam0", "cam1", "cam2"):
        paths = (dataset / camera / "rgb").glob("*.jpg")
        camera_indices.append({frame_number(path) for path in paths})
    if not camera_indices or any(not values for values in camera_indices):
        return []
    return sorted(set(gps).intersection(*camera_indices))


def raw_panorama(dataset: Path, index: int, stitch_state, timing: dict | None = None):
    read_started = time.perf_counter()
    images = {}
    depths = {}
    for camera in ("cam0", "cam1", "cam2"):
        path = dataset / camera / "rgb" / f"{index:06d}.jpg"
        images[camera] = cv2.imread(str(path))
        if images[camera] is None:
            return None
        depth_path = dataset / camera / "depth" / f"{index:06d}.npy"
        if depth_path.is_file():
            depth = np.load(depth_path, allow_pickle=False)
            depths[camera] = depth if depth.ndim == 2 else None
        else:
            depths[camera] = None
    if timing is not None:
        timing["image_read_s"] = time.perf_counter() - read_started
    stitch_started = time.perf_counter()
    # final_data_capture stores cam0=left, cam1=center, cam2=right.
    panorama = stitch_sift(
        images["cam0"], images["cam1"], images["cam2"],
        center_ts_us=index, state=stitch_state,
    )
    if timing is not None:
        timing["panorama_stitch_s"] = time.perf_counter() - stitch_started
    stitched_depth = stitch_depth_sift(
        depths.get("cam0"), depths.get("cam1"), depths.get("cam2"), stitch_state
    )
    return (panorama, depths, stitched_depth) if panorama is not None else None


def load_gps(path: Path) -> dict[int, dict[str, float]]:
    records: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                index = int(row["frame_idx"])
                records[index] = {
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "yaw_deg": float(row["yaw_deg"]),
                    "speed_mps": float(row["ins_speed"]),
                    "vel_n_mps": float(row["ins_vel_n"]),
                    "vel_e_mps": float(row["ins_vel_e"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return records


def interpolate(a: tuple[float, float], b: tuple[float, float], fraction: float) -> tuple[float, float]:
    return a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction


def future_point(
    indices: list[int],
    gps: dict[int, dict[str, float]],
    position: int,
    lookahead_m: float,
) -> tuple[tuple[float, float], int, float]:
    current_row = gps[indices[position]]
    start = (current_row["lat"], current_row["lon"])
    remaining = lookahead_m
    for next_position in range(position + 1, len(indices)):
        row = gps[indices[next_position]]
        end = (row["lat"], row["lon"])
        segment = haversine_m(start, end)
        if segment >= remaining and segment > 1e-6:
            target = interpolate(start, end, remaining / segment)
            return target, indices[next_position], lookahead_m
        remaining -= segment
        start = end
    current = (current_row["lat"], current_row["lon"])
    return start, indices[-1], haversine_m(current, start)


def local_point(current: dict[str, float], target: tuple[float, float]) -> tuple[float, float]:
    origin = (current["lat"], current["lon"])
    distance = haversine_m(origin, target)
    # Prefer VectorNav velocity course; raw yaw has a mounting offset here.
    if math.hypot(current["vel_n_mps"], current["vel_e_mps"]) > 0.15:
        heading = math.degrees(math.atan2(current["vel_e_mps"], current["vel_n_mps"])) % 360.0
    else:
        heading = bearing_deg(origin, target)
    relative = math.radians((bearing_deg(origin, target) - heading + 180.0) % 360.0 - 180.0)
    return distance * math.cos(relative), -distance * math.sin(relative)


def detections(result, output_shape: tuple[int, int] | None = None) -> list[tuple[Detection, str]]:
    output = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return output
    names = getattr(result, "names", {})
    source_height, source_width = getattr(result, "orig_shape", (1, 1))
    if output_shape is None:
        target_height, target_width = source_height, source_width
    else:
        target_height, target_width = output_shape
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        label = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
        score = float(boxes.conf[index].item())
        x1, y1, x2, y2 = boxes.xyxy[index].tolist()
        bbox = (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )
        output.append((Detection(label, score, bbox), label))
    return output


def detection_records(result, output_shape: tuple[int, int]) -> list[dict]:
    """Preserve every raw detector box, class ID, confidence, and track ID."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return []
    names = getattr(result, "names", {})
    source_height, source_width = getattr(result, "orig_shape", output_shape)
    target_height, target_width = output_shape
    scale_x, scale_y = target_width / source_width, target_height / source_height
    track_ids = getattr(boxes, "id", None)
    records = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        x1, y1, x2, y2 = boxes.xyxy[index].tolist()
        records.append(
            {
                "class_id": class_id,
                "class_name": str(
                    names.get(class_id, class_id)
                    if isinstance(names, dict)
                    else names[class_id]
                ),
                "confidence": float(boxes.conf[index].item()),
                "track_id": (
                    int(track_ids[index].item()) if track_ids is not None else None
                ),
                "bbox_xyxy": [
                    float(x1 * scale_x),
                    float(y1 * scale_y),
                    float(x2 * scale_x),
                    float(y2 * scale_y),
                ],
            }
        )
    return records


def filter_detections_to_valid_frame(
    found: list[tuple[Detection, str]], image: np.ndarray
) -> list[tuple[Detection, str]]:
    x_min, y_min, x_max, y_max = valid_frame_bounds(image)
    valid = valid_panorama_mask(image)
    filtered = []
    for detection, label_name in found:
        x1, y1, x2, y2 = detection.bbox
        if not detection_in_valid_panorama(detection.bbox, valid):
            continue
        clipped = Detection(
            detection.label,
            detection.confidence,
            (
                max(x_min, x1), max(y_min, y1),
                min(x_max, x2), min(y_max, y2),
            ),
        )
        filtered.append((clipped, label_name))
    return filtered


def overlay_instances(image: np.ndarray, result) -> np.ndarray:
    value = getattr(getattr(result, "masks", None), "data", None)
    if value is None:
        return image.copy()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    masks = np.asarray(value)
    output = image.copy()
    palette = ((255, 80, 80), (80, 255, 100), (80, 160, 255), (220, 80, 255))
    for index, mask in enumerate(masks):
        resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
        color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
        output[resized] = (0.55 * output[resized] + 0.45 * color).astype(np.uint8)
    return output


def color_depth(depth: np.ndarray, near_m: float = 0.25, far_m: float = 20.0) -> np.ndarray:
    """Render metric depth with near=red and far=blue."""
    valid = np.isfinite(depth) & (depth > 0.25)
    clipped = np.clip(depth, near_m, far_m)
    gray = ((far_m - clipped) * (255.0 / (far_m - near_m))).astype(np.uint8)
    gray[~valid] = 0
    result = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    result[~valid] = 0
    return result


def bev_panel(
    obstacles, goal_forward: float, goal_left: float, steering: float,
    semantic: np.ndarray | None = None, depth: np.ndarray | None = None,
    preview_goal: tuple[float, float] | None = None,
    esp_percent: float = 50.0,
) -> np.ndarray:
    height, width = 416, 560
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    cx, bottom = width // 2, height - 28
    scale_x, scale_z = 24.0, 18.0
    for metres in range(0, 21, 5):
        y = int(bottom - metres * scale_z)
        cv2.line(canvas, (30, y), (width - 30, y), (60, 60, 60), 1)
        cv2.putText(canvas, f"{metres}m", (5, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    for lateral in range(-10, 11, 5):
        x = int(cx - lateral * scale_x)
        cv2.line(canvas, (x, 22), (x, bottom), (50, 50, 50), 1)
    cv2.rectangle(canvas, (cx - 4, bottom - 10), (cx + 4, bottom), (255, 255, 255), -1)
    cv2.putText(canvas, "trike", (cx + 8, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if semantic is not None and depth is not None:
        draw_boundaries(canvas, semantic, depth)
    for obstacle in obstacles:
        x = int(cx - (-obstacle.x_m) * scale_x)
        y = int(bottom - obstacle.z_m * scale_z)
        if 0 <= x < width and 20 <= y < bottom:
            cv2.circle(canvas, (x, y), 6, (0, 80, 255), -1)
            cv2.putText(canvas, obstacle.label, (x + 7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    if preview_goal is not None:
        preview_forward, preview_left = preview_goal
        px = int(cx - preview_left * scale_x)
        py = int(bottom - preview_forward * scale_z)
        cv2.drawMarker(canvas, (px, py), (255, 120, 210), cv2.MARKER_DIAMOND, 18, 3)
        cv2.putText(canvas, "GPS 10m", (px + 8, max(38, py - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 210), 1)
    gx = int(cx - goal_left * scale_x)
    gy = int(bottom - goal_forward * scale_z)
    cv2.line(canvas, (cx, bottom), (gx, gy), (0, 255, 255), 3)
    cv2.drawMarker(canvas, (gx, gy), (0, 255, 255), cv2.MARKER_DIAMOND, 16, 2)
    cv2.putText(canvas, "PURE 5m", (gx + 8, max(38, gy - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(canvas, f"ESP {esp_percent:.1f}% {esp_direction(esp_percent)}", (12, height - 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.putText(canvas, f"pure pursuit steer {steering:+.3f}", (12, height - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return canvas


def esp_direction(percent: float, center_deadband: float = 2.5) -> str:
    if percent > 50.0 + center_deadband:
        return "LEFT"
    if percent < 50.0 - center_deadband:
        return "RIGHT"
    return "CENTER"


def label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> int:
    args = parse_args()
    gps = load_gps(args.dataset / "vn_gps.csv")
    stitched_frames = sorted(
        (args.dataset / "stitched" / "rgb").glob("*.jpg"), key=frame_number
    )
    legacy_frames = sorted(
        (args.dataset / "frames").glob("sift_equirectangular_*.png"),
        key=frame_number,
    )
    stitched_mode = bool(stitched_frames)
    frames = stitched_frames if stitched_mode else legacy_frames
    frames = [path for path in frames if frame_number(path) in gps]
    raw_indices = raw_frame_indices(args.dataset, gps) if not frames else []
    raw_mode = bool(raw_indices)
    stitch_state = None
    if raw_mode:
        stitch_state = create_stitch_state()
        # Lock the fixed-rig homographies once, then reuse them for every frame.
        calibration_ready = False
        for calibration_index in raw_indices:
            stitch_state.pano_state.next_retry_monotonic = 0.0
            if raw_panorama(args.dataset, calibration_index, stitch_state) is not None:
                calibration_ready = True
                break
        if not calibration_ready:
            raise RuntimeError("Could not calibrate the three-camera SIFT panorama")
    gps_indices = sorted(gps)
    positions = {index: position for position, index in enumerate(gps_indices)}
    selected = (raw_indices if raw_mode else frames)[:: args.stride]
    if args.max_frames:
        selected = selected[: args.max_frames]
    if not selected:
        raise RuntimeError("No frame-aligned images and GPS rows found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.jsonl or args.output.with_suffix(".jsonl")
    waypoint_path = args.waypoint_csv or args.output.with_name(f"{args.output.stem}_waypoints.csv")
    waypoint_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_arrays = None
    if args.artifacts_dir is not None:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_arrays = args.artifacts_dir / "arrays"
        artifact_arrays.mkdir(parents=True, exist_ok=True)
        artifact_manifest = {
            "dataset": str(args.dataset.resolve()),
            "source_mode": (
                "stitched_rgb_depth"
                if stitched_mode
                else ("raw_three_camera" if raw_mode else "legacy_panorama")
            ),
            "model_input_shape": [416, 1120],
            "detect_engine": str(args.detect_engine),
            "segmentation_engine": str(args.segmentation_engine),
            "depth_engine": str(args.depth_engine),
            "array_files": "arrays/<frame>.npz",
            "arrays": {
                "semantic_class_id": "int16 [416,1120]",
                "yolo_depth_m": "float32 [416,1120]",
                "oak_depth_mm": "uint16 [416,1120], zero where invalid",
                "fused_depth_m": "float32 [416,1120]",
            },
            "frame_metadata": str(log_path),
        }
        (args.artifacts_dir / "manifest.json").write_text(
            json.dumps(artifact_manifest, indent=2) + "\n", encoding="utf-8"
        )
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (1120, 832))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {args.output}")
    models = ParallelPerception(args.detect_engine, args.segmentation_engine, args.depth_engine)
    provider = FusedStereoDepthProvider(cam_hw=(400, 640))
    if raw_mode:
        if provider.ready and provider.idx_slot.shape != (416, 1120):
            output_size = (1120, 416)
            provider.idx_slot = cv2.resize(
                provider.idx_slot, output_size, interpolation=cv2.INTER_NEAREST
            ).astype(np.int16)
            provider.idx_u = cv2.resize(
                provider.idx_u, output_size, interpolation=cv2.INTER_NEAREST
            ).astype(np.int16)
            provider.idx_v = cv2.resize(
                provider.idx_v, output_size, interpolation=cv2.INTER_NEAREST
            ).astype(np.int16)
    state = PurePursuitState()
    config = PurePursuitConfig(
        wheelbase_m=WHEELBASE_M,
        max_wheel_angle_deg=MAX_WHEEL_ANGLE_DEG,
        lookahead_m=args.lookahead_m,
        gps_lateral_weight=args.gps_lateral_weight,
        gps_lateral_limit_m=args.gps_lateral_limit_m,
    )
    started = time.monotonic()
    esp = None
    dashboard = None
    try:
        if args.web_dashboard:
            dashboard = LiveDashboard(args.web_host, args.web_port, args.web_jpeg_quality)
            dashboard.start()
            print(f"[dashboard] http://<jetson-ip>:{args.web_port}", flush=True)
        if args.enable_esp:
            esp = TrikeEspController(
                args.esp_port, args.esp_baud, args.esp_rate_hz,
                invert=args.esp_invert,
                speed_bound_mps=args.esp_speed_bound_mps,
            )
            ok, detail = esp.preflight()
            if not ok:
                raise RuntimeError(f"ESP preflight failed: {detail}")
            print(f"[esp] preflight passed: {detail}", flush=True)
            print(f"[esp] CENTER HOLD for {args.esp_startup_seconds:.1f}s", flush=True)
            time.sleep(args.esp_startup_seconds)
            esp.set_armed(True)
            print("[esp] replay steering ARMED; Ctrl-C centers and stops", flush=True)
        with log_path.open("w", encoding="utf-8") as log, waypoint_path.open(
            "w", newline="", encoding="utf-8"
        ) as waypoint_stream:
            waypoint_writer = csv.DictWriter(
                waypoint_stream,
                fieldnames=[
                    "frame", "current_lat", "current_lon", "target_frame",
                    "target_5m_lat", "target_5m_lon", "target_10m_lat", "target_10m_lon",
                    "target_5m_forward_m", "target_5m_left_m",
                    "target_10m_forward_m", "target_10m_left_m",
                    "planned_forward_m", "planned_left_m", "steering",
                    "vision_center_left_m", "gps_hint_left_m",
                    "road_visible", "path_blocked",
                    "esp_normalized", "esp_percent",
                    "road_left_bound_m", "road_right_bound_m", "road_clamped", "reason",
                    "wall_time_start_s", "wall_time_end_s", "image_read_ms",
                    "panorama_stitch_ms", "panorama_resize_ms",
                    "detection_model_ms", "segmentation_model_ms", "depth_model_ms",
                    "parallel_perception_ms", "parallel_perception_fps",
                    "perception_postprocess_ms", "obstacle_bev_ms", "route_target_ms",
                    "road_bev_ms", "planner_ms", "trajectory_conversion_ms",
                    "render_ms", "video_write_ms", "trajectory_total_ms", "trajectory_fps",
                ],
            )
            waypoint_writer.writeheader()
            for output_index, item in enumerate(selected):
                wall_time_start_s = time.time()
                frame_started = time.perf_counter()
                io_timing = {}
                if raw_mode:
                    index = int(item)
                    raw_result = raw_panorama(args.dataset, index, stitch_state, io_timing)
                    if raw_result is None:
                        continue
                    image, recorded_depths, stitched_oak_depth = raw_result
                else:
                    index = frame_number(item)
                    read_started = time.perf_counter()
                    image = cv2.imread(str(item))
                    recorded_depths = None
                    stitched_depth_path = (
                        args.dataset / "stitched" / "depth" / f"{index:06d}.npy"
                    )
                    stitched_oak_depth = (
                        np.load(stitched_depth_path, allow_pickle=False)
                        if stitched_mode and stitched_depth_path.is_file()
                        else None
                    )
                    io_timing["image_read_s"] = time.perf_counter() - read_started
                    io_timing["panorama_stitch_s"] = 0.0
                if image is None:
                    continue

                stage_started = time.perf_counter()
                panorama = cv2.resize(image, (1120, 416), interpolation=cv2.INTER_AREA)
                panorama_resize_s = time.perf_counter() - stage_started

                outputs = models.infer(panorama, args.conf, args.iou)

                stage_started = time.perf_counter()
                depth = depth_array(outputs.depth, (416, 1120))
                semantic = semantic_array(outputs.segmentation, (416, 1120))
                if semantic is not None:
                    semantic = mask_semantic_to_valid_frame(semantic, panorama)
                if semantic is None:
                    raise RuntimeError("Semantic engine did not return a dense class map")
                if depth is None:
                    raise RuntimeError("Depth engine did not return a dense depth map")
                if stitched_oak_depth is not None:
                    provider.set_stitched_depth(stitched_oak_depth)
                else:
                    provider.set_stitched_depth(None)
                provider.set_valid_panorama_mask(valid_panorama_mask(panorama))
                provider.set_model_depth(outputs.depth)
                provider.set_segmentation(outputs.segmentation)
                fused_depth = provider.dense_depth()
                if fused_depth is None:
                    fused_depth = depth
                oak_depth_m = provider._dense_oak_m()
                oak_depth_mm = (
                    np.clip(oak_depth_m * 1000.0, 0, 65535).astype(np.uint16)
                    if oak_depth_m is not None
                    else np.zeros((416, 1120), dtype=np.uint16)
                )
                artifact_path = None
                if artifact_arrays is not None:
                    artifact_path = artifact_arrays / f"{index:06d}.npz"
                    np.savez_compressed(
                        artifact_path,
                        semantic_class_id=semantic.astype(np.int16),
                        yolo_depth_m=depth.astype(np.float32),
                        oak_depth_mm=oak_depth_mm,
                        fused_depth_m=fused_depth.astype(np.float32),
                    )
                found = filter_detections_to_valid_frame(
                    detections(outputs.detection), panorama
                )
                perception_postprocess_s = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                obstacles = detections_to_obstacles(
                    [item[0] for item in found],
                    provider,
                    BevConfig(max_range_m=20.0),
                    panorama.shape[1],
                )
                obstacle_bev_s = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                current = gps[index]
                target, target_frame, actual_lookahead = future_point(
                    gps_indices, gps, positions[index], args.lookahead_m
                )
                goal_forward, goal_left = local_point(current, target)
                preview_target, _, _ = future_point(gps_indices, gps, positions[index], 10.0)
                preview_forward, preview_left = local_point(current, preview_target)
                route_target_s = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                road_features = road_boundaries(semantic, fused_depth)
                semantic_bounds = corridor_bounds(road_features, goal_forward)
                road_bev_s = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                command = plan_pure_pursuit(
                    obstacles,
                    current["speed_mps"],
                    time.monotonic(),
                    state,
                    config,
                    route_goal_forward_m=max(0.25, goal_forward),
                    route_goal_left_m=goal_left,
                    road_bounds_left_m=semantic_bounds,
                )
                planner_s = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                esp_percent = normalized_to_percent(command.steering, not args.esp_invert)
                esp_normalized = esp_percent / 50.0 - 1.0
                trajectory_conversion_s = time.perf_counter() - stage_started
                if esp is not None:
                    esp.update(command.steering)

                stage_started = time.perf_counter()
                annotated = overlay_road(panorama, semantic)
                for detection, name in found:
                    x1, y1, x2, y2 = detection.bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        annotated,
                        f"{name} {detection.confidence:.2f}",
                        (x1, max(40, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                    )
                annotated = label(
                    annotated,
                    f"frame {index} | GPS {current['lat']:.7f}, {current['lon']:.7f} | "
                    f"target frame {target_frame} ({actual_lookahead:.2f}m)",
                )
                depth_tile = cv2.resize(
                    label(
                        color_depth(fused_depth),
                        "FUSED DEPTH: OAK + YOLO | NEAR RED, FAR BLUE | 0.25-20m",
                    ),
                    (560, 416),
                )
                bev = bev_panel(
                    obstacles,
                    command.goal_forward_m,
                    command.goal_left_m,
                    command.steering,
                    semantic,
                    fused_depth,
                    (preview_forward, preview_left),
                    esp_percent=esp_percent,
                )
                video_frame = cv2.vconcat([annotated, cv2.hconcat([depth_tile, bev])])
                render_s = time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                writer.write(video_frame)
                video_write_s = time.perf_counter() - stage_started

                if args.show:
                    cv2.imshow("Modular recorded replay - Q/Esc stops", video_frame)
                    preview_key = cv2.waitKey(1) & 0xFF
                    if preview_key in (ord("q"), 27):
                        print("[preview] stop requested; centering ESP", flush=True)
                        break
                trajectory_total_s = time.perf_counter() - frame_started
                wall_time_end_s = time.time()
                parallel_s = outputs.timings_s["parallel_wall_s"]
                timing = {
                    "wall_time_start_s": wall_time_start_s,
                    "wall_time_end_s": wall_time_end_s,
                    "image_read_s": io_timing.get("image_read_s", 0.0),
                    "panorama_stitch_s": io_timing.get("panorama_stitch_s", 0.0),
                    "panorama_resize_s": panorama_resize_s,
                    **outputs.timings_s,
                    "parallel_perception_fps": 1.0 / max(parallel_s, 1e-9),
                    "perception_postprocess_s": perception_postprocess_s,
                    "obstacle_bev_s": obstacle_bev_s,
                    "route_target_s": route_target_s,
                    "road_bev_s": road_bev_s,
                    "planner_s": planner_s,
                    "trajectory_conversion_s": trajectory_conversion_s,
                    "render_s": render_s,
                    "video_write_s": video_write_s,
                    "trajectory_total_s": trajectory_total_s,
                    "trajectory_fps": 1.0 / max(trajectory_total_s, 1e-9),
                }
                record = {
                    "frame": index,
                    "current_gps": current,
                    "target_frame": target_frame,
                    "target_gps": {"lat": target[0], "lon": target[1]},
                    "preview_10m_gps": {"lat": preview_target[0], "lon": preview_target[1]},
                    "lookahead_m": actual_lookahead,
                    "goal_forward_m": goal_forward,
                    "goal_left_m": goal_left,
                    "preview_forward_m": preview_forward,
                    "preview_left_m": preview_left,
                    "planned_forward_m": command.goal_forward_m,
                    "planned_left_m": command.goal_left_m,
                    "detections": len(found),
                    "detection_results": [
                        {
                            "label": detection.label,
                            "confidence": detection.confidence,
                            "bbox_xyxy": list(detection.bbox),
                        }
                        for detection, _ in found
                    ],
                    "all_detection_results": detection_records(
                        outputs.detection, (416, 1120)
                    ),
                    "artifact_npz": (
                        str(artifact_path.resolve()) if artifact_path is not None else None
                    ),
                    "obstacles": len(obstacles),
                    "depth_source": (
                        "oak_yolo_fused"
                        if provider.has_oak_depth
                        else "yolo_only"
                    ),
                    "depth_calibration_pixels": provider.calibration_pixels,
                    "depth_scale": provider.scale,
                    "depth_offset_m": provider.offset,
                    "bev_obstacles": [
                        {
                            "label": obstacle.label,
                            "x_m": obstacle.x_m,
                            "z_m": obstacle.z_m,
                            "bearing_deg": obstacle.bearing_deg,
                            "range_m": obstacle.range_m,
                        }
                        for obstacle in obstacles
                    ],
                    "steering": command.steering,
                    "vision_center_left_m": command.vision_center_left_m,
                    "gps_hint_left_m": command.gps_hint_left_m,
                    "road_visible": command.road_visible,
                    "path_blocked": command.path_blocked,
                    "esp_invert": args.esp_invert,
                    "esp_normalized": esp_normalized,
                    "esp_percent": esp_percent,
                    "reason": command.reason,
                    "timing": timing,
                }
                if dashboard is not None:
                    dashboard.publish(video_frame, {
                        "state": "REPLAY_ESP" if esp is not None else "REPLAY",
                        "planner": "pure-pursuit",
                        "auto_enabled": esp is not None,
                        "controller_armed": esp is not None,
                        "steering_normalized": command.steering,
                        "steering_direction": esp_direction(esp_percent),
                        "steering_target": None,
                        "throttle": 0.0,
                        "haptic": command.haptic,
                        "reason": command.reason,
                        "road_visible": command.road_visible,
                        "path_blocked": command.path_blocked,
                        "obstacle_count": len(obstacles),
                        "nearest_forward_m": command.nearest_forward_m,
                        "goal_forward_m": command.goal_forward_m,
                        "goal_left_m": command.goal_left_m,
                        "route": args.dataset.name,
                        "gps_valid": True,
                        "latitude": current["lat"],
                        "longitude": current["lon"],
                        "timings_s": outputs.timings_s,
                        "objects": [
                            {"label": name, "confidence": detection.confidence}
                            for detection, name in found
                        ],
                    })
                log.write(json.dumps(record, separators=(",", ":")) + "\n")
                log.flush()
                road_left, road_right = semantic_bounds if semantic_bounds is not None else (None, None)
                waypoint_writer.writerow({
                    "frame": index,
                    "current_lat": current["lat"],
                    "current_lon": current["lon"],
                    "target_frame": target_frame,
                    "target_5m_lat": target[0],
                    "target_5m_lon": target[1],
                    "target_10m_lat": preview_target[0],
                    "target_10m_lon": preview_target[1],
                    "target_5m_forward_m": goal_forward,
                    "target_5m_left_m": goal_left,
                    "target_10m_forward_m": preview_forward,
                    "target_10m_left_m": preview_left,
                    "planned_forward_m": command.goal_forward_m,
                    "planned_left_m": command.goal_left_m,
                    "steering": command.steering,
                    "vision_center_left_m": command.vision_center_left_m,
                    "gps_hint_left_m": command.gps_hint_left_m,
                    "road_visible": command.road_visible,
                    "path_blocked": command.path_blocked,
                    "esp_normalized": esp_normalized,
                    "esp_percent": esp_percent,
                    "road_left_bound_m": road_left,
                    "road_right_bound_m": road_right,
                    "road_clamped": command.road_clamped,
                    "reason": command.reason,
                    "wall_time_start_s": wall_time_start_s,
                    "wall_time_end_s": wall_time_end_s,
                    "image_read_ms": timing["image_read_s"] * 1000.0,
                    "panorama_stitch_ms": timing["panorama_stitch_s"] * 1000.0,
                    "panorama_resize_ms": panorama_resize_s * 1000.0,
                    "detection_model_ms": timing["detection_model_s"] * 1000.0,
                    "segmentation_model_ms": timing["segmentation_model_s"] * 1000.0,
                    "depth_model_ms": timing["depth_model_s"] * 1000.0,
                    "parallel_perception_ms": parallel_s * 1000.0,
                    "parallel_perception_fps": timing["parallel_perception_fps"],
                    "perception_postprocess_ms": perception_postprocess_s * 1000.0,
                    "obstacle_bev_ms": obstacle_bev_s * 1000.0,
                    "route_target_ms": route_target_s * 1000.0,
                    "road_bev_ms": road_bev_s * 1000.0,
                    "planner_ms": planner_s * 1000.0,
                    "trajectory_conversion_ms": trajectory_conversion_s * 1000.0,
                    "render_ms": render_s * 1000.0,
                    "video_write_ms": video_write_s * 1000.0,
                    "trajectory_total_ms": trajectory_total_s * 1000.0,
                    "trajectory_fps": timing["trajectory_fps"],
                })
                waypoint_stream.flush()
                print(
                    f"[{output_index + 1}/{len(selected)}] frame={index} target={target_frame} "
                    f"models={parallel_s * 1000.0:.1f}ms "
                    f"plan={planner_s * 1000.0:.2f}ms "
                    f"esp={trajectory_conversion_s * 1000.0:.3f}ms "
                    f"total={trajectory_total_s * 1000.0:.1f}ms "
                    f"steer={command.steering:+.3f} "
                    f"ESP={esp_percent:.1f}% {esp_direction(esp_percent)}",
                    flush=True,
                )
    finally:
        if esp is not None:
            esp.close()
        if dashboard is not None:
            dashboard.close()
        models.close()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()
    elapsed = time.monotonic() - started
    print(f"saved {args.output}, {log_path}, and {waypoint_path} in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
