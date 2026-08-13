"""Adapter for the authoritative final_data_capture panorama stitcher."""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


CAPTURE_ROOT = Path("/home/h2x/Documents/final_data_capture")
# Validated fixed-rig calibration selected from run_20260811_152449 frame 300.
# Keep a repository-owned copy so an unrelated capture calibration cannot
# silently change autonomous stitching.
STITCH_CONFIG_PATH = Path(__file__).resolve().parent / "sift_stitch_config.json"
STITCH_MODULE_PATH = CAPTURE_ROOT / "render_sift_panorama_frames.py"
MODEL_OUTPUT_SIZE = (1120, 416)


def _load_capture_stitch_module():
    capture_path = str(CAPTURE_ROOT)
    if capture_path not in sys.path:
        sys.path.insert(0, capture_path)
    spec = importlib.util.spec_from_file_location(
        "final_data_capture_batch_stitch", STITCH_MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load final_data_capture stitcher: {STITCH_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class PanoramaCompatibilityState:
    ready: bool = False
    status: str = "Loading final_data_capture stitch config"
    next_retry_monotonic: float = 0.0


@dataclass
class StitchState:
    pano_state: PanoramaCompatibilityState = field(default_factory=PanoramaCompatibilityState)
    renderer: object | None = None
    config: dict | None = None
    render_args: object | None = None
    source_shape: tuple[int, int] | None = None
    config_path: Path = STITCH_CONFIG_PATH

    def ensure_ready(self, source_shape: tuple[int, int]) -> None:
        if self.renderer is not None and self.source_shape == source_shape:
            return
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        expected = {"cam0": "left", "cam1": "center", "cam2": "right"}
        if config.get("camera_order") != expected:
            raise RuntimeError(f"{self.config_path} must define camera_order={expected}")
        # Keep the exact final_data_capture transforms/crop/blending/exposure
        # method, with only the final model-facing size changed from 1200 to
        # the fixed TensorRT engine input of 1120x416.
        config["output_width"], config["output_height"] = MODEL_OUTPUT_SIZE
        self.renderer = _load_capture_stitch_module()
        self.config = config
        self.render_args = SimpleNamespace(
            camera_hfov_deg=float(config["camera_hfov_deg"]),
            blend_width=float(config.get("blend_width", 6.0)),
            center_priority=float(config.get("center_priority", 2.2)),
            no_exposure_match=not bool(config.get("exposure_match", True)),
        )
        self.source_shape = source_shape
        self.pano_state.ready = True
        self.pano_state.status = (
            f"final_data_capture stitch ready: {self.config_path} -> "
            f"{MODEL_OUTPUT_SIZE[0]}x{MODEL_OUTPUT_SIZE[1]}"
        )


def create_stitch_state(config_path: Path = STITCH_CONFIG_PATH) -> StitchState:
    return StitchState(config_path=Path(config_path))


def stitch_sift(
    left_bgr: np.ndarray,
    center_bgr: np.ndarray,
    right_bgr: np.ndarray,
    center_ts_us: int,
    state: StitchState,
    recompute: bool = False,
):
    del center_ts_us, recompute
    state.ensure_ready(center_bgr.shape[:2])
    scale = float(state.config.get("work_scale", 1.0))
    if scale != 1.0:
        left_bgr = cv2.resize(left_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        center_bgr = cv2.resize(center_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        right_bgr = cv2.resize(right_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    left_h = np.asarray(state.config["left_to_center"], dtype=np.float64)
    right_h = np.asarray(state.config["right_to_center"], dtype=np.float64)
    panorama = state.renderer.stitch_panorama(
        left_bgr, center_bgr, right_bgr, left_h, right_h, state.render_args
    )
    return state.renderer.fit_output_size(
        panorama, MODEL_OUTPUT_SIZE[0], MODEL_OUTPUT_SIZE[1], "stretch"
    )


def stitch_depth_sift(
    left_depth: np.ndarray | None,
    center_depth: np.ndarray | None,
    right_depth: np.ndarray | None,
    state: StitchState,
) -> np.ndarray | None:
    if left_depth is None or center_depth is None or right_depth is None:
        return None
    state.ensure_ready(center_depth.shape[:2])
    scale = float(state.config.get("work_scale", 1.0))
    if scale != 1.0:
        left_depth = cv2.resize(left_depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        center_depth = cv2.resize(center_depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        right_depth = cv2.resize(right_depth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    renderer = state.renderer
    left_h = np.asarray(state.config["left_to_center"], dtype=np.float64)
    right_h = np.asarray(state.config["right_to_center"], dtype=np.float64)
    projected = []
    for depth in (left_depth, center_depth, right_depth):
        image, mask = renderer.cylindrical_project(depth, state.render_args.camera_hfov_deg)
        projected.append((image, mask))
    corners = [
        renderer.image_corners(projected[1][0]),
        cv2.perspectiveTransform(renderer.image_corners(projected[0][0]), left_h),
        cv2.perspectiveTransform(renderer.image_corners(projected[2][0]), right_h),
    ]
    points = np.vstack(corners).reshape(-1, 2)
    min_xy = np.floor(points.min(axis=0)).astype(int)
    max_xy = np.ceil(points.max(axis=0)).astype(int)
    translate = np.array(
        [[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=np.float64
    )
    size = (int(max_xy[0] - min_xy[0]), int(max_xy[1] - min_xy[1]))
    fused = np.full((size[1], size[0]), np.inf, dtype=np.float32)
    for (depth, mask), transform in zip(
        projected, (translate @ left_h, translate, translate @ right_h)
    ):
        valid = ((depth > 0).astype(np.uint8) * 255) & mask
        warped_depth = cv2.warpPerspective(
            depth.astype(np.float32), transform, size, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        warped_valid = cv2.warpPerspective(
            valid, transform, size, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ) > 0
        fused[warped_valid] = np.minimum(fused[warped_valid], warped_depth[warped_valid])
    fused[~np.isfinite(fused)] = 0
    resized = cv2.resize(fused, MODEL_OUTPUT_SIZE, interpolation=cv2.INTER_NEAREST)
    return np.clip(resized, 0, np.iinfo(np.uint16).max).astype(np.uint16)
