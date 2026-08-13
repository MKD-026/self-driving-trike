"""Parallel TensorRT perception and OAK-anchored dense metric depth."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import time

import cv2
import numpy as np

from bev import StereoDepthProvider


ENGINE_IMGSZ = (416, 1120)  # height, width of the fixed-shape TensorRT engines


@dataclass
class PerceptionOutputs:
    detection: object
    segmentation: object
    depth: object
    timings_s: dict[str, float]


class ParallelPerception:
    """Own one TensorRT context per model and launch all three together."""

    def __init__(self, detection: Path, segmentation: Path, depth: Path) -> None:
        from ultralytics import YOLO

        for label, path in (("detection", detection), ("segmentation", segmentation), ("depth", depth)):
            if not Path(path).is_file():
                raise FileNotFoundError(f"{label} engine not found: {path}")
        self.detection = YOLO(str(detection), task="detect")
        segmentation_task = "semantic" if "-sem" in Path(segmentation).stem else "segment"
        self.segmentation = YOLO(str(segmentation), task=segmentation_task)
        self.depth = YOLO(str(depth), task="depth")
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="trt-perception")
        self.detect_lock = Lock()

    @staticmethod
    def _predict(model, panorama: np.ndarray):
        return model.predict(panorama, imgsz=ENGINE_IMGSZ, verbose=False)[0]

    @staticmethod
    def prepare_input(image: np.ndarray) -> np.ndarray:
        """Prepare a native engine input, trimming excess height from the top."""
        target_height, target_width = ENGINE_IMGSZ
        height, width = image.shape[:2]
        if width == target_width and height >= target_height:
            # Keep the bottom of the driving image and remove sky/excess rows.
            return np.ascontiguousarray(image[height - target_height :, :])
        # Retain compatibility with lower-resolution live panoramas.
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    def _track(self, panorama: np.ndarray, conf: float, iou: float):
        # ByteTrack state belongs only to the detection engine.
        with self.detect_lock:
            return self.detection.track(
                panorama,
                persist=True,
                conf=conf,
                iou=iou,
                imgsz=ENGINE_IMGSZ,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

    @staticmethod
    def _timed(callable_, *args):
        started = time.perf_counter()
        result = callable_(*args)
        return result, time.perf_counter() - started

    def infer(self, panorama: np.ndarray, conf: float, iou: float) -> PerceptionOutputs:
        parallel_started = time.perf_counter()
        futures = {
            "detection": self.pool.submit(self._timed, self._track, panorama, conf, iou),
            "segmentation": self.pool.submit(self._timed, self._predict, self.segmentation, panorama),
            "depth": self.pool.submit(self._timed, self._predict, self.depth, panorama),
        }
        completed = {name: future.result() for name, future in futures.items()}
        timings = {f"{name}_model_s": value[1] for name, value in completed.items()}
        timings["parallel_wall_s"] = time.perf_counter() - parallel_started
        return PerceptionOutputs(
            detection=completed["detection"][0],
            segmentation=completed["segmentation"][0],
            depth=completed["depth"][0],
            timings_s=timings,
        )

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def depth_array(result, shape: tuple[int, int]) -> np.ndarray | None:
    """Extract the custom Ultralytics depth result as a dense float32 array."""
    value = getattr(getattr(result, "depth", None), "data", None)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.squeeze(np.asarray(value, dtype=np.float32))
    if array.ndim != 2:
        return None
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return array


def semantic_array(result, shape: tuple[int, int]) -> np.ndarray | None:
    """Extract a dense semantic class-id map from a -sem model result."""
    value = getattr(result, "semantic_mask", None)
    value = getattr(value, "data", value)
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.squeeze(np.asarray(value)).astype(np.int32)
    if array.ndim != 2:
        return None
    if array.shape != shape:
        array = cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return array

class FusedStereoDepthProvider(StereoDepthProvider):
    """Dense YOLO depth calibrated to OAK stereo, with OAK preferred in overlap."""

    def __init__(self, *args, max_oak_depth_m: float = 20.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_oak_depth_m = float(max_oak_depth_m)
        self.yolo_depth_m: np.ndarray | None = None
        self.stitched_oak_depth_m: np.ndarray | None = None
        self.instance_mask: np.ndarray | None = None
        self.valid_panorama_mask: np.ndarray | None = None
        self.fused_depth_m: np.ndarray | None = None
        self.scale = 1.0
        self.offset = 0.0
        self.calibration_pixels = 0
        self._generation = 0
        self._fused_generation = -1

    @property
    def ready(self) -> bool:
        """True for either legacy source-index depth or aligned stitched depth."""
        return self.stitched_oak_depth_m is not None or super().ready

    def _target_shape(self) -> tuple[int, int]:
        if self.idx_slot is not None:
            return self.idx_slot.shape
        if self.stitched_oak_depth_m is not None:
            return self.stitched_oak_depth_m.shape
        return ENGINE_IMGSZ

    def set_depths(self, center, left, right) -> None:
        super().set_depths(center, left, right)
        self.stitched_oak_depth_m = None
        self._generation += 1

    def set_stitched_depth(self, depth_mm: np.ndarray | None) -> None:
        """Set an already panorama-aligned uint16 OAK depth map."""
        if depth_mm is None:
            self.stitched_oak_depth_m = None
        else:
            depth = np.asarray(depth_mm)
            shape = self._target_shape()
            if depth.shape != shape:
                depth = cv2.resize(
                    depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
                )
            dense = depth.astype(np.float32) / 1000.0
            dense[
                ~np.isfinite(dense)
                | (dense < 0.25)
                | (dense > self.max_oak_depth_m)
            ] = 0.0
            self.stitched_oak_depth_m = dense
        self._generation += 1

    @property
    def has_oak_depth(self) -> bool:
        return self.stitched_oak_depth_m is not None or (
            self.ready and any(depth is not None for depth in self.depths)
        )

    def set_model_depth(self, result) -> None:
        shape = self._target_shape()
        self.yolo_depth_m = depth_array(result, shape)
        self._generation += 1

    def set_segmentation(self, result) -> None:
        """Union COCO instance masks for foreground-aware obstacle depth samples."""
        value = getattr(getattr(result, "masks", None), "data", None)
        if value is None:
            self.instance_mask = None
            return
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        masks = np.asarray(value)
        if masks.ndim != 3 or masks.shape[0] == 0:
            self.instance_mask = None
            return
        mask = np.any(masks > 0.5, axis=0).astype(np.uint8)
        shape = self._target_shape()
        if mask.shape != shape:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        self.instance_mask = mask.astype(bool)

    def set_valid_panorama_mask(self, mask: np.ndarray | None) -> None:
        shape = self._target_shape()
        if mask is None:
            self.valid_panorama_mask = None
        else:
            value = np.asarray(mask, dtype=np.uint8)
            if value.shape != shape:
                value = cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
            self.valid_panorama_mask = value.astype(bool)
        self._generation += 1

    def _dense_oak_m(self) -> np.ndarray | None:
        if self.stitched_oak_depth_m is not None:
            return self.stitched_oak_depth_m.copy()
        if not self.ready:
            return None
        dense = np.zeros(self.idx_slot.shape, dtype=np.float32)
        for slot, source in enumerate(self.depths):
            if source is None:
                continue
            mask = self.idx_slot == slot
            if not np.any(mask):
                continue
            height, width = source.shape[:2]
            u = np.clip(self.idx_u[mask], 0, width - 1)
            v = np.clip(self.idx_v[mask], 0, height - 1)
            dense[mask] = source[v, u].astype(np.float32) / 1000.0
        dense[~np.isfinite(dense) | (dense < 0.25) | (dense > self.max_oak_depth_m)] = 0.0
        return dense

    @staticmethod
    def _metric_fit(yolo: np.ndarray, oak: np.ndarray) -> tuple[float, float, int]:
        valid = np.isfinite(yolo) & (yolo > 0.25) & np.isfinite(oak) & (oak > 0.25)
        count = int(np.count_nonzero(valid))
        if count < 100:
            return 1.0, 0.0, count
        y_values, o_values = yolo[valid], oak[valid]
        near_cut, far_cut = np.percentile(o_values, (10.0, 90.0))
        near, far = o_values <= near_cut, o_values >= far_cut
        if np.count_nonzero(near) >= 20 and np.count_nonzero(far) >= 20:
            y_near, y_far = float(np.median(y_values[near])), float(np.median(y_values[far]))
            o_near, o_far = float(np.median(o_values[near])), float(np.median(o_values[far]))
            if abs(y_far - y_near) >= 0.25:
                scale = (o_far - o_near) / (y_far - y_near)
                offset = o_near - scale * y_near
                return float(np.clip(scale, 0.2, 5.0)), float(np.clip(offset, -20.0, 20.0)), count
        ratios = o_values / np.maximum(y_values, 0.25)
        return float(np.clip(np.median(ratios), 0.2, 5.0)), 0.0, count

    def _update_fusion(self) -> None:
        if self._fused_generation == self._generation:
            return
        self._fused_generation = self._generation
        oak = self._dense_oak_m()
        if self.yolo_depth_m is None:
            fused = None if oak is None else oak.copy()
        elif oak is None:
            fused = self.yolo_depth_m.copy()
        else:
            self.scale, self.offset, self.calibration_pixels = self._metric_fit(self.yolo_depth_m, oak)
            fused = self.yolo_depth_m * self.scale + self.offset
            fused = np.where(np.isfinite(fused) & (fused > 0.25), fused, 0.0).astype(np.float32)
            oak_valid = oak > 0.25
            fused[oak_valid] = oak[oak_valid]
        if fused is not None and self.valid_panorama_mask is not None:
            fused[~self.valid_panorama_mask] = 0.0
        self.fused_depth_m = fused

    def dense_depth(self) -> np.ndarray | None:
        self._update_fusion()
        return self.fused_depth_m

    def __call__(self, det):
        self._update_fusion()
        if self.fused_depth_m is None:
            return None
        height, width = self.fused_depth_m.shape
        x1, y1, x2, y2 = det.bbox
        x1, x2 = np.clip((int(x1), int(x2)), 0, width)
        y1, y2 = np.clip((int(y1), int(y2)), 0, height)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = self.fused_depth_m[y1:y2, x1:x2]
        if self.instance_mask is not None:
            foreground = self.instance_mask[y1:y2, x1:x2]
            if np.count_nonzero(foreground) >= 20:
                crop = crop[foreground]
        values = crop[np.isfinite(crop) & (crop >= 0.25) & (crop <= 40.0)]
        return float(np.percentile(values, 35.0)) if values.size else None
