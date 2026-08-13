"""Semantic-road boundary extraction and compact BEV rendering."""

from __future__ import annotations

import math
import cv2
import numpy as np

ROAD_CLASS_ID = 0
PANORAMA_HFOV_DEG = 140.0
FORWARD_MAX_M = 20.0
LEFT_MAX_M = 12.0
MIN_FORWARD_M = 0.5
MIN_ROAD_REGION_AREA_PX = 400


def valid_panorama_mask(image: np.ndarray, black_threshold: int = 15) -> np.ndarray:
    """Pixels inside the red panorama frame that contain real camera imagery."""
    x1, y1, x2, y2 = valid_frame_bounds(image)
    mask = np.zeros(image.shape[:2], dtype=bool)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask[y1:y2 + 1, x1:x2 + 1] = gray[y1:y2 + 1, x1:x2 + 1] > black_threshold
    return mask


def detection_in_valid_panorama(
    bbox: tuple[int, int, int, int],
    valid_mask: np.ndarray,
    min_valid_fraction: float = 0.50,
) -> bool:
    """Require a detection center and most of its box to lie on camera pixels."""
    height, width = valid_mask.shape
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0, min(width, int(x1))), max(0, min(width, int(x2)))))
    y1, y2 = sorted((max(0, min(height, int(y1))), max(0, min(height, int(y2)))))
    if x2 <= x1 or y2 <= y1:
        return False
    center_x = min(width - 1, (x1 + x2) // 2)
    center_y = min(height - 1, (y1 + y2) // 2)
    if not valid_mask[center_y, center_x]:
        return False
    return float(np.mean(valid_mask[y1:y2, x1:x2])) >= min_valid_fraction


def valid_frame_bounds(image: np.ndarray, inset_px: int = 4) -> tuple[int, int, int, int]:
    """Return the interior of the red valid-panorama outline."""
    blue, green, red = cv2.split(image)
    outline = (
        (red >= 170)
        & (red.astype(np.float32) >= 1.7 * green.astype(np.float32))
        & (red.astype(np.float32) >= 1.7 * blue.astype(np.float32))
    )
    row_counts = outline.sum(axis=1)
    column_counts = outline.sum(axis=0)
    border_rows = np.where(row_counts >= image.shape[1] * 0.40)[0]
    border_columns = np.where(column_counts >= image.shape[0] * 0.40)[0]
    if border_rows.size < 2 or border_columns.size < 2:
        return 0, 0, image.shape[1] - 1, image.shape[0] - 1
    x1, x2 = int(border_columns.min()) + inset_px, int(border_columns.max()) - inset_px
    y1, y2 = int(border_rows.min()) + inset_px, int(border_rows.max()) - inset_px
    if x2 <= x1 or y2 <= y1:
        return 0, 0, image.shape[1] - 1, image.shape[0] - 1
    return x1, y1, x2, y2


def mask_semantic_to_valid_frame(semantic: np.ndarray, image: np.ndarray) -> np.ndarray:
    if semantic.shape != image.shape[:2]:
        semantic = cv2.resize(
            semantic.astype(np.int32), (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    masked = np.full(semantic.shape, -1, dtype=np.int32)
    valid = valid_panorama_mask(image)
    masked[valid] = semantic[valid]
    return masked


def cleaned_road_mask(semantic: np.ndarray, dominant_only: bool = False) -> np.ndarray:
    road = cv2.morphologyEx(
        (semantic == ROAD_CLASS_ID).astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((7, 7), np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(road, 8)
    if count <= 1:
        return road
    areas = stats[1:, cv2.CC_STAT_AREA]
    threshold = MIN_ROAD_REGION_AREA_PX + 1
    if dominant_only:
        threshold = max(threshold, int(float(areas.max()) * 0.25))
    filtered = np.zeros_like(road, dtype=np.uint8)
    for label_id in range(1, count):
        if stats[label_id, cv2.CC_STAT_AREA] >= threshold:
            filtered[labels == label_id] = 1
    return filtered


def median_patch(depth: np.ndarray, u: int, v: int, radius: int = 7) -> float | None:
    values = depth[max(0, v-radius):min(depth.shape[0], v+radius+1), max(0, u-radius):min(depth.shape[1], u+radius+1)]
    values = values[np.isfinite(values) & (values >= 0.25) & (values <= 40.0)]
    return float(np.median(values)) if values.size else None


def pixel_to_bev(u: int, depth_m: float, width: int) -> tuple[float, float]:
    bearing = math.radians(((u + 0.5) / width - 0.5) * PANORAMA_HFOV_DEG)
    return depth_m * math.cos(bearing), -depth_m * math.sin(bearing)


def smooth(points: list[tuple[float, float]], window: int = 5) -> list[tuple[float, float]]:
    points = sorted(points)
    if len(points) < 4:
        return points
    lateral = np.asarray([point[1] for point in points], dtype=np.float32)
    half = window // 2
    keep = []
    for index, point in enumerate(points):
        local = lateral[max(0, index-half):min(len(points), index+half+1)]
        if abs(point[1] - float(np.median(local))) <= 2.75:
            keep.append(point)
    if len(keep) < 4:
        keep = points
    output = []
    for index, (forward, _) in enumerate(keep):
        local = keep[max(0, index-half):min(len(keep), index+half+1)]
        output.append((forward, float(np.median([point[1] for point in local]))))
    return output


def road_boundaries(semantic: np.ndarray, depth: np.ndarray) -> dict:
    if semantic.shape != depth.shape:
        semantic = cv2.resize(semantic.astype(np.int32), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
    road = cleaned_road_mask(semantic, dominant_only=True)

    left_raw, right_raw = [], []
    height, width = road.shape
    for v in range(height - 64, int(height * 0.50), -8):
        xs = np.where(road[v] > 0)[0]
        if xs.size < 16:
            continue
        left_u = int(min(xs[-1] - 2, xs[0] + 6))
        right_u = int(max(xs[0] + 2, xs[-1] - 6))
        for points, u in ((left_raw, left_u), (right_raw, right_u)):
            value = median_patch(depth, u, v)
            if value is None:
                continue
            forward, left = pixel_to_bev(u, value, width)
            if MIN_FORWARD_M <= forward <= FORWARD_MAX_M and abs(left) <= LEFT_MAX_M:
                points.append((forward, left))
    return {"left_raw": left_raw, "right_raw": right_raw, "left": smooth(left_raw), "right": smooth(right_raw)}


def corridor_bounds(features: dict, forward_m: float) -> tuple[float, float] | None:
    """Return right/left lateral road limits nearest the planner lookahead."""
    values = []
    for key in ("left", "right"):
        points = features.get(key, [])
        if points:
            values.append(min(points, key=lambda point: abs(point[0] - forward_m))[1])
    if len(values) != 2:
        return None
    low, high = sorted(values)
    return (low, high) if high - low >= 0.8 else None

def draw_boundaries(canvas: np.ndarray, semantic: np.ndarray, depth: np.ndarray) -> dict:
    features = road_boundaries(semantic, depth)
    height, width = canvas.shape[:2]
    cx, bottom = width // 2, height - 28
    def pixel(point):
        return int(cx - point[1] * 24.0), int(bottom - point[0] * 18.0)
    for point in sorted(features["left_raw"] + features["right_raw"]):
        x, y = pixel(point)
        if 0 <= x < width and 20 <= y < bottom:
            cv2.circle(canvas, (x, y), 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, (235, 235, 235), -1, cv2.LINE_AA)
    for key in ("left", "right"):
        points = [pixel(point) for point in features[key]]
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, np.int32)], False, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"road edges L/R {len(features['left'])}/{len(features['right'])}", (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    return features


def overlay_road(image: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    if semantic.shape != image.shape[:2]:
        semantic = cv2.resize(semantic.astype(np.int32), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    semantic = mask_semantic_to_valid_frame(semantic, image)
    road = cleaned_road_mask(semantic, dominant_only=False).astype(bool)
    output = image.copy()
    output[road] = (0.65 * output[road] + 0.35 * np.array([40, 210, 70], np.uint8)).astype(np.uint8)
    return output
