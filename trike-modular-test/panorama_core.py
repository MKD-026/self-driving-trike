from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

SIFT_WARMUP_FRAMES = 50
PANO_WORK_SCALE = 0.5
PANO_RENDER_SCALE = 0.5
PANO_WIDTH = 1120
PANO_HEIGHT = 416
PANO_YAW_FOV_DEG = 190.0
PANO_PITCH_FOV_DEG = 58.0
CYL_CAMERA_HFOV_DEG = 72.0
PANO_FILL_COLOR = (68, 68, 68)
PANO_BORDER_VALUE = (68.0, 68.0, 68.0, 0.0)
PANO_OUTLINE_COLOR = (0, 0, 255)
PANO_OUTLINE_THICKNESS = 3
SIFT_RATIO_TEST = 0.75
SIFT_RANSAC_THRESH = 4.0
SIFT_MIN_INLIERS = 18
SIFT_MIN_INLIER_RATIO = 0.28
PANO_EDGE_FEATHER = 35.0
SIFT_CACHE_FORMAT = "cylindrical_affine_v1"
PANORAMA_CONFIG_PATH = Path("/home/h2x/Documents/final_data_capture/panorama_config.json")


def load_panorama_config(path: Path = PANORAMA_CONFIG_PATH) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load panorama config from {path}: {exc}") from exc
    expected_order = {"left": "cam0", "center": "cam1", "right": "cam2"}
    if payload.get("camera_order") != expected_order:
        raise RuntimeError(f"{path} must define camera_order={expected_order}")
    if "sift_cache_path" not in payload:
        raise RuntimeError(f"{path} is missing sift_cache_path")
    return payload


PANORAMA_CONFIG = load_panorama_config()
PANO_HORIZONTAL_FLIP = bool(PANORAMA_CONFIG.get("horizontal_flip", False))
SIFT_CACHE_PATH = Path(PANORAMA_CONFIG["sift_cache_path"]).expanduser()


@dataclass
class StreamFrame:
    image: np.ndarray
    ts_us: int


@dataclass
class PanoramaState:
    warmup_count: int = 0
    ready: bool = False
    status: str = f"SIFT warmup: 0/{SIFT_WARMUP_FRAMES}"
    left_to_center: np.ndarray | None = None
    right_to_center: np.ndarray | None = None
    last_center_ts_us: int = -1
    next_retry_monotonic: float = 0.0
    last_render_center_ts_us: int = -1
    cached_panorama: np.ndarray | None = None


def scale_homography_to_fullres(h_scaled: np.ndarray, scale: float) -> np.ndarray:
    scale_mat = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    inv_scale_mat = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return inv_scale_mat @ h_scaled @ scale_mat


def scale_homography_for_resized_images(h_full: np.ndarray, scale: float) -> np.ndarray:
    scale_mat = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    inv_scale_mat = np.array([[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return scale_mat @ h_full @ inv_scale_mat


def image_corners(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    return np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)


def contrast_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


@lru_cache(maxsize=8)
def cylindrical_projection_maps(width: int, height: int, hfov_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focal = (width / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    x_coords = np.linspace(0, width - 1, width, dtype=np.float32)
    y_coords = np.linspace(0, height - 1, height, dtype=np.float32)
    x_grid = np.repeat(x_coords[None, :], height, axis=0)
    y_grid = np.repeat(y_coords[:, None], width, axis=1)
    theta = (x_grid - cx) / focal
    map_x = focal * np.tan(theta) + cx
    map_y = (y_grid - cy) / np.cos(theta) + cy
    valid = (map_x >= 0.0) & (map_x <= width - 1) & (map_y >= 0.0) & (map_y <= height - 1)
    return np.ascontiguousarray(map_x, dtype=np.float32), np.ascontiguousarray(map_y, dtype=np.float32), valid


def project_image_cylindrical(image: np.ndarray, hfov_deg: float = CYL_CAMERA_HFOV_DEG) -> tuple[np.ndarray, np.ndarray]:
    map_x, map_y, valid = cylindrical_projection_maps(image.shape[1], image.shape[0], hfov_deg)
    projected = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PANO_BORDER_VALUE)
    valid_mask = valid.astype(np.uint8) * 255
    return projected, valid_mask


def estimate_pair_homography(src_image: np.ndarray, dst_image: np.ndarray, scale: float = PANO_WORK_SCALE) -> tuple[np.ndarray | None, int, int]:
    src_projected, _ = project_image_cylindrical(src_image)
    dst_projected, _ = project_image_cylindrical(dst_image)
    src_small = cv2.resize(src_projected, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    dst_small = cv2.resize(dst_projected, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    src_gray = contrast_gray(src_small)
    dst_gray = contrast_gray(dst_small)
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.025, edgeThreshold=12)
    kp_src, desc_src = sift.detectAndCompute(src_gray, None)
    kp_dst, desc_dst = sift.detectAndCompute(dst_gray, None)
    if desc_src is None or desc_dst is None:
        return None, 0, 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(desc_src, desc_dst, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < SIFT_RATIO_TEST * second.distance:
            good_matches.append(first)
    good_matches.sort(key=lambda match: match.distance)
    if len(good_matches) < 8:
        return None, len(good_matches), 0
    src_pts = np.float32([kp_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    affine_scaled, mask = cv2.estimateAffine2D(
        src_pts.reshape(-1, 2),
        dst_pts.reshape(-1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=SIFT_RANSAC_THRESH,
        maxIters=3000,
        confidence=0.995,
        refineIters=10,
    )
    inliers = int(mask.ravel().sum()) if mask is not None else 0
    inlier_ratio = inliers / max(1, len(good_matches))
    if affine_scaled is None or inliers < SIFT_MIN_INLIERS or inlier_ratio < SIFT_MIN_INLIER_RATIO:
        return None, len(good_matches), inliers
    h_scaled = np.eye(3, dtype=np.float64)
    h_scaled[:2, :] = affine_scaled
    return scale_homography_to_fullres(h_scaled, scale), len(good_matches), inliers


def load_sift_cache(cache_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
        if payload.get("format") != SIFT_CACHE_FORMAT:
            return None
        if payload.get("width") != width or payload.get("height") != height:
            return None
        left_h = np.asarray(payload["left_to_center"], dtype=np.float64)
        right_h = np.asarray(payload["right_to_center"], dtype=np.float64)
        if left_h.shape != (3, 3) or right_h.shape != (3, 3):
            return None
        return left_h, right_h
    except Exception:
        return None


def save_sift_cache(cache_path: Path, width: int, height: int, left_h: np.ndarray, right_h: np.ndarray) -> None:
    payload = {
        "format": SIFT_CACHE_FORMAT,
        "width": width,
        "height": height,
        "left_to_center": left_h.tolist(),
        "right_to_center": right_h.tolist(),
        "updated_at": datetime.now().isoformat(),
    }
    cache_path.write_text(json.dumps(payload, indent=2))


@lru_cache(maxsize=4)
def equirect_remap_maps(width: int, height: int, yaw_fov_deg: float, pitch_fov_deg: float) -> tuple[np.ndarray, np.ndarray]:
    pitch = np.deg2rad(np.linspace(pitch_fov_deg / 2.0, -pitch_fov_deg / 2.0, height, dtype=np.float32))
    pitch_grid = np.repeat(pitch[:, None], width, axis=1)
    focal_y = (height / 2.0) / np.tan(np.deg2rad(pitch_fov_deg) / 2.0)
    center_y = (height - 1) / 2.0
    map_x = np.ascontiguousarray(np.tile(np.linspace(0, width - 1, width, dtype=np.float32), (height, 1)), dtype=np.float32)
    map_y = np.ascontiguousarray(center_y - focal_y * np.tan(pitch_grid), dtype=np.float32)
    return map_x, map_y


def cylindrical_strip_to_equirect(image: np.ndarray) -> np.ndarray:
    map_x, map_y = equirect_remap_maps(image.shape[1], image.shape[0], PANO_YAW_FOV_DEG, PANO_PITCH_FOV_DEG)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PANO_BORDER_VALUE)


def cylindrical_mask_to_equirect(mask: np.ndarray) -> np.ndarray:
    map_x, map_y = equirect_remap_maps(mask.shape[1], mask.shape[0], PANO_YAW_FOV_DEG, PANO_PITCH_FOV_DEG)
    return cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def draw_mask_rectangle(image: np.ndarray, mask: np.ndarray, color=PANO_OUTLINE_COLOR, thickness: int = PANO_OUTLINE_THICKNESS) -> np.ndarray:
    rows, cols = np.where(mask >= 128)
    if rows.size == 0 or cols.size == 0:
        return image
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    outlined = image.copy()
    cv2.rectangle(outlined, (left, top), (right, bottom), color, thickness, cv2.LINE_AA)
    return outlined


def canvas_transform(center_image: np.ndarray, left_h: np.ndarray, right_h: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    center_corners = image_corners(center_image)
    left_corners = cv2.perspectiveTransform(image_corners(center_image), left_h)
    right_corners = cv2.perspectiveTransform(image_corners(center_image), right_h)
    all_corners = np.vstack([center_corners, left_corners, right_corners]).reshape(-1, 2)
    min_xy = np.floor(all_corners.min(axis=0)).astype(int)
    max_xy = np.ceil(all_corners.max(axis=0)).astype(int)
    pad_x = int(center_image.shape[1] * 0.08)
    pad_y = int(center_image.shape[0] * 0.08)
    min_x, min_y = min_xy[0] - pad_x, min_xy[1] - pad_y
    max_x, max_y = max_xy[0] + pad_x, max_xy[1] + pad_y
    translate = np.array([[1.0, 0.0, -float(min_x)], [0.0, 1.0, -float(min_y)], [0.0, 0.0, 1.0]], dtype=np.float64)
    return translate, (max(1, max_x - min_x), max(1, max_y - min_y))


def feather_weight(mask: np.ndarray, edge_feather: float = PANO_EDGE_FEATHER) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if not mask_u8.any():
        return mask_u8.astype(np.float32)
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3).astype(np.float32)
    return np.clip(dist / max(edge_feather, 1.0), 0.0, 1.0)


def add_weighted_layer(accum: np.ndarray, weights: np.ndarray, image: np.ndarray, mask: np.ndarray, priority: float) -> None:
    weight = feather_weight(mask) * priority
    accum += image.astype(np.float32) * weight[..., None]
    weights += weight[..., None]


def build_stitched_panorama(left_image: np.ndarray, center_image: np.ndarray, right_image: np.ndarray, center_ts_us: int, state: PanoramaState) -> np.ndarray | None:
    if not state.ready or state.left_to_center is None or state.right_to_center is None:
        return None
    if state.cached_panorama is not None and state.last_render_center_ts_us == center_ts_us:
        return state.cached_panorama
    left_image, left_mask = project_image_cylindrical(left_image)
    center_image, center_mask = project_image_cylindrical(center_image)
    right_image, right_mask = project_image_cylindrical(right_image)
    scale = PANO_RENDER_SCALE
    left_image = cv2.resize(left_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    center_image = cv2.resize(center_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    right_image = cv2.resize(right_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    left_mask = cv2.resize(left_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    center_mask = cv2.resize(center_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    right_mask = cv2.resize(right_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    left_h = scale_homography_for_resized_images(state.left_to_center, scale)
    right_h = scale_homography_for_resized_images(state.right_to_center, scale)
    translate, canvas_size = canvas_transform(center_image, left_h, right_h)
    canvas_w, canvas_h = canvas_size
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weights = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
    center_layer = cv2.warpPerspective(center_image, translate, canvas_size, flags=cv2.INTER_LINEAR)
    center_mask = cv2.warpPerspective(center_mask, translate, canvas_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    add_weighted_layer(accum, weights, center_layer, center_mask, priority=1.8)
    for image, image_mask, homography in ((left_image, left_mask, left_h), (right_image, right_mask, right_h)):
        transform = translate @ homography
        warped = cv2.warpPerspective(image, transform, canvas_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=PANO_BORDER_VALUE)
        mask = cv2.warpPerspective(image_mask, transform, canvas_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        add_weighted_layer(accum, weights, warped, mask, priority=1.0)
    valid = weights[..., 0] > 1e-3
    panorama = np.full((canvas_h, canvas_w, 3), PANO_FILL_COLOR, dtype=np.uint8)
    panorama[valid] = np.clip(accum[valid] / weights[valid], 0, 255).astype(np.uint8)
    stitched = cv2.resize(panorama, (PANO_WIDTH, PANO_HEIGHT), interpolation=cv2.INTER_AREA)
    stitched_valid = cv2.resize((valid.astype(np.uint8) * 255), (PANO_WIDTH, PANO_HEIGHT), interpolation=cv2.INTER_NEAREST)
    equirect = cylindrical_strip_to_equirect(stitched)
    equirect_valid = cylindrical_mask_to_equirect(stitched_valid)
    equirect[equirect_valid < 128] = 0
    equirect = draw_mask_rectangle(equirect, equirect_valid)
    if PANO_HORIZONTAL_FLIP:
        equirect = np.ascontiguousarray(cv2.flip(equirect, 1))
    state.cached_panorama = equirect
    state.last_render_center_ts_us = center_ts_us
    return state.cached_panorama


def build_equirect_source_index(state: PanoramaState, cam_hw: tuple[int, int]):
    """Map every equirect pixel back to its source camera + pixel.

    Pushes each camera's (u, v) coordinate grid through the exact same geometry
    as build_stitched_panorama (cylindrical project -> scale -> warp -> resize ->
    equirect remap), with nearest-neighbour everywhere so coordinates are not
    blended. The result lets callers gather per-camera depth at any equirect
    location (e.g. a detection box) without inverting the warp at runtime.

    Static once the homography locks, so build it once and reuse. Returns
    (idx_slot, idx_u, idx_v) of shape (PANO_HEIGHT, PANO_WIDTH):
      idx_slot: -1 none, 0 center, 1 left, 2 right (matches the RGB stitch order)
      idx_u/idx_v: pixel in that camera's cam_hw frame (== the aligned depth frame)
    """
    if not state.ready or state.left_to_center is None or state.right_to_center is None:
        return None
    cam_h, cam_w = cam_hw
    map_x, map_y, valid = cylindrical_projection_maps(cam_w, cam_h, CYL_CAMERA_HFOV_DEG)
    uv_full = np.dstack([map_x, map_y]).astype(np.float32)          # source (u, v) at each cyl pixel
    valid_full = (valid.astype(np.uint8) * 255)
    scale = PANO_RENDER_SCALE
    uv_s = cv2.resize(uv_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    valid_s = cv2.resize(valid_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    sh, sw = uv_s.shape[:2]
    left_h = scale_homography_for_resized_images(state.left_to_center, scale)
    right_h = scale_homography_for_resized_images(state.right_to_center, scale)
    dummy = np.zeros((sh, sw, 3), dtype=np.uint8)
    translate, canvas_size = canvas_transform(dummy, left_h, right_h)
    canvas_w, canvas_h = canvas_size

    res_uv = np.zeros((canvas_h, canvas_w, 2), dtype=np.float32)
    res_slot = np.zeros((canvas_h, canvas_w), dtype=np.int16)  # 0 none; store slot+1
    # Write left, then right, then center last so center wins overlaps (it has
    # blend priority in the real panorama too).
    for store, transform in ((2, translate @ left_h), (3, translate @ right_h), (1, translate)):
        w_uv = cv2.warpPerspective(uv_s, transform, canvas_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        w_valid = cv2.warpPerspective(valid_s, transform, canvas_size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        m = w_valid > 0
        res_uv[m] = w_uv[m]
        res_slot[m] = store

    res_uv_p = cv2.resize(res_uv, (PANO_WIDTH, PANO_HEIGHT), interpolation=cv2.INTER_NEAREST)
    res_slot_p = cv2.resize(res_slot.astype(np.float32), (PANO_WIDTH, PANO_HEIGHT), interpolation=cv2.INTER_NEAREST)
    eqmx, eqmy = equirect_remap_maps(PANO_WIDTH, PANO_HEIGHT, PANO_YAW_FOV_DEG, PANO_PITCH_FOV_DEG)
    eq_uv = cv2.remap(res_uv_p, eqmx, eqmy, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    eq_slot = cv2.remap(res_slot_p, eqmx, eqmy, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    idx_u = np.clip(np.rint(eq_uv[..., 0]), 0, cam_w - 1).astype(np.int16)
    idx_v = np.clip(np.rint(eq_uv[..., 1]), 0, cam_h - 1).astype(np.int16)
    idx_slot = (np.rint(eq_slot).astype(np.int16) - 1)  # -1 none, 0 center, 1 left, 2 right
    if PANO_HORIZONTAL_FLIP:
        return (
            np.ascontiguousarray(idx_slot[:, ::-1]),
            np.ascontiguousarray(idx_u[:, ::-1]),
            np.ascontiguousarray(idx_v[:, ::-1]),
        )
    return idx_slot, idx_u, idx_v


def update_panorama_state(snap, state: PanoramaState, now_monotonic: float) -> PanoramaState:
    if state.ready:
        return state
    left = snap.get(1, {}).get("streams", {}).get("rgb")
    center = snap.get(0, {}).get("streams", {}).get("rgb")
    right = snap.get(2, {}).get("streams", {}).get("rgb")
    if left is None or center is None or right is None:
        state.status = "Waiting for all 3 RGB streams"
        return state
    if center.ts_us == state.last_center_ts_us:
        return state
    state.last_center_ts_us = center.ts_us
    state.warmup_count += 1
    if state.warmup_count < SIFT_WARMUP_FRAMES:
        state.status = f"SIFT warmup: {state.warmup_count}/{SIFT_WARMUP_FRAMES}"
        return state
    if now_monotonic < state.next_retry_monotonic:
        return state
    left_h, left_matches, left_inliers = estimate_pair_homography(left.image, center.image)
    right_h, right_matches, right_inliers = estimate_pair_homography(right.image, center.image)
    if left_h is None or right_h is None:
        state.next_retry_monotonic = now_monotonic + 1.0
        state.status = f"Panorama waiting for stronger SIFT features (L matches/inliers {left_matches}/{left_inliers}, R matches/inliers {right_matches}/{right_inliers})"
        return state
    state.left_to_center = left_h
    state.right_to_center = right_h
    state.ready = True
    state.status = f"Panorama ready (L matches/inliers {left_matches}/{left_inliers}, R matches/inliers {right_matches}/{right_inliers})"
    return state
