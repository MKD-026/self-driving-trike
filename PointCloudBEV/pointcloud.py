"""
3D point-cloud fusion for the OAK-D 3-camera rig.

Pipeline:
    1. Backproject each cam's depth (using its intrinsics) → 3D points in
       that cam's frame.
    2. Transform to body frame using extrinsics (caliscope R, t):
            P_body = R^T @ (P_cam - t)
    3. Voxel-grid filter: bin points into 3D cells and keep only cells with
       ≥ N member points (kills isolated salt-and-pepper noise).
    4. Multi-camera consensus (optional, the nuclear ghost killer): in voxels
       that lie inside the overlap of two cams' FOVs, require ≥ 2 distinct
       cam_ids to have contributed. A real surface visible to both cams must
       agree; a "ghost" only one cam saw is rejected.
    5. Render the cleaned cloud back to (a) a cylindrical panorama with a
       z-buffer or (b) a top-down BEV image.

All vectorized numpy + cv2; no torch / open3d. ~50 ms per frame on a laptop
for ~70k input points.

Self-contained: the calibration loader (caliscope camera-array TOML) and a
small frame-loader for the dataset format are included here, so this module
can be used without any other helper file.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# ─── Calibration + frame loader (self-contained) ─────────────────────────────

@dataclass
class CamCalib:
    K: np.ndarray         # 3x3 intrinsics (scaled to target image size)
    dist: np.ndarray      # distortion coeffs
    R: np.ndarray         # 3x3 rotation: body→camera (caliscope/OpenCV convention)
    t: np.ndarray         # 3-vec translation in cam frame (world origin in cam frame)


def load_calibration(toml_path: Path,
                     target_size: tuple[int, int]) -> dict[int, CamCalib]:
    """Load a caliscope camera-array TOML. `target_size` = (W, H) of the
    actual image streams; intrinsics are scaled accordingly.

    Convention used by caliscope (and OpenCV): for a 3D world point P,
    the camera-frame point is P_cam = R @ P_world + t. So R is body→camera
    and t is the body origin expressed in camera frame.
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    out: dict[int, CamCalib] = {}
    for key, cam in data["cameras"].items():
        cam_id = int(key)
        cw, ch = cam["size"]
        tw, th = target_size
        sx, sy = tw / cw, th / ch
        K = np.array(cam["matrix"], dtype=np.float64).copy()
        K[0, 0] *= sx; K[1, 1] *= sy
        K[0, 2] *= sx; K[1, 2] *= sy
        dist = np.array(cam["distortions"], dtype=np.float64)
        R, _ = cv2.Rodrigues(np.array(cam["rotation"], dtype=np.float64))
        t = np.array(cam["translation"], dtype=np.float64)
        out[cam_id] = CamCalib(K=K, dist=dist, R=R, t=t)
    return out


def load_frame(rec_dir: Path, frame_idx: int):
    """Read RGB + depth from cam0/1/2 at the given frame index.

    Expected layout (matches the project's recording format):
        <rec_dir>/cam<cam_id>/rgb/<frame:06d>.png
        <rec_dir>/cam<cam_id>/depth/<frame:06d>.npy   (uint16, mm)
    """
    rgb, depth = {}, {}
    for cam_id in (0, 1, 2):
        rgb_p = rec_dir / f"cam{cam_id}" / "rgb" / f"{frame_idx:06d}.png"
        d_p = rec_dir / f"cam{cam_id}" / "depth" / f"{frame_idx:06d}.npy"
        rgb[cam_id] = cv2.imread(str(rgb_p)) if rgb_p.exists() else None
        depth[cam_id] = np.load(d_p) if d_p.exists() else None
    return rgb, depth


def colorize_depth(depth_m: np.ndarray,
                   d_min: float = 0.5, d_max: float = 8.0) -> np.ndarray:
    """Depth (m) → BGR via TURBO colormap. Invalid pixels (≤0) are black.
    Closer points are warmer."""
    valid = (depth_m > 0) & np.isfinite(depth_m)
    norm = np.zeros_like(depth_m, dtype=np.float32)
    if valid.any():
        d = np.clip(depth_m, d_min, d_max)
        norm = (d - d_min) / max(d_max - d_min, 1e-6)
        norm = 1.0 - norm
    img8 = (np.clip(norm, 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


# ─── 1. Backproject all cams ─────────────────────────────────────────────────

def backproject_all(rgb_by_cam: dict, depth_by_cam: dict, calib: dict,
                    stride: int = 2,
                    depth_min: float = 0.4, depth_max: float = 12.0,
                    median_kernel: int = 5):
    """Returns (pts, colors, cam_ids):
        pts     : (N, 3) float32 — body-frame meters
        colors  : (N, 3) uint8 — BGR sampled from each cam's RGB
        cam_ids : (N,) int32 — which camera contributed each point
    """
    pts_all, col_all, cid_all = [], [], []
    for cam_id, depth in depth_by_cam.items():
        if depth is None or cam_id not in calib:
            continue
        c = calib[cam_id]
        rgb = rgb_by_cam.get(cam_id)

        d = depth.astype(np.float32) / 1000.0  # mm → m
        if median_kernel >= 3 and median_kernel % 2 == 1:
            d = cv2.medianBlur(d, median_kernel)

        H, W = d.shape
        # Subsample for speed.
        v_idx = np.arange(0, H, stride, dtype=np.int32)
        u_idx = np.arange(0, W, stride, dtype=np.int32)
        vv, uu = np.meshgrid(v_idx, u_idx, indexing="ij")
        z = d[vv, uu]
        valid = (z >= depth_min) & (z <= depth_max)
        if not valid.any():
            continue
        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)
        z = z[valid].astype(np.float32)

        # Backproject in cam frame.
        x = (u - c.K[0, 2]) * z / c.K[0, 0]
        y = (v - c.K[1, 2]) * z / c.K[1, 1]
        P_cam = np.column_stack([x, y, z])

        # Body frame: P_body = R^T @ (P_cam - t).
        # Row vec: P_body_row = (P_cam_row - t_row) @ R.
        P_body = (P_cam - c.t.astype(np.float32)) @ c.R.astype(np.float32)

        pts_all.append(P_body)

        if rgb is not None:
            cols = rgb[vv[valid], uu[valid]]
        else:
            cols = np.full((len(P_body), 3), 200, dtype=np.uint8)
        col_all.append(cols)
        cid_all.append(np.full(len(P_body), cam_id, dtype=np.int32))

    if not pts_all:
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros(0, dtype=np.int32))
    return (np.vstack(pts_all).astype(np.float32),
            np.vstack(col_all).astype(np.uint8),
            np.concatenate(cid_all))


# ─── 2. Voxel + consensus filter ─────────────────────────────────────────────

def voxel_consensus_filter(pts: np.ndarray, colors: np.ndarray,
                            cam_ids: np.ndarray,
                            voxel_size: float = 0.05,
                            min_points_per_voxel: int = 2,
                            require_min_cams: int = 1):
    """Voxel-grid filter with optional multi-camera consensus.

    Parameters:
        voxel_size            : metres per voxel cube side (5 cm default)
        min_points_per_voxel  : minimum input points to accept a voxel
        require_min_cams      : minimum *distinct* cam_ids that must have
                                contributed to a voxel (1 = no consensus,
                                2 = "ghost killer", 3 = strictly fused)

    Returns (filtered_pts, filtered_colors, n_pts_per_voxel, n_cams_per_voxel).
    """
    if len(pts) == 0:
        return (pts.copy(), colors.copy(),
                np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32))

    inv = 1.0 / voxel_size
    vx = np.floor(pts * inv).astype(np.int64)  # (N, 3)
    keys, inv_idx, counts = np.unique(vx, axis=0,
                                       return_inverse=True,
                                       return_counts=True)
    n_voxels = len(keys)

    # Per-voxel cam-id presence using a small flag matrix; OAK-D has 3 cams.
    n_cams_max = max(int(cam_ids.max()) + 1, 3)
    cam_present = np.zeros((n_voxels, n_cams_max), dtype=bool)
    cam_present[inv_idx, cam_ids] = True
    n_cams_per_voxel = cam_present.sum(axis=1).astype(np.int32)

    keep = (counts >= min_points_per_voxel) & (n_cams_per_voxel >= require_min_cams)
    if not keep.any():
        return (np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32))

    # Aggregate: mean point + mean color per voxel.
    sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
    sum_cols = np.zeros((n_voxels, 3), dtype=np.float64)
    np.add.at(sum_pts, inv_idx, pts.astype(np.float64))
    np.add.at(sum_cols, inv_idx, colors.astype(np.float64))
    mean_pts = (sum_pts / counts[:, None]).astype(np.float32)
    mean_cols = np.clip(sum_cols / counts[:, None], 0, 255).astype(np.uint8)

    return (mean_pts[keep], mean_cols[keep],
            counts[keep].astype(np.int32),
            n_cams_per_voxel[keep])


# ─── 3a. Project cloud → cylindrical panorama with z-buffer ──────────────────

def project_cloud_to_pano(pts: np.ndarray, colors: np.ndarray,
                          pano_size: tuple[int, int],
                          fov_h_deg: float = 130.0,
                          fov_v_deg: float = 45.0,
                          depth_max: float = 12.0,
                          point_radius_px: int = 1):
    """Z-buffered projection of a body-frame cloud onto a cylindrical panorama.

    Coordinate convention: body X right, Y down, Z forward. Azimuth 0 is
    looking down +Z; positive azimuth points toward +X (right).

    Returns (rgb_pano, depth_pano) where depth_pano is the body-frame
    distance from origin to the surface (m); 0 = no surface.
    """
    Wp, Hp = pano_size
    rgb = np.zeros((Hp, Wp, 3), dtype=np.uint8)
    depth = np.full((Hp, Wp), np.inf, dtype=np.float32)

    if len(pts) == 0:
        return rgb, np.zeros((Hp, Wp), dtype=np.float32)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    in_range = (z > 0.05) & (r > 0.05) & (r < depth_max)
    if not in_range.any():
        return rgb, np.zeros((Hp, Wp), dtype=np.float32)

    fov_h = np.radians(fov_h_deg)
    fov_v = np.radians(fov_v_deg)

    az = np.arctan2(x, z)
    el = np.arcsin(np.clip(-y / np.maximum(r, 1e-6), -1.0, 1.0))

    in_view = in_range & (az > -fov_h / 2) & (az < fov_h / 2) \
              & (el > -fov_v / 2) & (el < fov_v / 2)
    if not in_view.any():
        return rgb, np.zeros((Hp, Wp), dtype=np.float32)

    az = az[in_view]; el = el[in_view]
    r = r[in_view]; cols = colors[in_view]

    px = ((az / fov_h + 0.5) * Wp).astype(np.int32)
    py = ((-el / fov_v + 0.5) * Hp).astype(np.int32)
    np.clip(px, 0, Wp - 1, out=px)
    np.clip(py, 0, Hp - 1, out=py)

    # Sort far→near so closer points overwrite farther in the splat.
    order = np.argsort(-r)
    px = px[order]; py = py[order]; r = r[order]; cols = cols[order]

    if point_radius_px <= 0:
        rgb[py, px] = cols
        depth[py, px] = r
    else:
        # Splat each point as a small disc by writing to neighbours.
        for dy in range(-point_radius_px, point_radius_px + 1):
            for dx in range(-point_radius_px, point_radius_px + 1):
                if dy * dy + dx * dx > point_radius_px * point_radius_px:
                    continue
                yy = py + dy
                xx = px + dx
                ok = (yy >= 0) & (yy < Hp) & (xx >= 0) & (xx < Wp)
                if not ok.any():
                    continue
                rgb[yy[ok], xx[ok]] = cols[ok]
                depth[yy[ok], xx[ok]] = r[ok]

    out_d = np.where(np.isfinite(depth), depth, 0.0)
    return rgb, out_d


# ─── 3b. Top-down BEV ────────────────────────────────────────────────────────

def project_cloud_to_bev(pts: np.ndarray, colors: np.ndarray,
                         forward_m: float = 15.0, lateral_m: float = 12.0,
                         resolution_m: float = 0.04,
                         ground_y_range: tuple = (-0.5, 1.5),
                         point_radius_px: int = 1):
    """Top-down view; obstacles (Y < ground_y_range[0]) painted red,
    ground band painted with their original colors.
    """
    H = int(forward_m / resolution_m)
    W = int(lateral_m / resolution_m)
    img = np.full((H, W, 3), 30, dtype=np.uint8)
    if len(pts) == 0:
        return img

    half = lateral_m / 2.0
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    in_box = (z > 0) & (z < forward_m) & (x > -half) & (x < half)
    if not in_box.any():
        return img

    px = ((x[in_box] + half) / resolution_m).astype(np.int32)
    py = (H - 1 - z[in_box] / resolution_m).astype(np.int32)
    np.clip(px, 0, W - 1, out=px)
    np.clip(py, 0, H - 1, out=py)
    yy = y[in_box]
    cc = colors[in_box]

    above = yy < ground_y_range[0]
    on = (yy >= ground_y_range[0]) & (yy <= ground_y_range[1])

    if on.any():
        img[py[on], px[on]] = cc[on]
    if above.any():
        img[py[above], px[above]] = (40, 40, 240)

    cv2.drawMarker(img, (W // 2, H - 1), (0, 255, 255),
                   markerType=cv2.MARKER_TRIANGLE_UP, markerSize=14, thickness=2)
    for r_m in (5, 10, 15):
        if r_m < forward_m:
            r_px = int(r_m / resolution_m)
            cv2.line(img, (W // 2 - r_px, H - 1 - r_px),
                     (W // 2 + r_px, H - 1 - r_px), (60, 60, 60), 1)
            cv2.putText(img, f"{r_m}m", (W // 2 + r_px + 4, H - 1 - r_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    return img
