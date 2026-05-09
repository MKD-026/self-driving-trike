#!/usr/bin/env python3
"""
Render a panoramic + BEV MP4 from the *3D point-cloud* fusion pipeline. Each
frame:

    1. Backproject all 3 cams' depth → body-frame point cloud
    2. Voxel-grid filter (dedupe + isolated-noise removal)
    3. Project clean cloud to:
         - cylindrical RGB panorama  (z-buffered)
         - depth panorama            (body-frame distance)
         - top-down BEV
    4. Stack into one frame (3 rows: pano-RGB, pano-depth, BEV).

Run:
    python render_pointcloud_video.py \\
        --recording /path/to/recording \\
        --out /tmp/pointcloud.mp4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pointcloud import (
    backproject_all, colorize_depth, load_calibration, load_frame,
    project_cloud_to_bev, project_cloud_to_pano, voxel_consensus_filter,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recording", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("/tmp/pointcloud.mp4"))
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--stride", type=int, default=2,
                   help="Pixel stride when backprojecting depth")
    p.add_argument("--depth-min", type=float, default=0.4)
    p.add_argument("--depth-max", type=float, default=12.0)
    p.add_argument("--voxel-size", type=float, default=0.06)
    p.add_argument("--min-pts", type=int, default=3,
                   help="Min points per voxel to accept")
    p.add_argument("--require-min-cams", type=int, default=1)
    p.add_argument("--pano-w", type=int, default=1500)
    p.add_argument("--pano-h", type=int, default=540)
    p.add_argument("--fov-h-deg", type=float, default=130.0)
    p.add_argument("--fov-v-deg", type=float, default=45.0)
    p.add_argument("--splat-radius", type=int, default=2)
    p.add_argument("--bev-forward", type=float, default=15.0)
    p.add_argument("--bev-lateral", type=float, default=12.0)
    p.add_argument("--bev-resolution", type=float, default=0.04)
    args = p.parse_args()

    rec: Path = args.recording
    rgb_dir = rec / "cam1" / "rgb"
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        sys.exit(f"no PNGs in {rgb_dir}")
    n_frames = len(rgb_files)

    sample = cv2.imread(str(rgb_files[0]))
    image_size = (sample.shape[1], sample.shape[0])
    calib = load_calibration(
        rec / "calibration" / "caliscope_camera_array.toml", image_size,
    )
    print(f"[render] {n_frames} frames; voxel={args.voxel_size}m, min_pts={args.min_pts}")

    Wp, Hp = args.pano_w, args.pano_h
    bev_h = int(args.bev_forward / args.bev_resolution)
    bev_w = int(args.bev_lateral / args.bev_resolution)

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    end = n_frames if args.max_frames <= 0 else min(n_frames, args.start_frame + args.max_frames)
    indices = range(args.start_frame, end, args.frame_stride)

    t0 = time.time()
    n_done = 0
    for i in indices:
        rgb, depth = load_frame(rec, i)

        pts, cols, cids = backproject_all(
            rgb, depth, calib,
            stride=args.stride,
            depth_min=args.depth_min, depth_max=args.depth_max,
        )
        pts_v, cols_v, _, _ = voxel_consensus_filter(
            pts, cols, cids,
            voxel_size=args.voxel_size,
            min_points_per_voxel=args.min_pts,
            require_min_cams=args.require_min_cams,
        )

        pano_rgb, pano_d = project_cloud_to_pano(
            pts_v, cols_v,
            pano_size=(Wp, Hp),
            fov_h_deg=args.fov_h_deg, fov_v_deg=args.fov_v_deg,
            depth_max=args.depth_max,
            point_radius_px=args.splat_radius,
        )
        pano_d_color = colorize_depth(pano_d,
                                       d_min=args.depth_min,
                                       d_max=args.depth_max)
        bev = project_cloud_to_bev(
            pts_v, cols_v,
            forward_m=args.bev_forward,
            lateral_m=args.bev_lateral,
            resolution_m=args.bev_resolution,
        )

        # 3-row layout (all Wp wide):
        bev_resized_h = int(bev_h * (Wp / bev_w))
        bev_resized = cv2.resize(bev, (Wp, bev_resized_h))
        composed = np.vstack([pano_rgb, pano_d_color, bev_resized])
        cv2.putText(composed, f"frame {i:5d}  pts={len(pts_v):,}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            h, w = composed.shape[:2]
            writer = cv2.VideoWriter(str(args.out), fourcc, 20.0, (w, h))
            print(f"  output size {w}x{h}")
        writer.write(composed)

        n_done += 1
        if n_done % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {n_done} frames  {n_done/elapsed:.1f} fps  pts={len(pts_v):,}")

    if writer is not None:
        writer.release()
    elapsed = time.time() - t0
    print(f"[render] saved {args.out}  ({n_done} frames, {n_done/elapsed:.1f} fps avg)")


if __name__ == "__main__":
    sys.exit(main())
