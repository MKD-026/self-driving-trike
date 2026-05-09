#!/usr/bin/env python3
"""
Test the 3D point-cloud fusion pipeline on a single frame.

Outputs (in --out-dir):
    cam_strip.png            — 3 source RGBs side by side
    cloud_raw_bev.png        — top-down BEV of all backprojected points
    cloud_clean_bev.png      — BEV after voxel + multi-cam consensus filter
    cloud_clean_pano_rgb.png — re-projected cylindrical panorama from clean cloud
    cloud_clean_pano_d.png   — colorized depth pano from clean cloud
    cloud_clean_pano_combined.png — RGB + depth pano stacked

Run:
    python test_pointcloud.py \\
        --recording /path/to/recording \\
        --frame 1754
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
    p.add_argument("--frame", type=int, default=1754)
    p.add_argument("--stride", type=int, default=2,
                   help="Subsample stride when backprojecting (1 = every pixel)")
    p.add_argument("--depth-min", type=float, default=0.4)
    p.add_argument("--depth-max", type=float, default=12.0)
    p.add_argument("--voxel-size", type=float, default=0.05,
                   help="Voxel cube side in meters")
    p.add_argument("--min-pts", type=int, default=2,
                   help="Minimum points per voxel to keep")
    p.add_argument("--require-min-cams", type=int, default=1,
                   help="Min distinct cams contributing to a voxel "
                        "(2 = ghost killer for overlap regions, "
                        "1 = trust single-cam regions)")
    p.add_argument("--pano-w", type=int, default=1500)
    p.add_argument("--pano-h", type=int, default=540)
    p.add_argument("--fov-h-deg", type=float, default=130.0)
    p.add_argument("--fov-v-deg", type=float, default=45.0)
    p.add_argument("--bev-forward", type=float, default=15.0)
    p.add_argument("--bev-lateral", type=float, default=12.0)
    p.add_argument("--bev-resolution", type=float, default=0.04)
    p.add_argument("--out-dir", type=Path, default=Path("/tmp/pointcloud_test"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rgb, depth = load_frame(args.recording, args.frame)
    sample = next((v for v in rgb.values() if v is not None), None)
    if sample is None:
        sys.exit("no rgb frames found")
    image_size = (sample.shape[1], sample.shape[0])

    calib = load_calibration(
        args.recording / "calibration" / "caliscope_camera_array.toml",
        image_size,
    )

    # Source-cam strip for visual reference.
    strip = [rgb[c] for c in (0, 1, 2) if rgb.get(c) is not None]
    if strip:
        cv2.imwrite(str(args.out_dir / "cam_strip.png"), cv2.hconcat(strip))

    # ── Stage 1: backproject every cam. ─────────────────────────────────────
    t = time.time()
    pts_raw, cols_raw, cids_raw = backproject_all(
        rgb, depth, calib,
        stride=args.stride,
        depth_min=args.depth_min, depth_max=args.depth_max,
    )
    t_bp = (time.time() - t) * 1000
    print(f"[stage1] backprojected {len(pts_raw):,} points "
          f"(cam0={int((cids_raw==0).sum()):,}  "
          f"cam1={int((cids_raw==1).sum()):,}  "
          f"cam2={int((cids_raw==2).sum()):,})  in {t_bp:.0f} ms")

    # ── Stage 2: voxel filter (deduplicates + thins). ───────────────────────
    t = time.time()
    pts_v, cols_v, n_per_voxel, n_cams_per_voxel = voxel_consensus_filter(
        pts_raw, cols_raw, cids_raw,
        voxel_size=args.voxel_size,
        min_points_per_voxel=args.min_pts,
        require_min_cams=1,
    )
    t_v = (time.time() - t) * 1000
    print(f"[stage2] voxel filter ({args.voxel_size} m, min_pts={args.min_pts}): "
          f"{len(pts_raw):,} → {len(pts_v):,} points  ({t_v:.0f} ms)")
    if len(n_per_voxel):
        print(f"         pts/voxel: median={int(np.median(n_per_voxel))}  "
              f"max={int(n_per_voxel.max())}")

    # ── Stage 3: optional multi-cam consensus on the same voxel grid. ──────
    if args.require_min_cams >= 2:
        t = time.time()
        pts_c, cols_c, _, _ = voxel_consensus_filter(
            pts_raw, cols_raw, cids_raw,
            voxel_size=args.voxel_size,
            min_points_per_voxel=args.min_pts,
            require_min_cams=args.require_min_cams,
        )
        t_c = (time.time() - t) * 1000
        print(f"[stage3] multi-cam consensus (≥{args.require_min_cams} cams): "
              f"{len(pts_v):,} → {len(pts_c):,} points  ({t_c:.0f} ms)")
    else:
        pts_c = pts_v
        cols_c = cols_v

    # ── Renders. ────────────────────────────────────────────────────────────
    bev_raw = project_cloud_to_bev(pts_raw, cols_raw,
                                    forward_m=args.bev_forward,
                                    lateral_m=args.bev_lateral,
                                    resolution_m=args.bev_resolution)
    bev_clean = project_cloud_to_bev(pts_c, cols_c,
                                      forward_m=args.bev_forward,
                                      lateral_m=args.bev_lateral,
                                      resolution_m=args.bev_resolution)
    cv2.imwrite(str(args.out_dir / "cloud_raw_bev.png"), bev_raw)
    cv2.imwrite(str(args.out_dir / "cloud_clean_bev.png"), bev_clean)

    pano_rgb_clean, pano_d_clean = project_cloud_to_pano(
        pts_c, cols_c,
        pano_size=(args.pano_w, args.pano_h),
        fov_h_deg=args.fov_h_deg,
        fov_v_deg=args.fov_v_deg,
        depth_max=args.depth_max,
        point_radius_px=1,
    )
    pano_d_color = colorize_depth(pano_d_clean,
                                   d_min=args.depth_min, d_max=args.depth_max)
    cv2.imwrite(str(args.out_dir / "cloud_clean_pano_rgb.png"), pano_rgb_clean)
    cv2.imwrite(str(args.out_dir / "cloud_clean_pano_d.png"), pano_d_color)

    combined = np.vstack([pano_rgb_clean, pano_d_color])
    cv2.imwrite(str(args.out_dir / "cloud_clean_pano_combined.png"), combined)

    valid_pano = (pano_d_clean > 0).mean() * 100
    print(f"[render] pano depth coverage: {valid_pano:.1f}%")
    print(f"[render] outputs in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
