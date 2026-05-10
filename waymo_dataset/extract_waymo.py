"""
Waymo Open Dataset v2 — extract parquet segments to usable files.

Output layout per segment:
  <out_root>/<segment_id>/
    images/           JPEG per frame per camera
    lidar/            .npy per frame, TOP lidar only, shape (N, 4) x y z intensity
    labels/
      boxes_3d.csv    3D bounding boxes, vehicle frame, all frames
      boxes_2d.csv    2D bounding boxes per camera, all frames
      ego_pose.csv    4×4 world-from-vehicle transform per frame timestamp
      calibration/    camera_<name>.json and lidar_<name>.json
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}
LASER_NAMES  = {1: "TOP",   2: "FRONT",      3: "SIDE_LEFT",   4: "SIDE_RIGHT", 5: "REAR"}
CLASS_NAMES  = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "SIGN", 4: "CYCLIST"}


# ─── LiDAR range-image → point cloud ─────────────────────────────────────────

def _decode_range_image(values: np.ndarray, shape: list,
                        beam_inclinations: np.ndarray,
                        extrinsic: np.ndarray) -> np.ndarray:
    """Convert a range image to an (N, 4) array of [x, y, z, intensity] in vehicle frame."""
    H, W = int(shape[0]), int(shape[1])
    ri = values.reshape(H, W, -1)          # [..., 0]=range [..., 1]=intensity

    # azimuth: col 0 → +π, last col → -π (counterclockwise when viewed from above)
    col_idx = np.arange(W, dtype=np.float32)
    azimuth = np.pi - (col_idx + 0.5) / W * 2.0 * np.pi  # (W,)

    # inclination per row (bottom row = most negative)
    inclination = beam_inclinations[:H]                    # (H,)

    # broadcast to (H, W)
    az  = np.broadcast_to(azimuth[None, :],      (H, W))
    inc = np.broadcast_to(inclination[:, None],  (H, W))
    r   = ri[..., 0]
    intensity = ri[..., 1]

    valid = r > 0
    r   = r[valid];   az = az[valid];  inc = inc[valid];  intensity = intensity[valid]

    # sensor-frame XYZ
    x = r * np.cos(inc) * np.cos(az)
    y = r * np.cos(inc) * np.sin(az)
    z = r * np.sin(inc)

    pts_sensor = np.stack([x, y, z, np.ones_like(x)], axis=1)  # (N, 4)

    # rotate to vehicle frame via extrinsic (sensor→vehicle)
    R = extrinsic[:3, :3]
    t = extrinsic[:3,  3]
    pts_vehicle = (R @ pts_sensor[:, :3].T).T + t

    return np.column_stack([pts_vehicle, intensity]).astype(np.float32)


def extract_lidar(seg_dir: Path, out_dir: Path):
    lidar_df  = pd.read_parquet(seg_dir / "lidar.parquet")
    calib_df  = pd.read_parquet(seg_dir / "lidar_calibration.parquet")

    # build calibration dict keyed by laser_name
    calib = {}
    for _, row in calib_df.iterrows():
        name = int(row["key.laser_name"])
        ext  = np.array(row["[LiDARCalibrationComponent].extrinsic.transform"]).reshape(4, 4)
        inc_vals = np.array(row["[LiDARCalibrationComponent].beam_inclination.values"])
        calib[name] = {"extrinsic": ext, "beam_inclinations": inc_vals}

    # extract TOP lidar only (name=1) — largest, most useful for BEV
    top = lidar_df[lidar_df["key.laser_name"] == 1]
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in top.iterrows():
        ts   = int(row["key.frame_timestamp_micros"])
        vals = np.array(row["[LiDARComponent].range_image_return1.values"], dtype=np.float32)
        shp  = list(row["[LiDARComponent].range_image_return1.shape"])
        pts  = _decode_range_image(vals, shp, calib[1]["beam_inclinations"], calib[1]["extrinsic"])
        np.save(out_dir / f"{ts}.npy", pts)

    print(f"  lidar: {len(top)} frames → {out_dir}")


# ─── Camera images ────────────────────────────────────────────────────────────

def extract_images(seg_dir: Path, out_dir: Path):
    df = pd.read_parquet(seg_dir / "camera_image.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        ts   = int(row["key.frame_timestamp_micros"])
        cam  = CAMERA_NAMES.get(int(row["key.camera_name"]), str(row["key.camera_name"]))
        img_bytes = bytes(row["[CameraImageComponent].image"])
        path = out_dir / f"{ts}_{cam}.jpg"
        path.write_bytes(img_bytes)

    print(f"  images: {len(df)} files → {out_dir}")


# ─── 3D boxes ─────────────────────────────────────────────────────────────────

def extract_boxes_3d(seg_dir: Path, out_dir: Path):
    df = pd.read_parquet(seg_dir / "lidar_box.parquet")
    pfx = "[LiDARBoxComponent]."
    out = pd.DataFrame({
        "timestamp_us":  df["key.frame_timestamp_micros"],
        "object_id":     df["key.laser_object_id"],
        "class":         df[pfx + "type"].map(CLASS_NAMES),
        "cx":            df[pfx + "box.center.x"],
        "cy":            df[pfx + "box.center.y"],
        "cz":            df[pfx + "box.center.z"],
        "length":        df[pfx + "box.size.x"],
        "width":         df[pfx + "box.size.y"],
        "height":        df[pfx + "box.size.z"],
        "heading_rad":   df[pfx + "box.heading"],
        "vx":            df[pfx + "speed.x"],
        "vy":            df[pfx + "speed.y"],
        "n_lidar_pts":   df[pfx + "num_lidar_points_in_box"],
        "det_difficulty":df[pfx + "difficulty_level.detection"],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "boxes_3d.csv", index=False)
    print(f"  boxes_3d: {len(out)} rows → {out_dir / 'boxes_3d.csv'}")


# ─── 2D boxes ─────────────────────────────────────────────────────────────────

def extract_boxes_2d(seg_dir: Path, out_dir: Path):
    df = pd.read_parquet(seg_dir / "camera_box.parquet")
    pfx = "[CameraBoxComponent]."
    out = pd.DataFrame({
        "timestamp_us": df["key.frame_timestamp_micros"],
        "camera":       df["key.camera_name"].map(CAMERA_NAMES),
        "object_id":    df["key.camera_object_id"],
        "class":        df[pfx + "type"].map(CLASS_NAMES),
        "cx_px":        df[pfx + "box.center.x"],
        "cy_px":        df[pfx + "box.center.y"],
        "w_px":         df[pfx + "box.size.x"],
        "h_px":         df[pfx + "box.size.y"],
        "det_difficulty":df[pfx + "difficulty_level.detection"],
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "boxes_2d.csv", index=False)
    print(f"  boxes_2d: {len(out)} rows → {out_dir / 'boxes_2d.csv'}")


# ─── Ego pose ─────────────────────────────────────────────────────────────────

def extract_ego_pose(seg_dir: Path, out_dir: Path):
    df = pd.read_parquet(seg_dir / "vehicle_pose.parquet")
    rows = []
    for _, row in df.iterrows():
        T = np.array(row["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(4, 4)
        rows.append({
            "timestamp_us": int(row["key.frame_timestamp_micros"]),
            **{f"T{i}{j}": T[i, j] for i in range(4) for j in range(4)},
        })
    out_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "ego_pose.csv", index=False)
    print(f"  ego_pose: {len(out_df)} frames → {out_dir / 'ego_pose.csv'}")


# ─── Calibration ──────────────────────────────────────────────────────────────

def extract_calibration(seg_dir: Path, out_dir: Path):
    cam_df   = pd.read_parquet(seg_dir / "camera_calibration.parquet")
    lidar_df = pd.read_parquet(seg_dir / "lidar_calibration.parquet")
    cal_dir  = out_dir / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    for _, row in cam_df.iterrows():
        name = CAMERA_NAMES.get(int(row["key.camera_name"]), str(row["key.camera_name"]))
        pfx  = "[CameraCalibrationComponent]."
        ext  = np.array(row[pfx + "extrinsic.transform"]).reshape(4, 4).tolist()
        data = {
            "name": name,
            "width":  int(row[pfx + "width"]),
            "height": int(row[pfx + "height"]),
            "intrinsics": {
                "f_u": float(row[pfx + "intrinsic.f_u"]),
                "f_v": float(row[pfx + "intrinsic.f_v"]),
                "c_u": float(row[pfx + "intrinsic.c_u"]),
                "c_v": float(row[pfx + "intrinsic.c_v"]),
                "k1":  float(row[pfx + "intrinsic.k1"]),
                "k2":  float(row[pfx + "intrinsic.k2"]),
                "k3":  float(row[pfx + "intrinsic.k3"]),
                "p1":  float(row[pfx + "intrinsic.p1"]),
                "p2":  float(row[pfx + "intrinsic.p2"]),
            },
            "extrinsic_vehicle_to_camera": ext,   # 4×4 row-major
        }
        (cal_dir / f"camera_{name}.json").write_text(json.dumps(data, indent=2))

    for _, row in lidar_df.iterrows():
        name = LASER_NAMES.get(int(row["key.laser_name"]), str(row["key.laser_name"]))
        pfx  = "[LiDARCalibrationComponent]."
        ext  = np.array(row[pfx + "extrinsic.transform"]).reshape(4, 4).tolist()
        data = {
            "name": name,
            "extrinsic_vehicle_to_lidar": ext,
            "beam_inclination_min": float(row[pfx + "beam_inclination.min"]),
            "beam_inclination_max": float(row[pfx + "beam_inclination.max"]),
        }
        (cal_dir / f"lidar_{name}.json").write_text(json.dumps(data, indent=2))

    print(f"  calibration: {len(cam_df)} cameras, {len(lidar_df)} LiDARs → {cal_dir}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def extract_segment(seg_dir: str, out_root: str):
    seg_dir  = Path(seg_dir)
    seg_id   = seg_dir.name
    out_root = Path(out_root) / seg_id
    print(f"\n{'='*60}\nExtracting: {seg_id}\n{'='*60}")

    extract_images   (seg_dir, out_root / "images")
    extract_lidar    (seg_dir, out_root / "lidar")
    extract_boxes_3d (seg_dir, out_root / "labels")
    extract_boxes_2d (seg_dir, out_root / "labels")
    extract_ego_pose (seg_dir, out_root / "labels")
    extract_calibration(seg_dir, out_root / "labels")

    print(f"\nDone → {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segs",     nargs="+", required=True, help="Paths to segment parquet dirs")
    parser.add_argument("--out",      default="/home/dhruvil/BEV/dataset/waymo_extracted",
                        help="Output root directory")
    args = parser.parse_args()

    for seg in args.segs:
        extract_segment(seg, args.out)
