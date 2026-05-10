# Waymo Open Dataset — Download & Extract Pipeline

End-to-end scripts to download, extract, and visualise Waymo Open Dataset v2 segments for autonomous driving algorithm testing.

---

## Contents

| File | Purpose |
|---|---|
| `download_waymo.py` | Download raw `.parquet` files from GCS |
| `extract_waymo.py` | Extract parquets → images, LiDAR numpy, CSV labels |
| `visualise_segment.py` | Interactive BEV + camera viewer per frame |

---

## Prerequisites

### 1 — Google Cloud SDK (gsutil)

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud auth login    # use your Waymo-approved Google account
```

> Accept the Waymo Terms of Service first at **https://waymo.com/open** → "Get the Data"

### 2 — Python dependencies (in your conda env)

```bash
pip install pyarrow pandas numpy opencv-python matplotlib
```

---

## Step 1 — Download segments

### List available segments

```bash
python download_waymo.py --list --split validation
```

### Download specific segments by ID

```bash
python download_waymo.py \
  --segments 10203656353524179475_7625_000_7645_000 \
             1024360143612057520_3580_000_3600_000 \
  --split validation \
  --out ./WaymoDataset/raw
```

### Download N random segments

```bash
python download_waymo.py --random 3 --split validation --out ./WaymoDataset/raw
```

### If gsutil is not on your PATH

```bash
python download_waymo.py --gsutil ~/google-cloud-sdk/bin/gsutil \
  --random 2 --out ./WaymoDataset/raw
```

Each segment is **~440–520 MB** (5 camera streams + TOP LiDAR + all labels).

---

## Step 2 — Extract to usable format

```bash
python extract_waymo.py \
  --segs ./WaymoDataset/raw/10203656353524179475_7625_000_7645_000 \
         ./WaymoDataset/raw/1024360143612057520_3580_000_3600_000 \
  --out  ./WaymoDataset/extracted
```

### Output layout per segment

```
WaymoDataset/extracted/<segment_id>/
│
├── images/
│   └── {timestamp_us}_{CAMERA}.jpg     # FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT
│                                        # ~990–1000 files per segment (5 cams × ~200 frames)
│
├── lidar/
│   └── {timestamp_us}.npy              # shape (N, 4) — x, y, z, intensity (vehicle frame)
│                                        # TOP LiDAR only, ~200 files per segment
│
└── labels/
    ├── boxes_3d.csv                     # 3D bounding boxes (all frames)
    ├── boxes_2d.csv                     # 2D bounding boxes per camera (all frames)
    ├── ego_pose.csv                     # 4×4 world-from-vehicle transform per frame
    └── calibration/
        ├── camera_FRONT.json            # intrinsics + extrinsic (vehicle → camera)
        ├── camera_FRONT_LEFT.json
        ├── camera_FRONT_RIGHT.json
        ├── camera_SIDE_LEFT.json
        ├── camera_SIDE_RIGHT.json
        ├── lidar_TOP.json               # extrinsic (vehicle → LiDAR) + beam inclinations
        ├── lidar_FRONT.json
        ├── lidar_SIDE_LEFT.json
        ├── lidar_SIDE_RIGHT.json
        └── lidar_REAR.json
```

---

## Ground truth schema

### `labels/boxes_3d.csv`

| Column | Description |
|---|---|
| `timestamp_us` | Frame timestamp in microseconds |
| `object_id` | Persistent track ID (consistent across frames) |
| `class` | `VEHICLE` / `PEDESTRIAN` / `CYCLIST` / `SIGN` |
| `cx, cy, cz` | Box centre in **vehicle frame** (metres) |
| `length, width, height` | Box dimensions (metres) |
| `heading_rad` | Yaw angle in vehicle frame (radians) |
| `vx, vy` | Velocity in vehicle frame (m/s) |
| `n_lidar_pts` | LiDAR points inside the box |
| `det_difficulty` | Detection difficulty level (0 = easy) |

### `labels/boxes_2d.csv`

| Column | Description |
|---|---|
| `timestamp_us` | Frame timestamp |
| `camera` | Camera name (`FRONT`, `FRONT_LEFT`, …) |
| `object_id` | Track ID |
| `class` | Object class |
| `cx_px, cy_px` | Box centre in pixels |
| `w_px, h_px` | Box size in pixels |

### `labels/ego_pose.csv`

| Column | Description |
|---|---|
| `timestamp_us` | Frame timestamp |
| `T00…T33` | 4×4 world-from-vehicle transform (row-major) |

Reconstruct the 4×4 matrix in Python:
```python
import numpy as np, pandas as pd
pose = pd.read_csv("labels/ego_pose.csv")
row  = pose.iloc[0]
T    = np.array([row[f"T{i}{j}"] for i in range(4) for j in range(4)]).reshape(4, 4)
```

### `labels/calibration/camera_FRONT.json`

```json
{
  "name": "FRONT",
  "width": 1920,
  "height": 1280,
  "intrinsics": { "f_u": ..., "f_v": ..., "c_u": ..., "c_v": ...,
                  "k1": ..., "k2": ..., "k3": ..., "p1": ..., "p2": ... },
  "extrinsic_vehicle_to_camera": [[...4×4 row-major...]]
}
```

---

## Step 3 — Visualise a segment

### Interactive (keyboard navigation)

```bash
python visualise_segment.py \
  --seg ./WaymoDataset/extracted/10203656353524179475_7625_000_7645_000
```

Keys: `→` next frame · `←` previous frame · `Q` quit

### Save all frames as PNGs

```bash
python visualise_segment.py \
  --seg  ./WaymoDataset/extracted/10203656353524179475_7625_000_7645_000 \
  --save ./vis/seg1
```

Each saved frame shows the FRONT camera with 2D box overlays (left) and a BEV LiDAR map with 3D box footprints + ego trail (right).

---

## Quick Python usage

```python
import numpy as np
import pandas as pd
from pathlib import Path

seg = Path("WaymoDataset/extracted/10203656353524179475_7625_000_7645_000")

# Load all 3D boxes for this segment
boxes = pd.read_csv(seg / "labels" / "boxes_3d.csv")

# Get one frame's vehicle boxes
ts = boxes["timestamp_us"].unique()[0]
frame_vehicles = boxes[(boxes["timestamp_us"] == ts) & (boxes["class"] == "VEHICLE")]
print(frame_vehicles[["cx", "cy", "heading_rad", "length", "width"]])

# Load LiDAR point cloud for that frame
pts = np.load(seg / "lidar" / f"{ts}.npy")   # (N, 4) — x, y, z, intensity
bev_pts = pts[(pts[:, 2] > -2) & (pts[:, 2] < 1)]  # rough ground-plane slice

# Load front camera image
import cv2
img = cv2.imread(str(seg / "images" / f"{ts}_FRONT.jpg"))
```

---

## Notes

- The dataset uses **vehicle frame** coordinates: X forward, Y left, Z up.
- `heading_rad` is the yaw of the object in the vehicle frame.
- All 5 camera images are stored per frame. Timestamps align with LiDAR frames.
- Only the **TOP LiDAR** is decoded (64-beam, 360°). The 4 side LiDARs are in the raw parquets but not extracted by default.
- The raw parquet files (`WaymoDataset/raw/`) are excluded from git via `.gitignore`.

---

## Tested environment

- Python 3.11, conda env `bev_nav`
- Packages: `pyarrow`, `pandas`, `numpy`, `opencv-python`, `matplotlib`
- Dataset: Waymo Open Dataset v2.0, validation split
