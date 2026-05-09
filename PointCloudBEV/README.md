# PointCloudBEV

Multi-camera 3D point-cloud fusion for the OAK-D 3-camera trike rig.

Backprojects each camera's depth map into 3D using its intrinsics + extrinsics
(from a `caliscope_camera_array.toml`), fuses everything into one body-frame
point cloud, voxel-filters to deduplicate + remove ghost particles, then
re-projects the clean cloud back to:

- a **cylindrical RGB panorama** (z-buffered, ~120° forward),
- a **depth panorama** (body-frame distance from the trike origin), and
- a **top-down BEV** image (ground band + obstacles).

Why fuse in 3D rather than stitching depth maps in 2D pano space:

- The same world point seen by two cameras gives **different camera-Z values**
  (each cam measures along its own optical axis). A naive consistency check
  on those Z values rejects real surfaces. Fusing in 3D body coordinates is
  the proper apples-to-apples comparison.
- Voxel-grid binning **deduplicates** overlapping cam contributions: points
  from cam0 and cam1 hitting the same world cell collapse into a single
  output point.
- Setting `min_points_per_voxel ≥ 2` automatically discards isolated
  salt-and-pepper depth noise — the dominant ghost-particle source.

## Files

| File | What it does |
|---|---|
| `pointcloud.py` | Self-contained module: calibration loader, frame loader, backprojection, voxel-consensus filter, cloud→panorama, cloud→BEV. |
| `test_pointcloud.py` | Single-frame tester. Writes BEV (raw + clean) and a clean RGB+depth panorama to `/tmp/pointcloud_test/`. |
| `render_pointcloud_video.py` | Multi-frame MP4 renderer. 3-row output: pano RGB, pano depth, BEV. |

## Dependencies

- Python 3.11+ (uses `tomllib` from the stdlib)
- `numpy`, `opencv-python`

That's it. No torch, no open3d, no ROS.

## Expected recording layout

```
<recording_root>/
├── calibration/
│   └── caliscope_camera_array.toml
├── cam0/
│   ├── rgb/000000.png, 000001.png, ...
│   └── depth/000000.npy, 000001.npy, ...    # uint16, mm
├── cam1/
│   ├── rgb/...
│   └── depth/...
└── cam2/
    ├── rgb/...
    └── depth/...
```

`cam0` is the left camera, `cam1` the center, `cam2` the right.

## Quick start

```bash
# Single frame sanity check — writes outputs to /tmp/pointcloud_test/
python test_pointcloud.py \
    --recording /path/to/recording \
    --frame 1754

# Multi-frame MP4 — pano RGB + depth + BEV stacked, 3 rows
python render_pointcloud_video.py \
    --recording /path/to/recording \
    --start-frame 1500 \
    --max-frames 600 \
    --frame-stride 2 \
    --out /tmp/pointcloud.mp4
```

## Pipeline flags worth knowing

| Flag | Default | What it does |
|---|---|---|
| `--stride` | 2 | Pixel stride when backprojecting depth. Higher = fewer points = faster, less detail. |
| `--depth-min` / `--depth-max` | 0.4 / 12.0 m | Hard range gate on raw camera-Z values. |
| `--voxel-size` | 0.06 m | Voxel cube side. Smaller = sharper, less dedup; larger = cleaner, blurrier. |
| `--min-pts` | 3 | Minimum input points per voxel to accept. **Main ghost-particle knob.** |
| `--require-min-cams` | 1 | Min distinct cameras per voxel. Set to 2 for strict overlap-only fusion (loses a lot of points if extrinsics aren't sub-cm accurate). |
| `--splat-radius` | 2 px | Disc radius around each projected point in the panorama (denser visual). |
| `--pano-w / --pano-h` | 1500 × 540 | Panorama resolution. |
| `--fov-h-deg / --fov-v-deg` | 130 / 45 | Cylindrical-pano FOV. Match `--fov-v-deg` to the cameras' actual vertical FOV (≈45° for OAK-D 640×400). |
| `--bev-forward / --bev-lateral` | 15 / 12 m | BEV viewport size. |
| `--bev-resolution` | 0.04 m/px | BEV cell size. |

## Geometry conventions used here

- Body frame: **X = right, Y = down, Z = forward** (matches camera 1's
  natural frame on the OAK-D rig).
- Caliscope's `R, t` are **body→camera** (OpenCV convention `P_cam = R·P_world + t`).
- Backprojection: `P_cam = ((u−cx)/fx · Z, (v−cy)/fy · Z, Z)`
- Body-frame transform: `P_body = R^T · (P_cam − t)`
- Cylindrical-pano azimuth: `atan2(x, z)` (0 = forward, +X = right).
- Cylindrical-pano elevation: `asin(−y / |P|)` (Y down → +elevation = up).

## Per-frame timing (laptop CPU, ~150k input points)

| Stage | Time |
|---|---|
| Backproject 3 cams (stride=2) | ~13 ms |
| Voxel filter (np.unique) | ~120 ms |
| Pano + BEV render | ~25 ms |
| **Total** | **~160 ms ⇒ 6 fps** |

The voxel-filter is the bottleneck on CPU. A native-array hash or GPU port
gets this to 15-20 fps on a Jetson-class device with no algorithmic changes.

## Memory

Per-frame peak working-set ≈ 80 MB (3 RGB + 3 depth maps + ~150 k float32 points).
No accumulated state across frames.
