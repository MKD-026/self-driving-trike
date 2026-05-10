"""
Quick sanity-check visualiser for an extracted Waymo segment.

Shows one frame at a time:
  - Front camera image with 2D box overlays
  - BEV plot of LiDAR points + 3D box footprints + ego trail

Usage:
  python visualise_segment.py --seg /home/dhruvil/BEV/WaymoDataset/10203656353524179475_7625_000_7645_000
  python visualise_segment.py --seg <path> --frame 50   # jump to frame index
  python visualise_segment.py --seg <path> --save ./vis  # save PNGs instead of displaying
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASS_COLORS = {
    "VEHICLE":    (0, 200, 255),
    "PEDESTRIAN": (0, 255, 100),
    "CYCLIST":    (255, 160, 0),
    "SIGN":       (180, 0, 255),
}
BEV_COLORS = {
    "VEHICLE":    "cyan",
    "PEDESTRIAN": "lime",
    "CYCLIST":    "orange",
    "SIGN":       "violet",
}


def load_segment(seg_dir: Path):
    boxes3d = pd.read_csv(seg_dir / "labels" / "boxes_3d.csv")
    boxes2d = pd.read_csv(seg_dir / "labels" / "boxes_2d.csv")
    pose    = pd.read_csv(seg_dir / "labels" / "ego_pose.csv")
    ts_list = sorted(boxes3d["timestamp_us"].unique())
    return boxes3d, boxes2d, pose, ts_list


def draw_2d_boxes(img: np.ndarray, boxes: pd.DataFrame) -> np.ndarray:
    img = img.copy()
    for _, r in boxes.iterrows():
        cls   = str(r.get("class", ""))
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        x1    = int(r["cx_px"] - r["w_px"] / 2)
        y1    = int(r["cy_px"] - r["h_px"] / 2)
        x2    = int(r["cx_px"] + r["w_px"] / 2)
        y2    = int(r["cy_px"] + r["h_px"] / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, cls[:3], (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return img


def bev_box_corners(cx, cy, l, w, heading):
    """Return 4 corners of a BEV box as (4,2) array."""
    corners = np.array([[ l/2,  w/2],
                        [-l/2,  w/2],
                        [-l/2, -w/2],
                        [ l/2, -w/2]], dtype=np.float32)
    c, s = math.cos(heading), math.sin(heading)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (R @ corners.T).T + np.array([cx, cy])


def plot_bev(ax, lidar_pts, boxes3d_frame, pose_df, ts_list, frame_idx):
    ax.set_facecolor("#111111")
    ax.set_aspect("equal")

    # LiDAR ground plane (z filter)
    if lidar_pts is not None:
        mask = (lidar_pts[:, 2] > -3) & (lidar_pts[:, 2] < 3)
        pts  = lidar_pts[mask]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c="white", alpha=0.4, linewidths=0)

    # ego trail
    trail_ts = ts_list[max(0, frame_idx - 30): frame_idx + 1]
    trail    = pose_df[pose_df["timestamp_us"].isin(trail_ts)]
    xs = trail["T03"].values
    ys = trail["T13"].values
    if len(xs) > 1:
        ax.plot(xs, ys, color="yellow", linewidth=1.5, zorder=3)

    # current ego position
    cur = pose_df[pose_df["timestamp_us"] == ts_list[frame_idx]]
    if len(cur):
        ax.plot(cur["T03"].iloc[0], cur["T13"].iloc[0],
                "y*", markersize=10, zorder=4)

    # 3D box footprints
    for _, r in boxes3d_frame.iterrows():
        cls    = str(r.get("class", ""))
        color  = BEV_COLORS.get(cls, "white")
        corners = bev_box_corners(r["cx"], r["cy"], r["length"], r["width"], r["heading_rad"])
        poly    = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(poly)

    ax.set_xlim(-60, 60)
    ax.set_ylim(-40, 40)
    ax.set_title(f"BEV  frame {frame_idx}", color="white", fontsize=9)
    ax.tick_params(colors="gray", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


def visualise(seg_dir: Path, start_frame: int, save_dir: Path | None):
    boxes3d, boxes2d, pose_df, ts_list = load_segment(seg_dir)
    n_frames = len(ts_list)
    print(f"Segment has {n_frames} frames. Press → / ← to navigate, Q to quit.")

    matplotlib.use("Agg" if save_dir else "TkAgg")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a1a")

    frame_idx = [max(0, min(start_frame, n_frames - 1))]

    def render(idx):
        for ax in axes:
            ax.cla()

        ts = ts_list[idx]

        # --- camera image (FRONT) ---
        img_path = seg_dir / "images" / f"{ts}_FRONT.jpg"
        if img_path.exists():
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            boxes2d_frame = boxes2d[(boxes2d["timestamp_us"] == ts) &
                                    (boxes2d["camera"] == "FRONT")]
            img = draw_2d_boxes(img, boxes2d_frame)
            axes[0].imshow(img)
        else:
            axes[0].set_facecolor("#222")
            axes[0].text(0.5, 0.5, "No FRONT image", ha="center", va="center",
                         color="white", transform=axes[0].transAxes)
        axes[0].axis("off")
        axes[0].set_title(f"FRONT camera  ts={ts}", color="white", fontsize=9)

        # --- BEV ---
        lidar_path = seg_dir / "lidar" / f"{ts}.npy"
        lidar_pts  = np.load(lidar_path) if lidar_path.exists() else None
        boxes3d_f  = boxes3d[boxes3d["timestamp_us"] == ts]
        plot_bev(axes[1], lidar_pts, boxes3d_f, pose_df, ts_list, idx)

        plt.tight_layout(pad=0.5)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"frame_{idx:04d}.png", dpi=120, bbox_inches="tight")
        else:
            fig.canvas.draw()

    if save_dir:
        for i in range(n_frames):
            print(f"\r  saving frame {i+1}/{n_frames}", end="", flush=True)
            render(i)
        print(f"\nSaved to {save_dir}")
        return

    render(frame_idx[0])
    plt.show(block=False)

    def on_key(event):
        if event.key == "right":
            frame_idx[0] = min(frame_idx[0] + 1, n_frames - 1)
        elif event.key == "left":
            frame_idx[0] = max(frame_idx[0] - 1, 0)
        elif event.key in ("q", "escape"):
            plt.close("all")
            return
        render(frame_idx[0])
        fig.canvas.draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg",   required=True, help="Path to extracted segment directory")
    parser.add_argument("--frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--save",  default=None,   help="Save frames to this directory instead of displaying")
    args = parser.parse_args()
    visualise(Path(args.seg), args.frame, Path(args.save) if args.save else None)
