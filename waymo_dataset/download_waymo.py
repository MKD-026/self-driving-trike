"""
Download Waymo Open Dataset v2 segments from Google Cloud Storage.

Requirements:
  - Google Cloud SDK (gsutil): https://cloud.google.com/sdk/docs/install
  - Run `gcloud auth login` with a Waymo-approved Google account
  - Accept Waymo ToS at https://waymo.com/open before running

Usage:
  # List available validation segments
  python download_waymo.py --list --split validation

  # Download 2 specific segments (all ground-truth components)
  python download_waymo.py --segments 10203656353524179475_7625_000_7645_000 \\
                                       1024360143612057520_3580_000_3600_000 \\
                           --out /home/dhruvil/BEV/WaymoDataset/raw

  # Download N random segments from validation split
  python download_waymo.py --random 3 --split validation \\
                           --out /home/dhruvil/BEV/WaymoDataset/raw
"""

import argparse
import subprocess
import random
import sys
from pathlib import Path

BUCKET   = "gs://waymo_open_dataset_v_2_0_0"
GSUTIL   = "gsutil"   # override with full path if needed, e.g. ~/google-cloud-sdk/bin/gsutil

# Components to download — covers all ground truths needed for BEV testing
COMPONENTS = [
    "camera_image",
    "lidar",
    "lidar_box",
    "camera_box",
    "vehicle_pose",
    "camera_calibration",
    "lidar_calibration",
    "projected_lidar_box",
]


def run(cmd: list[str]):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def list_segments(split: str) -> list[str]:
    raw = run([GSUTIL, "ls", f"{BUCKET}/{split}/lidar_box/"])
    segs = []
    for line in raw.splitlines():
        name = line.split("/")[-1].replace(".parquet", "")
        if name:
            segs.append(name)
    return segs


def download_segment(seg: str, split: str, out_root: Path):
    seg_dir = out_root / seg
    seg_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for comp in COMPONENTS:
        src  = f"{BUCKET}/{split}/{comp}/{seg}.parquet"
        dst  = str(seg_dir / f"{comp}.parquet")
        procs.append(subprocess.Popen([GSUTIL, "cp", src, dst],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE))

    errors = []
    for p, comp in zip(procs, COMPONENTS):
        _, stderr = p.communicate()
        if p.returncode != 0:
            errors.append(f"  {comp}: {stderr.decode().strip()}")

    if errors:
        print(f"[WARN] Errors for {seg}:")
        print("\n".join(errors))
    else:
        print(f"  ✓ {seg}")


def main():
    parser = argparse.ArgumentParser(description="Download Waymo Open Dataset v2 segments")
    parser.add_argument("--split",    default="validation", choices=["training", "validation", "testing"])
    parser.add_argument("--out",      default="./raw",      help="Output directory for raw parquet files")
    parser.add_argument("--segments", nargs="+",            help="Explicit segment IDs to download")
    parser.add_argument("--random",   type=int,             help="Download N random segments")
    parser.add_argument("--list",     action="store_true",  help="List available segments and exit")
    parser.add_argument("--gsutil",   default=GSUTIL,       help="Path to gsutil binary")
    args = parser.parse_args()

    global GSUTIL
    GSUTIL = args.gsutil

    if args.list:
        segs = list_segments(args.split)
        print(f"\n{len(segs)} segments in {args.split}:\n")
        for s in segs:
            print(f"  {s}")
        return

    segs = args.segments or []
    if args.random:
        all_segs = list_segments(args.split)
        segs = random.sample(all_segs, min(args.random, len(all_segs)))
        print(f"Selected {len(segs)} random segments from {args.split}")

    if not segs:
        parser.error("Provide --segments, --random, or --list")

    out = Path(args.out)
    print(f"\nDownloading {len(segs)} segment(s) → {out}\n")
    for seg in segs:
        print(f"Downloading {seg} ...")
        download_segment(seg, args.split, out)

    print(f"\nDone. Raw parquets in: {out}")
    print(f"Next step: python extract_waymo.py --segs {out}/* --out ./extracted")


if __name__ == "__main__":
    main()
