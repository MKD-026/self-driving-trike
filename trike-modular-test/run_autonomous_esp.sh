#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${TRIKE_PYTHON:-/home/h2x/miniconda3/envs/yolo/bin/python}"
oak_site="/home/h2x/miniconda3/envs/oak/lib/python3.10/site-packages"

export PYTHONPATH="$oak_site${PYTHONPATH:+:$PYTHONPATH}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/h2x-ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
exec "$python_bin" "$project_dir/autonomous_driving_esp.py" "$@"
