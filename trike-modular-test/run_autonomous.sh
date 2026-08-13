#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${TRIKE_PYTHON:-/home/h2x/miniconda3/envs/yolo/bin/python}"
oak_site="/home/h2x/miniconda3/envs/oak/lib/python3.10/site-packages"

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    echo "Set TRIKE_PYTHON to an interpreter with the project requirements." >&2
    exit 2
fi

if [[ -d "$oak_site" ]]; then
    export PYTHONPATH="$oak_site${PYTHONPATH:+:$PYTHONPATH}"
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/h2x-ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"
exec "$python_bin" "$project_dir/autonomous_driving.py" --dry-run "$@"

