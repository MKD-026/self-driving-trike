#!/usr/bin/env python3
"""
Multi-OAK dataset recorder for a 3-camera rig.

Captures every per-device sensor stream simultaneously from all three OAK-D
Pro devices, plus the aligned stereo depth, and saves a timestamped run
folder. Frames are software-synced by DepthAI timestamp.

Output layout
-------------
  dataset/run_YYYYMMDD_HHMMSS/
    metadata.json
    calibration/
      oak_<mxid>.json              OAK factory calibration (from flash)
      caliscope_camera_array.toml  Caliscope inter-camera calibration
    cam<0|1|2>/
      rgb/        NNNNNN.png    CAM_A  BGR 8-bit
      mono_left/  NNNNNN.png    CAM_B  gray 8-bit
      mono_right/ NNNNNN.png    CAM_C  gray 8-bit
      depth/      NNNNNN.npy    uint16 NumPy array, value = mm
    imu.csv                     one row per (frame_idx, cam_id): latest
                                IMU sample at the moment that frame was saved
    gps.csv                     one row per frame: latest phone GPS fix at
                                the moment that frame was saved
    phone_imu.csv               one row per frame: latest phone IMU sample at
                                the moment that frame was saved
    timestamps.csv               frame_idx,cam_id,stream,ts_us,rel_us

Cameras are pinned to cam_id by MXID (LEFT=0, CENTER=1, RIGHT=2).

Usage
-----
  python record_dataset.py
  python record_dataset.py --fps 15 --duration 30
  python record_dataset.py --root /data/oak_runs --width 1280 --height 800
"""
from __future__ import annotations

import argparse
import csv
import json
import socket
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import depthai as dai
import numpy as np


# ── Fixed MXID → cam_id mapping (same as multi_cam_bev.py / caliscope) ────────
MXID_TO_CID = {
    "1944301041B9714800": 0,   # LEFT
    "19443010C12D654800": 1,   # CENTER
    "19443010A109654800": 2,   # RIGHT
}
CID_ROLE = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}

# Caliscope inter-camera calibration candidates to archive alongside each run.
CALISCOPE_TOML_CANDIDATES = (
    Path(__file__).with_name("camera_array.toml"),
    Path("/home/dhruvil/BEV/caliscope_ws/camera_array.toml"),
)

STREAMS = ("rgb", "mono_left", "mono_right", "depth")

# IMU sample rates (Hz). BNO085 supports up to 400 Hz raw accel/gyro;
# rotation-vector fusion typically runs at 100 Hz.
IMU_ACCEL_HZ = 200
IMU_GYRO_HZ  = 200
IMU_ROTV_HZ  = 100


# ── Runtime state ─────────────────────────────────────────────────────────────
@dataclass
class StreamFrame:
    image:  np.ndarray
    ts_us:  int


@dataclass
class IMUSample:
    ts_us: int = 0
    ax: float = 0.0; ay: float = 0.0; az: float = 0.0
    gx: float = 0.0; gy: float = 0.0; gz: float = 0.0
    qi: float = 0.0; qj: float = 0.0; qk: float = 0.0; qr: float = 1.0


@dataclass
class CamRuntime:
    cid:   int
    mxid:  str
    info:  dai.DeviceInfo
    lock:  threading.Lock       = field(default_factory=threading.Lock)
    latest: dict[str, StreamFrame] = field(default_factory=dict)
    latest_imu: IMUSample | None  = None
    ready:  threading.Event      = field(default_factory=threading.Event)
    failed: bool = False
    error: str = ""


@dataclass
class PhoneSensorRuntime:
    port: int = 5000
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_gps: dict | None = None
    latest_imu: dict | None = None
    latest_gps_host_us: int = 0
    latest_imu_host_us: int = 0
    packet_count: int = 0
    connected: bool = False


_running = True


# ── Phone GPS/IMU receiver over ADB TCP forward ───────────────────────────────
def host_time_us() -> int:
    return int(time.time() * 1e6)


def adb_forward(port: int) -> bool:
    try:
        result = subprocess.run(
            ["adb", "forward", f"tcp:{port}", f"tcp:{port}"],
            capture_output=True,
            timeout=8,
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("[phone] adb not found; continuing without phone GPS/IMU")
        return False
    except subprocess.TimeoutExpired:
        return False


def start_phone_sensor_service() -> None:
    try:
        subprocess.run(
            [
                "adb",
                "shell",
                "am",
                "start-foreground-service",
                "-n",
                "com.h2x.sensor/.SensorService",
            ],
            capture_output=True,
            timeout=8,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def phone_sensor_thread(rt: PhoneSensorRuntime):
    """Receive newline-delimited phone GPS/IMU JSON over an ADB-forwarded TCP port."""
    if not adb_forward(rt.port):
        return
    start_phone_sensor_service()
    print(f"[phone] waiting for GPS/IMU stream on tcp:{rt.port}")

    while _running:
        adb_forward(rt.port)
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(("127.0.0.1", rt.port))
            sock.settimeout(1)
            with rt.lock:
                rt.connected = True
            print("[phone] connected")

            buf = ""
            while _running:
                try:
                    chunk = sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                *lines, buf = buf.split("\n")
                now_us = host_time_us()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    packet_type = data.get("type")
                    with rt.lock:
                        if packet_type == "GPS":
                            rt.latest_gps = data
                            rt.latest_gps_host_us = now_us
                        elif packet_type == "IMU":
                            rt.latest_imu = data
                            rt.latest_imu_host_us = now_us
                        else:
                            continue
                        rt.packet_count += 1
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(2)
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
            with rt.lock:
                rt.connected = False
            if _running:
                time.sleep(0.5)

    try:
        subprocess.run(["adb", "forward", "--remove", f"tcp:{rt.port}"], capture_output=True, timeout=3)
    except Exception:
        pass
    print("[phone] stopped")


# ── Pipeline setup (one pipeline per OAK) ─────────────────────────────────────
def make_pipeline(device: dai.Device, width: int, height: int, fps: float):
    pipeline = dai.Pipeline(device)

    monoL = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoR = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    monoL_out = monoL.requestOutput((width, height), fps=fps)
    monoR_out = monoR.requestOutput((width, height), fps=fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    monoL_out.link(stereo.left)
    monoR_out.link(stereo.right)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(3)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(width, height)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)

    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    rgb_out = camRgb.requestOutput((width, height), dai.ImgFrame.Type.BGR888p, fps=fps)

    q_rgb   = rgb_out.createOutputQueue(maxSize=4, blocking=False)
    q_monoL = monoL_out.createOutputQueue(maxSize=4, blocking=False)
    q_monoR = monoR_out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

    # IMU — BNO085 on OAK-D Pro. Enabled sensors + sample rates are pinned
    # above.  Batch size is tuned to keep latency low without dropping.
    q_imu = None
    try:
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, IMU_ACCEL_HZ)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW,     IMU_GYRO_HZ)
        imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR,   IMU_ROTV_HZ)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        q_imu = imu.out.createOutputQueue(maxSize=50, blocking=False)
    except Exception as e:
        print(f"[imu] setup failed (continuing without IMU): {e}")

    return pipeline, q_rgb, q_monoL, q_monoR, q_depth, q_imu


# ── Per-device capture thread ─────────────────────────────────────────────────
def capture_thread(rt: CamRuntime,
                   width: int, height: int, fps: float,
                   run_dir: Path):
    """Open device, save its factory calibration, then stream 4 outputs
    (RGB, monoL, monoR, depth), updating the shared `latest` dict."""
    try:
        device = dai.Device(rt.info)
    except Exception as e:
        rt.failed = True
        rt.error = str(e)
        rt.ready.set()
        print(f"[cam{rt.cid}] open failed: {e}")
        return

    # OAK factory calibration (before pipeline start)
    try:
        cal_path = run_dir / "calibration" / f"oak_{rt.mxid}.json"
        calibration_json = device.readCalibration().eepromToJson()
        if not isinstance(calibration_json, str):
            calibration_json = json.dumps(calibration_json, indent=2)
        cal_path.write_text(calibration_json)
        print(f"[cal] cam{rt.cid} → {cal_path.name}")
    except Exception as e:
        print(f"[cal] cam{rt.cid} save failed: {e}")

    try:
        pipeline, q_rgb, q_monoL, q_monoR, q_depth, q_imu = make_pipeline(
            device, width, height, fps)
    except Exception as e:
        rt.failed = True
        rt.error = str(e)
        rt.ready.set()
        print(f"[cam{rt.cid}] pipeline failed: {e}")
        return

    streams = {"rgb": q_rgb, "mono_left": q_monoL,
               "mono_right": q_monoR, "depth": q_depth}

    def _ts_us(entry) -> int:
        try:
            return int(entry.getTimestampDevice().total_seconds() * 1e6)
        except Exception:
            try:
                return int(entry.getTimestamp().total_seconds() * 1e6)
            except Exception:
                return 0

    with pipeline:
        pipeline.start()
        try:
            pipeline.setIrLaserDotProjectorIntensity(0.8)
        except Exception:
            pass
        print(f"[cam{rt.cid}] streaming  mxid={rt.mxid}")
        rt.ready.set()

        while _running and pipeline.isRunning():
            got_any = False
            for name, q in streams.items():
                pkt = q.tryGet()
                if pkt is None:
                    continue
                got_any = True
                ts = pkt.getTimestamp()
                ts_us = int(ts.total_seconds() * 1e6)
                if name == "depth":
                    frame = pkt.getFrame().astype(np.uint16)
                else:
                    frame = pkt.getCvFrame()
                with rt.lock:
                    rt.latest[name] = StreamFrame(image=frame, ts_us=ts_us)

            # IMU — keep only the most recent sample (it'll be stamped into
            # whichever frame gets saved next by the main thread)
            if q_imu is not None:
                imu_pkt = q_imu.tryGet()
                if imu_pkt is not None:
                    got_any = True
                    # Pick the last packet's last sub-report set
                    for p in imu_pkt.packets:
                        accel = getattr(p, "acceleroMeter",  None)
                        gyro  = getattr(p, "gyroscope",      None)
                        rotv  = getattr(p, "rotationVector", None)
                        s = IMUSample(ts_us=_ts_us(accel or gyro or rotv))
                        if accel is not None:
                            s.ax, s.ay, s.az = accel.x, accel.y, accel.z
                        if gyro is not None:
                            s.gx, s.gy, s.gz = gyro.x,  gyro.y,  gyro.z
                        if rotv is not None:
                            s.qi, s.qj, s.qk, s.qr = rotv.i, rotv.j, rotv.k, rotv.real
                        with rt.lock:
                            rt.latest_imu = s

            if not got_any:
                time.sleep(0.002)

    print(f"[cam{rt.cid}] stopped")


# ── Disk I/O ──────────────────────────────────────────────────────────────────
def make_run_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run = root / stamp
    run.mkdir(parents=True, exist_ok=False)
    (run / "calibration").mkdir()
    for cid in MXID_TO_CID.values():
        for s in STREAMS:
            (run / f"cam{cid}" / s).mkdir(parents=True)
    return run


def copy_caliscope(run_dir: Path):
    target = run_dir / "calibration" / "caliscope_camera_array.toml"
    for source in CALISCOPE_TOML_CANDIDATES:
        if source.exists():
            shutil.copy(source, target)
            print(f"[cal] caliscope → {target.name} ({source})")
            return
    searched = ", ".join(str(path) for path in CALISCOPE_TOML_CANDIDATES)
    print(f"[cal] caliscope file not found; searched: {searched}")


def snapshot(runtimes):
    """Copy current latest-frame dict + IMU sample from every camera."""
    out = {}
    for rt in runtimes:
        with rt.lock:
            out[rt.cid] = {
                "streams": {name: sf for name, sf in rt.latest.items()},
                "imu":     rt.latest_imu,
            }
    return out


def snapshot_phone(phone_rt: PhoneSensorRuntime | None):
    if phone_rt is None:
        return None
    with phone_rt.lock:
        return {
            "gps": dict(phone_rt.latest_gps) if phone_rt.latest_gps is not None else None,
            "gps_host_us": phone_rt.latest_gps_host_us,
            "imu": dict(phone_rt.latest_imu) if phone_rt.latest_imu is not None else None,
            "imu_host_us": phone_rt.latest_imu_host_us,
            "packet_count": phone_rt.packet_count,
            "connected": phone_rt.connected,
        }


def write_phone_snapshot(phone_snap, idx: int, save_host_us: int, gps_writer, phone_imu_writer):
    if not phone_snap:
        return

    gps = phone_snap.get("gps")
    if gps is not None:
        gps_host_us = int(phone_snap.get("gps_host_us") or 0)
        gps_writer.writerow([
            idx,
            save_host_us,
            gps_host_us,
            (save_host_us - gps_host_us) / 1000.0 if gps_host_us else "",
            gps.get("ts", ""),
            gps.get("lat", ""),
            gps.get("lon", ""),
            gps.get("alt", ""),
            gps.get("acc", ""),
            gps.get("spd", ""),
            gps.get("brg", ""),
        ])

    imu = phone_snap.get("imu")
    if imu is not None:
        imu_host_us = int(phone_snap.get("imu_host_us") or 0)
        phone_imu_writer.writerow([
            idx,
            save_host_us,
            imu_host_us,
            (save_host_us - imu_host_us) / 1000.0 if imu_host_us else "",
            imu.get("ts", ""),
            imu.get("ax", ""),
            imu.get("ay", ""),
            imu.get("az", ""),
            imu.get("gx", ""),
            imu.get("gy", ""),
            imu.get("gz", ""),
            imu.get("mx", ""),
            imu.get("my", ""),
            imu.get("mz", ""),
        ])


def save_snapshot(snap, run_dir: Path, idx: int,
                  ts_writer, imu_writer, t0_us: int):
    base = f"{idx:06d}"
    for cid, cam in snap.items():
        cam_dir = run_dir / f"cam{cid}"
        for s, sf in cam["streams"].items():
            if s == "depth":
                np.save(str(cam_dir / s / f"{base}.npy"), sf.image)
            else:
                cv2.imwrite(str(cam_dir / s / f"{base}.png"), sf.image)
            ts_writer.writerow([idx, cid, s, sf.ts_us,
                                sf.ts_us - t0_us if t0_us > 0 else 0])
        imu = cam["imu"]
        if imu is not None:
            imu_writer.writerow([idx, cid, imu.ts_us,
                                 imu.ax, imu.ay, imu.az,
                                 imu.gx, imu.gy, imu.gz,
                                 imu.qi, imu.qj, imu.qk, imu.qr])


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    global _running
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=str, default="dataset",
                    help="Root folder for all runs (default: ./dataset)")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Save rate in Hz (default 10)")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Stop after N seconds (0 = until Q pressed)")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    phone_group = ap.add_mutually_exclusive_group()
    phone_group.add_argument("--phone-sensors", dest="phone_sensors", action="store_true",
                             help="Record Android phone GPS/IMU over ADB TCP forwarding")
    phone_group.add_argument("--no-phone-sensors", dest="phone_sensors", action="store_false",
                             help="Disable Android phone GPS/IMU recording")
    ap.set_defaults(phone_sensors=True)
    ap.add_argument("--phone-port", type=int, default=5000,
                    help="ADB-forwarded TCP port used by the phone sensor service")
    args = ap.parse_args()

    # Discover devices
    def _id(d):
        return str(d.getDeviceId() if hasattr(d, "getDeviceId") else d.getMxId())
    avail = {_id(d): d for d in dai.Device.getAllAvailableDevices()}
    print(f"[dai] available: {sorted(avail)}")

    runtimes: list[CamRuntime] = []
    for mxid, cid in MXID_TO_CID.items():
        if mxid not in avail:
            print(f"[warn] mxid {mxid} (cam{cid} {CID_ROLE[cid]}) not connected")
            continue
        runtimes.append(CamRuntime(cid=cid, mxid=mxid, info=avail[mxid]))

    if not runtimes:
        print("[err] no cameras to record")
        return 1

    # Prepare run folder + caliscope copy
    run_dir = make_run_dir(Path(args.root))
    copy_caliscope(run_dir)
    print(f"[rec] run folder: {run_dir}")

    # Launch capture threads
    threads = []
    phone_rt = PhoneSensorRuntime(port=args.phone_port) if args.phone_sensors else None
    if phone_rt is not None:
        t = threading.Thread(target=phone_sensor_thread, args=(phone_rt,), daemon=True)
        t.start()
        threads.append(t)

    for rt in runtimes:
        t = threading.Thread(target=capture_thread,
                             args=(rt, args.width, args.height, args.fps, run_dir),
                             daemon=True)
        t.start()
        threads.append(t)

    # Wait for all cameras to be ready (device open + pipeline started)
    print("[rec] waiting for pipelines to start...")
    try:
        deadline = time.time() + 15
        while time.time() < deadline:
            if all(rt.ready.is_set() for rt in runtimes):
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        _running = False
        print("\n[rec] interrupted during startup")
        for t in threads:
            t.join(timeout=3)
        return 130

    active_runtimes = []
    for rt in runtimes:
        if rt.failed:
            print(f"[err] cam{rt.cid} failed: {rt.error}")
        elif not rt.ready.is_set():
            rt.failed = True
            rt.error = "startup timeout"
            print(f"[err] cam{rt.cid} did not start")
        else:
            active_runtimes.append(rt)

    if not active_runtimes:
        _running = False
        print("[err] no cameras started; stopping")
        for t in threads:
            t.join(timeout=3)
        return 1
    if len(active_runtimes) != len(runtimes):
        active_ids = ", ".join(f"cam{rt.cid}" for rt in active_runtimes)
        print(f"[rec] continuing with active cameras: {active_ids}")
    runtimes = active_runtimes

    # Wait until every stream has at least one frame
    print("[rec] waiting for first frame on every stream...")
    deadline = time.time() + 20
    while time.time() < deadline:
        if all(len(rt.latest) == len(STREAMS) for rt in runtimes):
            break
        time.sleep(0.1)
    ok = all(len(rt.latest) == len(STREAMS) for rt in runtimes)
    print("[rec] all streams live" if ok else "[rec] starting with partial streams")

    # Write metadata
    meta = {
        "created":      datetime.now().isoformat(),
        "width":        args.width,
        "height":       args.height,
        "fps_target":   args.fps,
        "streams":      list(STREAMS),
        "cameras": [
            {"cam_id": rt.cid, "mxid": rt.mxid, "role": CID_ROLE[rt.cid]}
            for rt in runtimes
        ],
        "calibration": {
            "oak_factory": [f"calibration/oak_{rt.mxid}.json" for rt in runtimes],
            "caliscope":   "calibration/caliscope_camera_array.toml",
        },
        "phone_sensors": {
            "enabled": bool(phone_rt is not None),
            "port": args.phone_port if phone_rt is not None else None,
            "gps_csv": "gps.csv" if phone_rt is not None else None,
            "phone_imu_csv": "phone_imu.csv" if phone_rt is not None else None,
            "sync": "Rows contain the latest phone packet available at each saved frame and age_ms versus host save time.",
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # Open timestamps CSV
    ts_file = open(run_dir / "timestamps.csv", "w", newline="")
    ts_writer = csv.writer(ts_file)
    ts_writer.writerow(["frame_idx", "cam_id", "stream", "ts_us", "rel_us"])

    # Run-level IMU CSV — one row per (frame_idx, cam_id) using the latest
    # IMU sample at the moment that frame was saved.
    imu_file = open(run_dir / "imu.csv", "w", newline="")
    imu_writer = csv.writer(imu_file)
    imu_writer.writerow(["frame_idx", "cam_id", "ts_us",
                         "accel_x", "accel_y", "accel_z",
                         "gyro_x", "gyro_y", "gyro_z",
                         "quat_i", "quat_j", "quat_k", "quat_real"])

    gps_file = open(run_dir / "gps.csv", "w", newline="")
    gps_writer = csv.writer(gps_file)
    gps_writer.writerow([
        "frame_idx", "save_host_us", "packet_host_us", "age_ms", "phone_ts_ms",
        "lat", "lon", "alt", "acc", "spd", "brg",
    ])

    phone_imu_file = open(run_dir / "phone_imu.csv", "w", newline="")
    phone_imu_writer = csv.writer(phone_imu_file)
    phone_imu_writer.writerow([
        "frame_idx", "save_host_us", "packet_host_us", "age_ms", "phone_ts_ms",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z",
    ])

    # Preview window
    cv2.namedWindow("Recording", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Recording", min(args.width * 3, 1800), args.height + 120)

    period = 1.0 / args.fps
    t_start = time.time()
    t0_us = 0
    next_save = time.time()
    frame_idx = 0

    try:
        while _running:
            now = time.time()
            # --- Save at the target rate ---
            if now >= next_save:
                snap = snapshot(runtimes)
                complete = all(len(snap.get(rt.cid, {}).get("streams", {})) == len(STREAMS)
                               for rt in runtimes)
                if complete:
                    if t0_us == 0:
                        t0_us = min(s["streams"]["rgb"].ts_us for s in snap.values())
                    save_snapshot(snap, run_dir, frame_idx,
                                  ts_writer, imu_writer, t0_us)
                    write_phone_snapshot(
                        snapshot_phone(phone_rt),
                        frame_idx,
                        host_time_us(),
                        gps_writer,
                        phone_imu_writer,
                    )
                    frame_idx += 1
                next_save += period
                # If we've fallen behind, resync to now (avoid death-spiral)
                if now > next_save + 0.5:
                    next_save = now + period

            # --- Preview ---
            snap = snapshot(runtimes)
            tiles = []
            for cid in sorted(MXID_TO_CID.values()):
                cam = snap.get(cid, {"streams": {}, "imu": None})
                s = cam.get("streams", {})
                rgb = s.get("rgb")
                depth = s.get("depth")
                if rgb is None:
                    tile_rgb = np.zeros((args.height, args.width, 3), np.uint8)
                else:
                    tile_rgb = rgb.image.copy()
                if depth is None:
                    tile_dep = np.zeros((args.height, args.width, 3), np.uint8)
                else:
                    d = depth.image.astype(np.float32)
                    d = np.clip(d / 8000.0 * 255.0, 0, 255).astype(np.uint8)
                    tile_dep = cv2.applyColorMap(d, cv2.COLORMAP_JET)
                tile = cv2.vconcat([tile_rgb, tile_dep])
                cv2.putText(tile, f"cam{cid} {CID_ROLE.get(cid,'?')}",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                tiles.append(tile)
            if tiles:
                preview = cv2.hconcat(tiles)
                elapsed = time.time() - t_start
                hud = f"saved: {frame_idx}   {frame_idx/max(elapsed,1e-3):.1f} fps   {elapsed:5.1f}s"
                if phone_rt is not None:
                    phone_snap = snapshot_phone(phone_rt)
                    gps_ok = phone_snap and phone_snap.get("gps") is not None
                    imu_ok = phone_snap and phone_snap.get("imu") is not None
                    hud += f"   phone GPS:{'Y' if gps_ok else 'N'} IMU:{'Y' if imu_ok else 'N'}"
                if args.duration > 0:
                    hud += f" / {args.duration:.1f}s"
                hud += "   [Q to stop]"
                cv2.rectangle(preview, (0, 0), (preview.shape[1], 30), (0, 0, 0), -1)
                cv2.putText(preview, hud, (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
                cv2.imshow("Recording", preview)

            if args.duration > 0 and time.time() - t_start >= args.duration:
                print(f"[rec] reached duration {args.duration}s")
                break

            key = cv2.waitKey(5) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        _running = False
        ts_file.close()
        imu_file.close()
        gps_file.close()
        phone_imu_file.close()
        cv2.destroyAllWindows()
        for t in threads:
            t.join(timeout=3)
        elapsed = time.time() - t_start
        print(f"[rec] saved {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx / max(elapsed, 1e-3):.1f} fps)")
        print(f"[rec] output: {run_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
