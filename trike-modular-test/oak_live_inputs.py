"""Threaded three-OAK RGB/depth input."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import depthai as dai
import numpy as np

from inputs import CameraPacket, StreamFrame, host_time_us


CAMERA_ROLES_PATH = Path("/home/h2x/Documents/final_data_capture/camera_roles.json")


def load_mxid_to_role(path: Path = CAMERA_ROLES_PATH) -> dict[str, str]:
    """Load the authoritative physical camera layout used by data capture."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        roles = payload.get("roles", payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load camera roles from {path}: {exc}") from exc
    expected = {"left", "center", "right"}
    if not isinstance(roles, dict) or set(roles) != expected:
        raise RuntimeError(f"{path} must define exactly these roles: {sorted(expected)}")
    mxids = [str(roles[role]).strip() for role in expected]
    if any(not mxid for mxid in mxids) or len(set(mxids)) != len(mxids):
        raise RuntimeError(f"{path} must assign three distinct non-empty MXIDs")
    return {str(roles[role]).strip(): role for role in expected}


MXID_TO_ROLE = load_mxid_to_role()
ROLE_TO_CID = {"center": 0, "right": 1, "left": 2}
ROLE_ORDER = {"left": 0, "center": 1, "right": 2}


def device_id(info) -> str:
    if hasattr(info, "getDeviceId"):
        return str(info.getDeviceId())
    return str(getattr(info, "deviceId", getattr(info, "mxid", info)))


def socket(name: str):
    if hasattr(dai.CameraBoardSocket, name):
        return getattr(dai.CameraBoardSocket, name)
    legacy_name = {"CAM_A": "RGB", "CAM_B": "LEFT", "CAM_C": "RIGHT"}[name]
    return getattr(dai.CameraBoardSocket, legacy_name)


def packet_frame(packet) -> np.ndarray:
    frame = packet.getCvFrame() if hasattr(packet, "getCvFrame") else packet.getFrame()
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[2] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    return frame


def packet_timestamp_us(packet) -> int:
    try:
        return int(packet.getTimestamp().total_seconds() * 1e6)
    except Exception:
        return host_time_us()


def try_get(queue):
    if hasattr(queue, "tryGet"):
        return queue.tryGet()
    return queue.get() if queue.has() else None


@dataclass
class LiveRuntime:
    cid: int
    mxid: str
    role: str
    device: object
    pipeline: object
    rgb_queue: object
    depth_queue: object
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest: dict[str, StreamFrame] = field(default_factory=dict)
    ready: threading.Event = field(default_factory=threading.Event)
    failed: bool = False
    error: str = ""
    last_frame_host_us: int = 0

    def close(self) -> None:
        for obj in (self.pipeline, self.device):
            try:
                obj.close()
            except Exception:
                pass


def open_runtime(info, width: int, height: int, fps: float) -> LiveRuntime:
    mxid = device_id(info)
    role = MXID_TO_ROLE[mxid]
    device = dai.Device(mxid)
    pipeline = dai.Pipeline(device)

    rgb_camera = pipeline.create(dai.node.Camera).build(socket("CAM_A"))
    left_camera = pipeline.create(dai.node.Camera).build(socket("CAM_B"))
    right_camera = pipeline.create(dai.node.Camera).build(socket("CAM_C"))

    rgb = rgb_camera.requestOutput(
        (width, height), dai.ImgFrame.Type.BGR888p, dai.ImgResizeMode.CROP, fps
    )
    left = left_camera.requestOutput(
        (width, height), dai.ImgFrame.Type.GRAY8, dai.ImgResizeMode.CROP, fps
    )
    right = right_camera.requestOutput(
        (width, height), dai.ImgFrame.Type.GRAY8, dai.ImgResizeMode.CROP, fps
    )

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(socket("CAM_A"))
    stereo.setOutputSize(width, height)
    left.link(stereo.left)
    right.link(stereo.right)

    rgb_queue = rgb.createOutputQueue(maxSize=2, blocking=False)
    depth_queue = stereo.depth.createOutputQueue(maxSize=2, blocking=False)
    pipeline.start()
    print(f"[oak] opened {role:>6} mxid={mxid}", flush=True)
    return LiveRuntime(
        ROLE_TO_CID[role], mxid, role, device, pipeline, rgb_queue, depth_queue
    )


class OakCameraInputs:
    def __init__(self, fps: float, width: int, height: int, single: bool = False) -> None:
        self.fps = fps
        self.width = width
        self.height = height
        self.single = single
        self.stop_event = threading.Event()
        self.runtimes: list[LiveRuntime] = []
        self.threads: list[threading.Thread] = []
        self._open()

    def _open(self) -> None:
        infos = list(dai.Device.getAllAvailableDevices())
        known = [info for info in infos if device_id(info) in MXID_TO_ROLE]
        known.sort(key=lambda info: ROLE_ORDER[MXID_TO_ROLE[device_id(info)]])
        required_roles = {"center"} if self.single else set(ROLE_TO_CID)
        found_roles = {MXID_TO_ROLE[device_id(info)] for info in known}
        missing = sorted(required_roles - found_roles)
        if missing:
            discovered = ", ".join(device_id(info) for info in infos) or "none"
            raise RuntimeError(
                f"Missing required OAK role(s): {', '.join(missing)}. Discovered: {discovered}"
            )
        if self.single:
            known = [info for info in known if MXID_TO_ROLE[device_id(info)] == "center"]

        try:
            for info in known:
                self.runtimes.append(open_runtime(info, self.width, self.height, self.fps))
        except Exception:
            for runtime in reversed(self.runtimes):
                runtime.close()
            self.runtimes.clear()
            raise

        self.runtimes.sort(key=lambda runtime: runtime.cid)
        for runtime in self.runtimes:
            thread = threading.Thread(
                target=self._reader,
                args=(runtime,),
                name=f"oak-{runtime.role}-reader",
                daemon=True,
            )
            thread.start()
            self.threads.append(thread)

    def _reader(self, runtime: LiveRuntime) -> None:
        try:
            while not self.stop_event.is_set():
                got_packet = False
                for name, queue in (("rgb", runtime.rgb_queue), ("depth", runtime.depth_queue)):
                    packet = try_get(queue)
                    if packet is None:
                        continue
                    got_packet = True
                    frame = StreamFrame(packet_frame(packet), packet_timestamp_us(packet))
                    with runtime.lock:
                        runtime.latest[name] = frame
                        runtime.last_frame_host_us = host_time_us()
                        if "rgb" in runtime.latest and "depth" in runtime.latest:
                            runtime.ready.set()
                if not got_packet:
                    self.stop_event.wait(0.002)
        except Exception as exc:
            if not self.stop_event.is_set():
                with runtime.lock:
                    runtime.failed = True
                    runtime.error = str(exc)
                print(f"[oak] {runtime.role} reader failed: {exc}", flush=True)
        finally:
            runtime.ready.set()

    def _packet(self, cid: int) -> CameraPacket:
        runtime = next((item for item in self.runtimes if item.cid == cid), None)
        if runtime is None:
            return CameraPacket(None, None)
        with runtime.lock:
            return CameraPacket(runtime.latest.get("rgb"), runtime.latest.get("depth"))

    def camera1_image(self):
        return self._packet(ROLE_TO_CID["center"]).image

    def camera1_depth(self):
        return self._packet(ROLE_TO_CID["center"]).depth

    def camera2_image(self):
        return self._packet(ROLE_TO_CID["right"]).image

    def camera2_depth(self):
        return self._packet(ROLE_TO_CID["right"]).depth

    def camera3_image(self):
        return self._packet(ROLE_TO_CID["left"]).image

    def camera3_depth(self):
        return self._packet(ROLE_TO_CID["left"]).depth

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        deadline = time.monotonic() + 5.0
        for thread in self.threads:
            thread.join(timeout=max(0.1, deadline - time.monotonic()))
        self.threads.clear()
        for runtime in reversed(self.runtimes):
            runtime.close()


def get_camera_inputs(
    fps: float = 15.0,
    width: int = 640,
    height: int = 400,
    single: bool = False,
    **_kwargs,
):
    return OakCameraInputs(fps=fps, width=width, height=height, single=single)
