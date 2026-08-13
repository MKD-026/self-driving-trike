"""Small runtime types used by the autonomous driving application.

This intentionally contains only terminal and camera-frame helpers.  The old
project's ``inputs.py`` imported VectorNav at module load time even though the
person follower does not use it, making camera-only startup unnecessarily
depend on the IMU SDK.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass

import numpy as np


@dataclass
class StreamFrame:
    image: np.ndarray
    ts_us: int


@dataclass
class CameraPacket:
    image: StreamFrame | None
    depth: StreamFrame | None
    imu: object | None = None


def host_time_us() -> int:
    return int(time.time() * 1e6)


class KeyPoller:
    """Read one terminal key without blocking or requiring Enter."""

    def __init__(self) -> None:
        self.enabled = False
        self.fd: int | None = None
        self.original_settings = None

    def __enter__(self) -> "KeyPoller":
        if sys.stdin.isatty():
            self.fd = sys.stdin.fileno()
            self.original_settings = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            self.enabled = True
        return self

    def poll(self) -> str | None:
        if not self.enabled or self.fd is None:
            return None
        ready, _, _ = select.select([self.fd], [], [], 0)
        if not ready:
            return None
        return os.read(self.fd, 1).decode(errors="ignore")

    def __exit__(self, *_exc_info) -> None:
        if self.enabled and self.fd is not None and self.original_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_settings)
        self.enabled = False

