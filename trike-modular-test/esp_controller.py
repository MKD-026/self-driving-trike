"""Fail-centered runtime bridge for the ESP32 steering actuator."""

from __future__ import annotations

import threading
import time


def normalized_to_percent(steering: float, invert: bool = True) -> float:
    steering = max(-1.0, min(1.0, float(steering)))
    if invert:
        steering = -steering
    return (steering + 1.0) * 50.0


class EspController:
    def __init__(self, port: str, baud: int = 921600, rate_hz: float = 50.0,
                 stale_s: float = 0.25, invert: bool = True) -> None:
        import serial

        self.stream = serial.Serial(port, baud, timeout=0.05, write_timeout=0.1)
        self.period = 1.0 / max(1.0, float(rate_hz))
        self.stale_s = max(0.05, float(stale_s))
        self.invert = bool(invert)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.armed = False
        self.sequence = 0
        self.steering = 0.0
        self.updated = 0.0
        self.error = ""
        self.thread = threading.Thread(target=self._run, name="esp-control", daemon=True)
        with self.lock:
            self._send_locked(50.0)
        self.thread.start()

    def _send_locked(self, percent: float) -> None:
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        self.stream.write(f"CONTROL,{self.sequence},{percent:.2f},0.0,0\n".encode("ascii"))

    def update(self, steering: float) -> None:
        with self.lock:
            self.steering = max(-1.0, min(1.0, float(steering)))
            self.updated = time.monotonic()

    def set_armed(self, armed: bool) -> None:
        with self.lock:
            self.armed = bool(armed) and not self.error
            if not self.armed:
                self._send_locked(50.0)

    def _run(self) -> None:
        while not self.stop_event.wait(self.period):
            try:
                with self.lock:
                    fresh = time.monotonic() - self.updated <= self.stale_s
                    percent = normalized_to_percent(self.steering, self.invert) if self.armed and fresh else 50.0
                    self._send_locked(percent)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.armed = False
                self.stop_event.set()

    def close(self) -> None:
        try:
            self.set_armed(False)
            self.stop_event.wait(0.25)
        finally:
            self.stop_event.set()
            self.thread.join(timeout=1.0)
            try:
                with self.lock:
                    self._send_locked(50.0)
                    self.stream.write(b"STOP\n")
            finally:
                self.stream.close()
