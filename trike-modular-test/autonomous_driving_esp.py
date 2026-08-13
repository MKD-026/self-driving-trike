#!/usr/bin/env python3
"""Run the existing pure-pursuit pipeline with VLA-style ESP steering output."""

from __future__ import annotations

import argparse
import sys

import autonomous_driving as app
from esp_trike_adapter import TrikeEspController


DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_58A6082118-if00"


class EspCarController:
    """Drop-in replacement for the old PCA9685 CarController interface."""

    def __init__(self, _dry_run: bool) -> None:
        self.bridge = TrikeEspController(
            CONFIG.port, CONFIG.baud, CONFIG.rate_hz,
            invert=CONFIG.invert,
            speed_bound_mps=CONFIG.speed_bound_mps,
        )
        ok, detail = self.bridge.preflight()
        if not ok:
            self.bridge.close()
            raise RuntimeError(f"ESP preflight failed: {detail}")
        print(f"[esp] preflight passed: {detail}", flush=True)
        self.armed = False
        self.current_steering = app.STEERING_CENTER
        self.current_throttle = app.THROTTLE_NEUTRAL
        print(f"[esp] CENTER HOLD; mapping={'flipped' if CONFIG.invert else 'original'}", flush=True)

    def arm(self) -> None:
        self.armed = True
        self.bridge.set_armed(True)

    def disarm(self) -> None:
        self.armed = False
        self.current_steering = app.STEERING_CENTER
        self.current_throttle = app.THROTTLE_NEUTRAL
        self.bridge.set_armed(False)

    def command(self, steering: float, _throttle: float) -> None:
        self.current_steering = app.clamp(steering, app.STEERING_MIN, app.STEERING_MAX)
        self.current_throttle = app.THROTTLE_NEUTRAL
        normalized = app.clamp(
            (self.current_steering - app.STEERING_CENTER) / app.STEERING_MAX_OFFSET,
            -1.0,
            1.0,
        )
        self.bridge.update(normalized)

    def brake_then_stop(self, _duration_s: float) -> None:
        self.current_throttle = app.THROTTLE_NEUTRAL

    def snapshot(self):
        return self.armed and not self.bridge.error, self.current_steering, self.current_throttle, self.bridge.error

    def close(self) -> None:
        self.disarm()
        self.bridge.close()


def parse_wrapper_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--enable-esp", action="store_true")
    parser.add_argument("--esp-port", dest="port", default=DEFAULT_PORT)
    parser.add_argument("--esp-baud", dest="baud", type=int, default=921600)
    parser.add_argument("--esp-rate-hz", dest="rate_hz", type=float, default=50.0)
    parser.add_argument("--esp-speed-bound-mps", dest="speed_bound_mps", type=float, default=2.0)
    parser.add_argument("--esp-invert", dest="invert", action="store_true")
    parser.add_argument("--esp-no-invert", dest="invert", action="store_false")
    parser.set_defaults(invert=False)
    config, remaining = parser.parse_known_args(argv)
    if not config.enable_esp:
        parser.error("explicit --enable-esp is required")
    if config.rate_hz <= 0:
        parser.error("--esp-rate-hz must be positive")
    if "--dry-run" in remaining:
        parser.error("--dry-run cannot be combined with --enable-esp")
    return config, remaining


CONFIG, REMAINING = parse_wrapper_args(sys.argv[1:])


def main() -> int:
    sys.argv = [sys.argv[0], *REMAINING]
    app.CarController = EspCarController
    return app.main()


if __name__ == "__main__":
    raise SystemExit(main())
