"""EspController-compatible adapter for the curvature-based trike firmware.

WHY THIS EXISTS
  esp_controller.EspController targets a different ESP program:

      CONTROL,<seq>,<percent 0..100>,0.0,0     @ 115200
      STOP / STATUS handshake, requires cal=1

  trike_controller.ino speaks curvature instead:

      K<kappa>[,<v>]   kappa in 1/m, v = UPPER bound on speed   @ 921600
      Z straight   X stop+disable   S status   E constants

  Same class surface (update / set_armed / close) so autonomous_driving.py
  can use either without changing the control loop.

WHY CURVATURE RATHER THAN A NORMALISED STEER
  bev.py computes curvature and then throws it away:

      curvature = 2*sin(alpha)/lookahead
      wheel_angle = atan(wheelbase_m * curvature)
      return -wheel_angle / max_angle          # normalised -1..1

  That normalisation bakes in wheelbase_m=1.20 and max_wheel_angle=34.0,
  neither of which matches this trike (0.965 m, 30.5 deg). Sending the
  normalised value would apply that error twice. Curvature is
  wheelbase-free, so passing it straight through cancels the mismatch
  exactly instead of needing a patch. See HANDOFF section 6.

SIGN
  bev.py is +right. This firmware is +kappa = LEFT. Inverted here, once,
  explicitly. Getting this wrong steers into whatever you were avoiding.
"""

from __future__ import annotations

import math
import threading
import time

from trike_geometry import (
    WHEELBASE_M, MAX_WHEEL_ANGLE_DEG, KAPPA_MECH, ESP_BAUD,
    normalized_to_kappa, clamp_kappa,
)

# Inversion uses THE SAME geometry bev.py is configured with (both import
# trike_geometry), so the round trip kappa -> raw -> kappa is an exact
# identity and the two sites cannot drift apart.
normalized_steer_to_kappa = normalized_to_kappa


class TrikeEspController:
    """Drop-in for EspController, speaking K<kappa> at 921600.

    update() accepts EITHER a normalised bev steer (default, so existing
    call sites work untouched) or a curvature directly, which is what a
    route follower should send.
    """

    def __init__(self, port: str, baud: int = ESP_BAUD, rate_hz: float = 50.0,
                 stale_s: float = 0.25, invert: bool = False,
                 speed_bound_mps: float = 2.0,
                 accept_curvature: bool = False) -> None:
        import serial

        self.stream = serial.Serial(port, baud, timeout=0.05, write_timeout=0.5)
        self.period = 1.0 / max(1.0, float(rate_hz))
        self.stale_s = max(0.05, float(stale_s))
        self.invert = bool(invert)
        self.accept_curvature = bool(accept_curvature)
        self.speed_bound = max(0.5, float(speed_bound_mps))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.armed = False
        self.kappa = 0.0
        self.updated = 0.0
        self.error = ""
        self.rx_buffer = bytearray()
        self.last_rx_line = ""
        time.sleep(2.0)                     # let the board settle after open
        with self.lock:
            self._send_locked(0.0)
        self.thread = threading.Thread(target=self._run, name="trike-esp", daemon=True)
        self.thread.start()

    # ---- protocol -------------------------------------------------
    def _drain_rx_locked(self) -> None:
        """Drain replies/telemetry so the ESP USB serial path stays flowing."""
        waiting = self.stream.in_waiting
        if waiting <= 0:
            return
        self.rx_buffer.extend(self.stream.read(waiting))
        while b"\n" in self.rx_buffer:
            raw, _, remainder = self.rx_buffer.partition(b"\n")
            self.rx_buffer = bytearray(remainder)
            line = raw.decode("ascii", "ignore").strip()
            if line:
                self.last_rx_line = line
        if len(self.rx_buffer) > 4096:
            self.rx_buffer.clear()

    def _send_locked(self, kappa: float) -> None:
        self._drain_rx_locked()
        k = clamp_kappa(kappa)
        # v is an UPPER BOUND, not an estimate. Overestimating only makes
        # the firmware's rollover clamp stricter - it costs conservatism,
        # never stability.
        self.stream.write(f"K{k:.4f},{self.speed_bound:.2f}\n".encode("ascii"))

    def set_speed_bound(self, v_mps: float) -> None:
        with self.lock:
            self.speed_bound = max(0.5, float(v_mps))

    # ---- same surface as EspController -----------------------------
    def update(self, steering: float) -> None:
        with self.lock:
            k = float(steering) if self.accept_curvature \
                else normalized_steer_to_kappa(steering)
            if self.invert:
                k = -k
            self.kappa = clamp_kappa(k)
            self.updated = time.monotonic()

    def update_percent(self, percent: float) -> None:
        """Accept the legacy display scale: 100=left, 50=center, 0=right."""
        p = max(0.0, min(100.0, float(percent)))
        self.update(-(p / 50.0 - 1.0))

    def set_armed(self, armed: bool) -> None:
        with self.lock:
            was_armed = self.armed
            self.armed = bool(armed) and not self.error
            if was_armed and not self.armed:
                self._send_locked(0.0)

    def _run(self) -> None:
        while not self.stop_event.wait(self.period):
            try:
                with self.lock:
                    fresh = time.monotonic() - self.updated <= self.stale_s
                    # Keep sending even when disarmed: K also feeds the
                    # firmware watchdog, and silence for 300 ms makes it
                    # re-centre anyway. Explicit zero beats implicit.
                    self._send_locked(self.kappa if (self.armed and fresh) else 0.0)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.armed = False
                self.stop_event.set()

    def status(self, timeout_s: float = 1.0) -> str:
        """Read one S line. Motion-free."""
        with self.lock:
            self.stream.reset_input_buffer()
            self.stream.write(b"S\n")
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                line = self.stream.readline().decode("ascii", "ignore").strip()
                if line.startswith("mode="):
                    return line
                time.sleep(0.02)
        return ""

    def preflight(self) -> tuple[bool, str]:
        """Replaces the cal=1 handshake, which this firmware never emits.

        Checks what it DOES report: that it answers, is not in a latched
        stall fault, and has a plausible pot reading.
        """
        line = self.status()
        if not line:
            return False, "no status response from the trike firmware"
        fields = dict(
            part.split("=", 1) for part in line.split() if "=" in part
        )
        try:
            pot = int(fields.get("pot", "-1"))
            fault = int(fields.get("fault", "1"))
            mode = int(fields.get("mode", "1"))
        except ValueError:
            return False, f"unparseable status: {line}"
        if mode != 0:
            return False, f"manual joystick override active; center/calibrate joystick: {line}"
        if fault:
            return False, f"latched stall fault - send F to clear: {line}"
        if not (200 <= pot <= 3300):
            return False, f"pot {pot} outside soft limits 200..3300: {line}"
        return True, line

    def close(self) -> None:
        try:
            self.set_armed(False)
            self.stop_event.wait(0.25)
        finally:
            self.stop_event.set()
            self.thread.join(timeout=1.0)
            try:
                with self.lock:
                    self._send_locked(0.0)
                    self.stream.write(b"Z\n")     # straight ahead
            finally:
                self.stream.close()
