"""Live VectorNav/NMEA reader with merged, age-gated position fixes."""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import asdict, dataclass

try:
    import serial
except ImportError:
    serial = None


def _number(value: str) -> float | None:
    try:
        return float(value) if value != "" else None
    except (TypeError, ValueError):
        return None


def _integer(value: str) -> int | None:
    value = _number(value)
    return int(value) if value is not None else None


def _parts(line: str) -> list[str]:
    text = line.strip().split("*", 1)[0].lstrip("$")
    return text.split(",") if text else []


def _nmea_coordinate(value: str, hemisphere: str) -> float | None:
    try:
        raw = float(value)
        degrees = int(raw // 100)
        coordinate = degrees + (raw - degrees * 100) / 60.0
        return -coordinate if hemisphere.upper() in ("S", "W") else coordinate
    except (TypeError, ValueError):
        return None


def _speed(north, east, down) -> float | None:
    try:
        return math.sqrt(float(north) ** 2 + float(east) ** 2 + float(down) ** 2)
    except (TypeError, ValueError):
        return None


def parse_vectornav_or_nmea(line: str) -> dict[str, object]:
    """Parse the same VNINS/VNGPS/NMEA forms used by the recorder."""
    values = _parts(line)
    if not values:
        return {}
    sentence, fields = values[0].upper(), values[1:]
    record: dict[str, object] = {"parse_type": sentence, "raw_line": line.strip()}

    if sentence.endswith("GGA") and len(fields) >= 9:
        record.update(
            lat=_nmea_coordinate(fields[1], fields[2]),
            lon=_nmea_coordinate(fields[3], fields[4]),
            fix_status=_integer(fields[5]),
            num_sats=_integer(fields[6]),
            altitude=_number(fields[8]),
        )
    elif sentence.endswith("RMC") and len(fields) >= 8:
        speed_knots = _number(fields[6])
        record.update(
            lat=_nmea_coordinate(fields[2], fields[3]),
            lon=_nmea_coordinate(fields[4], fields[5]),
            speed_mps=speed_knots * 0.514444 if speed_knots is not None else None,
            yaw_deg=_number(fields[7]),
            fix_status=1 if fields[1].upper() == "A" else 0,
        )
    elif sentence == "VNINS" and len(fields) == 15:
        record.update(
            parse_type="VNINS_COMPACT",
            fix_status=fields[2],
            yaw_deg=_number(fields[3]),
            pitch_deg=_number(fields[4]),
            roll_deg=_number(fields[5]),
            lat=_number(fields[6]),
            lon=_number(fields[7]),
            altitude=_number(fields[8]),
            vel_n=_number(fields[9]),
            vel_e=_number(fields[10]),
            vel_d=_number(fields[11]),
        )
    elif sentence == "VNINS" and len(fields) >= 16:
        record.update(
            fix_status=_integer(fields[3]),
            num_sats=_integer(fields[4]),
            yaw_deg=_number(fields[5]),
            pitch_deg=_number(fields[6]),
            roll_deg=_number(fields[7]),
            lat=_number(fields[8]),
            lon=_number(fields[9]),
            altitude=_number(fields[10]),
            vel_n=_number(fields[11]),
            vel_e=_number(fields[12]),
            vel_d=_number(fields[13]),
        )
    elif sentence == "VNGPS" and len(fields) >= 13:
        record.update(
            fix_status=_integer(fields[2]),
            num_sats=_integer(fields[3]),
            lat=_number(fields[4]),
            lon=_number(fields[5]),
            altitude=_number(fields[6]),
            vel_n=_number(fields[7]),
            vel_e=_number(fields[8]),
            vel_d=_number(fields[9]),
        )

    if all(record.get(key) is not None for key in ("vel_n", "vel_e", "vel_d")):
        record["speed_mps"] = _speed(record["vel_n"], record["vel_e"], record["vel_d"])
    return record


@dataclass(frozen=True)
class GPSFix:
    lat: float
    lon: float
    yaw_deg: float | None
    altitude: float | None
    speed_mps: float | None
    fix_status: object | None
    num_sats: int | None
    source: str
    port: str
    received_ns: int
    age_ms: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class VectorNavReader(threading.Thread):
    def __init__(
        self,
        port: str = "",
        baud: int = 115200,
        timeout: float = 0.2,
    ) -> None:
        super().__init__(name="vectornav-reader", daemon=True)
        self.requested_port = port
        self.baud = baud
        self.timeout = timeout
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest: dict[str, object] = {}
        self.active_port = ""
        self.last_error = ""

    def _ports(self) -> list[str]:
        if self.requested_port.strip():
            return [self.requested_port.strip()]
        candidates = (
            "/dev/ttyUSB0",
            "/dev/ttyUSB1",
            "/dev/ttyACM0",
            "/dev/ttyACM1",
            "/dev/ttyTHS0",
            "/dev/ttyTHS1",
        )
        return [path for path in candidates if os.path.exists(path)]

    def run(self) -> None:
        if serial is None:
            self.last_error = "pyserial is not installed"
            return
        while not self.stop_event.is_set():
            ports = self._ports()
            if not ports:
                self.last_error = "no VectorNav serial port found"
                self.stop_event.wait(1.0)
                continue
            for port in ports:
                if self.stop_event.is_set():
                    return
                try:
                    print(f"[gps] opening {port} @ {self.baud}", flush=True)
                    with serial.Serial(port, self.baud, timeout=self.timeout) as stream:
                        self.active_port = port
                        self.last_error = ""
                        print(f"[gps] connected: {port}", flush=True)
                        while not self.stop_event.is_set():
                            raw = stream.readline()
                            if not raw:
                                continue
                            line = raw.decode("ascii", errors="replace").strip()
                            parsed = parse_vectornav_or_nmea(line)
                            if not parsed:
                                continue
                            received_ns = time.time_ns()
                            with self.lock:
                                # VectorNav may interleave complementary sentence types.
                                # Retain the newest non-empty value for each field.
                                for key, value in parsed.items():
                                    if value not in (None, ""):
                                        self.latest[key] = value
                                self.latest["received_ns"] = received_ns
                                self.latest["port"] = port
                except Exception as exc:
                    self.active_port = ""
                    self.last_error = f"{port}: {exc}"
                    print(f"[gps] {self.last_error}", flush=True)
                    self.stop_event.wait(0.5)

    def snapshot(self, max_age_s: float = 1.0) -> GPSFix | None:
        with self.lock:
            data = dict(self.latest)
        lat, lon = data.get("lat"), data.get("lon")
        received_ns = int(data.get("received_ns") or 0)
        if lat is None or lon is None or received_ns <= 0:
            return None
        age_ms = (time.time_ns() - received_ns) / 1_000_000.0
        if age_ms > max_age_s * 1000.0:
            return None
        return GPSFix(
            lat=float(lat),
            lon=float(lon),
            yaw_deg=float(data["yaw_deg"]) if data.get("yaw_deg") is not None else None,
            altitude=float(data["altitude"]) if data.get("altitude") is not None else None,
            speed_mps=float(data["speed_mps"]) if data.get("speed_mps") is not None else None,
            fix_status=data.get("fix_status"),
            num_sats=int(data["num_sats"]) if data.get("num_sats") is not None else None,
            source=str(data.get("parse_type") or "unknown"),
            port=str(data.get("port") or self.active_port),
            received_ns=received_ns,
            age_ms=age_ms,
        )

    def close(self) -> None:
        self.stop_event.set()
        self.join(timeout=max(1.0, self.timeout + 0.5))

