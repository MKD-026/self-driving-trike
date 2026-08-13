"""Route-file loading and a metric lookahead target for OmniVLA GPS modality."""

from __future__ import annotations

import json
import math
from pathlib import Path


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(value))


def bearing_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


class RouteFollower:
    """Choose a point a fixed distance ahead while preventing route-index regressions."""

    def __init__(self, route_file: Path, route_name: str | None, lookahead_m: float = 5.0) -> None:
        data = json.loads(route_file.read_text(encoding="utf-8"))
        routes = data.get("routes", {})
        if not routes:
            raise ValueError(f"No routes in {route_file}")
        self.name = route_name or next(iter(routes))
        if self.name not in routes:
            raise ValueError(f"Unknown route {self.name!r}; choices: {', '.join(routes)}")
        self.points = [(float(point[0]), float(point[1])) for point in routes[self.name]]
        if len(self.points) < 2:
            raise ValueError("Route needs at least two waypoints")
        self.lookahead_m = float(lookahead_m)
        self.progress_index = 0

    def target(self, lat: float, lon: float) -> dict[str, object]:
        current = (float(lat), float(lon))
        search_start = max(0, self.progress_index - 1)
        nearest = min(
            range(search_start, len(self.points)),
            key=lambda index: haversine_m(current, self.points[index]),
        )
        self.progress_index = max(self.progress_index, nearest)
        remaining = self.lookahead_m
        start = current
        target_index = self.progress_index
        target = self.points[target_index]
        for index in range(self.progress_index, len(self.points)):
            end = self.points[index]
            segment = haversine_m(start, end)
            if segment >= remaining and segment > 1e-6:
                fraction = remaining / segment
                target = (
                    start[0] + (end[0] - start[0]) * fraction,
                    start[1] + (end[1] - start[1]) * fraction,
                )
                target_index = index
                break
            remaining -= segment
            start = end
            target = end
            target_index = index
        heading_target = self.points[min(target_index + 1, len(self.points) - 1)]
        if heading_target == target and target_index > 0:
            heading_start = self.points[target_index - 1]
        else:
            heading_start = target
        distance_to_end = haversine_m(current, self.points[-1])
        return {
            "lat": target[0],
            "lon": target[1],
            "yaw_deg": bearing_deg(heading_start, heading_target),
            "route": self.name,
            "route_index": target_index,
            "lookahead_m": min(self.lookahead_m, haversine_m(current, target)),
            "distance_to_end_m": distance_to_end,
            "complete": distance_to_end <= 1.5,
        }
