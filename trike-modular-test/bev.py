"""Bird's-eye-view obstacle map and avoidance steering.

This does NOT warp the camera footage to BEV. It places *detections* on a
top-down grid using a (range, bearing) per detection:

  - bearing comes for free from the detection's column in the equirectangular
    panorama (the equirect x-axis is yaw), so no inverse warp is needed;
  - range comes from a depth provider (OAK stereo today, optionally fused with a
    monocular model later) sampled at the detection box.

From the resulting obstacle list we compute a steering bias on the same discrete
ladder the planner uses (-30..+30, steps of 5), scaled by the speed mode so the
trike is gentle when slow/pushing and decisive at speed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from panorama_core import PANO_WIDTH, PANO_YAW_FOV_DEG

# Minimum physical widths stop a narrow/far 2-D box from collapsing to a BEV
# point. Projected bbox width is still retained whenever it is larger.
NOMINAL_OBSTACLE_WIDTH_M = {
    "person": 0.60,
    "bicycle": 0.80,
    "bike": 0.80,
    "motorcycle": 0.90,
    "car": 1.90,
    "vehicle": 1.90,
    "van": 2.20,
    "truck": 2.60,
    "bus": 2.70,
    "train": 3.00,
    "dog": 0.60,
    "bench": 1.50,
    "fire hydrant": 0.50,
    "traffic light": 0.40,
    "stop sign": 0.75,
}
DEFAULT_OBSTACLE_WIDTH_M = 0.60

# A depth provider maps a detection to a forward range in metres, or None when
# depth is unavailable for that detection (so avoidance simply ignores it).
DepthProvider = Callable[["object"], Optional[float]]


@dataclass
class BevConfig:
    max_range_m: float = 10.0          # grid extent forward and to each side
    cell_m: float = 0.1                # grid resolution
    min_range_m: float = 0.3           # ignore depth closer than this (noise)
    corridor_half_width_m: float = 0.8 # forward corridor that triggers avoidance
    corridor_range_m: float = 6.0      # how far ahead we look for obstacles


@dataclass
class BevObstacle:
    x_m: float          # footprint center, + = right of the trike
    z_m: float          # forward distance
    bearing_deg: float  # + = right of straight-ahead
    range_m: float
    label: str = ""
    left_m: float = 0.0   # minimum lateral x (physical left edge)
    right_m: float = 0.0  # maximum lateral x (physical right edge)
    width_m: float = 0.0


@dataclass
class PurePursuitConfig:
    max_range_m: float = 10.0
    min_range_m: float = 0.3
    wheelbase_m: float = 1.20
    max_wheel_angle_deg: float = 34.0
    lookahead_m: float = 4.5
    corridor_half_width_m: float = 0.95
    obstacle_padding_m: float = 0.75
    max_lateral_goal_m: float = 2.4
    road_half_width_m: float = 1.75
    road_edge_margin_m: float = 0.30
    normal_steer_limit: float = 0.42
    low_speed_steer_limit: float = 0.70
    high_speed_steer_limit: float = 0.30
    low_speed_mps: float = 1.0
    high_speed_mps: float = 3.0
    derivative_gain_s: float = 0.10
    max_steer_rate_per_s: float = 1.8
    haptic_low_ttc_s: float = 3.5
    haptic_mid_ttc_s: float = 2.0
    haptic_high_ttc_s: float = 1.0
    haptic_low_distance_m: float = 8.0
    haptic_mid_distance_m: float = 4.5
    haptic_high_distance_m: float = 2.0
    vision_priority: bool = True
    gps_lateral_weight: float = 0.15
    gps_lateral_limit_m: float = 0.50


@dataclass
class PurePursuitState:
    previous_raw_steer: float = 0.0
    previous_command_steer: float = 0.0
    previous_time_s: float | None = None


@dataclass
class PurePursuitCommand:
    steering: float
    raw_steering: float
    haptic: str
    goal_forward_m: float
    goal_left_m: float
    nearest_forward_m: float | None
    reason: str
    obstacle_count: int
    road_clamped: bool = False
    road_visible: bool = False
    vision_center_left_m: float | None = None
    gps_hint_left_m: float = 0.0
    path_blocked: bool = False


def bearing_deg_from_x(x_px: float, width: int = PANO_WIDTH, yaw_fov_deg: float = PANO_YAW_FOV_DEG) -> float:
    """Equirect column -> bearing relative to straight-ahead, degrees.

    Left half of the panorama is negative (left), right half positive (right),
    matching the planner's steer sign (+ = right)."""
    return (x_px / max(1, width) - 0.5) * yaw_fov_deg


def polar_to_xz(range_m: float, bearing_deg: float) -> tuple[float, float]:
    """(range, bearing) -> (x lateral +right, z forward) in the trike frame."""
    theta = math.radians(bearing_deg)
    return range_m * math.sin(theta), range_m * math.cos(theta)


def _steer_limit_for_speed(speed_mps: float, config: PurePursuitConfig) -> float:
    if speed_mps <= config.low_speed_mps:
        return config.low_speed_steer_limit
    if speed_mps >= config.high_speed_mps:
        return config.high_speed_steer_limit
    ratio = (speed_mps - config.low_speed_mps) / max(0.01, config.high_speed_mps - config.low_speed_mps)
    return config.low_speed_steer_limit + ratio * (config.high_speed_steer_limit - config.low_speed_steer_limit)


def _road_limits(
    road_bounds_left_m: tuple[float, float] | None,
    config: PurePursuitConfig,
) -> tuple[float, float, bool]:
    """Return safe centerline limits in left-positive trike coordinates."""
    fixed = max(0.0, config.road_half_width_m - config.road_edge_margin_m)
    fixed_lower, fixed_upper = -fixed, fixed
    if road_bounds_left_m is None:
        return fixed_lower, fixed_upper, False
    lower, upper = sorted(map(float, road_bounds_left_m))
    lower = max(fixed_lower, lower + config.road_edge_margin_m)
    upper = min(fixed_upper, upper - config.road_edge_margin_m)
    if upper - lower < 0.25:
        return fixed_lower, fixed_upper, False
    return lower, upper, True


def _footprint_intersects_corridor(
    obstacle: BevObstacle,
    corridor_half_width_m: float,
    padding_m: float = 0.0,
    corridor_center_x_m: float = 0.0,
) -> bool:
    corridor_left = float(corridor_center_x_m) - float(corridor_half_width_m)
    corridor_right = float(corridor_center_x_m) + float(corridor_half_width_m)
    obstacle_left = obstacle.left_m - float(padding_m)
    obstacle_right = obstacle.right_m + float(padding_m)
    return obstacle_right >= corridor_left and obstacle_left <= corridor_right


def _nearest_corridor_obstacle(
    obstacles: list[BevObstacle],
    config: PurePursuitConfig,
    path_goal_left_m: float = 0.0,
) -> BevObstacle | None:
    # Obstacles use +right while path goals use +left.
    path_center_x_m = -float(path_goal_left_m)
    candidates = [
        obstacle
        for obstacle in obstacles
        if 0.0 < obstacle.z_m <= config.haptic_low_distance_m
        and _footprint_intersects_corridor(
            obstacle,
            config.corridor_half_width_m,
            config.obstacle_padding_m,
            path_center_x_m,
        )
    ]
    return min(candidates, key=lambda obstacle: obstacle.z_m) if candidates else None


def _vision_first_base_goal(
    route_goal_left_m: float,
    road_bounds_left_m: tuple[float, float] | None,
    config: PurePursuitConfig,
) -> tuple[float, float | None, float, tuple[float, float], bool, str]:
    lower, upper, road_visible = _road_limits(road_bounds_left_m, config)
    if not config.vision_priority:
        goal = max(lower, min(upper, float(route_goal_left_m)))
        return goal, None, float(route_goal_left_m), (lower, upper), road_visible, "gps_route"
    if not road_visible:
        # A missing/invalid road mask must not turn a bad GPS fix into a lateral
        # command. Decay toward straight while perception recovers.
        return 0.0, None, 0.0, (lower, upper), False, "vision_straight_fallback"
    vision_center = 0.5 * (lower + upper)
    gps_delta = max(
        -config.gps_lateral_limit_m,
        min(config.gps_lateral_limit_m, float(route_goal_left_m) - vision_center),
    )
    gps_hint = max(0.0, min(1.0, config.gps_lateral_weight)) * gps_delta
    goal = max(lower, min(upper, vision_center + gps_hint))
    reason = "vision_road_center" if abs(gps_hint) < 1e-4 else "vision_road_center_gps_hint"
    return goal, vision_center, gps_hint, (lower, upper), True, reason


def _avoidance_goal_left_m(
    obstacles: list[BevObstacle],
    base_goal_left_m: float,
    safe_limits_left_m: tuple[float, float],
    config: PurePursuitConfig,
) -> tuple[float, str, BevObstacle | None, bool]:
    nearest = _nearest_corridor_obstacle(obstacles, config, base_goal_left_m)
    if nearest is None:
        return base_goal_left_m, "clear", None, False

    lower, upper = safe_limits_left_m
    # Convert the obstacle footprint from +right to a +left interval. The
    # padding includes vehicle clearance; candidate values are trike centers.
    obstacle_low_left = -nearest.right_m
    obstacle_high_left = -nearest.left_m
    candidates = [
        (obstacle_high_left + config.obstacle_padding_m, "left"),
        (obstacle_low_left - config.obstacle_padding_m, "right"),
    ]
    feasible = [
        (max(lower, min(upper, candidate)), side)
        for candidate, side in candidates
        if lower <= candidate <= upper
    ]
    if not feasible:
        label = nearest.label or "object"
        return base_goal_left_m, f"blocked_{label}_{nearest.z_m:.1f}m", nearest, True
    goal, side = min(feasible, key=lambda item: abs(item[0] - base_goal_left_m))
    label = nearest.label or "object"
    return goal, f"avoid_{label}_{side}_{nearest.z_m:.1f}m", nearest, False

def _clamp_goal_to_road(
    goal_left_m: float,
    reason: str,
    config: PurePursuitConfig,
) -> tuple[float, str, bool]:
    allowed = max(0.0, config.road_half_width_m - config.road_edge_margin_m)
    clamped = max(-allowed, min(allowed, goal_left_m))
    if abs(clamped - goal_left_m) <= 1e-6:
        return goal_left_m, reason, False
    return clamped, f"{reason}_road_clamped", True


def _pure_pursuit_raw_steer(goal_forward_m: float, goal_left_m: float, config: PurePursuitConfig) -> float:
    lookahead = max(0.25, math.hypot(goal_forward_m, goal_left_m))
    alpha = math.atan2(goal_left_m, goal_forward_m)
    curvature = 2.0 * math.sin(alpha) / lookahead
    wheel_angle = math.atan(config.wheelbase_m * curvature)
    max_angle = math.radians(max(1.0, config.max_wheel_angle_deg))
    # BEV goals use left-positive lateral coordinates. The existing trike
    # command convention is negative=left, positive=right.
    return max(-1.0, min(1.0, -wheel_angle / max_angle))


def haptic_level(
    nearest_forward_m: float | None,
    speed_mps: float,
    config: PurePursuitConfig,
) -> str:
    if nearest_forward_m is None:
        return "none"
    if nearest_forward_m <= config.haptic_high_distance_m:
        return "high"
    if nearest_forward_m <= config.haptic_mid_distance_m:
        return "mid"
    if nearest_forward_m <= config.haptic_low_distance_m:
        return "low"

    ttc = nearest_forward_m / max(0.2, speed_mps)
    if ttc <= config.haptic_high_ttc_s:
        return "high"
    if ttc <= config.haptic_mid_ttc_s:
        return "mid"
    if ttc <= config.haptic_low_ttc_s:
        return "low"
    return "none"


def plan_pure_pursuit(
    obstacles: list[BevObstacle],
    speed_mps: float,
    now_s: float,
    state: PurePursuitState,
    config: PurePursuitConfig,
    route_goal_forward_m: float | None = None,
    route_goal_left_m: float = 0.0,
    road_bounds_left_m: tuple[float, float] | None = None,
) -> PurePursuitCommand:
    (
        base_goal_left_m,
        vision_center_left_m,
        gps_hint_left_m,
        safe_limits,
        road_visible,
        base_reason,
    ) = _vision_first_base_goal(route_goal_left_m, road_bounds_left_m, config)
    goal_left_m, avoidance_reason, nearest, path_blocked = _avoidance_goal_left_m(
        obstacles, base_goal_left_m, safe_limits, config
    )
    reason = base_reason if avoidance_reason == "clear" else f"{base_reason}_{avoidance_reason}"

    lower, upper = safe_limits
    bounded = max(lower, min(upper, goal_left_m))
    road_clamped = abs(bounded - goal_left_m) > 1e-6
    goal_left_m = bounded
    goal_left_m, reason, fixed_clamped = _clamp_goal_to_road(goal_left_m, reason, config)
    road_clamped = road_clamped or fixed_clamped

    # Vision priority always aims forward at a stable metric lookahead. GPS
    # controls route progress, but a noisy fix cannot make the local goal point
    # sideways or behind the trike.
    goal_forward_m = config.lookahead_m if config.vision_priority else (
        config.lookahead_m
        if route_goal_forward_m is None
        else max(0.25, float(route_goal_forward_m))
    )
    raw = _pure_pursuit_raw_steer(goal_forward_m, goal_left_m, config)

    dt = 0.05 if state.previous_time_s is None else max(0.01, now_s - state.previous_time_s)
    lead = raw + config.derivative_gain_s * (raw - state.previous_raw_steer) / dt
    steer_limit = min(config.normal_steer_limit, _steer_limit_for_speed(speed_mps, config))
    if speed_mps <= config.low_speed_mps and avoidance_reason != "clear":
        steer_limit = min(config.low_speed_steer_limit, max(steer_limit, 0.55))
    limited = max(-steer_limit, min(steer_limit, lead))

    max_delta = config.max_steer_rate_per_s * dt
    command = max(
        state.previous_command_steer - max_delta,
        min(state.previous_command_steer + max_delta, limited),
    )
    command = max(-1.0, min(1.0, command))

    nearest_forward = nearest.z_m if nearest is not None else None
    haptic = haptic_level(nearest_forward, speed_mps, config)
    if path_blocked:
        haptic = "high"
    elif not road_visible and haptic in {"none", "low"}:
        haptic = "mid"
    elif road_clamped and haptic in {"none", "low"}:
        haptic = "mid"

    state.previous_raw_steer = raw
    state.previous_command_steer = command
    state.previous_time_s = now_s

    return PurePursuitCommand(
        steering=command,
        raw_steering=raw,
        haptic=haptic,
        goal_forward_m=goal_forward_m,
        goal_left_m=goal_left_m,
        nearest_forward_m=nearest_forward,
        reason=reason,
        obstacle_count=len(obstacles),
        road_clamped=road_clamped,
        road_visible=road_visible,
        vision_center_left_m=vision_center_left_m,
        gps_hint_left_m=gps_hint_left_m,
        path_blocked=path_blocked,
    )


def _bbox_center_x(det) -> float:
    x1, _, x2, _ = det.bbox
    return (x1 + x2) / 2.0


def detections_to_obstacles(
    detections,
    depth_provider: DepthProvider,
    config: BevConfig,
    pano_width: int = PANO_WIDTH,
) -> list[BevObstacle]:
    """Turn panorama detections into BEV obstacles using the depth provider."""
    obstacles: list[BevObstacle] = []
    for det in detections:
        range_m = depth_provider(det)
        if range_m is None or not math.isfinite(range_m):
            continue
        if range_m < config.min_range_m or range_m > config.max_range_m:
            continue
        x1, _, x2, _ = det.bbox
        bearing = bearing_deg_from_x(_bbox_center_x(det), pano_width)
        left_bearing = bearing_deg_from_x(min(x1, x2), pano_width)
        right_bearing = bearing_deg_from_x(max(x1, x2), pano_width)
        x_m, z_m = polar_to_xz(range_m, bearing)
        projected_edges = sorted((
            polar_to_xz(range_m, left_bearing)[0],
            polar_to_xz(range_m, right_bearing)[0],
        ))
        label = str(getattr(det, "label", ""))
        nominal_width = NOMINAL_OBSTACLE_WIDTH_M.get(label.lower(), DEFAULT_OBSTACLE_WIDTH_M)
        projected_width = max(0.0, projected_edges[1] - projected_edges[0])
        width_m = max(projected_width, nominal_width)
        left_m = x_m - width_m * 0.5
        right_m = x_m + width_m * 0.5
        obstacles.append(BevObstacle(
            x_m=x_m,
            z_m=z_m,
            bearing_deg=bearing,
            range_m=range_m,
            label=label,
            left_m=left_m,
            right_m=right_m,
            width_m=width_m,
        ))
    return obstacles



class StereoDepthProvider:
    """Depth for panorama detections via the static equirect->camera index map.

    Build the index once after the SIFT homography locks, push in the latest
    per-camera depth frames each loop, then call the instance with a detection to
    get a forward range in metres (median of valid depth inside its box), or None.
    """

    def __init__(self, cam_hw: tuple[int, int] = (400, 640), max_samples_per_side: int = 25):
        self.cam_hw = cam_hw
        self.max_samples = max_samples_per_side
        self.idx_slot = None
        self.idx_u = None
        self.idx_v = None
        self.depths: list = [None, None, None]  # [center, left, right] uint16 mm

    @property
    def ready(self) -> bool:
        return self.idx_slot is not None

    def build_index(self, pano_state) -> None:
        from panorama_core import build_equirect_source_index
        maps = build_equirect_source_index(pano_state, self.cam_hw)
        if maps is not None:
            self.idx_slot, self.idx_u, self.idx_v = maps

    def set_depths(self, center, left, right) -> None:
        """Per-camera depth arrays (uint16 mm) matching the RGB stitch order.
        Pass already-fused depth here if using a DepthFuser (see depth.py)."""
        self.depths = [center, left, right]

    def __call__(self, det) -> Optional[float]:
        if not self.ready:
            return None
        h_eq, w_eq = self.idx_slot.shape
        x1, y1, x2, y2 = det.bbox
        x1 = max(0, min(w_eq - 1, int(x1))); x2 = max(0, min(w_eq - 1, int(x2)))
        y1 = max(0, min(h_eq - 1, int(y1))); y2 = max(0, min(h_eq - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        ys = np.linspace(y1, y2 - 1, min(self.max_samples, y2 - y1)).astype(int)
        xs = np.linspace(x1, x2 - 1, min(self.max_samples, x2 - x1)).astype(int)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        slot = self.idx_slot[gy, gx]
        u = self.idx_u[gy, gx]
        v = self.idx_v[gy, gx]
        collected = []
        for s in (0, 1, 2):
            d = self.depths[s]
            if d is None:
                continue
            m = slot == s
            if not m.any():
                continue
            dh, dw = d.shape[:2]
            du = np.clip(u[m], 0, dw - 1)
            dv = np.clip(v[m], 0, dh - 1)
            vals = d[dv, du].astype(np.float32)
            vals = vals[vals > 0]  # 0 = invalid stereo depth
            if vals.size:
                collected.append(vals)
        if not collected:
            return None
        allv = np.concatenate(collected)
        return float(np.median(allv)) / 1000.0  # mm -> m


def render_bev(obstacles: list[BevObstacle], config: BevConfig, size_px: int = 400):
    """Top-down debug image: trike at bottom-centre, +z up, obstacles as dots."""
    img = np.full((size_px, size_px, 3), 30, dtype=np.uint8)
    scale = size_px / (2.0 * config.max_range_m)  # px per metre
    cx = size_px // 2
    cy = size_px - 1
    # corridor shading
    for o in obstacles:
        px1 = int(cx + o.left_m * scale)
        px2 = int(cx + o.right_m * scale)
        py = int(cy - o.z_m * scale)
        if 0 <= py < size_px and px2 >= 0 and px1 < size_px:
            in_corr = (
                _footprint_intersects_corridor(o, config.corridor_half_width_m)
                and o.z_m <= config.corridor_range_m
            )
            color = (0, 0, 255) if in_corr else (0, 200, 0)
            left_px = max(0, min(px1, px2))
            right_px = min(size_px - 1, max(px1, px2))
            img[max(0, py - 3):py + 4, left_px:right_px + 1] = color
    # trike marker
    img[cy - 4:cy, cx - 2:cx + 3] = (255, 255, 255)
    return img
