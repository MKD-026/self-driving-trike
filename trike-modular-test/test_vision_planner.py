#!/usr/bin/env python3
"""Deterministic checks for vision-priority local planning."""

import unittest

from bev import BevObstacle, PurePursuitConfig, PurePursuitState, plan_pure_pursuit
from trike_geometry import MAX_WHEEL_ANGLE_DEG, WHEELBASE_M


class VisionPlannerTests(unittest.TestCase):
    def config(self):
        return PurePursuitConfig(
            wheelbase_m=WHEELBASE_M,
            max_wheel_angle_deg=MAX_WHEEL_ANGLE_DEG,
            lookahead_m=5.0,
            derivative_gain_s=0.0,
            max_steer_rate_per_s=100.0,
            gps_lateral_weight=0.15,
            gps_lateral_limit_m=0.50,
        )

    def plan(self, *, road=(-1.75, 1.75), gps_left=0.0, obstacles=None):
        return plan_pure_pursuit(
            obstacles or [], 1.0, 1.0, PurePursuitState(), self.config(),
            route_goal_forward_m=0.25,
            route_goal_left_m=gps_left,
            road_bounds_left_m=road,
        )

    def test_large_bad_gps_is_bounded(self):
        cmd = self.plan(gps_left=20.0)
        self.assertTrue(cmd.road_visible)
        self.assertAlmostEqual(cmd.vision_center_left_m, 0.0)
        self.assertAlmostEqual(cmd.gps_hint_left_m, 0.075)
        self.assertAlmostEqual(cmd.goal_left_m, 0.075)
        self.assertEqual(cmd.goal_forward_m, 5.0)

    def test_missing_road_ignores_gps_and_goes_straight(self):
        cmd = self.plan(road=None, gps_left=20.0)
        self.assertFalse(cmd.road_visible)
        self.assertEqual(cmd.goal_left_m, 0.0)
        self.assertIn('vision_straight_fallback', cmd.reason)
        self.assertEqual(cmd.haptic, 'mid')

    def test_shifted_road_center_is_followed(self):
        cmd = self.plan(road=(0.2, 2.0), gps_left=0.0)
        self.assertTrue(cmd.road_visible)
        self.assertGreater(cmd.goal_left_m, 0.0)
        self.assertLess(cmd.steering, 0.0)  # planner negative means left

    def test_person_on_path_produces_road_valid_detour(self):
        person = BevObstacle(
            x_m=0.0, z_m=4.0, bearing_deg=0.0, range_m=4.0,
            label='person', left_m=-0.30, right_m=0.30, width_m=0.60,
        )
        cmd = self.plan(obstacles=[person])
        self.assertFalse(cmd.path_blocked)
        self.assertGreater(abs(cmd.goal_left_m), 0.5)
        self.assertLessEqual(abs(cmd.goal_left_m), 1.45)
        self.assertIn('avoid_person', cmd.reason)

    def test_no_road_valid_gap_is_reported_blocked(self):
        car = BevObstacle(
            x_m=0.0, z_m=3.0, bearing_deg=0.0, range_m=3.0,
            label='car', left_m=-0.95, right_m=0.95, width_m=1.90,
        )
        cmd = self.plan(obstacles=[car])
        self.assertTrue(cmd.path_blocked)
        self.assertEqual(cmd.haptic, 'high')
        self.assertIn('blocked_car', cmd.reason)


if __name__ == '__main__':
    unittest.main()
