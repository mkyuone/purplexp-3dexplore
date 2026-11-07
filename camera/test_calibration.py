"""Deterministic calibration checks that run without a camera or MediaPipe."""

from dataclasses import replace
from math import cos, dist, radians, sin
import json
import unittest

from calibration import Calibration, CalibrationFitError, Pose, TARGETS, fit, fit_corners, steady_pose


TILT = radians(35)


def pose(x: float, y: float, travel: float = 0.0, *, width: float = 0.50, height: float = 0.38) -> Pose:
    """A camera below the screen with mirrored RGB and a slightly skewed reach."""
    across = (0.5 - x) * width + (y - 0.5) * 0.03
    down = (y - 0.5) * height
    tip = (
        0.04 + across,
        -0.10 + down * cos(TILT) - travel * sin(TILT),
        0.78 + down * sin(TILT) + travel * cos(TILT),
    )
    palm = tip[0] + 0.02, tip[1] + 0.07 * cos(TILT), tip[2] + 0.07 * sin(TILT)
    return Pose(tip, palm)


def ray_pose(x: float, y: float, depth: float) -> Pose:
    tip = (0.5 - x) * 0.60 * depth, (y - 0.5) * 0.45 * depth, depth
    return Pose(tip, (tip[0], tip[1] + 0.04, depth + 0.025))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.points = [pose(x, y) for x, y in TARGETS]
        self.calibration = fit(self.points)

    def assertProjected(self, hand, expected):
        point = self.calibration.project(hand)
        self.assertIsNotNone(point)
        for actual, value in zip(point, expected):
            self.assertAlmostEqual(actual, value)

    def test_reach_and_depth_are_screen_relative_with_camera_tilt_and_skew(self):
        self.assertFalse(self.calibration.uses_ray_projection)
        self.assertFalse(self.calibration.center_adjusted)
        for x, y in (*TARGETS, (0, 0), (1, 0), (1, 1), (0, 1)):
            self.assertProjected(pose(x, y), (x, y, 0))
        self.assertAlmostEqual(self.calibration.viewport_width_m, 0.50)
        self.assertProjected(pose(0.5, 0.5, 0.08), (0.5, 0.5, 0.16))
        self.assertProjected(pose(0.5, 0.5, -0.08), (0.5, 0.5, -0.16))

    def test_overscan_clamps_edges_but_rejects_hands_outside_the_reach_volume(self):
        self.assertProjected(pose(-0.05, 1.05), (0, 1, 0))
        self.assertIsNone(self.calibration.project(pose(-0.11, 0.5)))
        self.assertIsNone(self.calibration.project(pose(0.5, 1.11)))
        self.assertIsNone(self.calibration.project(pose(0.5, 0.5, 0.35)))
        self.assertIsNone(self.calibration.project(Pose((0, 0, float('nan')), (0, 0, 0.7))))

    def test_calibration_accepts_a_compact_comfortable_rectangle(self):
        def compact(x, y, travel=0):
            return pose(x, y, travel, width=0.24, height=0.18)

        calibration = fit([compact(x, y) for x, y in TARGETS])
        point = calibration.project(compact(0.5, 0.5, 0.11))
        self.assertIsNotNone(point)
        self.assertAlmostEqual(point[2], 0.11 / 0.24)

    def test_calibration_accepts_modest_depth_and_aim_variation(self):
        offsets = ((0.025, 0.01, 0.048), (-0.02, 0.03, -0.032), (0.03, -0.02, 0.050),
                   (-0.025, -0.015, -0.025), (0.035, -0.03, -0.055))
        samples = [pose(x + dx, y + dy, depth) for (x, y), (dx, dy, depth) in zip(TARGETS, offsets)]
        calibration = fit(samples)
        for sample, target in zip(samples, TARGETS):
            self.assertLess(dist(calibration.aim(sample.tip), target), 0.10)
            self.assertIsNotNone(calibration.project(sample))

    def test_folded_corner_order_rejects_a_mapping(self):
        a, b, c, d, center = self.points
        with self.assertRaises(CalibrationFitError) as failure:
            fit([a, c, b, d, center])
        self.assertTrue(failure.exception.targets)
        self.assertNotIn(4, failure.exception.targets)

    def test_rejects_collapsed_and_incomplete_captures(self):
        with self.assertRaisesRegex(ValueError, 'spread the left and right'):
            fit([self.points[-1]] * len(TARGETS))
        with self.assertRaisesRegex(ValueError, 'five points'):
            fit(self.points[:-1])

    def test_center_is_advisory_when_corners_define_a_usable_plane(self):
        for center in (pose(0.5, 0.5, -0.12), pose(0.76, 0.53),
                       replace(self.points[-1], palm=pose(0.5, 0.5, 0.15).palm)):
            with self.subTest(center=center):
                calibration = fit([*self.points[:-1], center])
                self.assertTrue(calibration.center_adjusted)
                self.assertFalse(calibration.uses_ray_projection)
                neutral = calibration.project(self.points[-1])
                self.assertIsNotNone(neutral)
                self.assertAlmostEqual(neutral[2], 0)
                self.assertEqual(Calibration.from_dict(calibration.to_dict()), calibration)

    def test_one_palm_depth_outlier_does_not_erase_valid_corner_aim(self):
        points = list(self.points)
        points[1] = replace(points[1], palm=pose(*TARGETS[1], 0.18).palm)
        self.assertIsNone(fit_corners(points[:4]))
        calibration = fit(points)
        self.assertFalse(calibration.uses_ray_projection)
        self.assertAlmostEqual(calibration.project(self.points[-1])[2], 0)

    def test_projection_fallback_keeps_well_ordered_rays_at_irregular_depths(self):
        depths = (0.55, 0.82, 0.56, 0.84, 0.96)
        points = [ray_pose(x, y, depth) for (x, y), depth in zip(TARGETS, depths)]
        self.assertIsNone(fit_corners(points[:4]))
        calibration = fit(points)
        self.assertTrue(calibration.uses_ray_projection)
        self.assertTrue(calibration.center_adjusted)
        for point, (x, y) in zip(points, TARGETS):
            actual = calibration.aim(point.tip)
            self.assertAlmostEqual(actual[0], x)
            self.assertAlmostEqual(actual[1], y)
        shallow = calibration.project(ray_pose(0.24, 0.60, 0.65))
        deep = calibration.project(ray_pose(0.24, 0.60, 0.77))
        self.assertAlmostEqual(shallow[0], deep[0])
        self.assertAlmostEqual(shallow[1], deep[1])
        self.assertGreater(deep[2], shallow[2])
        self.assertEqual(Calibration.from_dict(calibration.to_dict()), calibration)

        with self.assertRaises(CalibrationFitError):
            fit_corners([points[0], points[2], points[1], points[3]])
        with self.assertRaises(CalibrationFitError):
            fit_corners([ray_pose(0.5, y, depth) for (_, y), depth in zip(TARGETS, depths)])

    def test_steady_capture_requires_five_valid_samples_and_rejects_palm_motion(self):
        center = self.points[-1]
        samples = [pose(0.5 + offset, 0.5) for offset in (-0.01, 0, 0.005, -0.003, 0.01)]
        steady = steady_pose(samples)
        self.assertEqual(steady, center)
        self.assertIsNone(steady_pose(samples[:4]))
        self.assertIsNone(steady_pose([*samples[:-1], replace(center, palm=pose(0.5, 0.5, 0.04).palm)]))
        self.assertIsNone(steady_pose([*samples[:-1], replace(center, tip=(0, 0, 2))]))

    def test_profiles_round_trip_and_revalidate_saved_points(self):
        saved = json.loads(json.dumps(self.calibration.to_dict()))
        self.assertEqual(Calibration.from_dict(saved), self.calibration)
        saved['points'][0]['tip'][2] = float('inf')
        with self.assertRaisesRegex(ValueError, 'valid depth'):
            Calibration.from_dict(saved)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            Calibration.from_dict({'version': 2})


if __name__ == '__main__':
    unittest.main()
