"""Deterministic calibration checks that run without a camera or MediaPipe."""

from dataclasses import replace
from math import cos, radians, sin
import json
import unittest

from calibration import Calibration, Pose, TARGETS, fit, steady_pose


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

    def test_rejects_collapsed_nonplanar_and_inconsistent_palm_captures(self):
        with self.assertRaisesRegex(ValueError, 'spread the left and right'):
            fit([self.points[-1]] * len(TARGETS))
        with self.assertRaisesRegex(ValueError, 'same plane'):
            fit([*self.points[:-1], pose(0.5, 0.5, -0.12)])
        center = self.points[-1]
        far_palm = replace(center, palm=pose(0.5, 0.5, 0.12).palm)
        with self.assertRaisesRegex(ValueError, 'palm depth'):
            fit([*self.points[:-1], far_palm])
        with self.assertRaisesRegex(ValueError, 'five points'):
            fit(self.points[:-1])

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
