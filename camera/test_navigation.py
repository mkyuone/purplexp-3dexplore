from dataclasses import replace
from math import exp, isclose
from unittest import TestCase

from calibration import Pose, TARGETS, fit
from depth_range import DepthRange
from navigation import CalibratedMapper, DEPTH_WIDTH_M, FuzzyMapper, IMAGE_BOTTOM, IMAGE_LEFT, IMAGE_RIGHT, IMAGE_TOP, soft_axis
from tracking import Candidate, GripFeatures, Hand, HandTracker


def hand(identifier='one', x=0.5, y=0.5, depth=0.7, grip='open'):
    return Hand(identifier, Pose((0, 0, depth), (0, 0.04, depth)), grip, True, (x, y))


def pinch(x=0.5, depth=0.7, *, side='Left', palm_x=0):
    pose = Pose((palm_x, 0, depth), (palm_x, 0.04, depth))
    return Candidate(pose, side, GripFeatures(0.24, 1, 1), 0.020, True, (x, 0.5), (x, 0.5), 0.15)


class CustomDepthRangeTests(TestCase):
    def test_saved_endpoints_require_a_supported_and_ordered_twenty_centimeter_span(self):
        minimum = DepthRange(0.4, 0.6)
        self.assertEqual(DepthRange.from_dict(minimum.to_dict()), minimum)
        for value in (None, {}, {'nearM': '0.4', 'farM': 0.8}, {'nearM': True, 'farM': 1.5},
                      {'nearM': float('nan'), 'farM': 0.8}, {'nearM': 0.2, 'farM': 0.8},
                      {'nearM': 0.4, 'farM': 1.6}, {'nearM': 0.4, 'farM': 0.59},
                      {'nearM': 0.9, 'farM': 0.4}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DepthRange.from_dict(value)

    def test_fuzzy_depth_range_saturates_while_smoothing_and_rgb_fallback_still_apply(self):
        mapper = FuzzyMapper(DepthRange(0.5, 0.9))
        measured = hand(depth=0.42, grip='pinch')
        near, = mapper.update([measured], 1)
        self.assertEqual((near.depth, near.depth_valid), (-0.5, True))

        moved = hand(x=0.60, depth=0.7, grip='pinch')
        middle, = mapper.update([moved], 1.04)
        self.assertLess(near.depth, middle.depth)
        self.assertLess(middle.depth, 0)

        dropout = replace(moved, pose=None, pointing=False, image=(0.65, 0.5))
        image_only, = mapper.update([dropout], 1.08)
        self.assertFalse(image_only.depth_valid)
        self.assertLess(image_only.x, middle.x)

        far, = mapper.update([hand(x=0.7, depth=1.0, grip='pinch')], 1.12)
        self.assertEqual((far.depth, far.depth_valid), (0.5, True))
        self.assertIsNone(mapper.neutral_depth_m)

    def test_calibrated_range_replaces_only_depth_reach_and_keeps_rgb_pinch_continuity(self):
        def point(x, y, palm_depth=0.7):
            x, y = (0.5 - x) * 0.5, (y - 0.5) * 0.4
            return Pose((x, y, 0.7), (x, y + 0.04, palm_depth))

        calibration = fit([point(x, y) for x, y in TARGETS])
        custom = CalibratedMapper(calibration, lambda x, y, z: point(x, y).tip,
                                  DepthRange(0.5, 0.9))
        outside_legacy_reach = point(0.5, 0.5, 1.2)
        self.assertIsNone(calibration.project(outside_legacy_reach))
        pinch = replace(hand(grip='pinch'), pose=outside_legacy_reach)
        mapped, = custom.update([pinch], 1, 1)
        self.assertEqual((mapped.x, mapped.y, mapped.depth, mapped.depth_valid), (0.5, 0.5, 0.5, True))

        dropout = replace(pinch, pose=None, pointing=False, image=(0.6, 0.5))
        continued, = custom.update([dropout], 1.04, 2)
        self.assertAlmostEqual(continued.x, 0.6)
        self.assertFalse(continued.depth_valid)
        out_of_bounds = replace(pinch, pose=point(1.2, 0.5))
        self.assertEqual(custom.update([out_of_bounds], 1.08, 3), ())


class CalibratedMappingTests(TestCase):
    def setUp(self):
        points = [Pose(((0.5 - x) * 0.68, (y - 0.5) * 0.48, 0.7),
                       ((0.5 - x) * 0.68, (y - 0.5) * 0.48 + 0.04, 0.72)) for x, y in TARGETS]
        self.calibration = fit(points)

    @staticmethod
    def project_image(x, y, z):
        return (0.5 - x) * 0.95 * z / 0.7, (y - 0.5) * 0.48 * z / 0.7, z

    def test_reset_gesture_does_not_require_spatial_reach(self):
        for palm in ((0.65, 0, 0.7), (0, 0, 1.2)):
            with self.subTest(palm=palm):
                mapper = CalibratedMapper(self.calibration, self.project_image)
                outside = replace(hand(), pose=Pose(palm, palm))
                self.assertIsNone(self.calibration.project(outside.pose))

                reset, = mapper.update([replace(outside, pointing=False, thumbs_up=True)], 1, 1)
                self.assertEqual((reset.id, reset.grip, reset.thumbs_up, reset.depth_valid), ('one', 'open', True, True))
                self.assertTrue(0 <= reset.x <= 1 and 0 <= reset.y <= 1)

                self.assertEqual(mapper.update([replace(outside, grip='fist')], 1.04, 2), ())
                self.assertEqual(mapper.update([replace(outside, grip='pinch')], 1.08, 3), ())
                self.assertEqual(mapper.update([replace(outside, pose=None, pointing=False, thumbs_up=True)], 1.12, 4), ())

    def test_rgb_pinch_starts_after_an_open_aim_with_or_without_measured_fingertip_depth(self):
        for measured_open in (True, False):
            with self.subTest(measured_open=measured_open):
                mapper = CalibratedMapper(self.calibration, self.project_image)
                tip = self.project_image(0.55, 0.5, 0.8)
                palm = (tip[0], 0.04, 0.82)
                observed = replace(hand(x=0.55), pose=Pose(tip, palm) if measured_open else None,
                                   pointing=measured_open)
                opened, = mapper.update([observed], 1, 1)
                self.assertEqual(opened.depth_valid, measured_open)

                # Palm depth still exists but neither fingertip can drive the calibrated aim.
                optical = replace(observed, pose=Pose(palm, palm), pointing=False,
                                  grip='pinch', image=(0.57, 0.5))
                grabbed, = mapper.update([optical], 1.04, 2)
                self.assertEqual((grabbed.id, grabbed.grip, grabbed.depth_valid), (opened.id, 'pinch', False))
                self.assertGreater(grabbed.x, opened.x)

                tip = self.project_image(*optical.image, 0.95)
                measured = replace(optical, pose=Pose(tip, (tip[0], 0.04, 0.97)), pointing=True)
                restored, = mapper.update([measured], 1.08, 3)
                self.assertTrue(restored.depth_valid)
                self.assertAlmostEqual(restored.x, grabbed.x)
                self.assertAlmostEqual(restored.y, grabbed.y)

    def test_outside_optical_aim_does_not_arm_a_pinch_and_depth_reach_only_pauses_held_depth(self):
        mapper = CalibratedMapper(self.calibration, self.project_image)
        outside = replace(hand(x=0.98), pose=None, pointing=False)
        self.assertEqual(mapper.update([outside], 1, 1), ())
        self.assertEqual(mapper.update([replace(outside, image=(0.5, 0.5), grip='pinch')], 1.04, 2), ())

        initial, = mapper.update([hand(grip='pinch')], 1.08, 3)
        outside_depth = replace(hand(x=0.55, grip='pinch'), pose=Pose((0, 0, 1.05), (0, 0.04, 1.1)))
        continued, = mapper.update([outside_depth], 1.12, 4)
        self.assertFalse(continued.depth_valid)
        self.assertGreater(continued.x, initial.x)
        edge, = mapper.update([replace(outside_depth, image=outside.image)], 1.16, 5)
        self.assertEqual((edge.x, edge.grip, edge.depth_valid), (1.0, 'pinch', False))
        released, = mapper.update([outside], 1.20, 6)
        self.assertEqual(released.grip, 'open')
        self.assertEqual(mapper.update([replace(outside, image=(0.5, 0.5), grip='pinch')], 1.24, 7), ())


class FuzzyMappingTests(TestCase):
    def test_aim_mirrors_a_broad_region_and_reaches_soft_edges(self):
        mapper = FuzzyMapper()
        center, = mapper.update([hand()], 1)
        self.assertEqual((center.x, center.y, center.depth), (0.5, 0.5, 0))
        upper_right, = mapper.update([hand(x=IMAGE_LEFT, y=IMAGE_TOP)], 1.1)
        self.assertEqual((upper_right.x, upper_right.y), (1, 0))
        lower_left, = mapper.update([hand(x=IMAGE_RIGHT, y=IMAGE_BOTTOM)], 1.2)
        self.assertEqual((lower_left.x, lower_left.y), (0, 1))
        linear = 0.01
        eased = soft_axis(IMAGE_LEFT + linear * (IMAGE_RIGHT - IMAGE_LEFT), IMAGE_LEFT, IMAGE_RIGHT)
        self.assertGreater(eased, 0)
        self.assertLess(eased, linear)

    def test_cursor_deadzone_keeps_a_still_hand_stable_without_losing_larger_motion(self):
        mapper = FuzzyMapper()
        center, = mapper.update([hand()], 1)
        jittered, = mapper.update([hand(x=0.5005, y=0.5005)], 1.04)
        self.assertEqual((jittered.x, jittered.y), (center.x, center.y))
        moved, = mapper.update([hand(x=0.55)], 1.08)
        self.assertLess(moved.x, 0.45)
        self.assertTrue(isclose(moved.y, 0.5))

    def test_neutral_depth_stays_fixed_in_grips_and_adapts_when_open(self):
        mapper = FuzzyMapper()
        mapper.update([hand()], 1)
        grabbed, = mapper.update([hand(depth=0.80, grip='pinch')], 1.1)
        self.assertAlmostEqual(grabbed.depth, 0.10 / DEPTH_WIDTH_M)
        mapper.update([hand(depth=0.90, grip='pinch')], 2)
        self.assertAlmostEqual(mapper.neutral_depth_m, 0.7)
        mapper.update([hand(depth=0.90)], 2.1)
        self.assertGreater(mapper.neutral_depth_m, 0.7)
        self.assertLess(mapper.neutral_depth_m, 0.9)

    def test_both_hands_share_one_depth_reference_and_keep_grip_semantics(self):
        mapper = FuzzyMapper()
        first, second = mapper.update([hand('a', depth=0.6), hand('b', depth=0.8)], 1)
        self.assertAlmostEqual(second.depth - first.depth, 0.2 / DEPTH_WIDTH_M)
        first, second = mapper.update([hand('a', depth=0.6, grip='fist'), hand('b', depth=0.8, grip='fist')], 1.1)
        self.assertEqual((first.grip, second.grip), ('fist', 'fist'))
        self.assertAlmostEqual(second.depth - first.depth, 0.2 / DEPTH_WIDTH_M)
        first, second = mapper.update([hand('a', depth=0.6, grip='fist'), hand('b', depth=0.9, grip='fist')], 1.2)
        self.assertGreater(second.depth - first.depth, 0.48)
        self.assertAlmostEqual(mapper.neutral_depth_m, 0.7)

    def test_duplicate_frames_are_idempotent_and_restart_clears_old_reference(self):
        mapper = FuzzyMapper()
        mapper.update([hand()], 1)
        current = mapper.update([hand(depth=0.8)], 1.1)
        neutral = mapper.neutral_depth_m
        for _ in range(10):
            self.assertEqual(mapper.update([hand(depth=0.8)], 1.1), current)
        self.assertEqual(mapper.neutral_depth_m, neutral)
        self.assertEqual(mapper.update([hand()], 0.5), ())
        mapper.reset()
        restarted, = mapper.update([hand(depth=0.9)], 0.1)
        self.assertEqual(restarted.depth, 0)

    def test_missing_or_invalid_samples_drop_without_publishing_stale_hands(self):
        mapper = FuzzyMapper()
        mapper.update([hand()], 1)
        self.assertEqual(mapper.update([], 1.04), ())
        invalid = [replace(hand(), image=None), hand(x=float('nan')), hand(depth=2)]
        self.assertEqual(mapper.update(invalid, 1.08), ())
        returned, = mapper.update([hand(x=0.7)], 1.12)
        self.assertLess(returned.x, 0.3)

    def test_image_only_hands_cannot_establish_or_bias_the_measured_neutral(self):
        mapper = FuzzyMapper()
        image_only = replace(hand('rgb', x=0.6), pose=None, pointing=False)
        fallback, = mapper.update([image_only], 1)
        self.assertAlmostEqual(fallback.x, 1 - soft_axis(0.6, IMAGE_LEFT, IMAGE_RIGHT))
        self.assertEqual((fallback.id, fallback.depth, fallback.depth_valid), ('rgb', 0, False))
        self.assertIsNone(mapper.neutral_depth_m)

        _, measured = mapper.update([image_only, hand('measured', depth=0.7)], 1.1)
        self.assertTrue(measured.depth_valid)
        self.assertAlmostEqual(mapper.neutral_depth_m, 0.7)
        mapper.update([image_only, hand('measured', depth=0.9)], 1.2)
        self.assertAlmostEqual(mapper.neutral_depth_m, 0.7 + 0.2 * (1 - exp(-0.1 / 1.8)))

    def test_xy_smoothing_continues_as_measured_depth_disappears_and_returns(self):
        mapper = FuzzyMapper()
        mapper.update([hand(depth=0.7)], 1)
        measured, = mapper.update([hand(depth=0.82, grip='pinch')], 1.04)
        dropout = replace(hand(x=0.55, y=0.52, grip='pinch'), pose=None, pointing=False)
        fallback, = mapper.update([dropout], 1.08)
        target_x = 1 - soft_axis(0.55, IMAGE_LEFT, IMAGE_RIGHT)
        self.assertLess(target_x, fallback.x)
        self.assertLess(fallback.x, measured.x)
        self.assertGreater(fallback.y, measured.y)
        self.assertEqual((fallback.depth, fallback.depth_valid), (0, False))
        self.assertEqual(mapper.update([dropout], 1.08), (fallback,))

        returned, = mapper.update([hand(x=0.56, y=0.53, depth=0.82, grip='pinch')], 1.12)
        self.assertLess(1 - soft_axis(0.56, IMAGE_LEFT, IMAGE_RIGHT), returned.x)
        self.assertLess(returned.x, fallback.x)
        self.assertTrue(returned.depth_valid)
        self.assertAlmostEqual(returned.depth, measured.depth)
        self.assertAlmostEqual(mapper.neutral_depth_m, 0.7)

    def test_still_hand_noise_is_damped_in_image_position_and_depth(self):
        mapper = FuzzyMapper()
        mapper.update([hand(grip='pinch')], 1)
        samples = []
        for frame in range(1, 91):
            direction = 1 if frame % 2 else -1
            sample, = mapper.update(
                [hand(x=0.5 + direction * 0.004, depth=0.7 + direction * 0.006, grip='pinch')],
                1 + frame / 30,
            )
            if frame > 30:
                samples.append(sample)
        self.assertLess(max(sample.x for sample in samples) - min(sample.x for sample in samples), 0.003)
        self.assertLess(max(sample.depth for sample in samples) - min(sample.depth for sample in samples), 0.004)
        self.assertEqual(mapper.neutral_depth_m, 0.7)

    def test_fast_deliberate_move_settles_promptly_and_duplicate_frames_do_not_filter_twice(self):
        mapper = FuzzyMapper()
        mapper.update([hand(grip='pinch')], 1)
        moved = hand(x=0.7, depth=0.82, grip='pinch')
        first = mapper.update([moved], 1 + 1 / 30)
        self.assertEqual(mapper.update([moved], 1 + 1 / 30), first)
        mapper.update([moved], 1 + 2 / 30)
        settled, = mapper.update([moved], 1.1)
        target_x = 1 - soft_axis(0.7, IMAGE_LEFT, IMAGE_RIGHT)
        self.assertLess(abs(settled.x - target_x), 0.008)
        self.assertLess(abs(settled.depth - 0.12 / DEPTH_WIDTH_M), 0.015)

    def test_established_tracking_and_mapping_respond_promptly_to_rigid_pair_movement(self):
        tracker, mapper = HandTracker(), FuzzyMapper()
        initial = [pinch(.38, .64, palm_x=-.12), pinch(.62, .76, side='Right', palm_x=.12)]
        moved = [pinch(.416, .67, palm_x=-.12), pinch(.656, .79, side='Right', palm_x=.12)]
        _, hands = tracker.update(initial, 1)
        before = mapper.update(hands, 1)
        for index in range(1, 6):
            at = 1 + index / 30
            status, hands = tracker.update(moved[::-1], at)
            self.assertEqual((status, [item.grip for item in hands]), ('tracking', ['pinch', 'pinch']))
            after = mapper.update(hands, at)
            self.assertAlmostEqual(after[1].x - after[0].x, before[1].x - before[0].x)
            self.assertAlmostEqual(after[1].depth - after[0].depth, before[1].depth - before[0].depth)
            if index == 1:
                self.assertGreater(before[0].x - after[0].x, 0.0235)
                self.assertGreater(after[0].depth - before[0].depth, 0.0225)
        self.assertGreater(before[0].x - after[0].x, 0.042)
        self.assertGreater(after[0].depth - before[0].depth, 0.040)

    def test_established_tracking_still_damps_small_image_and_depth_noise(self):
        for fps in (30, 60):
            with self.subTest(fps=fps):
                tracker, mapper = HandTracker(), FuzzyMapper()
                _, hands = tracker.update([pinch()], 1)
                mapper.update(hands, 1)
                samples = []
                for index in range(1, 91):
                    at = 1 + index / fps
                    direction = 1 if index % 2 else -1
                    _, hands = tracker.update([pinch(.5 + direction * .004, .7 + direction * .006)], at)
                    current, = mapper.update(hands, at)
                    if index > 30:
                        samples.append(current)
                self.assertLess(max(item.x for item in samples) - min(item.x for item in samples), 0.0005)
                self.assertLess(max(item.depth for item in samples) - min(item.depth for item in samples), 0.0005)

    def test_common_depth_filter_preserves_hand_separation_and_grip_baselines(self):
        mapper = FuzzyMapper()
        before = mapper.update([hand('a', depth=0.6, grip='fist'), hand('b', depth=0.8, grip='fist')], 1)
        separation = before[1].depth - before[0].depth
        for frame in range(1, 20):
            shift = 0.008 if frame % 2 else -0.008
            first, second = mapper.update(
                [hand('a', depth=0.6 + shift, grip='fist'), hand('b', depth=0.8 + shift, grip='fist')],
                1 + frame / 30,
            )
            self.assertAlmostEqual(second.depth - first.depth, separation)
        single = FuzzyMapper()
        single.update([hand()], 1)
        grabbed, = single.update([hand(x=0.7, depth=0.82, grip='pinch')], 1.04)
        self.assertAlmostEqual(grabbed.x, 1 - soft_axis(0.7, IMAGE_LEFT, IMAGE_RIGHT))
        self.assertAlmostEqual(grabbed.depth, 0.12 / DEPTH_WIDTH_M)
        unchanged, = single.update([hand(x=0.7, depth=0.82, grip='pinch')], 1.08)
        self.assertEqual(grabbed, unchanged)
