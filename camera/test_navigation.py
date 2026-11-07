from dataclasses import replace
from math import exp, isclose
from unittest import TestCase

from calibration import Pose
from navigation import DEPTH_WIDTH_M, FuzzyMapper, IMAGE_BOTTOM, IMAGE_LEFT, IMAGE_RIGHT, IMAGE_TOP, soft_axis
from tracking import Hand


def hand(identifier='one', x=0.5, y=0.5, depth=0.7, grip='open'):
    return Hand(identifier, Pose((0, 0, depth), (0, 0.04, depth)), grip, True, (x, y))


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
