from dataclasses import replace
from unittest import TestCase

from calibration import Pose
from tracking import Candidate, GripFeatures, HandTracker


OPEN = GripFeatures(0.8, 0, 0)
CLOSED = GripFeatures(0.30, 1, 2)


def sample(*, x=0.0, gap=0.06, features=OPEN):
    return Candidate(Pose((x, 0, 0.70), (x, 0.04, 0.72)), None, features, gap, True,
                     (0.5 + x, 0.4), (0.5 + x, 0.5))


def optical_sample(*, x=0.0, ratio=0.16, features=CLOSED):
    candidate = sample(x=x, gap=None, features=features)
    return replace(candidate, pose=Pose(candidate.pose.palm, candidate.pose.palm), pointing=False,
                   optical_pinch_ratio=ratio)


class PinchTrackingTests(TestCase):
    def pinch(self):
        tracker = HandTracker()
        tracker.update([sample()], 1.00)
        closed = sample(gap=0.025, features=CLOSED)
        tracker.update([closed], 1.04)
        _, (held,) = tracker.update([closed], 1.08)
        self.assertEqual(held.grip, 'pinch')
        return tracker, held, closed

    def test_agreeing_open_landmarks_release_even_if_measured_gap_stays_small(self):
        tracker, held, closed = self.pinch()
        disagreeing = replace(closed, optical_pinch_ratio=.63)
        _, (still_held,) = tracker.update([disagreeing], 1.12)
        self.assertEqual(still_held.grip, 'pinch')
        opening = replace(disagreeing, features=GripFeatures(.62, 1, 2))
        _, (released,) = tracker.update([opening], 1.16)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))

    def test_fresh_rgb_only_pinch_confirms_and_releases_with_other_fingers_curled(self):
        tracker = HandTracker()
        closed = replace(optical_sample(), pose=None)
        self.assertEqual(tracker.update([closed], 1), ('lost', ()))
        self.assertEqual(tracker.update([closed], 1.02), ('lost', ()))
        status, (held,) = tracker.update([closed], 1.04)
        self.assertEqual((status, held.grip), ('tracking', 'pinch'))
        self.assertIsNone(held.pose)

        no_current_closure = replace(closed, optical_pinch_ratio=None)
        self.assertEqual(tracker.update([no_current_closure], 1.08), ('lost', ()))
        opening = replace(closed, optical_pinch_ratio=.65, features=GripFeatures(.62, 3, 3))
        status, (released,) = tracker.update([opening], 1.12)
        self.assertEqual((status, released.id, released.grip), ('tracking', held.id, 'open'))
        self.assertIsNone(released.pose)
        _, (still_open,) = tracker.update([opening], 1.16)
        self.assertEqual(still_open.grip, 'open')

    def test_open_to_pinch_requires_brief_continuous_measured_confirmation(self):
        tracker = HandTracker()
        tracker.update([sample()], 1.00)
        closed = sample(gap=0.030, features=GripFeatures(0.39, 1, 2))
        _, (pending,) = tracker.update([closed], 1.04)
        self.assertEqual(pending.grip, 'open')
        _, (still_pending,) = tracker.update([closed], 1.06)
        self.assertEqual(still_pending.grip, 'open')
        _, (held,) = tracker.update([closed], 1.09)
        self.assertEqual(held.grip, 'pinch')
        self.assertEqual(held.id, pending.id)

    def test_brief_false_closure_is_canceled_by_an_open_sample(self):
        tracker = HandTracker()
        tracker.update([sample()], 1.00)
        closed = sample(gap=0.025, features=CLOSED)
        tracker.update([closed], 1.04)
        tracker.update([sample()], 1.08)
        _, (pending,) = tracker.update([closed], 1.12)
        self.assertEqual(pending.grip, 'open')
        _, (held,) = tracker.update([closed], 1.16)
        self.assertEqual(held.grip, 'pinch')

    def test_two_optical_pinches_confirm_independently_with_measured_palms(self):
        tracker = HandTracker()
        opening = [optical_sample(x=x, ratio=0.75, features=OPEN) for x in (-0.12, 0.12)]
        closed = [optical_sample(x=x) for x in (-0.12, 0.12)]
        _, initial = tracker.update(opening, 1.00)
        _, pending = tracker.update([closed[0], opening[1]], 1.04)
        self.assertEqual([hand.grip for hand in pending], ['open', 'open'])
        _, joining = tracker.update(closed[::-1], 1.08)
        self.assertEqual([hand.grip for hand in joining], ['pinch', 'open'])
        _, held = tracker.update(closed, 1.12)
        self.assertEqual([hand.grip for hand in held], ['pinch', 'pinch'])
        self.assertEqual([hand.id for hand in held], [hand.id for hand in initial])
        self.assertTrue(all(not hand.pointing and hand.pose.tip == hand.pose.palm for hand in held))

    def test_pinch_can_confirm_with_the_other_three_fingers_curled(self):
        tracker = HandTracker()
        closed = optical_sample(features=GripFeatures(.30, 3, 3))
        self.assertEqual(tracker.update([closed], 1.00), ('lost', ()))
        _, (pinch,) = tracker.update([closed], 1.04)
        self.assertEqual(pinch.grip, 'pinch')

    def test_new_optical_pinch_never_publishes_pending_neutral(self):
        tracker = HandTracker()
        first = optical_sample(x=-0.12, ratio=0.70, features=OPEN)
        second = optical_sample(x=0.12)
        _, (open_hand,) = tracker.update([first], 1.00)
        self.assertEqual(tracker.update([first, second], 1.04), ('tracking', (open_hand,)))
        _, hands = tracker.update([second, first], 1.06)
        self.assertEqual([hand.grip for hand in hands], ['open'])
        _, hands = tracker.update([first, second], 1.08)
        self.assertEqual([hand.grip for hand in hands], ['open', 'pinch'])
        self.assertEqual(hands[0].id, open_hand.id)

        tracker.reset()
        self.assertEqual(tracker.update([second], 2.00), ('lost', ()))
        _, (cold_pinch,) = tracker.update([second], 2.04)
        self.assertEqual(cold_pinch.grip, 'pinch')

    def test_new_optical_pinch_confirmation_cannot_span_an_interruption(self):
        tracker = HandTracker()
        closed = optical_sample()
        self.assertEqual(tracker.update([closed], 1.00), ('lost', ()))
        self.assertEqual(tracker.update([], 1.04), ('lost', ()))
        self.assertEqual(tracker.update([closed], 1.08), ('lost', ()))
        self.assertEqual(tracker.update([closed], 1.33), ('lost', ()))
        _, (pinch,) = tracker.update([closed], 1.37)
        self.assertEqual(pinch.grip, 'pinch')

    def test_two_pinches_can_confirm_at_a_slower_fresh_detector_cadence(self):
        tracker = HandTracker()
        opening = [sample(x=x) for x in (-.12, .12)]
        closed = [sample(x=x, gap=.025, features=CLOSED) for x in (-.12, .12)]
        tracker.update(opening, 1.00)
        _, pending = tracker.update(closed, 1.18)
        self.assertTrue(all(hand.grip == 'open' for hand in pending))
        _, held = tracker.update(closed, 1.36)
        self.assertEqual([hand.grip for hand in held], ['pinch', 'pinch'])

    def test_optical_overlap_or_conflicting_measured_gap_does_not_acquire(self):
        closed = optical_sample()
        for uncertain in (
            replace(closed, optical_pinch_ratio=float('nan')),
            replace(closed, optical_pinch_ratio=0.35),
            replace(closed, image_palm=None),
            replace(closed, features=GripFeatures(0.55, 1, 2)),
            replace(closed, features=GripFeatures(0.20, 3, 4)),
            replace(closed, pointing=True, pinch_gap=0.08),
        ):
            with self.subTest(uncertain=uncertain):
                tracker = HandTracker()
                tracker.update([sample()], 1.00)
                for at in (1.04, 1.08):
                    _, hands = tracker.update([uncertain], at)
                    self.assertFalse(any(hand.grip == 'pinch' for hand in hands))

    def test_fresh_optical_evidence_sustains_pinch_and_opening_releases_promptly(self):
        tracker = HandTracker()
        closed = optical_sample()
        tracker.update([closed], 1.00)
        _, (pinch,) = tracker.update([closed], 1.04)
        for at in (1.08, 1.12, 1.16, 1.20):
            noisy_closed = optical_sample(ratio=0.34, features=GripFeatures(0.46, 2, 3))
            _, (held,) = tracker.update([noisy_closed], at)
            self.assertEqual((held.id, held.grip), (pinch.id, 'pinch'))
        # RGB clearly separates before the world-landmark ratio catches up.
        _, (released,) = tracker.update([optical_sample(ratio=0.65)], 1.24)
        self.assertEqual((released.id, released.grip), (pinch.id, 'open'))

    def test_unknown_depth_never_starts_a_pinch_or_manufactures_a_neutral_for_a_new_closed_hand(self):
        tracker = HandTracker()
        closed = sample(gap=None, features=CLOSED)
        for at in (1.00, 1.04, 1.08):
            _, (hand,) = tracker.update([closed], at)
            self.assertEqual(hand.grip, 'open')
        tracker.reset()
        _, (new_closed,) = tracker.update([replace(closed, pinch_gap=0.025)], 2.00)
        self.assertEqual(new_closed.grip, 'pinch')

    def test_thumb_depth_dropout_has_a_bounded_grace_with_fresh_pose_updates(self):
        tracker, held, closed = self.pinch()
        missing = replace(closed, pinch_gap=None, pose=Pose((0.01, 0, 0.70), (0.01, 0.04, 0.72)))
        _, (continued,) = tracker.update([missing], 1.12)
        self.assertEqual((continued.id, continued.grip), (held.id, 'pinch'))
        self.assertGreater(continued.pose.palm[0], held.pose.palm[0])
        self.assertEqual(tracker.update([missing], 1.21), ('lost', ()))

    def test_noisy_held_gap_is_tolerated_and_intentional_opening_releases_immediately(self):
        tracker, held, closed = self.pinch()
        noisy = replace(closed, pinch_gap=0.049, features=GripFeatures(0.60, 1, 2))
        _, (continued,) = tracker.update([noisy], 1.12)
        self.assertEqual((continued.id, continued.grip), (held.id, 'pinch'))
        _, (released,) = tracker.update([sample()], 1.16)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))

        tracker, held, closed = self.pinch()
        visible_opening = replace(closed, pinch_gap=None, features=OPEN)
        _, (released,) = tracker.update([visible_opening], 1.12)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))

    def test_conflicting_pinch_readings_need_confirmation_before_release(self):
        for conflicting in (sample(gap=0.057, features=CLOSED), sample(gap=0.025, features=GripFeatures(0.67, 1, 2))):
            with self.subTest(conflicting=conflicting):
                tracker, held, closed = self.pinch()
                _, (noisy,) = tracker.update([conflicting], 1.12)
                self.assertEqual((noisy.id, noisy.grip), (held.id, 'pinch'))
                _, (recovered,) = tracker.update([closed], 1.16)
                self.assertEqual(recovered.grip, 'pinch')

                for at in (1.20, 1.24):
                    _, (uncertain,) = tracker.update([conflicting], at)
                    self.assertEqual(uncertain.grip, 'pinch')
                _, (released,) = tracker.update([conflicting], 1.28)
                self.assertEqual((released.id, released.grip), (held.id, 'open'))

    def test_corroborated_pinch_opening_skips_release_confirmation(self):
        tracker, held, _ = self.pinch()
        opening = sample(gap=0.057, features=GripFeatures(0.67, 1, 2))
        _, (released,) = tracker.update([opening], 1.12)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))

    def test_optical_index_and_measured_palm_hold_a_pinch_only_during_depth_grace(self):
        tracker, held, closed = self.pinch()
        palm = (0.02, 0.04, 0.72)
        optical = replace(closed, pose=Pose(palm, palm), pointing=False, pinch_gap=None)
        _, (continued,) = tracker.update([optical], 1.12)
        self.assertEqual((continued.id, continued.grip), (held.id, 'pinch'))
        self.assertFalse(continued.pointing)
        self.assertEqual(continued.pose.tip, continued.pose.palm)
        self.assertGreater(continued.pose.palm[0], held.pose.palm[0])
        self.assertEqual(tracker.update([optical], 1.21), ('lost', ()))
        self.assertEqual(tracker.update([optical], 1.25), ('lost', ()))

    def test_clearly_open_optical_aim_releases_without_index_depth(self):
        tracker, held, _ = self.pinch()
        opening = replace(sample(), pointing=False, pinch_gap=None)
        _, (released,) = tracker.update([opening], 1.12)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))
        self.assertFalse(released.pointing)

    def test_missing_index_observations_or_palm_depth_interrupt_without_grace(self):
        tracker, _, closed = self.pinch()
        no_index = replace(closed, pointing=False, pinch_gap=None, image_index=None)
        self.assertEqual(tracker.update([no_index], 1.12), ('lost', ()))
        tracker, _, closed = self.pinch()
        invalid_pose = replace(closed, pose=Pose((0, 0, float('nan')), (0, 0.04, 0.72)))
        self.assertEqual(tracker.update([invalid_pose], 1.12), ('lost', ()))
