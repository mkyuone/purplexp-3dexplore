from dataclasses import replace
from unittest import TestCase

from calibration import Pose
from tracking import Candidate, GripFeatures, HandTracker


OPEN = GripFeatures(0.8, 0, 0)
PINCH = GripFeatures(0.25, 1, 2)
FIST = GripFeatures(0.2, 4, 4)


def candidate(x=0.0, *, grip=PINCH, side=None):
    return Candidate(Pose((x, 0, 0.70), (x, 0.04, 0.72)), side, grip,
                     0.06 if grip == OPEN else 0.02, True, (0.5 + x, 0.4), (0.5 + x, 0.5))


class TrackingRecoveryTests(TestCase):
    def test_rgb_fallback_and_metric_return_follow_palm_across_grip_anchors(self):
        tracker = HandTracker()
        open_hand = replace(candidate(grip=OPEN, side='Left'), image_index=(0.26, 0.20))
        _, (before,) = tracker.update([open_hand], 1)
        fist = replace(candidate(.02, grip=FIST, side='Left'), pose=None, pointing=False, pinch_gap=None)
        status, (optical,) = tracker.update([fist], 1.04)
        self.assertEqual((status, optical.id, optical.grip), ('tracking', before.id, 'fist'))
        self.assertIsNone(optical.pose)
        self.assertEqual(optical.image, fist.image_palm)

        recovered = candidate(.04, grip=FIST, side='Left')
        status, (measured,) = tracker.update([recovered], 1.08)
        self.assertEqual((status, measured.id, measured.grip), ('tracking', before.id, 'fist'))
        self.assertEqual(measured.pose.palm, recovered.pose.palm)
        self.assertEqual(measured.pose.tip, measured.pose.palm)

    def test_optical_match_rejects_conflicting_handedness_and_distant_palms(self):
        original = candidate(grip=OPEN, side='Left')
        for replacement in (
            candidate(grip=OPEN, side='Right'),
            candidate(.15, grip=OPEN, side='Left'),
        ):
            with self.subTest(replacement=replacement):
                tracker = HandTracker()
                _, (before,) = tracker.update([original], 1)
                optical = replace(replacement, pose=None, pointing=False, pinch_gap=None)
                status, (after,) = tracker.update([optical], 1.04)
                self.assertEqual(status, 'tracking')
                self.assertNotEqual(after.id, before.id)
                self.assertIsNone(after.pose)

    def test_mixed_depth_tracks_keep_identity_when_reordered_and_reject_overlapping_palms(self):
        tracker = HandTracker()
        left = candidate(-.12, grip=OPEN, side='Left')
        right = candidate(.12, grip=FIST, side='Right')
        _, (first, second) = tracker.update([left, right], 1)
        optical = replace(right, pose=None, pointing=False, pinch_gap=None)
        status, (same_first, same_second) = tracker.update([optical, left], 1.04)
        self.assertEqual(status, 'tracking')
        self.assertEqual((same_first.id, same_second.id), (first.id, second.id))
        self.assertIsNotNone(same_first.pose)
        self.assertIsNone(same_second.pose)
        overlapping = replace(optical, image_palm=(.40, .5))
        self.assertEqual(tracker.update([left, overlapping], 1.08), ('ambiguous', ()))

    def test_brief_missing_frames_pause_output_then_resume_measured_pose_with_same_identity(self):
        tracker = HandTracker()
        _, (before,) = tracker.update([candidate()], 1.00)
        self.assertEqual(tracker.update([], 1.04), ('lost', ()))
        self.assertEqual(tracker.update([None], 1.08), ('lost', ()))
        _, (after,) = tracker.update([candidate(0.02)], 1.12)
        self.assertEqual((after.id, after.grip), (before.id, 'pinch'))
        self.assertEqual(after.pose.palm[0], 0.02)

    def test_partial_two_hand_detection_publishes_only_each_verified_survivor(self):
        tracker = HandTracker()
        left = candidate(-0.12, side='Left')
        right = candidate(0.12, grip=FIST, side='Right')
        _, before = tracker.update([left, right], 1.00)
        status, (observed_left,) = tracker.update([left], 1.04)
        self.assertEqual((status, observed_left.id), ('tracking', before[0].id))
        status, (observed_right,) = tracker.update([None, right], 1.08)
        self.assertEqual((status, observed_right.id), ('tracking', before[1].id))
        _, after = tracker.update([right, left], 1.12)
        self.assertEqual([(hand.id, hand.grip) for hand in after], [(hand.id, hand.grip) for hand in before])

    def test_one_tracked_hand_cannot_keep_an_absent_hand_identity_alive(self):
        tracker = HandTracker()
        left = candidate(-.12, side='Left')
        right = candidate(.12, side='Right')
        _, before = tracker.update([left, right], 1)
        for at in (1.06, 1.12, 1.18, 1.24, 1.30):
            status, (survivor,) = tracker.update([left, None], at)
            self.assertEqual((status, survivor.id), ('tracking', before[0].id))
        _, after = tracker.update([left, right], 1.34)
        self.assertEqual(after[0].id, before[0].id)
        self.assertNotEqual(after[1].id, before[1].id)

    def test_secondary_depth_loss_does_not_cancel_a_valid_primary_pinch(self):
        tracker = HandTracker()
        left = candidate(-.12, side='Left')
        right = candidate(.12, side='Right')
        _, before = tracker.update([left, right], 1)
        unmeasured = replace(right, pinch_gap=None, pointing=False, image_index=None)
        status, (survivor,) = tracker.update([left, unmeasured], 1.16)
        self.assertEqual((status, survivor.id, survivor.grip), ('tracking', before[0].id, 'pinch'))

    def test_partial_frame_with_ambiguous_identity_still_pauses_every_hand(self):
        tracker = HandTracker()
        left, right = candidate(-.08), candidate(.08)
        _, before = tracker.update([left, right], 1.00)
        self.assertEqual(tracker.update([candidate(0), None], 1.04), ('ambiguous', ()))
        _, after = tracker.update([right, left], 1.08)
        self.assertEqual([(hand.id, hand.grip) for hand in after], [(hand.id, 'pinch') for hand in before])

    def test_overlapping_palms_pause_then_recover_both_pinches_only_within_identity_timeout(self):
        tracker = HandTracker()
        separated = [candidate(-.06, side='Left'), candidate(.06, side='Right')]
        overlapping = [candidate(-.01, side='Left'), candidate(.01, side='Right')]
        _, before = tracker.update(separated, 1.00)
        self.assertEqual(tracker.update(overlapping, 1.04), ('ambiguous', ()))
        _, recovered = tracker.update(separated[::-1], 1.08)
        self.assertEqual([(hand.id, hand.grip) for hand in recovered], [(hand.id, 'pinch') for hand in before])

        for at in (1.16, 1.24, 1.33):
            self.assertEqual(tracker.update(overlapping, at), ('ambiguous', ()))
        _, after_timeout = tracker.update(separated, 1.36)
        self.assertTrue({hand.id for hand in before}.isdisjoint(hand.id for hand in after_timeout))

    def test_two_missing_frames_at_a_slower_camera_rate_can_recover_identity(self):
        tracker = HandTracker()
        _, (before,) = tracker.update([candidate()], 1.00)
        self.assertEqual(tracker.update([], 1.067), ('lost', ()))
        self.assertEqual(tracker.update([None], 1.134), ('lost', ()))
        _, (after,) = tracker.update([candidate(0.02)], 1.201)
        self.assertEqual((after.id, after.grip), (before.id, 'pinch'))

    def test_missing_observations_cannot_extend_the_reacquisition_deadline(self):
        tracker = HandTracker()
        _, (before,) = tracker.update([candidate()], 1.00)
        for at in (1.04, 1.08, 1.12, 1.16, 1.20, 1.25):
            self.assertEqual(tracker.update([], at), ('lost', ()))
        _, (after,) = tracker.update([candidate()], 1.27)
        self.assertNotEqual(after.id, before.id)
        self.assertEqual(after.grip, 'pinch')

    def test_long_gap_replaces_identity_even_without_an_explicit_missing_frame(self):
        tracker = HandTracker()
        _, (before,) = tracker.update([candidate()], 1.00)
        _, (after,) = tracker.update([candidate()], 1.25)
        self.assertNotEqual(after.id, before.id)

    def test_distant_or_conflicting_hand_does_not_inherit_a_suspended_grip(self):
        for replacement in (candidate(0.12, grip=OPEN, side='Left'), candidate(grip=OPEN, side='Right')):
            with self.subTest(replacement=replacement):
                tracker = HandTracker()
                _, (before,) = tracker.update([candidate(side='Left')], 1.00)
                tracker.update([], 1.04)
                _, (after,) = tracker.update([replacement], 1.08)
                self.assertNotEqual(after.id, before.id)
                self.assertEqual(after.grip, 'open')

    def test_explicit_open_releases_a_suspended_pinch_immediately(self):
        tracker = HandTracker()
        _, (before,) = tracker.update([candidate()], 1.00)
        tracker.update([], 1.04)
        _, (after,) = tracker.update([candidate(grip=OPEN)], 1.08)
        self.assertEqual((after.id, after.grip), (before.id, 'open'))

    def test_held_fist_stays_on_its_palm_through_a_borderline_finger_count(self):
        tracker = HandTracker()
        fist = candidate(grip=FIST)
        uncertain = candidate(0.02, grip=GripFeatures(0.40, 1, 2))
        _, (held,) = tracker.update([fist], 1.00)
        _, (noisy,) = tracker.update([uncertain], 1.04)
        self.assertEqual((noisy.id, noisy.grip), (held.id, 'fist'))
        self.assertEqual(noisy.pose.tip, noisy.pose.palm)
        self.assertEqual(noisy.image[1], 0.5)
        tracker.update([fist], 1.08)
        for at in (1.12, 1.16):
            _, (borderline,) = tracker.update([uncertain], at)
            self.assertEqual(borderline.grip, 'fist')
        _, (released,) = tracker.update([uncertain], 1.20)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))

    def test_clearly_open_hand_releases_held_fist_immediately(self):
        tracker = HandTracker()
        _, (held,) = tracker.update([candidate(grip=FIST)], 1.00)
        _, (released,) = tracker.update([candidate(grip=OPEN)], 1.04)
        self.assertEqual((released.id, released.grip), (held.id, 'open'))

    def test_pending_pinch_confirmation_does_not_accumulate_missing_time(self):
        tracker = HandTracker()
        tracker.update([candidate(grip=OPEN)], 1.00)
        tracker.update([candidate()], 1.04)
        tracker.update([None], 1.07)
        _, (pending,) = tracker.update([candidate()], 1.13)
        self.assertEqual(pending.grip, 'open')
        _, (held,) = tracker.update([candidate()], 1.17)
        self.assertEqual(held.grip, 'pinch')

    def test_missing_index_observations_pause_without_publishing_the_old_pose(self):
        tracker = HandTracker()
        _, (before,) = tracker.update([candidate()], 1.00)
        missing_index = replace(candidate(), pointing=False, pinch_gap=None, image_index=None)
        self.assertEqual(tracker.update([missing_index], 1.04), ('lost', ()))
        _, (after,) = tracker.update([candidate()], 1.08)
        self.assertEqual(after.id, before.id)

    def test_open_hand_can_aim_with_valid_optical_index_and_measured_palm(self):
        tracker = HandTracker()
        open_aim = replace(candidate(grip=OPEN), pointing=False, pinch_gap=None)
        status, (hand,) = tracker.update([open_aim], 1.00)
        self.assertEqual(status, 'tracking')
        self.assertEqual(hand.grip, 'open')
        self.assertEqual(hand.image, open_aim.image_index)
        self.assertEqual(hand.pose.tip, hand.pose.palm)
        self.assertFalse(hand.pointing)

        uncertain = replace(open_aim, features=PINCH)
        self.assertEqual(tracker.update([uncertain], 1.04), ('lost', ()))
        unavailable = replace(open_aim, image_index=None)
        self.assertEqual(tracker.update([unavailable], 1.08), ('lost', ()))
