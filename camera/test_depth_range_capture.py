from collections import deque
from unittest import TestCase

from depth_range import DepthRange
from depth_range_capture import DepthRangeCapture
from realsense import Frame, Snapshot
from tracking import Hand


class Replay:
    def __init__(self):
        self.sequence = 0
        self.capture = DepthRangeCapture(started_at=0)
        self.frames = deque(maxlen=48)

    def emit(self, depth_m, *, frames=1, thumbs_up=False, hand_id='one', hand_count=1):
        for _ in range(frames):
            self.sequence += 1
            hand = Hand(hand_id, None, 'open', False, thumbs_up=thumbs_up)
            hands = (hand,) if hand_count == 1 else ()
            tracking = 'tracking' if hands else 'lost'
            self.frames.append(Frame(self.sequence, self.sequence / 30, tracking, hands,
                                     detected_hands=hand_count, thumb_depth_m=depth_m))

    def poll(self):
        snapshot = Snapshot(None, self.frames[-1], (), None, tuple(self.frames))
        self.capture.observe(snapshot)
        return self.capture

    def confirm(self, depth_m):
        self.emit(depth_m, frames=12)
        self.emit(depth_m, thumbs_up=True, frames=22)
        return self.poll()


class DepthRangeCaptureTests(TestCase):
    def test_capture_uses_open_thumb_distance_and_requires_a_new_open_pose_for_far(self):
        replay = Replay()
        replay.emit(0.45, frames=5)
        replay.emit(0.56)  # A single bad depth patch should not bias the stable distance.
        replay.emit(0.45, frames=6)
        self.assertAlmostEqual(replay.poll().capture_m, 0.45)

        replay.emit(0.48, thumbs_up=True, frames=10)
        self.assertTrue(replay.poll().confirming)
        replay.emit(0.49, thumbs_up=True, frames=13)
        capture = replay.poll()
        self.assertEqual(capture.step, 'far')
        self.assertAlmostEqual(capture.near_m, 0.45)

        replay.emit(0.90, thumbs_up=True, frames=30)
        self.assertIsNone(replay.poll().result)
        self.assertEqual(replay.confirm(0.90).result, DepthRange(0.45, 0.90))

    def test_tracking_loss_extra_hands_and_switched_identity_cannot_finish_a_hold(self):
        for change in ({'hand_count': 0}, {'hand_count': 2}, {'hand_id': 'two'}):
            with self.subTest(change=change):
                replay = Replay()
                replay.emit(0.55, frames=12)
                replay.emit(0.55, thumbs_up=True, frames=12)
                replay.emit(0.55, thumbs_up=True, **change)
                replay.emit(0.55, thumbs_up=True, frames=14)
                self.assertIsNone(replay.poll().near_m)
                self.assertFalse(replay.poll().confirming)
                self.assertAlmostEqual(replay.confirm(0.55).near_m, 0.55)

    def test_thumb_motion_and_stalled_frames_do_not_confirm(self):
        replay = Replay()
        replay.emit(0.50, frames=12)
        replay.emit(0.50, thumbs_up=True, frames=10)
        self.assertTrue(replay.poll().confirming)
        for _ in range(30):
            self.assertIsNone(replay.poll().near_m)

        replay.emit(0.57, thumbs_up=True, frames=22)
        self.assertIsNone(replay.poll().near_m)
        replay.emit(0.50, frames=12)
        replay.emit(0.50, thumbs_up=True, frames=10)
        self.assertTrue(replay.poll().confirming)
        replay.sequence += 8  # No camera observations during this part of the hold.
        replay.emit(0.50, thumbs_up=True, frames=22)
        self.assertIsNone(replay.poll().near_m)

    def test_brief_missing_thumb_depth_pauses_hold_until_fresh_depth_returns(self):
        replay = Replay()
        replay.emit(0.55, frames=12)
        replay.emit(0.55, thumbs_up=True, frames=18)
        self.assertTrue(replay.poll().confirming)

        replay.emit(None, thumbs_up=True, frames=2)
        capture = replay.poll()
        self.assertFalse(capture.confirming)
        self.assertIsNone(capture.near_m)

        replay.emit(0.55, thumbs_up=True)
        self.assertIsNone(replay.poll().near_m)
        replay.emit(0.55, thumbs_up=True, frames=4)
        self.assertAlmostEqual(replay.poll().near_m, 0.55)

    def test_out_of_order_or_short_far_capture_leaves_near_unchanged_for_retry(self):
        replay = Replay()
        self.assertAlmostEqual(replay.confirm(0.55).near_m, 0.55)
        for distance in (0.50, 0.67):
            with self.subTest(distance=distance):
                capture = replay.confirm(distance)
                self.assertEqual(capture.step, 'far')
                self.assertAlmostEqual(capture.near_m, 0.55)
                self.assertIsNone(capture.result)
                self.assertIn('20 cm', capture.hint)
        self.assertEqual(replay.confirm(0.87).result, DepthRange(0.55, 0.87))
