from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose, TARGETS
from realsense import CameraInfo, Frame, Snapshot
from server import CameraService, Profiles, valid_display
from tracking import Candidate, GripFeatures, Hand, HandTracker, classify_grip


def candidate(x, side=None, gap=0.06, features=GripFeatures(0.7, 0, 0)):
    return Candidate(Pose((x, 0.0, 0.70), (x, 0.04, 0.73)), side, features, gap, True)


class TrackingTests(TestCase):
    def test_continuity_survives_detection_order_changes(self):
        tracker = HandTracker()
        left = candidate(-0.12, 'Left')
        right = candidate(0.12, 'Right')
        status, hands = tracker.update([left, right], 1.0)
        self.assertEqual(status, 'tracking')
        ids = tuple(hand.id for hand in hands)
        status, hands = tracker.update([right, left], 1.04)
        self.assertEqual(status, 'tracking')
        self.assertEqual(tuple(hand.id for hand in hands), ids)
        self.assertLess(hands[0].pose.palm[0], hands[1].pose.palm[0])

    def test_close_palms_or_lost_pinch_depth_invalidate_old_identity(self):
        tracker = HandTracker()
        pinch = GripFeatures(0.2, 1, 1)
        _, (hand,) = tracker.update([candidate(0, gap=0.012, features=pinch)], 1)
        self.assertEqual(hand.grip, 'pinch')
        self.assertEqual(tracker.update([candidate(0.005, gap=None, features=pinch)], 1.04), ('lost', ()))
        _, (new_hand,) = tracker.update([candidate(0.005)], 1.08)
        self.assertNotEqual(new_hand.id, hand.id)
        self.assertEqual(tracker.update([candidate(0), candidate(0.02)], 1.12), ('ambiguous', ()))

    def test_grips_are_hysteretic_and_cannot_switch_directly(self):
        pinch = candidate(0, gap=0.02, features=GripFeatures(0.27, 1, 2))
        barely_closed = candidate(0, gap=0.037, features=GripFeatures(0.47, 1, 2))
        fist = candidate(0, gap=0.019, features=GripFeatures(0.20, 4, 4))
        self.assertEqual(classify_grip(pinch), 'pinch')
        self.assertEqual(classify_grip(barely_closed), 'open')
        self.assertEqual(classify_grip(barely_closed, 'pinch'), 'pinch')
        self.assertEqual(classify_grip(fist), 'fist')
        self.assertEqual(classify_grip(fist, 'pinch'), 'open')
        self.assertEqual(classify_grip(pinch, 'fist'), 'open')

    def test_fist_uses_palm_when_fingertip_moves_or_disappears(self):
        tracker = HandTracker()
        tracker.update([candidate(0)], 1.0)
        features = GripFeatures(0.2, 4, 4)
        observed = candidate(0.01, gap=0.02, features=features)
        _, (fist,) = tracker.update([observed], 1.04)
        self.assertEqual(fist.grip, 'fist')
        self.assertEqual(fist.pose.tip, fist.pose.palm)
        occluded = Candidate(Pose((0.3, 0.2, 0.8), observed.pose.palm), None, features, None, False)
        _, (continued,) = tracker.update([occluded], 1.08)
        self.assertEqual(continued.id, fist.id)
        self.assertEqual(continued.pose.tip, continued.pose.palm)


class StubFeed:
    def __init__(self):
        self.now = 10
        self.set_target(TARGETS[0])

    def set_target(self, target):
        x, y = target
        pose = Pose(((x - 0.5) * 0.68, (y - 0.5) * 0.48, 0.7), ((x - 0.5) * 0.68, (y - 0.5) * 0.48 + 0.04, 0.72))
        hand = Hand('hand-1', pose, 'open', True)
        self.history = tuple(Frame(i, self.now - 0.14 + i * 0.02, 'tracking', (hand,)) for i in range(1, 7))

    def snapshot(self):
        return Snapshot(CameraInfo('camera:profile', 'session-1'), self.history[-1], self.history, None)


class CalibrationBridgeTests(TestCase):
    def test_capture_persists_display_profile_and_cancel_preserves_it(self):
        display = '1280x720@1.25:1920x1080:-1920,0'
        feed = StubFeed()
        self.assertTrue(valid_display(display))
        self.assertFalse(valid_display('unknown-display'))
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            path = Path(folder) / 'profiles.json'
            service = CameraService(feed, Profiles(path))
            self.assertEqual(service.poses(display)['status'], 'calibration')
            self.assertEqual(service.calibrate(display, 'begin')['step'], 0)
            self.assertEqual(service.calibrate(display, 'capture')['step'], 0)
            for index, target in enumerate(TARGETS):
                feed.now += 1
                feed.set_target(target)
                result = service.calibrate(display, 'capture')
                if index < 4:
                    self.assertEqual(result['step'], index + 1)
                else:
                    self.assertEqual(result['status'], 'ready')
            self.assertEqual(service.poses(display)['hands'][0]['x'], 0.5)
            self.assertEqual(service.poses(display)['hands'][0]['depth'], 0.0)

            reopened = CameraService(feed, Profiles(path))
            self.assertEqual(reopened.poses(display)['status'], 'ready')
            self.assertEqual(reopened.poses('1280x720@1.25:1920x1080:0,0')['status'], 'calibration')
            reopened.calibrate(display, 'begin')
            self.assertEqual(reopened.poses(display)['status'], 'calibration')
            self.assertEqual(reopened.calibrate(display, 'cancel')['status'], 'ready')
            reopened.calibrate(display, 'begin')
            for elapsed in (60, 61):
                feed.now += elapsed
                feed.set_target(TARGETS[-1])
                status = 'calibration' if elapsed == 60 else 'ready'
                self.assertEqual(reopened.poses(display)['status'], status)
            feed.now += 1
            self.assertEqual(reopened.poses(display)['status'], 'offline')
