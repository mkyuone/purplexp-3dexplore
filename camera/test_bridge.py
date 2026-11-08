from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose, TARGETS, fit
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

    def test_lost_pinch_evidence_pauses_until_an_open_hand_and_close_palms_invalidate_identity(self):
        tracker = HandTracker()
        pinch = GripFeatures(0.2, 1, 1)
        tracker.update([candidate(0)], 0.96)
        _, (pending,) = tracker.update([candidate(0, gap=0.012, features=pinch)], 1)
        self.assertEqual(pending.grip, 'open')
        _, (hand,) = tracker.update([candidate(0, gap=0.012, features=pinch)], 1.04)
        self.assertEqual(hand.grip, 'pinch')
        _, (brief_dropout,) = tracker.update([candidate(0.005, gap=None, features=pinch)], 1.08)
        self.assertEqual((brief_dropout.id, brief_dropout.grip), (hand.id, 'pinch'))
        self.assertEqual(tracker.update([candidate(0.005, gap=None, features=pinch)], 1.17), ('lost', ()))
        _, (new_hand,) = tracker.update([candidate(0.005)], 1.21)
        self.assertEqual((new_hand.id, new_hand.grip), (hand.id, 'open'))
        self.assertEqual(tracker.update([candidate(0), candidate(0.02)], 1.25), ('ambiguous', ()))

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

    def set_target(self, target, *, depth=0.0, palm_depth=0.0):
        x, y = target
        pose = Pose(((x - 0.5) * 0.68, (y - 0.5) * 0.48, 0.7 + depth),
                    ((x - 0.5) * 0.68, (y - 0.5) * 0.48 + 0.04, 0.72 + depth + palm_depth))
        hand = Hand('hand-1', pose, 'open', True)
        self.history = tuple(
            Frame(i, self.now - 0.18 + i * 0.026, 'tracking', (hand,), calibration_pose=pose, detected_hands=1)
            for i in range(1, 7)
        )

    def replace_sample(self, index, **changes):
        frames = list(self.history)
        frames[index] = replace(frames[index], **changes)
        self.history = tuple(frames)

    def snapshot(self):
        return Snapshot(CameraInfo('camera:profile', 'session-1'), self.history[-1], self.history, None)


class CalibrationBridgeTests(TestCase):
    def test_rgb_only_startup_is_ready_in_fuzzy_mode_and_waits_for_depth_to_calibrate(self):
        feed = StubFeed()
        snapshot = replace(feed.snapshot(), info=CameraInfo(None, 'session-1'))
        with (TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now),
              patch.object(feed, 'snapshot', return_value=snapshot)):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            self.assertEqual(service.poses('1280x720@1')['status'], 'ready')
            self.assertIn('depth stream', service.poses('1280x720@1', 'calibrated')['message'])
            self.assertEqual(service.calibrate('1280x720@1', 'begin')['status'], 'offline')

    def test_pose_cursor_waits_without_holding_the_service_lock_and_reads_the_latest_frame(self):
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))

            def publish_during_wait(session, after, timeout):
                self.assertFalse(service.lock.locked())
                self.assertEqual((session, after), ('session-1', 6))
                self.assertGreater(timeout, 0)
                self.assertLess(timeout, 0.16)
                feed.replace_sample(-1, sequence=8)

            with patch.object(feed, 'wait_for_pose', publish_during_wait, create=True):
                response = service.poses('1280x720@1', session='session-1', after=6)
            self.assertEqual(response['sequence'], 8)

    def test_connected_stale_stream_waits_for_fresh_frames_without_republishing_hands(self):
        display = '1280x720@1'
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            self.assertEqual(service.poses(display)['status'], 'ready')
            feed.now += 0.3
            self.assertEqual(service.poses(display), {'status': 'recovering'})
            feed.set_target(TARGETS[0])
            self.assertEqual(service.poses(display)['status'], 'ready')
            feed.now += 0.7
            self.assertEqual(service.poses(display)['status'], 'offline')
            disconnected = replace(feed.snapshot(), error='Camera disconnected.')
            with patch.object(feed, 'snapshot', return_value=disconnected):
                self.assertEqual(service.poses(display), {'status': 'offline', 'message': 'Camera disconnected.'})

    def test_calibrated_navigation_requires_fingertip_depth_except_for_fists(self):
        display = '1280x720@1'
        feed = StubFeed()
        points = []
        for target in TARGETS:
            feed.set_target(target)
            points.append(feed.history[-1].calibration_pose)
        hand = replace(feed.history[-1].hands[0], image=(0.5, 0.5), pointing=False)
        feed.replace_sample(-1, hands=(hand,))
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            profiles = Profiles(Path(folder) / 'profiles.json')
            profiles.save(f'camera:profile|{display}', fit(points))
            service = CameraService(feed, profiles)
            self.assertEqual(len(service.poses(display)['hands']), 1)
            self.assertEqual(service.poses(display, 'calibrated')['tracking'], 'lost')
            self.assertEqual(service.poses(display, 'calibrated')['hands'], [])
            feed.replace_sample(-1, hands=(replace(hand, grip='fist'),))
            self.assertTrue(service.poses(display, 'calibrated')['hands'][0]['fist'])

    def test_fuzzy_navigation_is_ready_without_any_calibration(self):
        display = '1280x720@1'
        feed = StubFeed()
        hand = replace(feed.history[-1].hands[0], image=(0.5, 0.5))
        feed.replace_sample(-1, hands=(hand,))
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            path = Path(folder) / 'profiles.json'
            service = CameraService(feed, Profiles(path))
            result = service.poses(display)
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(len(result['hands']), 1)
            self.assertAlmostEqual(result['hands'][0]['x'], 0.5)
            self.assertAlmostEqual(result['hands'][0]['y'], 0.5)
            self.assertAlmostEqual(result['hands'][0]['depth'], 0)
            self.assertFalse(path.exists())
            self.assertFalse(service.attempts)
            self.assertEqual(service.poses(display, 'calibrated')['status'], 'calibration')
            service.calibrate(display, 'begin')
            self.assertEqual(service.poses(display)['status'], 'ready')
            self.assertEqual(service.poses(display, 'calibrated')['status'], 'calibration')

    def test_rgb_only_fist_publishes_xy_in_fuzzy_and_calibrated_waits_for_measured_depth(self):
        display = '1280x720@1'
        feed = StubFeed()
        points = []
        for target in TARGETS:
            feed.set_target(target)
            points.append(feed.history[-1].calibration_pose)
        measured = replace(feed.history[-1].hands[0], image=(0.6, 0.5), grip='fist', pointing=False)
        feed.replace_sample(-1, hands=(replace(measured, pose=None),), calibration_pose=None)

        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            profiles = Profiles(Path(folder) / 'profiles.json')
            profiles.save(f'camera:profile|{display}', fit(points))
            service = CameraService(feed, profiles)
            fuzzy = service.poses(display)
            self.assertEqual((fuzzy['status'], fuzzy['tracking']), ('ready', 'tracking'))
            hand, = fuzzy['hands']
            self.assertEqual(hand['id'], measured.id)
            self.assertLess(hand['x'], 0.5)
            self.assertEqual((hand['depth'], hand['depthValid'], hand['fist']), (0, False, True))
            self.assertIsNone(service.fuzzy.neutral_depth_m)
            calibrated = service.poses(display, 'calibrated')
            self.assertEqual((calibrated['status'], calibrated['tracking'], calibrated['hands']), ('ready', 'lost', []))

            feed.now += 0.04
            feed.set_target(TARGETS[-1])
            feed.replace_sample(-1, hands=(measured,))
            self.assertTrue(service.poses(display)['hands'][0]['depthValid'])
            hand, = service.poses(display, 'calibrated')['hands']
            self.assertEqual((hand['id'], hand['depthValid'], hand['fist']), (measured.id, True, True))

    def test_off_center_capture_finishes_with_corner_depth(self):
        display = '1280x720@1'
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            service.calibrate(display, 'begin')
            for target in TARGETS[:4]:
                feed.now += 1
                feed.set_target(target)
                service.calibrate(display, 'capture')

            feed.now += 1
            feed.set_target(TARGETS[3], palm_depth=0.12)
            self.assertEqual(service.calibrate(display, 'capture')['status'], 'ready')
            feed.now += 1
            feed.set_target(TARGETS[4])
            self.assertAlmostEqual(service.poses(display, 'calibrated')['hands'][0]['depth'], 0)

    def test_bad_corner_retries_before_center_and_keeps_other_corners(self):
        display = '1280x720@1'
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            service.calibrate(display, 'begin')
            for index, target in enumerate(TARGETS[:4]):
                feed.now += 1
                feed.set_target((0.65, 0.80) if index == 1 else target)
                result = service.calibrate(display, 'capture')

            self.assertLess(result['step'], 4)
            self.assertIn('other captures were kept', result['message'])
            self.assertEqual(sum(point is not None for point in service.attempts[display].points[:4]), 3)
            self.assertEqual(service.poses(display, 'calibrated')['step'], result['step'])
            for _ in range(4):
                feed.now += 1
                feed.set_target(TARGETS[result['step']])
                result = service.calibrate(display, 'capture')
                if result['step'] == 4:
                    break
            self.assertEqual(result['step'], 4)
            feed.now += 1
            feed.set_target(TARGETS[4])
            self.assertEqual(service.calibrate(display, 'capture')['status'], 'ready')

    def test_capture_persists_display_profile_and_cancel_preserves_it(self):
        display = '1280x720@1.25:1920x1080:-1920,0'
        feed = StubFeed()
        self.assertTrue(valid_display(display))
        self.assertFalse(valid_display('unknown-display'))
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            path = Path(folder) / 'profiles.json'
            service = CameraService(feed, Profiles(path))
            self.assertEqual(service.poses(display, 'calibrated')['status'], 'calibration')
            beginning = service.calibrate(display, 'begin')
            self.assertEqual(beginning['step'], 0)
            self.assertFalse(beginning['captureReady'])
            self.assertEqual(service.calibrate(display, 'capture')['step'], 0)
            for index, target in enumerate(TARGETS):
                feed.now += 1
                feed.set_target(target)
                self.assertTrue(service.poses(display, 'calibrated')['captureReady'])
                result = service.calibrate(display, 'capture')
                if index < 4:
                    self.assertEqual(result['step'], index + 1)
                    self.assertFalse(result['captureReady'])
                else:
                    self.assertEqual(result['status'], 'ready')
            self.assertEqual(service.poses(display, 'calibrated')['hands'][0]['x'], 0.5)
            self.assertEqual(service.poses(display, 'calibrated')['hands'][0]['depth'], 0.0)

            reopened = CameraService(feed, Profiles(path))
            self.assertEqual(reopened.poses(display, 'calibrated')['status'], 'ready')
            self.assertEqual(reopened.poses('1280x720@1.25:1920x1080:0,0', 'calibrated')['status'], 'calibration')
            reopened.calibrate(display, 'begin')
            self.assertEqual(reopened.poses(display, 'calibrated')['status'], 'calibration')
            self.assertEqual(reopened.calibrate(display, 'cancel')['status'], 'ready')
            reopened.calibrate(display, 'begin')
            for elapsed in (60, 61):
                feed.now += elapsed
                feed.set_target(TARGETS[-1])
                status = 'calibration' if elapsed == 60 else 'ready'
                self.assertEqual(reopened.poses(display, 'calibrated')['status'], status)
            feed.now += 1
            self.assertEqual(reopened.poses(display, 'calibrated')['status'], 'offline')

    def test_capture_uses_fresh_raw_pose_and_tolerates_an_earlier_outlier(self):
        display = '1280x720@1'
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            service.calibrate(display, 'begin')
            feed.now += 1
            feed.set_target(TARGETS[0])
            outlier = Pose((-0.36, -0.22, 0.77), (-0.36, -0.18, 0.79))

            feed.replace_sample(-1, calibration_pose=outlier)
            rejected = service.calibrate(display, 'capture')
            self.assertFalse(rejected['captureReady'])
            self.assertIn('still', rejected['captureHint'])
            self.assertEqual(rejected['step'], 0)

            feed.set_target(TARGETS[0])
            feed.replace_sample(2, calibration_pose=outlier)
            # Calibration uses the depth-valid raw sample even if gesture
            # classification rejects the frame for interaction.
            feed.replace_sample(-1, tracking='lost', hands=())
            self.assertTrue(service.poses(display, 'calibrated')['captureReady'])
            self.assertEqual(service.calibrate(display, 'capture')['step'], 1)

            feed.now += 1
            feed.set_target(TARGETS[0])
            duplicate = service.calibrate(display, 'capture')
            self.assertFalse(duplicate['captureReady'])
            self.assertIn('next highlighted', duplicate['captureHint'])
            self.assertEqual(duplicate['step'], 1)

    def test_capture_reports_missing_depth_and_other_hands_without_using_old_samples(self):
        display = '1280x720@1'
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            service.calibrate(display, 'begin')
            feed.now += 1
            for detected, hint in ((0, 'into the camera'), (1, 'needs depth'), (2, 'only your pointing hand')):
                feed.set_target(TARGETS[0])
                feed.replace_sample(-1, detected_hands=detected, calibration_pose=None)
                status = service.poses(display, 'calibrated')
                self.assertFalse(status['captureReady'])
                self.assertIn(hint, status['captureHint'])
                self.assertEqual(service.calibrate(display, 'capture')['step'], 0)
