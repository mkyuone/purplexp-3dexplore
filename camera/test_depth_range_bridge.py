from collections import deque
from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose
from realsense import CameraInfo, Frame, Snapshot
from server import CameraService, Profiles
from test_preview import DISPLAY, request
from tracking import Hand


class DepthFeed:
    def __init__(self):
        self.now = 10.0
        self.info = CameraInfo('camera:640x480:848:30:color', 'session-1', serial='camera')
        self.frames = deque(maxlen=48)
        self.thumb_requests = 0
        self.publish(0.48)

    def publish(self, z, *, thumbs_up=False):
        self.now += 1 / 30
        sequence = self.frames[-1].sequence + 1 if self.frames else 1
        pose = Pose((0.0, 0.0, z), (0.0, 0.05, z))
        hand = Hand('hand-1', pose, 'open', True, (0.5, 0.5), thumbs_up)
        self.frames.append(Frame(sequence, self.now, 'tracking', (hand,), detected_hands=1, thumb_depth_m=z))

    def snapshot(self, *, include_gestures=False):
        return Snapshot(self.info, self.frames[-1], (), None, tuple(self.frames) if include_gestures else ())

    def request_thumb_depth(self):
        self.thumb_requests += 1


class DepthRangeBridgeTests(TestCase):
    def test_thumb_capture_persists_and_does_not_send_a_visitor_reset(self):
        feed = DepthFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            path = Path(folder) / 'profiles.json'
            service = CameraService(feed, Profiles(path))
            original = service.poses(DISPLAY)
            self.assertEqual(service.depth_range(), {'range': None, 'session': 'session-1'})
            self.assertEqual(feed.thumb_requests, 0)
            status = service.depth_range({'action': 'begin', 'session': 'session-1'})
            self.assertEqual(status['step'], 'near')
            self.assertNotIn('captureM', status)

            def hold(z, *, thumbs_up=False):
                for _ in range(7):
                    for _ in range(5):
                        feed.publish(z, thumbs_up=thumbs_up)
                    status = service.depth_range()
                return status

            hold(0.48)
            status = hold(0.49, thumbs_up=True)
            self.assertEqual((status['step'], status['nearM'], status['range']), ('far', 0.48, None))
            self.assertFalse(service.poses(DISPLAY)['hands'][0]['thumbsUp'])
            self.assertEqual(hold(0.49, thumbs_up=True)['step'], 'far')
            hold(0.84)
            status = hold(0.85, thumbs_up=True)
            self.assertTrue(status['confirmed'])
            self.assertEqual(status['range'], {'nearM': 0.48, 'farM': 0.84})
            self.assertNotIn('step', status)
            updated = service.poses(DISPLAY)
            self.assertGreater(updated['depthRangeRevision'], original['depthRangeRevision'])
            self.assertFalse(updated['hands'][0]['thumbsUp'])
            self.assertEqual(updated['hands'][0]['depth'], 0.5)

            feed.publish(0.84)
            service.poses(DISPLAY)
            feed.publish(0.85, thumbs_up=True)
            self.assertTrue(service.poses(DISPLAY)['hands'][0]['thumbsUp'])

            reloaded = CameraService(feed, Profiles(path))
            feed.info = CameraInfo(None, 'session-2', serial='camera')
            warming = reloaded.poses(DISPLAY)
            feed.info = CameraInfo('camera:640x480:848:60:color', 'session-2', serial='camera')
            self.assertEqual(reloaded.poses(DISPLAY)['depthRangeRevision'], warming['depthRangeRevision'])
            self.assertEqual(reloaded.depth_range()['range'], status['range'])
            feed.info = CameraInfo('other-camera:640x480:848:30:color', 'session-3', serial='other-camera')
            self.assertIsNone(reloaded.depth_range()['range'])
            feed.info = CameraInfo('camera:640x480:848:60:color', 'session-4', serial='camera')
            reset = reloaded.depth_range({'action': 'reset', 'session': 'session-4'})
            self.assertIsNone(reset['range'])
            self.assertIsNone(CameraService(feed, Profiles(path)).depth_range()['range'])

    def test_http_actions_reject_stale_sessions_and_abandon_a_disconnected_capture(self):
        feed = DepthFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            self.assertEqual(request(service, 'OPTIONS', '/depth-range')[0], 204)
            status, _, body = request(service, 'GET', '/depth-range')
            self.assertEqual((status, loads(body)['session']), (200, 'session-1'))

            action = {'action': 'begin', 'session': 'session-1'}
            self.assertEqual(request(service, 'POST', '/depth-range', origin=None, payload=action)[0], 403)
            self.assertEqual(request(service, 'POST', '/depth-range', payload={'action': 'begin'})[0], 400)
            status, _, body = request(service, 'POST', '/depth-range', payload=action)
            self.assertEqual((status, loads(body)['step']), (200, 'near'))

            feed.info = CameraInfo('other-camera:640x480:848:30:color', 'session-2', serial='other-camera')
            status, _, body = request(service, 'GET', '/depth-range')
            self.assertEqual((status, loads(body)), (200, {'range': None, 'session': 'session-2'}))
            self.assertEqual(request(service, 'POST', '/depth-range', payload=action)[0], 409)
