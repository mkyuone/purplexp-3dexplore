from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose
from diagnostics import DiagnosticHand, diagnostic_hands, hand_overlay
from estimators import EstimatedHand, HandSide
from realsense import CameraFeed, CameraInfo, Frame, Snapshot
from server import CameraService, Handler, Profiles
from test_preview import JPEG, StubFeed, request
from tracking import Candidate, GripFeatures, Hand, Track


class DiagnosticFeed(StubFeed):
    def __init__(self):
        super().__init__()
        self.error = None
        self.status_requests = 0

    def snapshot(self, *, watch_hand_diagnostics=False):
        if watch_hand_diagnostics:
            self.status_requests += 1
        return Snapshot(self.info, self.frame, (), self.error)

    def request_diagnostic_rgb(self):
        self.requests.append('rgb')
        return JPEG

    def request_infrared(self, index):
        self.requests.append(index)
        return JPEG if index == 1 else None

    def infrared_status(self):
        return 'Waiting for IR 2.'


class DiagnosticsTests(TestCase):
    def test_reports_stable_anatomical_side_and_distinguishes_detection_from_tracking(self):
        pose = Pose((.1, .2, .4), (.12, .21, .423))
        features = GripFeatures(.2, 1, 2)
        sample = Candidate(pose, 'Right', features, .0114, True, (.73, .48), (.77, .52))
        hand = Hand('hand-1', pose, 'open', True, (.73, .48), pinch_pending=True)
        track = Track(hand, pose.palm, 'Left')
        points = [SimpleNamespace(x=.73, y=.48)] * 20 + [SimpleNamespace(x=1.1, y=.5)]
        detected = [EstimatedHand(points, features, (HandSide('Right', .92),)),
                    EstimatedHand(points, None, (HandSide('Right', .91),)),
                    EstimatedHand(points, None, (HandSide('Left', .53),))]

        observed = diagnostic_hands(detected, [sample, None, None], (track,))
        self.assertEqual([row.row() for row in observed], [
            {'side': 'Left', 'grip': 'pinch pending', 'depthMm': 423, 'gapMm': 11},
            {'side': 'Right', 'grip': 'untracked', 'depthMm': None, 'gapMm': None, 'confidence': .91},
            {'side': None, 'grip': 'untracked', 'depthMm': None, 'gapMm': None},
        ])
        overlays = hand_overlay(detected, observed)
        self.assertEqual(overlays[0], ('Left', ((.73, .48),) * 20 + (None,)))

    def test_diagnostics_routes_require_local_origin_and_hide_stale_hands(self):
        feed = DiagnosticFeed()
        feed.frame = Frame(1, 10.0, 'lost', (), hand_diagnostics=(
            DiagnosticHand('Left', 'pinch', 423, 11, .91), DiagnosticHand(None, 'untracked', None, None, None),
        ))
        with TemporaryDirectory() as folder, patch('server.monotonic', return_value=10.05) as clock:
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            self.assertEqual(request(service, 'GET', '/diagnostics', origin=None)[0], 403)
            self.assertEqual(request(service, 'GET', '/diagnostics/frame?stream=rgb', origin=None)[0], 403)
            self.assertEqual(request(service, 'GET', '/diagnostics/frame?stream=ir1', origin='https://example.org')[0], 403)
            self.assertEqual(request(service, 'GET', '/diagnostics/frame?stream=ir1', host='example.org:4182')[0], 403)
            self.assertEqual(request(service, 'GET', '/diagnostics/frame?stream=ir1&stream=ir2')[0], 400)
            self.assertEqual(feed.requests, [])
            self.assertEqual(feed.status_requests, 0)

            status, headers, data = request(service, 'GET', '/diagnostics')
            self.assertEqual(status, 200)
            self.assertEqual(feed.status_requests, 1)
            self.assertIn(b'Cache-Control: no-store', headers)
            self.assertEqual(loads(data), {
                'status': 'ready', 'hands': [{'side': 'Left', 'grip': 'pinch', 'depthMm': 423, 'gapMm': 11, 'confidence': .91}],
                'unassignedHands': 1, 'infraredMessage': 'Waiting for IR 2.',
            })
            self.assertEqual(request(service, 'OPTIONS', '/diagnostics/frame?stream=rgb')[0], 204)
            for stream in ('rgb', 'ir1'):
                status, headers, image = request(service, 'GET', f'/diagnostics/frame?stream={stream}')
                self.assertEqual((status, image), (200, JPEG))
                self.assertIn(b'Content-Type: image/jpeg', headers)
                self.assertIn(b'Access-Control-Allow-Origin: http://localhost:4180', headers)
            self.assertEqual(request(service, 'GET', '/diagnostics/frame?stream=ir2')[0], 204)
            self.assertEqual(feed.requests, ['rgb', 1, 2])

            clock.return_value = 10.3
            _, _, data = request(service, 'GET', '/diagnostics')
            self.assertEqual((loads(data)['status'], loads(data)['hands']), ('waiting', []))
            self.assertEqual(feed.status_requests, 2)
            feed.error = 'Unplugged.'
            _, _, data = request(service, 'GET', '/diagnostics')
            self.assertEqual(loads(data), {'status': 'offline', 'message': 'Unplugged.', 'hands': []})

    def test_diagnostics_rgb_is_on_demand_and_rejects_stale_frames(self):
        feed = object.__new__(CameraFeed)
        feed.lock = Lock()
        feed.wake = Event()
        feed.info = CameraInfo('camera:profile', 'session-1')
        feed.diagnostic_rgb = None
        feed.diagnostic_rgb_until = 0.0
        feed.next_diagnostic_rgb_at = 0.0
        with patch('realsense.monotonic', return_value=10.0) as clock:
            self.assertFalse(feed.diagnostic_rgb_due(10.0))
            self.assertIsNone(feed.request_diagnostic_rgb())
            self.assertTrue(feed.wake.is_set())
            self.assertTrue(feed.diagnostic_rgb_due(10.0))
            self.assertFalse(feed.diagnostic_rgb_due(10.1))
            feed.publish_diagnostic_rgb(10.0, JPEG, 'old-session')
            self.assertIsNone(feed.request_diagnostic_rgb())
            feed.publish_diagnostic_rgb(10.0, JPEG, 'session-1')
            self.assertEqual(feed.request_diagnostic_rgb(), JPEG)
            clock.return_value = 10.7
            self.assertFalse(feed.diagnostic_rgb_due(10.7))
            self.assertIsNone(feed.request_diagnostic_rgb())

    def test_diagnostics_distinguish_a_thumb_held_after_depth_setup_from_a_reset(self):
        feed = DiagnosticFeed()
        pose = Pose((.1, .2, .4), (.12, .21, .423))
        thumb = Hand('hand-1', pose, 'open', False, (.73, .48), thumbs_up=True)
        opened = Hand('hand-1', pose, 'open', True, (.73, .48))
        feed.frame = Frame(1, 10.0, 'tracking', (thumb,),
                           hand_diagnostics=(DiagnosticHand('Left', 'thumbs up', 423, None, .91),))
        with TemporaryDirectory() as folder, patch('server.monotonic', return_value=10.05):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            service.suppress_thumb_session = feed.info.session
            reading = service.diagnostics()
            self.assertEqual(reading['hands'][0]['grip'], 'thumbs up')
            self.assertTrue(reading['resetSuppressed'])
            feed.frame = Frame(2, 10.04, 'tracking', (opened,))
            self.assertNotIn('resetSuppressed', service.diagnostics())
