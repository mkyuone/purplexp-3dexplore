from io import BytesIO
from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose
from realsense import CameraFeed, CameraInfo, Frame, Snapshot, preview_markers
from server import CameraService, Handler, Profiles
from tracking import Candidate, GripFeatures, HandTracker

DISPLAY = '1280x720@1'
PREVIEW_URL = f'/preview?display={DISPLAY}'
ORIGIN = 'http://localhost:4180'
JPEG = b'\xff\xd8test-image\xff\xd9'


class StubFeed:
    def __init__(self):
        self.info = CameraInfo('camera:profile', 'session-1')
        self.frame = Frame(1, 10.0, 'lost', ())
        self.requests = []
        self.ended = []

    def snapshot(self):
        return Snapshot(self.info, self.frame, (), None)

    def request_preview(self, session, after, client='calibration'):
        self.requests.append((session, after, client))
        return JPEG

    def end_preview(self, client):
        self.ended.append(client)


class MemoryConnection:
    def __init__(self, request):
        self.input = BytesIO(request)
        self.output = bytearray()

    def makefile(self, *_):
        return self.input

    def sendall(self, data):
        self.output.extend(data)


def request(service, method, path, origin=ORIGIN, token=None, payload=None, host='127.0.0.1:4182'):
    headers = [f'Host: {host}']
    if origin is not None:
        headers.append(f'Origin: {origin}')
    if token is not None:
        headers.append(f'X-Calibration-Preview: {token}')
    body = b'' if payload is None else dumps(payload).encode()
    if body:
        headers.extend(('Content-Type: application/json', f'Content-Length: {len(body)}'))
    message = '\r\n'.join((f'{method} {path} HTTP/1.0', *headers, '', '')).encode() + body
    connection = MemoryConnection(message)
    Handler(connection, ('127.0.0.1', 42000), SimpleNamespace(server_port=4182, service=service))
    response_headers, response_body = bytes(connection.output).split(b'\r\n\r\n', 1)
    return int(response_headers.split()[1]), response_headers, response_body


class PreviewTests(TestCase):
    def test_http_preview_requires_local_origin_and_current_attempt_token(self):
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', return_value=10.05):
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            status, headers, _ = request(service, 'OPTIONS', PREVIEW_URL)
            self.assertEqual(status, 204)
            self.assertIn(b'Access-Control-Allow-Headers: Content-Type, X-Calibration-Preview', headers)
            self.assertIn(b'Access-Control-Max-Age: 300', headers)

            _, _, result = request(service, 'POST', '/calibration', payload={'display': DISPLAY, 'action': 'begin'})
            token = loads(result)['previewToken']
            self.assertGreaterEqual(len(token), 32)
            _, _, result = request(service, 'GET', f'/poses?display={DISPLAY}')
            self.assertNotIn('previewToken', loads(result))
            _, _, result = request(service, 'POST', '/calibration', payload={'display': DISPLAY, 'action': 'capture'})
            self.assertNotIn('previewToken', loads(result))

            self.assertEqual(request(service, 'GET', PREVIEW_URL, origin=None, token=token)[0], 403)
            self.assertEqual(request(service, 'GET', PREVIEW_URL, origin='https://other.example', token=token)[0], 403)
            self.assertEqual(request(service, 'GET', PREVIEW_URL, host='other.example:4182', token=token)[0], 403)
            self.assertEqual(request(service, 'GET', PREVIEW_URL)[0], 204)
            self.assertEqual(request(service, 'GET', PREVIEW_URL, token='wrong')[0], 204)
            self.assertEqual(request(service, 'GET', '/preview?display=640x480@1', token=token)[0], 204)
            self.assertFalse(feed.requests)

            status, headers, image = request(service, 'GET', PREVIEW_URL, token=token)
            self.assertEqual(status, 200)
            self.assertEqual(image, JPEG)
            self.assertEqual(feed.requests, [('session-1', 10.0, token)])
            self.assertIn(b'Content-Type: image/jpeg', headers)
            self.assertIn(b'Cache-Control: no-store', headers)
            self.assertIn(b'Access-Control-Allow-Origin: http://localhost:4180', headers)

            _, _, result = request(service, 'POST', '/calibration', payload={'display': DISPLAY, 'action': 'begin'})
            new_token = loads(result)['previewToken']
            self.assertNotEqual(new_token, token)
            self.assertEqual(request(service, 'GET', PREVIEW_URL, token=token)[0], 204)
            feed.info = CameraInfo(feed.info.profile, 'session-2')
            self.assertEqual(request(service, 'GET', PREVIEW_URL, token=new_token)[0], 204)
            _, _, result = request(service, 'POST', '/calibration', payload={'display': DISPLAY, 'action': 'begin'})
            token = loads(result)['previewToken']
            request(service, 'POST', '/calibration', payload={'display': DISPLAY, 'action': 'cancel'})
            self.assertEqual(request(service, 'GET', PREVIEW_URL, token=token)[0], 204)
            self.assertEqual(len(feed.requests), 1)

    def test_operator_previews_share_a_display_without_revoking_other_tabs(self):
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', return_value=10.05) as clock:
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            _, _, result = request(service, 'POST', '/calibration', payload={'display': DISPLAY, 'action': 'begin'})
            calibration_token = loads(result)['previewToken']
            feed.info = None
            _, _, result = request(service, 'POST', '/preview', payload={'display': DISPLAY, 'action': 'begin'})
            token = loads(result)['previewToken']
            self.assertNotEqual(token, calibration_token)
            self.assertEqual(service.preview(DISPLAY, token), None)
            feed.info = CameraInfo('camera:profile', 'session-1')
            self.assertEqual(request(service, 'GET', PREVIEW_URL, token=token)[2], JPEG)
            self.assertEqual(feed.requests[-1], ('session-1', 10.05, token))
            self.assertEqual(request(service, 'GET', '/preview?display=640x480@1', token=token)[0], 204)
            self.assertEqual(request(service, 'POST', '/preview', origin=None, payload={'display': DISPLAY, 'action': 'begin'})[0], 403)
            self.assertEqual(request(service, 'POST', '/preview', payload={'display': DISPLAY, 'action': 'end'})[0], 400)
            request(service, 'POST', '/preview', payload={'display': DISPLAY, 'action': 'end', 'previewToken': token})
            self.assertEqual(service.preview(DISPLAY, token), None)
            self.assertIn(token, feed.ended)
            self.assertEqual(service.preview(DISPLAY, calibration_token), JPEG)

            _, _, result = request(service, 'POST', '/preview', payload={'display': DISPLAY, 'action': 'begin'})
            token = loads(result)['previewToken']
            _, _, result = request(service, 'POST', '/preview', payload={'display': DISPLAY, 'action': 'begin'})
            replacement = loads(result)['previewToken']
            self.assertNotEqual(replacement, token)
            self.assertEqual(service.preview(DISPLAY, token), JPEG)
            request(service, 'POST', '/preview', payload={'display': DISPLAY, 'action': 'end', 'previewToken': token})
            self.assertEqual(service.preview(DISPLAY, token), None)
            self.assertEqual(service.preview(DISPLAY, replacement), JPEG)
            for at in (15, 20, 25):
                clock.return_value = at
                self.assertEqual(service.preview(DISPLAY, replacement), JPEG)
            clock.return_value = 32
            self.assertEqual(service.preview(DISPLAY, replacement), None)
            self.assertIn(replacement, feed.ended)

    def test_operator_preview_sessions_are_bounded_by_oldest_activity(self):
        feed = StubFeed()
        with TemporaryDirectory() as folder, patch('server.monotonic', return_value=10.05) as clock:
            service = CameraService(feed, Profiles(Path(folder) / 'profiles.json'))
            tokens = []
            for _ in range(8):
                result = service.preview_session(DISPLAY, 'begin', None)
                tokens.append(result['previewToken'])
            clock.return_value = 10.1
            self.assertEqual(service.preview(DISPLAY, tokens[0]), JPEG)
            latest = service.preview_session(DISPLAY, 'begin', None)['previewToken']
            self.assertEqual(len(service.operator_previews), 8)
            self.assertEqual(service.preview(DISPLAY, tokens[1]), None)
            self.assertIn(tokens[1], feed.ended)
            self.assertEqual(service.preview(DISPLAY, tokens[0]), JPEG)
            self.assertEqual(service.preview(DISPLAY, latest), JPEG)

    def test_frame_lease_is_throttled_and_does_not_return_stale_jpeg(self):
        feed = object.__new__(CameraFeed)
        feed.lock = Lock()
        feed.info = CameraInfo('camera:profile', 'session-1')
        feed.preview = None
        feed.preview_clients = {}
        feed.preview_until = 0.0
        feed.next_preview_at = 0.0
        with patch('realsense.monotonic', return_value=10.0) as clock:
            self.assertFalse(feed.preview_due(10.0))
            self.assertIsNone(feed.request_preview('wrong-session', 9.0))
            self.assertFalse(feed.preview_due(10.0))
            self.assertIsNone(feed.request_preview('session-1', 9.0))
            self.assertTrue(feed.preview_due(10.0))
            feed.publish_preview(10.0, JPEG)
            self.assertEqual(feed.request_preview('session-1', 9.0), JPEG)
            self.assertIsNone(feed.request_preview('session-1', 10.0))
            self.assertFalse(feed.preview_due(10.1))
            self.assertTrue(feed.preview_due(10.17))
            feed.request_preview('session-1', 9.0, 'operator')
            feed.end_preview('calibration')
            self.assertIsNotNone(feed.preview)
            feed.end_preview('operator')
            self.assertIsNone(feed.preview)
            self.assertFalse(feed.preview_due(10.18))
            feed.request_preview('session-1', 9.0, 'operator')
            feed.publish_preview(10.2, JPEG)
            clock.return_value = 10.7
            self.assertFalse(feed.preview_due(10.7))
            self.assertIsNone(feed.preview)

    def test_two_tracked_fists_have_valid_palm_markers_and_lost_landmarks_are_amber(self):
        features = GripFeatures(.2, 4, 4)
        left = Candidate(Pose((-.1,0,.7),(-.1,.04,.72)), None, features, .02, True, (.2,.3), (.25,.5))
        right = Candidate(Pose((.1,0,.7),(.1,.04,.72)), None, features, .02, True, (.8,.3), (.75,.5))
        tracker = HandTracker()
        tracker.update([left, right], 1.0)
        points = [
            [SimpleNamespace(x=.2,y=.3)] * 21,
            [SimpleNamespace(x=.8,y=.3)] * 21,
        ]
        markers = preview_markers(points, [left,right], tracker.tracks)
        self.assertEqual([marker.point for marker in markers], [(.25,.5),(.75,.5)])
        self.assertTrue(all(marker.valid and marker.fist for marker in markers))
        markers = preview_markers(points, [None,None], ())
        self.assertEqual([marker.point for marker in markers], [(.2,.3),(.8,.3)])
        self.assertTrue(all(not marker.valid for marker in markers))
