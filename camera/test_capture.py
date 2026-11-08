from collections import deque
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from realsense import CameraError, CameraFeed, CameraSample, DepthMatcher, DepthStream, PreviewEncoder


def sample(number, timestamp, domain='camera', arrival=None):
    return CameraSample(SimpleNamespace(number=number), number, timestamp, domain, arrival)


class CaptureTests(TestCase):
    def test_matching_uses_one_fresh_depth_sample_and_recovers_after_a_gap(self):
        pending = deque()
        queue = SimpleNamespace(poll_for_frame=lambda: pending.popleft() if pending else None)
        matcher = DepthMatcher(None, queue, 30)

        with patch.object(CameraSample, 'read', side_effect=lambda _, frame: frame):
            newest = sample(3, 136)
            pending.extend((sample(1, float('nan')), sample(2, 100), newest))
            self.assertIs(matcher.match(sample(10, 132)), newest.frame)
            self.assertIsNone(matcher.match(sample(11, 150)))
            pending.extend((sample(3, 136), sample(4, 172)))
            self.assertIsNone(matcher.match(sample(12, 215)))
            pending.append(sample(5, 270))
            self.assertEqual(matcher.match(sample(13, 268)).number, 5)

    def test_matches_different_domains_only_with_shared_host_timing(self):
        pending = deque((sample(1, 50, 'device'), sample(2, 51, 'device', 120)))
        queue = SimpleNamespace(poll_for_frame=lambda: pending.popleft() if pending else None)
        matcher = DepthMatcher(None, queue, 60)
        with patch.object(CameraSample, 'read', side_effect=lambda _, frame: frame):
            self.assertIsNone(matcher.match(sample(10, 50, 'global')))
            self.assertEqual(matcher.match(sample(11, 123, 'global', 123)).number, 2)

    def test_expired_or_out_of_order_rgb_arrivals_never_reach_inference(self):
        base_wall = 1000
        inputs = iter(((1, 0, 0), (2, .03, -.40), (3, .06, -.06), (4, .09, -6), (5, .12, .117)))
        clock = [0.0]
        detected = []
        published = []

        def receive(_):
            number, at, arrival = next(inputs)
            clock[0] = at
            return True, SimpleNamespace(number=number, arrival=(base_wall + arrival) * 1000, get_data=lambda: number)

        color = SimpleNamespace(try_wait_for_frame=receive)
        depth = SimpleNamespace(match=lambda _: (None, None))
        estimator = SimpleNamespace(detect=lambda image, _: detected.append(image) or [])
        feed = SimpleNamespace(wanted=lambda _: len(published) < 2, publish=lambda frame, _: published.append(frame))
        numpy = SimpleNamespace(ascontiguousarray=lambda data: data)
        with (
            patch('realsense.time', side_effect=lambda: base_wall + clock[0]),
            patch('realsense.monotonic', side_effect=lambda: clock[0]),
            patch.object(CameraSample, 'read', side_effect=lambda _, f: CameraSample(f, f.number, 0, 'camera', f.arrival)),
        ):
            CameraFeed.track_frames(feed, None, numpy, estimator, color, depth, None, 0)
        self.assertEqual(detected, [1, 5])
        self.assertEqual(published[-1].diagnostics['captureDroppedFrames'], 3)
        self.assertAlmostEqual(published[-1].diagnostics['queueMs'], 3.0)
        self.assertAlmostEqual(published[-1].at, .117)

    def test_failed_depth_start_retries_without_changing_the_rgb_session(self):
        attempts = []
        profile_received = Event()
        sdk = SimpleNamespace(distortion=SimpleNamespace(modified_brown_conrady='modified'))
        color = SimpleNamespace(intrinsics=SimpleNamespace(model='none'))
        widths = []

        @contextmanager
        def open_depth(rs, device, fps, widths):
            attempts.append(widths)
            if len(attempts) == 1:
                raise CameraError('Depth unavailable')
            yield SimpleNamespace(width=lambda: 640), None

        def activate(width):
            widths.append(width)
            profile_received.set()

        with (
            patch('realsense.open_depth', side_effect=open_depth),
            patch('realsense.DEPTH_RETRY_DELAY', .001),
            patch('realsense.logging.warning'),
        ):
            stream = DepthStream(sdk, None, color, 30, activate)
            try:
                self.assertTrue(profile_received.wait(1))
                self.assertEqual(attempts, [(848, 640), (848, 640)])
                self.assertEqual(widths, [640])
            finally:
                stream.close()
        self.assertFalse(stream.thread.is_alive())

    def test_preview_replaces_backlog_and_owns_its_frame(self):
        started, release, delivered = Event(), Event(), Event()
        received = []

        def encode(_, rgb, markers):
            if rgb[0] == 1:
                started.set()
                release.wait(1)
            return bytes(rgb)

        def publish(at, jpeg):
            received.append((at, jpeg))
            if at == 3:
                delivered.set()

        with patch('realsense.encode_preview', side_effect=encode):
            encoder = PreviewEncoder(None, publish)
            try:
                encoder.submit(1, bytearray([1]), ())
                self.assertTrue(started.wait(1))
                encoder.submit(2, bytearray([2]), ())
                newest = bytearray([3])
                encoder.submit(3, newest, ())
                newest[0] = 4
                release.set()
                self.assertTrue(delivered.wait(1))
            finally:
                release.set()
                encoder.close()
        self.assertEqual(received, [(1, b'\x01'), (3, b'\x03')])
