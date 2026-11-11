from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from calibration import Pose
from diagnostics import diagnostic_hands
from estimators import EstimatedHand, HandSide
from infrared_preview import InfraredPreview
from realsense import CameraError, CameraFeed, CameraInfo, CameraSample, DepthMatcher, DepthStream, PreviewEncoder, image_projector, open_depth
from tracking import Candidate, GripFeatures


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
        feed = SimpleNamespace(wanted=lambda _: len(published) < 2, publish=lambda frame, _: published.append(frame),
                               hand_diagnostics_until=0.0)
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

    def test_hand_summaries_follow_diagnostics_viewers_and_keep_both_rgb_previews_working(self):
        with patch('realsense.Thread'):
            feed = CameraFeed(Path('unused-model'))
        feed.info = CameraInfo('camera:profile', 'camera-session')
        feed.publish = Mock(wraps=feed.publish)
        actions = [(10.0, 'preview'), (10.1, 'watch'), (10.5, 'watch'), (10.8, 'pose'),
                   (11.15, 'idle'), (11.2, 'overlay'), (11.23, 'idle'), (11.9, 'idle')]
        feed.wanted = lambda _: feed.publish.call_count < len(actions)
        inputs = iter(enumerate(actions, start=1))
        clock = [10.0]

        def receive(_):
            number, (at, action) = next(inputs)
            clock[0] = at
            if action == 'preview':
                feed.request_preview('camera-session', 0)
            elif action == 'watch':
                feed.snapshot(watch_hand_diagnostics=True)
            elif action == 'pose':
                feed.snapshot()
            elif action == 'overlay':
                feed.request_diagnostic_rgb()
            return True, SimpleNamespace(number=number, get_data=lambda: number)

        point = SimpleNamespace(x=.5, y=.5)
        features = GripFeatures(.8, 0, 0)
        estimate = EstimatedHand([point] * 21, features, (HandSide('Left', .99),))
        candidate = Candidate(Pose((0, 0, .7), (0, .04, .7)), 'Left', features, .06, True, (.5, .5), (.5, .5))
        estimator = SimpleNamespace(detect=lambda *_: [estimate])
        color = SimpleNamespace(try_wait_for_frame=receive)
        depth = SimpleNamespace(match=lambda _: (None, None))
        numpy = SimpleNamespace(ascontiguousarray=lambda data: data)
        preview, overlay = Mock(), Mock()
        with (
            patch('realsense.monotonic', side_effect=lambda: clock[0]),
            patch('realsense.fuse_hand', return_value=candidate) as fuse,
            patch('realsense.diagnostic_hands', wraps=diagnostic_hands) as summaries,
            patch.object(CameraSample, 'read', side_effect=lambda _, f: CameraSample(f, f.number, 0, 'camera', None)),
        ):
            feed.track_frames(None, numpy, estimator, color, depth, None, 0, preview, overlay)

        frames = [call.args[0] for call in feed.publish.call_args_list]
        self.assertTrue(all(frame.tracking == 'tracking' for frame in frames))
        self.assertEqual(summaries.call_count, 4)
        self.assertEqual([call.kwargs['notes'] is not None for call in fuse.call_args_list],
                         [False, True, True, True, False, False, False, False])
        self.assertEqual([frame.at for frame in frames if frame.hand_diagnostics], [10.1, 10.5, 10.8, 11.2])
        self.assertEqual(frames[1].hand_diagnostics[0].side, 'Left')
        self.assertTrue(preview.submit.call_args_list[0].args[2][0].valid)
        overlay.submit.assert_called_once_with(11.2, 6, (('Left', ((.5, .5),) * 21),))

    def test_failed_depth_start_retries_without_changing_the_rgb_session(self):
        attempts = []
        profile_received = Event()
        sdk = SimpleNamespace(distortion=SimpleNamespace(modified_brown_conrady='modified'))
        color = SimpleNamespace(intrinsics=SimpleNamespace(model='none'))
        widths = []
        projections = []

        @contextmanager
        def open_depth(rs, device, fps, widths):
            attempts.append(widths)
            if len(attempts) == 1:
                raise CameraError('Depth unavailable')
            yield SimpleNamespace(width=lambda: 640), None

        def activate(width, project_image):
            widths.append(width)
            projections.append(project_image)
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
                self.assertTrue(callable(projections[0]))
            finally:
                stream.close()
        self.assertFalse(stream.thread.is_alive())

    def test_infrared_uses_both_matched_profiles_and_routes_only_depth_to_the_matcher(self):
        def profile(stream, fmt, index, width=848):
            value = SimpleNamespace(stream_type=lambda: stream, format=lambda: fmt, fps=lambda: 30,
                                    stream_index=lambda: index, width=lambda: width, height=lambda: 480)
            value.as_video_stream_profile = lambda: value
            return value

        depth, first, second = (profile('depth', 'z16', 0), profile('infrared', 'y8', 1), profile('infrared', 'y8', 2))
        sensor = Mock()
        sensor.get_stream_profiles.return_value = [profile('infrared', 'y8', 1, 640), second, depth, first]
        queue, infrared = Mock(), Mock()
        sdk = SimpleNamespace(stream=SimpleNamespace(depth='depth', infrared='infrared'),
                              format=SimpleNamespace(z16='z16', y8='y8'), frame_queue=lambda _: queue,
                              option=SimpleNamespace(global_time_enabled='clock', emitter_enabled='emitter'))

        with open_depth(sdk, SimpleNamespace(first_depth_sensor=lambda: sensor), 30, (848,), infrared=infrared) as (opened, output):
            self.assertIs(opened, depth)
            self.assertIs(output, queue)
            sensor.open.assert_called_once_with([depth, first, second])
            receive, = sensor.start.call_args.args
            for current in (first, depth, second):
                frame = SimpleNamespace(get_profile=lambda current=current: current)
                receive(frame)
            self.assertEqual(queue.enqueue.call_count, 1)
            self.assertIs(queue.enqueue.call_args.args[0].get_profile(), depth)
            self.assertEqual([call.args[0] for call in infrared.submit.call_args_list], [1, 2])
        sensor.stop.assert_called_once()
        sensor.close.assert_called_once()

    def test_infrared_lease_returns_only_fresh_frames_and_expires(self):
        preview = InfraredPreview()
        with patch('infrared_preview.monotonic', return_value=10) as now:
            self.assertFalse(preview.wanted())
            self.assertIsNone(preview.request(1))
            preview.publish(1, 10, b'left')
            preview.publish(2, 10, b'right')
            self.assertTrue(preview.fresh_pair(.65))
            self.assertEqual(preview.request(2), b'right')
            now.return_value = 10.8
            self.assertIsNone(preview.request(1))
            now.return_value = 12.5
            self.assertFalse(preview.wanted())
            self.assertIsNone(preview.request(2))

    def test_unavailable_infrared_restores_normal_depth_during_the_same_viewer_lease(self):
        attempts = []
        depth_started = Event()
        sdk = SimpleNamespace(distortion=SimpleNamespace(modified_brown_conrady='modified'))
        color = SimpleNamespace(intrinsics=SimpleNamespace(model='none'))
        infrared = Mock()
        encoder = Mock()

        @contextmanager
        def open_modes(rs, device, fps, widths, *, infrared=None):
            attempts.append(infrared)
            if infrared is not None:
                raise CameraError('Infrared profiles unavailable')
            yield SimpleNamespace(width=lambda: 848), None

        with (patch('realsense.open_depth', side_effect=open_modes),
              patch('realsense.InfraredEncoder', return_value=encoder), patch('realsense.logging.warning')):
            stream = DepthStream(sdk, None, color, 30, lambda *_: depth_started.set(), infrared=infrared, cv2=object())
            try:
                self.assertTrue(depth_started.wait(1))
                self.assertEqual(attempts, [encoder, None])
                infrared.fail.assert_called_once()
                encoder.close.assert_called_once()
            finally:
                stream.close()

    def test_project_image_uses_normalized_color_pixels_at_the_supplied_depth(self):
        intrinsics = SimpleNamespace(width=640, height=480)
        deproject = Mock(return_value=(.15, -.1, .7))
        sdk = SimpleNamespace(rs2_deproject_pixel_to_point=deproject)
        project = image_projector(sdk, intrinsics)
        self.assertEqual(project(.25, .75, .7), (.15, -.1, .7))
        self.assertIsNone(project(float('nan'), .75, .7))
        self.assertIsNone(project(.25, .75, .1))
        deproject.assert_called_once_with(intrinsics, [160, 360], .7)

    def test_project_image_preserves_depth_camera_space_for_modified_color_intrinsics(self):
        calls = []
        def depth_space(x, y, z):
            calls.append((x, y, z))
            return .11, -.04, z

        project = image_projector(None, SimpleNamespace(width=640, height=480), depth_space)
        self.assertEqual(project(.5, .25, .6), (.11, -.04, .6))
        self.assertEqual(calls, [(320, 120, .6)])

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
