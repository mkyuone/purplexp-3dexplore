from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from estimators import GpuInferenceError, MediaPipeEstimator, create_estimator, tracker_choices


def landmarks(width=640, height=480):
    points = [SimpleNamespace(x=320 / width, y=410 / height, z=0.0) for _ in range(21)]
    for base, x in ((1, 180), (5, 240), (9, 300), (13, 360), (17, 420)):
        for offset in range(4):
            points[base + offset] = SimpleNamespace(x=x / width, y=(300 - offset * 45) / height, z=0.0)
    return points


class EstimatorTests(TestCase):
    def test_mediapipe_keeps_two_detections_and_uses_world_features(self):
        first, second = landmarks(), landmarks()
        world = [SimpleNamespace(x=point.x, y=point.y, z=point.z) for point in first]
        world[4].z = 0.4
        result = SimpleNamespace(
            hand_landmarks=[first, second], hand_world_landmarks=[world, []],
            handedness=[[SimpleNamespace(category_name='Left', score=0.9)], []],
        )
        calls = []
        detector = SimpleNamespace(detect_for_video=lambda image, at: calls.append((image, at)) or result)
        mp = SimpleNamespace(Image=lambda **kw: kw, ImageFormat=SimpleNamespace(SRGB='srgb'))
        estimator = MediaPipeEstimator(mp, detector)
        hands = estimator.detect('rgb-frame', 1000)
        self.assertEqual(len(hands), 2)
        self.assertIs(hands[0].landmarks, first)
        self.assertGreater(hands[0].features.pinch_ratio, 1 / 3)
        self.assertEqual(hands[0].handedness[0].category_name, 'Left')
        self.assertIsNone(hands[1].features)
        self.assertEqual(estimator.diagnostics, {'rawEstimatedHands': 2, 'acceptedHands': 2, 'invalidFeatures': 1})
        self.assertEqual(calls, [({'image_format': 'srgb', 'data': 'rgb-frame'}, 1000)])

    def test_choices_validate_availability_without_constructing_an_estimator(self):
        with patch('estimators.find_spec', return_value=object()), patch.object(Path, 'is_file', return_value=True):
            choices = tracker_choices(Path('/model.task'))
        self.assertEqual([(choice['value'], choice['available']) for choice in choices], [('mediapipe', True)])
        with patch('estimators.find_spec', return_value=None):
            missing, = tracker_choices(Path('/model.task'))
        self.assertFalse(missing['available'])
        self.assertIn('Missing', missing['reason'])

    def test_factory_rejects_removed_trackers_before_loading_dependencies(self):
        with self.assertRaisesRegex(RuntimeError, 'Unknown hand tracker'):
            with create_estimator('rtmpose', Path('/model.task')):
                self.fail('An unsupported estimator cannot enter its context.')

    def test_gpu_option_is_applied_to_the_native_task_and_startup_failure_is_recoverable(self):
        base_options = Mock(Delegate=SimpleNamespace(GPU='gpu'))
        detector = Mock()
        factory = Mock(return_value=detector)
        vision = SimpleNamespace(
            HandLandmarkerOptions=lambda **options: options,
            RunningMode=SimpleNamespace(VIDEO='video'),
            HandLandmarker=SimpleNamespace(create_from_options=factory),
        )
        python = SimpleNamespace(BaseOptions=base_options, vision=vision)
        modules = {
            'mediapipe': SimpleNamespace(),
            'mediapipe.tasks': SimpleNamespace(python=python),
            'mediapipe.tasks.python': python,
        }
        with patch('estimators.tracker_problem', return_value=None), patch.dict('sys.modules', modules):
            with create_estimator('mediapipe', Path('/model.task'), num_hands=4, delegate='gpu'):
                base_options.assert_called_once_with(model_asset_path='/model.task', delegate='gpu')
                self.assertEqual(factory.call_args.args[0]['num_hands'], 4)
            detector.close.assert_called_once()

            init_failure = RuntimeError('EGL initialization failed')
            factory.side_effect = init_failure
            with self.assertRaisesRegex(GpuInferenceError, 'initialize the GPU hand tracker') as failure:
                with create_estimator('mediapipe', Path('/model.task'), delegate='gpu'):
                    self.fail('A failed native task cannot enter its context.')
            self.assertIs(failure.exception.__cause__, init_failure)
            factory.side_effect = FileNotFoundError('No model file')
            with self.assertRaisesRegex(RuntimeError, 'No model file') as failure:
                with create_estimator('mediapipe', Path('/model.task'), delegate='gpu'):
                    self.fail('A missing model cannot enter its context.')
            self.assertNotIsInstance(failure.exception, GpuInferenceError)

    def test_only_native_gpu_processing_errors_trigger_recovery(self):
        detector = Mock()
        mp = SimpleNamespace(Image=lambda **options: options, ImageFormat=SimpleNamespace(SRGB='srgb'))
        gpu_estimator = MediaPipeEstimator(mp, detector, delegate='gpu')
        cpu_estimator = MediaPipeEstimator(mp, detector)

        inference_failure = RuntimeError('Context lost')
        detector.detect_for_video.side_effect = inference_failure
        with self.assertRaisesRegex(GpuInferenceError, 'process a hand frame on the GPU') as failure:
            gpu_estimator.detect('rgb-frame', 1000)
        self.assertIs(failure.exception.__cause__, inference_failure)
        with self.assertRaisesRegex(RuntimeError, 'Context lost') as failure:
            cpu_estimator.detect('rgb-frame', 1000)
        self.assertNotIsInstance(failure.exception, GpuInferenceError)

        detector.detect_for_video.side_effect = ValueError('Invalid timestamp')
        with self.assertRaisesRegex(ValueError, 'Invalid timestamp'):
            gpu_estimator.detect('rgb-frame', 1000)

    def test_mediapipe_factory_loads_selected_bundle_with_two_hand_tracking(self):
        closed, options_seen = [], []
        detector = SimpleNamespace(close=lambda: closed.append(True))
        vision = SimpleNamespace(
            HandLandmarkerOptions=lambda **options: options,
            RunningMode=SimpleNamespace(VIDEO='video'),
            HandLandmarker=SimpleNamespace(
                create_from_options=lambda options: options_seen.append(options) or detector),
        )
        python = SimpleNamespace(BaseOptions=lambda **options: options, vision=vision)
        modules = {
            'mediapipe': SimpleNamespace(),
            'mediapipe.tasks': SimpleNamespace(python=python),
            'mediapipe.tasks.python': python,
        }
        with patch('estimators.tracker_problem', return_value=None), patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(ValueError, 'capture interrupted'):
                with create_estimator('mediapipe', Path('/camera/hand_landmarker.task')) as estimator:
                    self.assertIs(estimator.detector, detector)
                    raise ValueError('capture interrupted')
        options, = options_seen
        self.assertEqual(options['base_options']['model_asset_path'], '/camera/hand_landmarker.task')
        self.assertEqual((options['num_hands'], options['running_mode']), (2, 'video'))
        self.assertEqual(closed, [True])
