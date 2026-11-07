from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from estimators import MediaPipeEstimator, RTMPoseAdapter, create_estimator, image_features, tracker_choices


def landmarks(width=640, height=480):
    points = [SimpleNamespace(x=320 / width, y=410 / height, z=0.0) for _ in range(21)]
    for base, x in ((1, 180), (5, 240), (9, 300), (13, 360), (17, 420)):
        for offset in range(4):
            points[base + offset] = SimpleNamespace(x=x / width, y=(300 - offset * 45) / height, z=0.0)
    return points


class EstimatorTests(TestCase):
    def test_image_features_preserve_pixel_geometry_across_aspect_ratios(self):
        landscape = image_features(landmarks(640, 480), 640, 480)
        wide = image_features(landmarks(1280, 480), 1280, 480)
        self.assertIsNotNone(landscape)
        self.assertAlmostEqual(landscape.pinch_ratio, wide.pinch_ratio)
        self.assertEqual((landscape.folded, landscape.loosely_folded), (0, 0))
        self.assertEqual(landscape, wide)

    def test_image_features_reject_missing_or_collapsed_geometry_and_ignore_model_z(self):
        points = landmarks()
        self.assertIsNone(image_features(points[:20], 640, 480))
        self.assertIsNone(image_features(points, 0, 480))
        collapsed = [SimpleNamespace(x=0.5, y=0.5, z=0.0)] * 21
        self.assertIsNone(image_features(collapsed, 640, 480))
        invalid = list(points)
        invalid[8] = SimpleNamespace(x=float('nan'), y=0.4, z=0.0)
        self.assertIsNone(image_features(invalid, 640, 480))
        invented_depth = [SimpleNamespace(x=point.x, y=point.y, z=index) for index, point in enumerate(points)]
        self.assertEqual(image_features(points, 640, 480), image_features(invented_depth, 640, 480))

    def test_partial_non_aim_finger_visibility_preserves_palm_and_pinch_geometry(self):
        points = landmarks()
        expected = image_features(points, 640, 480)
        for index, point in (
            (14, points[13]),
            (15, SimpleNamespace(x=-0.05, y=0.5, z=0.0)),
            (19, SimpleNamespace(x=float('nan'), y=0.5, z=0.0)),
        ):
            with self.subTest(index=index):
                partial = list(points)
                partial[index] = point
                features = image_features(partial, 640, 480)
                self.assertIsNotNone(features)
                self.assertEqual(features, expected)
        for index in (0, 4, 5, 8, 9, 17):
            with self.subTest(essential=index):
                partial = list(points)
                partial[index] = SimpleNamespace(x=1.02, y=0.5, z=0.0)
                self.assertIsNone(image_features(partial, 640, 480))

    def test_degenerate_finger_chain_cannot_supply_evidence_for_a_full_fist(self):
        points = landmarks()
        for base in (5, 9, 13, 17):
            x = points[base].x
            points[base:base + 4] = [SimpleNamespace(x=x, y=y / 480, z=0.0) for y in (300, 260, 290, 330)]
        full = image_features(points, 640, 480)
        self.assertEqual((full.folded, full.loosely_folded), (4, 4))
        points[14] = points[13]
        partial = image_features(points, 640, 480)
        self.assertEqual((partial.folded, partial.loosely_folded), (3, 3))
        self.assertEqual(partial.pinch_ratio, full.pinch_ratio)

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
        self.assertGreater(hands[0].features.pinch_ratio, image_features(first, 640, 480).pinch_ratio)
        self.assertEqual(hands[0].handedness[0].category_name, 'Left')
        self.assertIsNone(hands[1].features)
        self.assertEqual(estimator.diagnostics, {'rawEstimatedHands': 2, 'acceptedHands': 2, 'invalidFeatures': 1})
        self.assertEqual(calls, [({'image_format': 'srgb', 'data': 'rgb-frame'}, 1000)])

    def test_rtmpose_adapter_forwards_camera_and_uses_only_image_geometry(self):
        points = landmarks()
        second = [SimpleNamespace(x=point.x + 0.05, y=point.y, z=point.z) for point in points]
        seen = []
        detector = SimpleNamespace(
            set_camera=lambda camera: seen.append(camera),
            detect=lambda image, at: [SimpleNamespace(landmarks=hand, side=None, score=0.98) for hand in (points, second)],
            diagnostics={'detectorRegions': 2, 'acceptedHands': 2},
        )
        adapter = RTMPoseAdapter(detector)
        camera = {'width': 640, 'height': 480, 'fx': 400, 'fy': 400}
        adapter.set_camera(camera)
        hands = adapter.detect(SimpleNamespace(shape=(480, 640, 3)), 1000)
        self.assertEqual(seen, [camera])
        self.assertIs(adapter.diagnostics, detector.diagnostics)
        self.assertEqual(len(hands), 2)
        self.assertEqual(hands[0].features, image_features(points, 640, 480))
        self.assertEqual(hands[1].features, image_features(second, 640, 480))
        self.assertTrue(all(hand.handedness == () for hand in hands))

    def test_choices_validate_availability_without_constructing_an_estimator(self):
        with patch('estimators.find_spec', return_value=object()), patch.object(Path, 'is_file', return_value=True):
            rtmpose = SimpleNamespace(model_problem=lambda path: 'Download both RTMPose model files.')
            with patch.dict('sys.modules', {'rtmpose': rtmpose}):
                choices = tracker_choices(Path('/model.task'))
        self.assertEqual([choice['value'] for choice in choices], ['mediapipe', 'rtmpose'])
        self.assertTrue(choices[0]['available'])
        self.assertFalse(choices[1]['available'])
        self.assertIn('both RTMPose', choices[1]['reason'])
        with patch('estimators.find_spec', return_value=None):
            choices = tracker_choices(Path('/model.task'))
        self.assertTrue(all(not choice['available'] and 'Missing' in choice['reason'] for choice in choices))

    def test_factory_reports_validation_before_switch_and_always_closes_runtime(self):
        with self.assertRaisesRegex(RuntimeError, 'Unknown hand tracker'):
            with create_estimator('unknown', Path('/model.task')):
                self.fail('An unknown estimator cannot enter its context.')
        closed = []
        detector = SimpleNamespace(close=lambda: closed.append(True))
        rtmpose = SimpleNamespace(RTMPoseEstimator=lambda path: detector)
        with patch('estimators.tracker_problem', return_value=None), patch.dict('sys.modules', {'rtmpose': rtmpose}):
            with self.assertRaisesRegex(ValueError, 'capture interrupted'):
                with create_estimator('rtmpose', Path('/model.task')) as estimator:
                    self.assertIs(estimator.detector, detector)
                    raise ValueError('capture interrupted')
        self.assertEqual(closed, [True])

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
            with create_estimator('mediapipe', Path('/camera/hand_landmarker.task')) as estimator:
                self.assertIs(estimator.detector, detector)
        options, = options_seen
        self.assertEqual(options['base_options']['model_asset_path'], '/camera/hand_landmarker.task')
        self.assertEqual((options['num_hands'], options['running_mode']), (2, 'video'))
        self.assertEqual(closed, [True])
