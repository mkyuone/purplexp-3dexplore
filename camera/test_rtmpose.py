from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from rtmpose import (MODEL_NAMES, RGB_MEAN, RGB_STD, RTMPoseEstimator, decode_pose,
                     detector_input, hand_boxes, model_problem, pose_input)


class Detector:
    def __init__(self):
        self.boxes = np.array([[[50, 20, 130, 100, 0.9], [51, 21, 131, 101, 0.8],
                                [190, 30, 270, 110, 0.8]]], dtype=np.float32)
        self.inputs = []

    def get_inputs(self):
        return [SimpleNamespace(name='input')]

    def run(self, names, inputs):
        self.inputs.append(inputs['input'])
        return [self.boxes, np.zeros(self.boxes.shape[:2], dtype=np.int64)]


class Pose:
    def __init__(self):
        self.inputs = []
        self.invalid = False
        self.invalid_sample = None

    def get_inputs(self):
        return [SimpleNamespace(name='input')]

    def run(self, names, inputs):
        self.inputs.append(inputs['input'])
        count = inputs['input'].shape[0]
        x, y = np.zeros((2, count, 21, 512), dtype=np.float32)
        for sample in range(count):
            for joint in range(21):
                x[sample, joint, 160 + (joint % 5) * 32 + sample * 12] = 0.04 * (sample + 1)
                y[sample, joint, 160 + (joint // 5) * 32 + sample * 12] = 0.06 * (sample + 1)
        if self.invalid:
            x[:] = np.nan
        if self.invalid_sample is not None:
            x[self.invalid_sample] = np.nan
        return x, y


def estimator():
    result = RTMPoseEstimator.__new__(RTMPoseEstimator)
    result.detector, result.pose = Detector(), Pose()
    result.last_timestamp = None
    result.last_result = ()
    result.diagnostics = {}
    return result


class RTMPoseTests(TestCase):
    def test_actual_pipeline_channel_orders_and_padding_are_preserved(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        detector, ratio = detector_input(rgb)
        self.assertEqual(detector.shape, (1, 3, 320, 320))
        self.assertEqual(ratio, 0.5)
        np.testing.assert_allclose(detector[0, :, 0, 0], (np.array([0, 0, 255]) - RGB_MEAN[::-1]) / RGB_STD[::-1])
        np.testing.assert_allclose(detector[0, :, 300, 0], (114 - RGB_MEAN[::-1]) / RGB_STD[::-1])
        pose, center, extent = pose_input(rgb, [100, 100, 260, 180])
        self.assertEqual(pose.shape, (1, 3, 256, 256))
        np.testing.assert_allclose(pose[0, :, 128, 128], (np.array([255, 0, 0]) - RGB_MEAN) / RGB_STD)
        np.testing.assert_allclose(center, [180, 140])
        self.assertEqual(extent, 200)

    def test_duplicate_boxes_do_not_consume_the_second_hand_slot(self):
        detector = Detector()
        boxes, count = hand_boxes(detector.boxes, 0.5, 640, 480)
        self.assertEqual(count, 3)
        self.assertEqual(len(boxes), 2)
        np.testing.assert_allclose(boxes[0][:4], [100, 40, 260, 200])
        np.testing.assert_allclose(boxes[1][:4], [380, 60, 540, 220])

    def test_simcc_argmax_mapping_keeps_soft_response_confidence_and_missing_joints(self):
        pose = Pose()
        x, y = pose.run(None, {'input': np.empty((1, 3, 256, 256))})
        center = np.array([200, 150])
        points, confidence = decode_pose(x, y, center, 200, 640, 480)
        self.assertAlmostEqual(points[0].x, (160 / 2 / 256 * 200 + 100) / 640)
        self.assertAlmostEqual(points[0].y, (160 / 2 / 256 * 200 + 50) / 480)
        self.assertAlmostEqual(confidence[0], 0.05)
        self.assertEqual(points[0].z, 0)
        x[0, 4] = y[0, 4] = 0
        points, _ = decode_pose(x, y, center, 200, 640, 480)
        self.assertTrue(np.isnan(points[4].x))
        x[0, 0, 0] = np.nan
        self.assertIsNone(decode_pose(x, y, center, 200, 640, 480))

    def test_two_independent_hands_are_current_and_carry_no_invented_handedness(self):
        value = estimator()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = value.detect(rgb, 1000)
        self.assertEqual(len(hands), 2)
        self.assertLess(hands[0].landmarks[0].x, hands[1].landmarks[0].x)
        self.assertTrue(all(hand.side is None and len(hand.landmarks) == 21 for hand in hands))
        self.assertEqual(value.diagnostics['acceptedHands'], 2)
        self.assertEqual(len(value.pose.inputs), 1)
        self.assertEqual(value.pose.inputs[0].shape, (2, 3, 256, 256))
        self.assertAlmostEqual(hands[0].confidences[0], 0.05)
        self.assertAlmostEqual(hands[1].confidences[0], 0.10)
        self.assertAlmostEqual(hands[0].landmarks[0].x, (160 / 2 / 256 * 200 + 80) / 640)
        self.assertAlmostEqual(hands[1].landmarks[0].x, (172 / 2 / 256 * 200 + 360) / 640)
        self.assertIs(value.detect(rgb, 1000), hands)
        self.assertEqual(len(value.pose.inputs), 1)
        value.detector.boxes[..., 4] = 0
        self.assertEqual(value.detect(rgb, 1033), ())
        self.assertEqual(len(value.pose.inputs), 1)
        self.assertEqual(value.diagnostics['acceptedHands'], 0)

    def test_invalid_pose_does_not_return_previous_coordinates(self):
        value = estimator()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self.assertEqual(len(value.detect(rgb, 1000)), 2)
        value.pose.invalid = True
        self.assertEqual(value.detect(rgb, 1033), ())

    def test_invalid_second_batch_member_keeps_independently_valid_first_hand(self):
        value = estimator()
        value.pose.invalid_sample = 1
        hands = value.detect(np.zeros((480, 640, 3), dtype=np.uint8), 1000)
        self.assertEqual(len(hands), 1)
        self.assertAlmostEqual(hands[0].score, 0.9)
        self.assertAlmostEqual(hands[0].confidences[0], 0.05)
        self.assertEqual(value.diagnostics['acceptedHands'], 1)

    def test_single_hand_still_uses_a_single_crop_batch(self):
        value = estimator()
        value.detector.boxes = value.detector.boxes[:, :1]
        self.assertEqual(len(value.detect(np.zeros((480, 640, 3), dtype=np.uint8), 1000)), 1)
        self.assertEqual(len(value.pose.inputs), 1)
        self.assertEqual(value.pose.inputs[0].shape, (1, 3, 256, 256))

    def test_missing_model_or_pointer_reports_an_actionable_error(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.assertIn('missing', model_problem(folder))
            (folder / MODEL_NAMES[0]).write_text('version https://git-lfs.github.com/spec/v1\n')
            self.assertIn('Git LFS pointer', model_problem(folder))
