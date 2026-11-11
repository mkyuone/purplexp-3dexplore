from types import SimpleNamespace
from unittest import TestCase

from visitor_pose import VisitorPoseEstimator, body_poses


def landmark(x, y, visibility=1.0):
    return SimpleNamespace(x=x, y=y, visibility=visibility, presence=1.0)


class VisitorPoseTests(TestCase):
    def test_preserves_owner_anchor_when_wrist_is_hidden(self):
        pose = [landmark(.5, .5)] * 33
        pose[11] = landmark(.25, .5)
        pose[12] = landmark(.75, .5)
        pose[15] = landmark(.1, .5, visibility=.2)
        body, = body_poses([pose])
        self.assertEqual(body.center, (.5, .5))
        self.assertEqual(body.scale, .5)
        self.assertIsNone(body.left.wrist)
        self.assertEqual(body.right.wrist, (.5, .5))

        pose[12] = landmark(1.1, .5)
        self.assertEqual(body_poses([pose]), ())

    def test_throttles_submissions_and_rejects_old_results(self):
        sent = []
        mp = SimpleNamespace(Image=lambda **fields: fields, ImageFormat=SimpleNamespace(SRGB='srgb'))
        estimator = VisitorPoseEstimator(mp)
        estimator.detector = SimpleNamespace(detect_async=lambda image, timestamp: sent.append((image, timestamp)))
        empty_result = SimpleNamespace(pose_landmarks=[])

        self.assertIsNone(estimator.update('frame1', 1.0))
        estimator.receive(empty_result, None, 1000)
        estimator.receive(empty_result, None, 900)
        self.assertEqual(estimator.update('frame2', 1.04).at, 1.0)
        self.assertIsNone(estimator.update('frame3', 1.26))
        self.assertEqual(sent, [({'image_format': 'srgb', 'data': 'frame1'}, 1000),
                                ({'image_format': 'srgb', 'data': 'frame3'}, 1260)])
