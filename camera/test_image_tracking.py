from dataclasses import replace
from math import exp
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose
from realsense import fuse_hand
from tracking import Candidate, GripFeatures, Hand, HandTracker

OPEN = GripFeatures(0.8, 0, 0)
PINCH = GripFeatures(0.2, 1, 2)
FIST = GripFeatures(0.2, 4, 4)


def candidate(x=0, *, index=(0.4, 0.3), palm=(0.42, 0.5), features=OPEN, gap=0.06, pointing=True):
    return Candidate(Pose((x, 0, 0.7), (x, 0.04, 0.72)), None, features, gap, pointing, index, palm)


class ImageTrackingTests(TestCase):
    def test_optional_metadata_and_image_smoothing_preserve_physical_identity(self):
        pose = Pose((0, 0, 0.7), (0, 0.04, 0.72))
        self.assertIsNone(Hand('legacy', pose, 'open', True).image)
        tracker = HandTracker()
        _, (first,) = tracker.update([candidate()], 1)
        self.assertEqual(first.image, (0.4, 0.3))
        _, (moved,) = tracker.update([candidate(.01, index=(0.46, 0.36))], 1.045)
        self.assertEqual(moved.id, first.id)
        alpha = 1 - exp(-1)
        self.assertAlmostEqual(moved.image[0], 0.4 + 0.06 * alpha)
        self.assertAlmostEqual(moved.image[1], 0.3 + 0.06 * alpha)
        self.assertTrue(moved.pose.valid())

    def test_fist_changes_to_palm_anchor_without_index_interpolation(self):
        tracker = HandTracker()
        tracker.update([candidate(index=(0.4, 0.2), palm=(0.42, 0.6))], 1)
        fist = candidate(index=(0.4, 0.3), palm=(0.42, 0.6), features=FIST, gap=.02)
        _, (held,) = tracker.update([fist], 1.04)
        self.assertEqual(held.grip, 'fist')
        self.assertEqual(held.image, (0.42, 0.6))
        occluded = replace(fist, pointing=False, image_index=None, pinch_gap=None)
        _, (continued,) = tracker.update([occluded], 1.08)
        self.assertEqual(continued.id, held.id)
        self.assertEqual(continued.image, held.image)
        _, (released,) = tracker.update([candidate(index=(.4, .2))], 1.12)
        self.assertEqual(released.grip, 'open')
        self.assertEqual(released.image, (.4, .2))

    def test_detection_reordering_keeps_each_image_attached_to_its_hand(self):
        tracker = HandTracker()
        left = candidate(-.12, index=(.25,.3))
        right = candidate(.12, index=(.75,.3))
        _, before = tracker.update([left, right], 1)
        _, after = tracker.update([right, left], 1.04)
        self.assertEqual([(hand.id, hand.image) for hand in after], [(hand.id, hand.image) for hand in before])
        tracker.reset()
        _, (invalid,) = tracker.update([replace(left, image_index=(float('nan'), .3))], 1.08)
        self.assertIsNone(invalid.image)


class FusedImageTests(TestCase):
    def setUp(self):
        self.landmarks = [SimpleNamespace(x=.5, y=.5, z=0) for _ in range(21)]
        self.landmarks[0] = SimpleNamespace(x=.5, y=.75, z=0)
        for index, x in ((5,.4),(9,.5),(17,.6)):
            self.landmarks[index] = SimpleNamespace(x=x, y=.55, z=0)
        self.landmarks[8] = SimpleNamespace(x=.37, y=.25, z=0)
        profile = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(intrinsics=None))
        self.depth = SimpleNamespace(profile=profile, get_data=lambda: SimpleNamespace(shape=(480, 640)), get_units=lambda: .001)
        self.np = SimpleNamespace(asanyarray=lambda data: data)

    def test_fusion_exposes_original_image_ray_for_valid_depth(self):
        with patch('realsense.grip_features', return_value=OPEN), patch('realsense.locate_depth', return_value=(0,.02,.7)):
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertEqual(hand.image_index, (.37,.25))
        self.assertAlmostEqual(hand.image_palm[0], .5)
        self.assertAlmostEqual(hand.image_palm[1], .65)

    def test_fusion_exposes_pixel_aspect_correct_pinch_ratio(self):
        self.landmarks[4] = SimpleNamespace(x=.38, y=.25, z=0)
        with patch('realsense.grip_features', return_value=PINCH), patch('realsense.locate_depth', return_value=(0,.02,.7)):
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertAlmostEqual(hand.optical_pinch_ratio, .05)
        self.landmarks[4] = SimpleNamespace(x=1.01, y=.25, z=0)
        with patch('realsense.grip_features', return_value=PINCH), patch('realsense.locate_depth', return_value=(0,.02,.7)):
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertIsNone(hand.optical_pinch_ratio)

    def test_missing_finger_depth_preserves_optical_aim_and_measured_palm(self):
        palm = (0,.02,.7)
        with patch('realsense.grip_features', return_value=OPEN), patch('realsense.locate_depth', side_effect=[palm,palm,palm,None,None]):
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertFalse(hand.pointing)
        self.assertIsNone(hand.pinch_gap)
        self.assertEqual(hand.pose.palm, palm)
        self.assertEqual(hand.image_index, (.37,.25))
        status, (aiming,) = HandTracker().update([hand], 1.0)
        self.assertEqual(status, 'tracking')
        self.assertEqual((aiming.grip, aiming.image, aiming.pointing), ('open', hand.image_index, False))
        with patch('realsense.grip_features', return_value=FIST), patch('realsense.locate_depth', side_effect=[palm,palm,palm,None,None]):
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertFalse(hand.pointing)
        self.assertEqual(hand.image_index, (.37,.25))
        self.assertEqual(hand.image_palm, (.5,.65))

    def test_measured_distal_pads_recover_pinch_gap_at_both_fingertip_edges(self):
        palm = (0,.02,.7)
        index = (0,0,.7)
        thumb = (.012,0,.7)
        measured = [palm, palm, palm, None, index, None, thumb]
        with patch('realsense.grip_features', return_value=PINCH), patch('realsense.locate_depth', side_effect=measured) as locate:
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertTrue(hand.pointing)
        self.assertEqual(hand.pose.tip, index)
        self.assertAlmostEqual(hand.pinch_gap, .012)
        self.assertEqual(locate.call_args_list[4].args[-1], (.435, .375))
        self.assertEqual(locate.call_args_list[6].args[-1], (.5, .5))

    def test_missing_thumb_and_pad_depth_preserve_index_pose_with_no_gap(self):
        palm = (0,.02,.7)
        index = (0,0,.7)
        with patch('realsense.grip_features', return_value=PINCH), patch('realsense.locate_depth', side_effect=[palm,palm,palm,index,None,None]):
            hand = fuse_hand(None, self.np, self.landmarks, [], [], self.depth)
        self.assertTrue(hand.pointing)
        self.assertEqual(hand.pose.tip, index)
        self.assertIsNone(hand.pinch_gap)
