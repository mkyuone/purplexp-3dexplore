from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from calibration import Pose, TARGETS, fit
from estimators import image_features, is_thumbs_up
from navigation import DEPTH_WIDTH_M, FuzzyMapper, soft_axis, IMAGE_LEFT, IMAGE_RIGHT
from realsense import fuse_hand
from server import CameraService, Profiles
from test_bridge import StubFeed
from tracking import Candidate, GripFeatures, Hand, HandTracker, classify_grip


FOLDED = GripFeatures(1.1, 4, 4)
PIXELS = (
    (244, 273), (242, 218), (219, 168), (191, 132), (181, 102),
    (156, 191), (121, 181), (143, 202), (184, 211),
    (150, 223), (117, 216), (138, 230), (180, 229),
    (150, 251), (125, 246), (144, 255), (177, 253),
    (150, 277), (130, 268), (143, 277), (170, 279),
)


def landmarks(width=382, height=406, angle=0, mirror=False):
    tilt = radians(angle)
    points = []
    for x, y in PIXELS:
        dx, dy = x - 200, y - 200
        x = dx * cos(tilt) - dy * sin(tilt) + 200
        y = dx * sin(tilt) + dy * cos(tilt) + 200
        if mirror:
            x = width - x
        points.append(SimpleNamespace(x=x / width, y=y / height, z=0.0))
    return points


def candidate(*, pointing=False):
    return Candidate(
        Pose((0.02, -0.10, 0.70), (0.01, 0.02, 0.72)), None, FOLDED, None, pointing,
        image_index=(0.30, 0.25), image_palm=(0.55, 0.65), thumbs_up=True,
    )


class ThumbsUpTests(TestCase):
    def test_upward_thumb_works_for_both_hands_aspects_and_moderate_tilt(self):
        for width, height, angle, mirror in (
            (382, 406, 0, False), (382, 406, 0, True),
            (764, 406, 0, False), (382, 812, 0, False), (382, 406, 15, False),
        ):
            with self.subTest(width=width, height=height, angle=angle, mirror=mirror):
                points = landmarks(width, height, angle, mirror)
                features = image_features(points, width, height)
                self.assertEqual(features.loosely_folded, 4)
                self.assertTrue(is_thumbs_up(points, width, height, features))

    def test_sideways_downward_bent_or_tucked_thumbs_are_not_up(self):
        self.assertFalse(is_thumbs_up(landmarks(angle=90), 382, 406, FOLDED))
        self.assertFalse(is_thumbs_up(landmarks(angle=180), 382, 406, FOLDED))
        for changes in (
            {2: (215, 190), 3: (191, 192), 4: (170, 197)},
            {3: (240, 100)},
            {3: (210, 158), 4: (205, 150)},
        ):
            with self.subTest(changes=changes):
                points = landmarks()
                for index, (x, y) in changes.items():
                    points[index] = SimpleNamespace(x=x / 382, y=y / 406, z=0.0)
                self.assertFalse(is_thumbs_up(points, 382, 406, FOLDED))

    def test_open_pinch_v_and_missing_thumb_geometry_cannot_be_thumbs_up(self):
        for features in (GripFeatures(1.1, 0, 0), GripFeatures(0.2, 1, 2), GripFeatures(1.1, 2, 2)):
            self.assertFalse(is_thumbs_up(landmarks(), 382, 406, features))
        for index in (0, 2, 3, 4, 8, 9, 17):
            points = landmarks()
            points[index] = SimpleNamespace(x=float('nan'), y=0.5, z=0.0)
            self.assertFalse(is_thumbs_up(points, 382, 406, FOLDED))
        self.assertFalse(is_thumbs_up(landmarks()[:20], 382, 406, FOLDED))

    def test_thumb_up_publishes_at_palm_without_fingertip_depth_and_suppresses_grips(self):
        sample = candidate()
        for previous in ('open', 'pinch', 'fist'):
            self.assertEqual(classify_grip(sample, previous), 'open')
        tracker = HandTracker()
        status, (hand,) = tracker.update([sample], 1.0)
        self.assertEqual(status, 'tracking')
        self.assertTrue(hand.thumbs_up)
        self.assertEqual((hand.grip, hand.pointing, hand.image), ('open', False, sample.image_palm))
        self.assertEqual(hand.pose.tip, hand.pose.palm)
        self.assertEqual(tracker.update([], 1.04), ('lost', ()))
        invalid = replace(sample, pose=Pose((0, 0, 0.1), (0, 0, 0.1)))
        self.assertEqual(tracker.update([invalid], 1.08), ('lost', ()))

    def test_thumb_up_ends_a_fist_or_pinch_and_switches_anchor_without_blending(self):
        for previous in (
            replace(candidate(), thumbs_up=False),
            replace(candidate(pointing=True), features=GripFeatures(0.2, 1, 2), pinch_gap=0.02, thumbs_up=False),
        ):
            with self.subTest(previous=previous):
                tracker = HandTracker()
                _, (before,) = tracker.update([previous], 1.0)
                _, (action,) = tracker.update([candidate()], 1.04)
                self.assertEqual(action.id, before.id)
                self.assertEqual((action.grip, action.thumbs_up), ('open', True))
                self.assertEqual(action.image, candidate().image_palm)
                opening = replace(candidate(pointing=True), thumbs_up=False, features=GripFeatures(1.1, 0, 0), pinch_gap=0.08)
                _, (released,) = tracker.update([opening], 1.08)
                self.assertFalse(released.thumbs_up)
                self.assertEqual(released.image, opening.image_index)
                self.assertEqual(released.pose.tip, opening.pose.tip)

    def test_fusion_requires_real_palm_but_not_closed_index_depth(self):
        profile = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(intrinsics=None))
        depth = SimpleNamespace(profile=profile, get_data=lambda: SimpleNamespace(shape=(406, 382)), get_units=lambda: 0.001)
        np = SimpleNamespace(asanyarray=lambda data: data)
        palm = (0.01, 0.02, 0.72)
        with patch('realsense.locate_depth', side_effect=[palm, palm, palm, None]):
            sample = fuse_hand(None, np, landmarks(), [], [], depth, FOLDED)
        self.assertTrue(sample.thumbs_up)
        self.assertFalse(sample.pointing)
        self.assertEqual(sample.pose.tip, palm)
        status, (hand,) = HandTracker().update([sample], 1)
        self.assertEqual((status, hand.thumbs_up, hand.grip), ('tracking', True, 'open'))
        with patch('realsense.locate_depth', return_value=None):
            self.assertIsNone(fuse_hand(None, np, landmarks(), [], [], depth, FOLDED))

    def test_thumb_up_keeps_neutral_depth_fixed_and_changes_mapping_anchor_immediately(self):
        mapper = FuzzyMapper()
        pose = Pose((0, 0, 0.7), (0, 0.02, 0.7))
        open_hand = Hand('one', pose, 'open', True, (0.3, 0.3))
        mapper.update([open_hand], 1.0)
        action = replace(open_hand, pose=Pose((0, 0, 0.8), (0, 0.02, 0.8)), image=(0.7, 0.6), thumbs_up=True)
        held, = mapper.update([action], 1.04)
        self.assertTrue(held.thumbs_up)
        self.assertAlmostEqual(held.x, 1 - soft_axis(0.7, IMAGE_LEFT, IMAGE_RIGHT))
        self.assertAlmostEqual(held.depth, 0.1 / DEPTH_WIDTH_M)
        mapper.update([action], 2.0)
        self.assertEqual(mapper.neutral_depth_m, 0.7)
        released, = mapper.update([replace(action, thumbs_up=False, image=(0.3, 0.3))], 2.04)
        self.assertFalse(released.thumbs_up)
        self.assertAlmostEqual(released.x, 1 - soft_axis(0.3, IMAGE_LEFT, IMAGE_RIGHT))
        self.assertGreater(mapper.neutral_depth_m, 0.7)

    def test_both_json_modes_publish_thumb_up_independently_without_index_depth(self):
        display = '1280x720@1'
        feed = StubFeed()
        poses = []
        for target in TARGETS:
            feed.set_target(target)
            poses.append(feed.history[-1].calibration_pose)
        hand = replace(feed.history[-1].hands[0], image=(0.5, 0.5), pointing=False, thumbs_up=True)
        hand = replace(hand, pose=Pose(hand.pose.palm, hand.pose.palm))
        feed.replace_sample(-1, hands=(hand,))
        with TemporaryDirectory() as folder, patch('server.monotonic', lambda: feed.now):
            profiles = Profiles(Path(folder) / 'profiles.json')
            profiles.save(f'camera:profile|{display}', fit(poses))
            service = CameraService(feed, profiles)
            for mode in ('fuzzy', 'calibrated'):
                published, = service.poses(display, mode)['hands']
                self.assertTrue(published['thumbsUp'])
                self.assertFalse(published['fist'])
                self.assertFalse(published['pinched'])
