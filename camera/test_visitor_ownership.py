from dataclasses import replace
from types import SimpleNamespace
from unittest import TestCase

from estimators import EstimatedHand
from tracking import Candidate, GripFeatures
from visitor_ownership import VisitorOwnership
from visitor_pose import BodyArm, BodyPose


def visitor(x=.5, *, left=(.37, .32), right=(.63, .62), depth=None):
    return BodyPose(
        (x, .59), BodyArm((x - .10, .42), (x - .13, .54), left),
        BodyArm((x + .10, .42), (x + .13, .54), right), .20, depth_m=depth,
    )


def palm(wrist, *, closed=False):
    x, y = wrist
    points = [(x, y)] * 5
    for offset in (-.030, -.010, .010, .030):
        points.extend((x + offset, y - .045 - i * .028) for i in range(4))
    landmarks = [SimpleNamespace(x=px, y=py, z=0.) for px, py in points]
    features = GripFeatures(.17, 0, 0) if closed else GripFeatures(.8, 0, 0)
    sample = Candidate(None, None, features, None, False, points[8], (x, y - .035), .13 if closed else .8)
    return EstimatedHand(landmarks, features, ()), sample


class VisitorOwnershipTests(TestCase):
    def setUp(self):
        self.ownership = VisitorOwnership()

    def update(self, at, visitors, hands=()):
        detections = [detection for detection, _ in hands]
        samples = [sample for _, sample in hands]
        return self.ownership.update(visitors, detections, samples, at)

    def hold(self, start, duration, visitors, hands):
        for frame in range(round(duration * 10) + 1):
            result = self.update(start + frame / 10, visitors, hands)
        return result

    def test_competing_claims_wait_then_only_the_owner_can_supply_hands(self):
        owner = visitor(.40, left=(.30, .31), right=(.48, .61))
        bystander = visitor(.67, left=(.64, .35), right=(.80, .64))
        held = self.hold(1., .8, [owner, bystander], [palm(owner.left.wrist), palm(bystander.left.wrist)])
        self.assertEqual((held.phase, held.revision, held.indices), ('ready', 0, ()))

        bystander = visitor(.67, left=(.63, .62), right=(.80, .64))
        owned = self.hold(2., .7, [owner, bystander], [palm(owner.left.wrist)])
        self.assertEqual((owned.phase, owned.revision), ('active', 1))
        both = self.update(2.8, [bystander, owner], [palm(bystander.left.wrist), palm(owner.right.wrist),
                                                    palm(owner.left.wrist)])
        self.assertEqual(set(both.indices), {1, 2})
        self.assertEqual(both.revision, 1)
        rested = self.update(2.9, [owner, bystander])
        self.assertEqual((rested.phase, rested.revision), ('active', 1))

    def test_ambiguous_body_and_lost_owner_require_same_visitor_or_new_claim(self):
        owner = visitor(.45)
        self.hold(1., .7, [owner], [palm(owner.left.wrist)])
        competitor = visitor(.47)
        paused = self.update(1.8, [owner, competitor], [palm(owner.left.wrist)])
        self.assertEqual((paused.phase, paused.revision, paused.indices), ('paused', 1, ()))
        recovered = self.update(1.9, [owner], [palm(owner.left.wrist)])
        self.assertEqual((recovered.phase, recovered.revision, recovered.indices), ('active', 1, (0,)))

        arriving = visitor(.67, left=(.62, .35), right=(.81, .64))
        released = self.update(3.5, [arriving], [palm(arriving.left.wrist)])
        self.assertEqual((released.phase, released.revision), ('ready', 2))
        raised_through_release = self.hold(3.6, .8, [arriving], [palm(arriving.left.wrist)])
        self.assertEqual(raised_through_release.phase, 'ready')
        self.update(4.5, [visitor(.67, left=(.63, .62), right=(.81, .64))])
        fresh_claim = self.hold(4.6, .7, [arriving], [palm(arriving.left.wrist)])
        self.assertEqual((fresh_claim.phase, fresh_claim.revision), ('active', 3))

    def test_known_bystander_cannot_replace_an_occluded_owner(self):
        owner = visitor(.45)
        bystander = visitor(.56, left=(.48, .62), right=(.68, .62))
        self.hold(1., .7, [owner, bystander], [palm(owner.left.wrist)])
        for at, x in ((1.8, .51), (1.9, .47)):
            crossing = visitor(x, left=owner.left.wrist)
            result = self.update(at, [crossing], [palm(crossing.left.wrist)])
            self.assertEqual((result.phase, result.revision, result.indices), ('paused', 1, ()))
        recovered = self.update(2., [bystander, owner], [palm(owner.left.wrist)])
        self.assertEqual((recovered.phase, recovered.revision, recovered.indices), ('active', 1, (0,)))

    def test_bystander_cannot_pass_and_owner_can_pass_without_leaving(self):
        owner = visitor()
        self.hold(1., .7, [owner], [palm(owner.left.wrist)])
        bystander = visitor(.68, left=(.28, .70), right=(.80, .63))
        stranger = self.hold(1.8, .9, [owner, bystander], [palm(bystander.left.wrist)])
        self.assertEqual((stranger.phase, stranger.revision, stranger.pass_progress), ('active', 1, 0.))

        owner = visitor(left=(.2552, .67), right=(.63, .32))
        hands = [palm(owner.left.wrist), palm(owner.right.wrist)]
        blocked_by_grip = self.hold(2.8, .9, [owner], [hands[0], palm(owner.right.wrist, closed=True)])
        self.assertEqual(blocked_by_grip.pass_progress, 0.)
        passing = self.hold(3.8, .3, [owner], hands)
        self.assertAlmostEqual(passing.pass_cursor[0], .84)
        self.assertGreater(passing.pass_progress, 0.)
        released = self.hold(4.2, .5, [owner], hands)
        self.assertEqual((released.phase, released.revision), ('ready', 2))
        self.assertEqual(self.hold(4.8, .8, [owner], hands).phase, 'ready')

        arriving = visitor(.68, left=(.52, .33), right=(.82, .62))
        switched = self.hold(5.7, .7, [owner, arriving], [*hands, palm(arriving.left.wrist)])
        self.assertEqual((switched.phase, switched.revision, switched.indices), ('active', 3, (2,)))

    def test_frame_gap_and_operator_setup_discard_incomplete_claims(self):
        owner = visitor()
        self.hold(1., .4, [owner], [palm(owner.left.wrist)])
        after_gap = self.update(2., [owner], [palm(owner.left.wrist)])
        self.assertEqual((after_gap.phase, after_gap.progress), ('claiming', 0.))
        self.hold(2.1, .6, [owner], [palm(owner.left.wrist)])
        self.ownership.suspend()
        self.update(3., [])
        self.assertEqual(self.hold(3.1, .7, [owner], [palm(owner.left.wrist)]).phase, 'ready')

    def test_body_association_survives_hip_landmark_loss(self):
        owner = visitor()
        self.hold(1., .7, [owner], [palm(owner.left.wrist)])
        shoulders_only = replace(owner, center=(.5, .42))
        self.assertEqual(self.ownership.preview_owner([shoulders_only], 1.8), shoulders_only)
        result = self.update(1.8, [shoulders_only], [palm(owner.left.wrist)])
        self.assertEqual((result.phase, result.revision, result.indices), ('active', 1, (0,)))
