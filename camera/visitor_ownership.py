"""Give one visitor at a time control of the exhibit's hand gestures."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import dist, isfinite
from typing import Literal

from estimators import EstimatedHand, visible_landmark
from navigation import fuzzy_image_aim
from tracking import Candidate, Vec2, classify_grip, valid_image
from visitor_pose import BodyArm, BodyPose


CLAIM_SECONDS = .6
PASS_SECONDS = .8
OWNER_LOST_SECONDS = 1.5
MAX_HOLD_GAP_SECONDS = .25
BODY_AMBIGUITY_MARGIN = .038
PASS_TARGET = (.75, .95, .58, .82)

VisitorPhase = Literal['ready', 'claiming', 'active', 'paused']
ArmSide = Literal['left', 'right']
ARM_SIDES: tuple[ArmSide, ArmSide] = ('left', 'right')
ArmSlot = tuple[int, ArmSide]


@dataclass(frozen=True)
class VisitorUpdate:
    phase: VisitorPhase
    revision: int
    indices: tuple[int, ...] = ()
    progress: float = 0.
    pass_progress: float = 0.
    pass_cursor: Vec2 | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        result = {
            'phase': self.phase, 'revision': self.revision, 'progress': self.progress,
            'passProgress': self.pass_progress,
        }
        if self.pass_cursor is not None:
            result['passCursor'] = {'x': self.pass_cursor[0], 'y': self.pass_cursor[1]}
        if self.reason is not None:
            result['reason'] = self.reason
        return result


@dataclass(frozen=True)
class _RaisedArm:
    body: BodyPose
    side: ArmSide
    at: float


@dataclass(frozen=True)
class _Claim:
    body: BodyPose
    side: ArmSide
    start: Vec2
    at: float


def _in_area(point: Vec2, area: tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = area
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


def _in_claim_area(body: BodyPose) -> bool:
    return _in_area(body.center, (.29, .71, .17, .90))


def _in_turn_area(body: BodyPose) -> bool:
    # Allow visitors to shift their weight around the edges without a handoff.
    return _in_area(body.center, (.22, .78, .12, .95))


def _raised(arm: BodyArm) -> bool:
    if arm.wrist is None:
        return False
    if arm.shoulder is not None:
        return arm.wrist[1] <= arm.shoulder[1] + .07
    return arm.elbow is not None and arm.wrist[1] <= arm.elbow[1] - .06


def _shoulder_center(body: BodyPose) -> Vec2 | None:
    left, right = body.left.shoulder, body.right.shoulder
    if left is None or right is None:
        return None
    return (left[0] + right[0]) / 2, (left[1] + right[1]) / 2


def _body_candidates(previous: BodyPose, bodies: Sequence[BodyPose]) -> list[tuple[float, int]]:
    ranked = []
    previous_shoulders = _shoulder_center(previous)
    for index, body in enumerate(bodies):
        shoulders = _shoulder_center(body)
        # The estimated torso center shifts when the hips leave the image.
        center_distance = (dist(previous_shoulders, shoulders) if previous_shoulders is not None and shoulders is not None
                           else dist(previous.center, body.center))
        if center_distance > min(.15, max(.065, previous.scale * .60)):
            continue
        scale_difference = abs(body.scale - previous.scale)
        if scale_difference > max(.05, previous.scale * .38):
            continue
        depth_difference = 0.
        if previous.depth_m is not None and body.depth_m is not None:
            depth_difference = abs(body.depth_m - previous.depth_m)
            if depth_difference > .28:
                continue
        ranked.append((center_distance + scale_difference * .6 + depth_difference * .23, index))
    ranked.sort()
    return ranked


def _unique_body(ranked: list[tuple[float, int]]) -> int | None:
    if not ranked or (len(ranked) > 1 and ranked[1][0] - ranked[0][0] < BODY_AMBIGUITY_MARGIN):
        return None
    return ranked[0][1]


def _match_hands(
    bodies: Sequence[BodyPose],
    detections: Sequence[EstimatedHand],
    samples: Sequence[Candidate | None],
) -> tuple[dict[ArmSlot, int], set[int]]:
    matches: dict[ArmSlot, int] = {}
    uncertain: set[int] = set()
    contested: set[ArmSlot] = set()
    for detection_index, (detection, sample) in enumerate(zip(detections, samples)):
        if sample is None or not detection.landmarks or not visible_landmark(detection.landmarks[0]):
            continue
        wrist = detection.landmarks[0].x, detection.landmarks[0].y
        ranked = []
        for body_index, body in enumerate(bodies):
            for side in ARM_SIDES:
                arm = getattr(body, side)
                if arm.wrist is None:
                    continue
                distance = dist(wrist, arm.wrist)
                if distance > min(.085, max(.036, body.scale * .36)):
                    continue
                if sample.pose is not None and arm.wrist_depth_m is not None:
                    depth_difference = abs(sample.pose.palm[2] - arm.wrist_depth_m)
                    if depth_difference > .22:
                        continue
                    distance += depth_difference * .15
                ranked.append((distance, (body_index, side)))
        ranked.sort()
        if not ranked:
            continue
        distance, slot = ranked[0]
        competitors = [other for score, other in ranked[1:] if score - distance < .026]
        if competitors:
            uncertain.add(slot[0])
            uncertain.update(other[0] for other in competitors)
            continue
        if slot in matches or slot in contested:
            matches.pop(slot, None)
            contested.add(slot)
            uncertain.add(slot[0])
        else:
            matches[slot] = detection_index
    return matches, uncertain


def _open_palm(detection: EstimatedHand, sample: Candidate | None) -> bool:
    if (sample is None or sample.thumbs_up or sample.features.loosely_folded > 1
            or sample.features.pinch_ratio < .60 or not valid_image(sample.image_palm)
            or len(detection.landmarks) != 21 or not all(map(visible_landmark, detection.landmarks))):
        return False
    if sample.optical_pinch_ratio is not None and sample.optical_pinch_ratio < .50:
        return False
    points = [(landmark.x * 4 / 3, landmark.y) for landmark in detection.landmarks]
    palm_width = dist(points[5], points[17])
    extended = 0
    for base in (5, 9, 13, 17):
        finger = points[base:base + 4]
        length = sum(dist(start, end) for start, end in zip(finger, finger[1:]))
        reach = dist(finger[0], finger[-1])
        if reach >= max(.75 * palm_width, .83 * length, .015):
            extended += 1
    return extended >= 3


class VisitorOwnership:
    def __init__(self) -> None:
        self.revision = 0
        self._owner: BodyPose | None = None
        self._owner_seen_at = float('-inf')
        self._owner_paused = False
        self._bystanders: tuple[BodyPose, ...] = ()
        self._claim: _Claim | None = None
        self._pass_side: ArmSide | None = None
        self._pass_at = 0.
        self._blocked: list[_RaisedArm] = []
        self._block_initial_raise = False
        self._last_at: float | None = None

    def suspend(self) -> None:
        """Yield to operator setup and require a fresh claim after it closes."""
        if self._owner is not None:
            self.revision += 1
        self._owner = None
        self._bystanders = ()
        self._owner_paused = False
        self._claim = None
        self._pass_side = None
        self._blocked.clear()
        self._block_initial_raise = True
        self._last_at = None

    def preview_owner(self, poses: Sequence[BodyPose], at: float) -> BodyPose | None:
        """Return a confident body to guide where the hand detector looks."""
        if self._owner is not None:
            if at - self._owner_seen_at >= OWNER_LOST_SECONDS:
                return None
            index = self._match_owner(poses)
            return poses[index] if index is not None and _in_turn_area(poses[index]) else None
        eligible = [body for body in poses if _in_claim_area(body)]
        return eligible[0] if len(eligible) == 1 else None

    def update(
        self, poses: Sequence[BodyPose], detections: Sequence[EstimatedHand],
        samples: Sequence[Candidate | None], at: float,
    ) -> VisitorUpdate:
        if not isfinite(at) or (self._last_at is not None and at <= self._last_at):
            phase = 'paused' if self._owner is not None else 'ready'
            return VisitorUpdate(phase, self.revision, reason='Waiting for a fresh camera frame.')
        if self._last_at is not None and at - self._last_at > MAX_HOLD_GAP_SECONDS:
            self._claim = None
            self._pass_side = None
        self._last_at = at

        if self._block_initial_raise and poses:
            self._block_initial_raise = False
            self._block_raised(poses, at)
        if self._owner is None:
            matches, uncertain = _match_hands(poses, detections, samples)
            return self._try_claim(poses, detections, samples, at, matches, uncertain)

        owner_index = self._match_owner(poses)
        if at - self._owner_seen_at >= OWNER_LOST_SECONDS:
            return self._release(poses, at)
        if owner_index is not None:
            self._owner = poses[owner_index]
            self._bystanders = tuple(body for index, body in enumerate(poses) if index != owner_index)
        if owner_index is None or not _in_turn_area(self._owner):
            self._owner_paused = True
            self._pass_side = None
            reason = ('The visitor match is uncertain.' if owner_index is None
                      else 'The active visitor stepped outside the interaction area.')
            return VisitorUpdate('paused', self.revision, reason=reason)
        self._owner_seen_at = at
        self._owner_paused = False

        matches, uncertain = _match_hands(poses, detections, samples)
        owned = {side: index for (body_index, side), index in matches.items() if body_index == owner_index}
        if owner_index in uncertain:
            self._pass_side = None
            return VisitorUpdate('active' if owned else 'paused', self.revision, tuple(owned.values()),
                                 reason='A hand could not be matched confidently to the active visitor.')
        return self._try_pass(poses, detections, samples, at, owned)

    def _match_owner(self, poses: Sequence[BodyPose]) -> int | None:
        assert self._owner is not None
        owner_candidates = _body_candidates(self._owner, poses)
        owner_index = _unique_body(owner_candidates)
        if owner_index is None:
            return None
        for bystander in self._bystanders:
            candidates = _body_candidates(bystander, poses)
            if not candidates or candidates[0][1] != owner_index:
                continue
            # A previously nearby visitor can move into the owner's last
            # position while the camera loses the owner. Keep the turn paused
            # when the new observation can still belong to that visitor.
            if (candidates[0][0] <= owner_candidates[0][0] + BODY_AMBIGUITY_MARGIN
                    or (self._owner_paused and len(poses) == 1)):
                return None
        return owner_index

    def _try_claim(self, poses, detections, samples, at, matches, uncertain) -> VisitorUpdate:
        blocked = self._still_blocked(poses, at)
        raised = [(index, side) for index, body in enumerate(poses) if _in_claim_area(body)
                  for side in ARM_SIDES if (index, side) not in blocked and _raised(getattr(body, side))]
        visitors = {index for index, _ in raised}
        if len(visitors) > 1:
            self._claim = None
            return VisitorUpdate('ready', self.revision, reason='More than one visitor is raising a hand.')
        candidates = [slot for slot in raised if slot[0] not in uncertain and slot in matches
                      and _open_palm(detections[matches[slot]], samples[matches[slot]])]
        if not candidates:
            self._claim = None
            return VisitorUpdate('ready', self.revision)
        previous = self._claim
        continuation = None if previous is None else (_unique_body(_body_candidates(previous.body, poses)), previous.side)
        slot = continuation if continuation in candidates else candidates[0]
        body_index, side = slot
        body = poses[body_index]
        palm = samples[matches[slot]].image_palm
        if previous is None or slot != continuation or dist(previous.start, palm) > .085:
            self._claim = _Claim(body, side, palm, at)
        else:
            self._claim = _Claim(body, side, previous.start, previous.at)
        elapsed = at - self._claim.at
        if elapsed < CLAIM_SECONDS:
            return VisitorUpdate('claiming', self.revision, progress=elapsed / CLAIM_SECONDS)

        self._owner = body
        self._owner_seen_at = at
        self._bystanders = tuple(pose for index, pose in enumerate(poses) if index != body_index)
        self._owner_paused = False
        self._claim = None
        self._blocked.clear()
        self.revision += 1
        # Start the gesture stream with the open palm that claimed the turn.
        return VisitorUpdate('active', self.revision, (matches[slot],))

    def _try_pass(self, poses, detections, samples, at, owned) -> VisitorUpdate:
        indices = tuple(owned.values())
        if any(samples[index].thumbs_up or classify_grip(samples[index]) != 'open' for index in indices):
            self._pass_side = None
            return VisitorUpdate('active', self.revision, indices)
        cursors = {}
        for side, index in owned.items():
            if _open_palm(detections[index], samples[index]):
                cursor = fuzzy_image_aim(samples[index].image_palm)
                if _in_area(cursor, (.67, .99, .49, .90)):
                    cursors[side] = cursor
        eligible = {side: cursor for side, cursor in cursors.items() if _in_area(cursor, PASS_TARGET)}
        if not eligible:
            self._pass_side = None
            return VisitorUpdate('active', self.revision, indices, pass_cursor=next(iter(cursors.values()), None))

        side = self._pass_side if self._pass_side in eligible else next(iter(eligible))
        if side != self._pass_side:
            self._pass_side = side
            self._pass_at = at
        elapsed = at - self._pass_at
        if elapsed >= PASS_SECONDS:
            return self._release(poses, at)
        return VisitorUpdate('active', self.revision, indices, pass_progress=elapsed / PASS_SECONDS,
                             pass_cursor=eligible[side])

    def _release(self, poses: Sequence[BodyPose], at: float) -> VisitorUpdate:
        outgoing = self._owner
        self.revision += 1
        self._owner = None
        self._bystanders = ()
        self._owner_paused = False
        self._claim = None
        self._pass_side = None
        self._block_raised(poses, at)
        if outgoing is not None and _unique_body(_body_candidates(outgoing, poses)) is None:
            self._blocked.extend(_RaisedArm(outgoing, side, at) for side in ARM_SIDES)
        return VisitorUpdate('ready', self.revision)

    def _block_raised(self, poses: Sequence[BodyPose], at: float) -> None:
        self._blocked = [_RaisedArm(body, side, at) for body in poses for side in ARM_SIDES
                         if _raised(getattr(body, side))]

    def _still_blocked(self, poses: Sequence[BodyPose], at: float) -> set[ArmSlot]:
        blocked: set[ArmSlot] = set()
        recent = []
        for item in self._blocked:
            ranked = _body_candidates(item.body, poses)
            body_index = _unique_body(ranked)
            if body_index is None:
                blocked.update((index, item.side) for _, index in ranked)
                if at - item.at < OWNER_LOST_SECONDS:
                    recent.append(item)
                continue
            body = poses[body_index]
            arm = getattr(body, item.side)
            if arm.wrist is None or _raised(arm):
                blocked.add((body_index, item.side))
                recent.append(_RaisedArm(body, item.side, at))
        self._blocked = recent
        return blocked
