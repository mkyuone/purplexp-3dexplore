"""Associate depth-valid hand poses and classify mutually exclusive grips."""

from dataclasses import dataclass, replace
from itertools import product
from math import dist, exp, isfinite
from typing import Literal

from calibration import Pose, Vec3

Tracking = Literal['tracking', 'lost', 'ambiguous']
Grip = Literal['open', 'pinch', 'fist']
Vec2 = tuple[float, float]

MAX_HAND_TRAVEL = 0.15
AMBIGUOUS_MATCH_MARGIN = 0.028
MIN_PALM_SEPARATION = 0.045
TRACK_TIMEOUT = 0.24
REACQUIRE_DISTANCE_M = 0.085
PINCH_START_GAP_M = 0.032
PINCH_END_GAP_M = 0.052
PINCH_START_RATIO = 0.42
PINCH_END_RATIO = 0.64
PINCH_CONFIRM_SECONDS = 0.035
PINCH_CONFIRM_MAX_INTERVAL = TRACK_TIMEOUT
PINCH_EVIDENCE_GRACE = 0.12
OPTICAL_PINCH_START_RATIO = 0.28
OPTICAL_PINCH_HOLD_RATIO = 0.38
OPTICAL_PINCH_OPEN_RATIO = 0.50
OPTICAL_WORLD_START_RATIO = 0.34
OPTICAL_WORLD_HOLD_RATIO = 0.52
GRIP_RELEASE_CONFIRM_SECONDS = 0.065
PINCH_CLEAR_OPEN_GAP_M = 0.072
PINCH_CLEAR_OPEN_RATIO = 0.76


@dataclass(frozen=True)
class GripFeatures:
    pinch_ratio: float
    folded: int
    loosely_folded: int


@dataclass(frozen=True)
class Candidate:
    pose: Pose
    side: str | None
    features: GripFeatures
    pinch_gap: float | None
    pointing: bool
    image_index: Vec2 | None = None
    image_palm: Vec2 | None = None
    optical_pinch_ratio: float | None = None
    thumbs_up: bool = False


@dataclass(frozen=True)
class Hand:
    id: str
    pose: Pose
    grip: Grip
    pointing: bool
    # Unmirrored normalized aim in the aligned RGB image.
    image: Vec2 | None = None
    thumbs_up: bool = False


@dataclass(frozen=True)
class Track:
    hand: Hand
    palm: Vec3
    side: str | None
    pinch_since: float | None = None
    pinch_evidence_at: float | None = None
    release_since: float | None = None
    published: bool = True
    observed_at: float | None = None
    interrupted: bool = False


def grip_features(points: list[Vec3]) -> GripFeatures | None:
    if len(points) != 21 or not all(isfinite(value) for point in points for value in point):
        return None
    palm_width = dist(points[5], points[17])
    if palm_width < 1e-5:
        return None

    folded = loosely_folded = 0
    wrist = points[0]
    for base in (5, 9, 13, 17):
        mcp, pip, dip, tip = points[base:base + 4]
        first = tuple(b - a for a, b in zip(mcp, pip))
        second = tuple(b - a for a, b in zip(pip, dip))
        span = dist(mcp, pip) * dist(pip, dip)
        wrist_span = dist(wrist, mcp)
        if span < 1e-10 or wrist_span < 1e-5:
            return None
        bend = sum(a * b for a, b in zip(first, second)) / span
        reach = dist(wrist, tip) / wrist_span
        folded += bend < 0.55 and reach < 1.25
        loosely_folded += bend < 0.78 and reach < 1.45

    return GripFeatures(dist(points[4], points[8]) / palm_width, folded, loosely_folded)


def classify_grip(candidate: Candidate, previous: Grip = 'open') -> Grip:
    if candidate.thumbs_up:
        return 'open'
    features = candidate.features
    gap = candidate.pinch_gap
    if previous == 'fist':
        return 'fist' if features.loosely_folded >= 3 else 'open'
    if previous != 'pinch' and features.folded == 4:
        return 'fist'
    if gap is None:
        # Both optical estimates must agree. An actual measured gap takes
        # precedence when both fingertips have depth.
        ratio = valid_optical_ratio(candidate)
        if ratio is None or not candidate.pose.valid() or features.folded == 4 or features.loosely_folded == 4:
            return 'open'
        if previous == 'pinch':
            closed = ratio <= OPTICAL_PINCH_HOLD_RATIO and 0 <= features.pinch_ratio <= OPTICAL_WORLD_HOLD_RATIO
        else:
            closed = ratio <= OPTICAL_PINCH_START_RATIO and 0 <= features.pinch_ratio <= OPTICAL_WORLD_START_RATIO
    elif previous == 'pinch':
        closed = gap < PINCH_END_GAP_M and features.pinch_ratio < PINCH_END_RATIO and features.folded < 4
    else:
        closed = gap <= PINCH_START_GAP_M and features.pinch_ratio <= PINCH_START_RATIO
    return 'pinch' if closed else 'open'


def uncertain_release(candidate: Candidate, previous: Grip) -> bool:
    if candidate.thumbs_up:
        return False
    features = candidate.features
    if previous == 'fist':
        return features.loosely_folded == 2
    gap = candidate.pinch_gap
    if previous != 'pinch' or gap is None or not candidate.pointing or features.folded == 4:
        return False
    if gap >= PINCH_CLEAR_OPEN_GAP_M or features.pinch_ratio >= PINCH_CLEAR_OPEN_RATIO:
        return False
    gap_closed = gap < PINCH_END_GAP_M
    ratio_closed = features.pinch_ratio < PINCH_END_RATIO
    return gap_closed != ratio_closed


def blend(start: Vec3, end: Vec3, alpha: float) -> Vec3:
    return (
        start[0] + (end[0] - start[0]) * alpha,
        start[1] + (end[1] - start[1]) * alpha,
        start[2] + (end[2] - start[2]) * alpha,
    )


def valid_image(point: Vec2 | None) -> bool:
    return point is not None and len(point) == 2 and all(isfinite(value) and 0 <= value <= 1 for value in point)


def valid_optical_ratio(candidate: Candidate) -> float | None:
    ratio = candidate.optical_pinch_ratio
    if ratio is None or not isfinite(ratio) or ratio < 0:
        return None
    if not valid_image(candidate.image_index) or not valid_image(candidate.image_palm):
        return None
    return ratio


def blend_image(start: Vec2, end: Vec2, alpha: float) -> Vec2:
    return start[0] + (end[0] - start[0]) * alpha, start[1] + (end[1] - start[1]) * alpha


def match_cost(sample: Candidate, track: Track) -> float | None:
    if track.interrupted and not same_reacquired_hand(sample, track):
        return None
    distance = dist(sample.pose.palm, track.palm)
    if distance > MAX_HAND_TRAVEL:
        return None
    side_changed = sample.side is not None and track.side is not None and sample.side != track.side
    return distance + (0.045 if side_changed else 0)


def same_reacquired_hand(sample: Candidate, track: Track) -> bool:
    return dist(sample.pose.palm, track.palm) <= REACQUIRE_DISTANCE_M and (
        sample.side is None or track.side is None or sample.side == track.side
    )


class HandTracker:
    def __init__(self) -> None:
        self.tracks: tuple[Track, ...] = ()
        self.next_id = 0
        self.last_at: float | None = None

    def reset(self) -> None:
        self.tracks = ()
        self.last_at = None

    def pause(self, at: float, tracking: Tracking) -> tuple[Tracking, tuple[Hand, ...]]:
        # Keep identity privately, never publish an old pose as a current one.
        # Repeated missing frames cannot extend the last valid observation.
        if self.tracks and self.last_at is not None and 0 <= at - self.last_at < TRACK_TIMEOUT:
            self.tracks = tuple(
                replace(track, interrupted=True) for track in self.tracks
                if track.observed_at is not None and 0 <= at - track.observed_at < TRACK_TIMEOUT
            )
        else:
            self.reset()
        return tracking, ()

    def assignments(self, samples: list[Candidate]) -> tuple[int, ...] | None:
        # There are at most two hands. Rank the small set of one-to-one
        # assignments by continuity, using MediaPipe's handedness as a hint.
        candidates: list[tuple[int, float, tuple[int, ...]]] = []
        for assignment in product(range(-1, len(self.tracks)), repeat=len(samples)):
            matched = [index for index in assignment if index >= 0]
            if len(set(matched)) != len(matched):
                continue
            costs = [
                match_cost(sample, self.tracks[index])
                for sample, index in zip(samples, assignment) if index >= 0
            ]
            if any(cost is None for cost in costs):
                continue
            candidates.append((-len(matched), sum(cost for cost in costs if cost is not None), assignment))

        candidates.sort()
        count, cost, assignment = candidates[0]
        if len(candidates) > 1 and candidates[1][0] == count and candidates[1][1] - cost < AMBIGUOUS_MATCH_MARGIN:
            return None
        return assignment

    def update(self, samples: list[Candidate | None], at: float) -> tuple[Tracking, tuple[Hand, ...]]:
        if not isfinite(at) or (self.last_at is not None and at < self.last_at):
            self.reset()
            return 'lost', ()
        if not samples:
            return self.pause(at, 'lost')
        if len(samples) > 2:
            self.reset()
            return 'ambiguous', ()
        observed = [sample for sample in samples if sample is not None and sample.pose.valid()]
        if not observed:
            return self.pause(at, 'ambiguous' if len(samples) > 1 else 'lost')
        if len(observed) == 2 and dist(observed[0].pose.palm, observed[1].pose.palm) < MIN_PALM_SEPARATION:
            self.reset()
            return 'ambiguous', ()
        if self.last_at is not None and at - self.last_at >= TRACK_TIMEOUT:
            self.reset()
        self.tracks = tuple(
            track for track in self.tracks
            if track.observed_at is not None and 0 <= at - track.observed_at < TRACK_TIMEOUT
        )

        assignment = self.assignments(observed)
        if assignment is None:
            self.reset()
            return 'ambiguous', ()
        next_tracks = []
        for sample, index in zip(observed, assignment):
            previous = self.tracks[index] if index >= 0 else None
            resuming = previous is not None and previous.interrupted
            elapsed = at - previous.observed_at if previous is not None else 0
            alpha = 1.0 if resuming else 1 - exp(-max(0.001, elapsed) / 0.045)
            previous_grip = previous.hand.grip if previous else 'open'
            grip = classify_grip(sample, previous_grip)
            features = sample.features
            thumbs_up = sample.thumbs_up and valid_image(sample.image_palm)
            optical_aim = valid_image(sample.image_index)
            optical_ratio = valid_optical_ratio(sample)
            optically_open = optical_ratio is not None and optical_ratio >= OPTICAL_PINCH_OPEN_RATIO
            optical_pinch = sample.pinch_gap is None and grip == 'pinch'
            private_candidate = not previous.published if previous else optical_pinch
            pinch_since = None
            pinch_evidence_at = None
            release_since = None
            pinch_without_gap = (
                previous_grip == 'pinch' and sample.pinch_gap is None and features.folded < 4
                and not optical_pinch and not optically_open and not thumbs_up
                and (features.pinch_ratio < PINCH_END_RATIO if sample.pointing
                     else optical_aim and features.pinch_ratio <= PINCH_START_RATIO)
            )
            if pinch_without_gap:
                if previous.pinch_evidence_at is None or at - previous.pinch_evidence_at > PINCH_EVIDENCE_GRACE:
                    continue
                grip = 'pinch'
                pinch_evidence_at = previous.pinch_evidence_at
                release_since = previous.release_since
            elif grip == 'pinch':
                pinch_evidence_at = at
                if (previous is not None or optical_pinch) and previous_grip != 'pinch':
                    pinch_since = at
                    if previous and not resuming and elapsed <= PINCH_CONFIRM_MAX_INTERVAL and previous.pinch_since is not None:
                        pinch_since = previous.pinch_since
                    if at - pinch_since < PINCH_CONFIRM_SECONDS:
                        grip = 'open'
            if previous is not None and grip == 'open' and uncertain_release(sample, previous_grip):
                release_since = previous.release_since if previous.release_since is not None else at
                if at - release_since < GRIP_RELEASE_CONFIRM_SECONDS:
                    grip = previous_grip
                    pinch_evidence_at = previous.pinch_evidence_at
            clearly_open = (grip == 'open' and features.loosely_folded < 3
                            and (optically_open or features.pinch_ratio >= PINCH_END_RATIO
                                 or sample.pinch_gap is not None and sample.pinch_gap >= PINCH_CLEAR_OPEN_GAP_M))
            # A hand first seen already closed has no neutral observation to
            # publish. Keep its identity private until it confirms or opens.
            published = not (private_candidate and grip == 'open' and not (clearly_open or thumbs_up))
            if published and not sample.pointing and grip != 'fist' and not (optical_pinch or pinch_without_gap or thumbs_up):
                if not optical_aim or not clearly_open:
                    continue

            tip, palm = sample.pose.tip, sample.pose.palm
            palm_anchor = grip == 'fist' or thumbs_up
            image = sample.image_palm if palm_anchor else sample.image_index
            if not valid_image(image):
                image = None
            if previous:
                old = previous.hand
                identifier = old.id
                palm = blend(old.pose.palm, palm, alpha)
                old_palm_anchor = old.grip == 'fist' or old.thumbs_up
                if not palm_anchor and not old_palm_anchor and sample.pointing and old.pointing:
                    tip = blend(old.pose.tip, tip, alpha)
                if image is not None and old.image is not None and palm_anchor == old_palm_anchor:
                    image = blend_image(old.image, image, alpha)
            else:
                self.next_id += 1
                identifier = f'hand-{self.next_id}'

            side = sample.side if sample.side is not None else previous.side if previous else None
            # Optical-only aiming carries the measured palm as its physical anchor.
            pose = Pose(tip if not palm_anchor and sample.pointing else palm, palm)
            hand = Hand(identifier, pose, grip, sample.pointing, image, thumbs_up)
            next_tracks.append(Track(
                hand, sample.pose.palm, side, pinch_since, pinch_evidence_at, release_since,
                published, at,
            ))

        hands = tuple(track.hand for track in sorted(next_tracks, key=lambda track: track.hand.id) if track.published)
        # A missing hand retains identity for a bounded return, but its old
        # position never appears in this frame or refreshes its observation time.
        if all(index >= 0 for index in assignment):
            observed_ids = {track.hand.id for track in next_tracks}
            next_tracks.extend(
                replace(track, interrupted=True) for track in self.tracks
                if track.hand.id not in observed_ids
            )
        self.last_at = at
        self.tracks = tuple(sorted(next_tracks, key=lambda track: track.hand.id))
        return ('tracking', hands) if hands else ('lost', ())
