"""Associate current hand observations and classify mutually exclusive grips."""

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
MAX_IMAGE_TRAVEL = 0.10
REACQUIRE_IMAGE_TRAVEL = 0.055
MIN_IMAGE_PALM_SEPARATION = 0.06
CLOSE_PAIR_IMAGE_SEPARATION = 0.025
CLOSE_PAIR_IMAGE_TRAVEL = 0.025
IMAGE_MATCH_PENALTY = 0.03
PINCH_START_GAP_M = 0.024
PINCH_END_GAP_M = 0.052
PINCH_START_RATIO = 0.42
PINCH_END_RATIO = 0.64
PINCH_CONFIRM_SECONDS = 0.035
PINCH_CONFIRM_MAX_INTERVAL = TRACK_TIMEOUT
PINCH_EVIDENCE_GRACE = 0.12
OPTICAL_PINCH_START_RATIO = 0.24
OPTICAL_PINCH_HOLD_RATIO = 0.38
OPTICAL_PINCH_OPEN_RATIO = 0.50
OPTICAL_WORLD_START_RATIO = 0.34
OPTICAL_WORLD_HOLD_RATIO = 0.52
OPTICAL_CLEAR_OPEN_RATIO = 0.60
OPTICAL_WORLD_CLEAR_OPEN_RATIO = 0.60
GRIP_RELEASE_CONFIRM_SECONDS = 0.065
PINCH_CLEAR_OPEN_GAP_M = 0.072
PINCH_CLEAR_OPEN_RATIO = 0.76
TRACK_STILL_SECONDS = 0.045
TRACK_MOVING_SECONDS = 0.016


@dataclass(frozen=True)
class GripFeatures:
    pinch_ratio: float
    folded: int
    loosely_folded: int


@dataclass(frozen=True)
class Candidate:
    pose: Pose | None
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
    pose: Pose | None
    grip: Grip
    pointing: bool
    # Unmirrored normalized aim in the aligned RGB image.
    image: Vec2 | None = None
    thumbs_up: bool = False


@dataclass(frozen=True)
class Track:
    hand: Hand
    palm: Vec3 | None
    side: str | None
    pinch_since: float | None = None
    pinch_evidence_at: float | None = None
    release_since: float | None = None
    published: bool = True
    observed_at: float | None = None
    interrupted: bool = False
    image_palm: Vec2 | None = None
    pinch_had_depth: bool = False
    release_handoff: bool = False


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
    if candidate.thumbs_up and candidate.pose is not None:
        return 'open'
    features = candidate.features
    gap = candidate.pinch_gap if candidate.pose is not None else None
    if previous == 'fist':
        return 'fist' if features.loosely_folded >= 3 else 'open'
    if previous == 'pinch' and clear_pinch_opening(candidate):
        return 'open'
    if previous != 'pinch' and features.folded == 4:
        return 'fist'
    if gap is None:
        # Both optical estimates must agree. An actual measured gap takes
        # precedence when both fingertips have depth.
        ratio = valid_optical_ratio(candidate)
        if (ratio is None or (candidate.pose is not None and not candidate.pose.valid())
                or features.folded == 4 or features.loosely_folded == 4):
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
    if candidate.thumbs_up and candidate.pose is not None:
        return False
    features = candidate.features
    if previous == 'fist':
        return features.loosely_folded == 2
    if previous != 'pinch' or features.folded == 4 or clear_pinch_opening(candidate):
        return False
    ratio = valid_optical_ratio(candidate)
    gap = candidate.pinch_gap if candidate.pose is not None and candidate.pointing else None
    if features.pinch_ratio >= PINCH_CLEAR_OPEN_RATIO:
        return False
    if gap is None:
        # A change just outside the optical hold band can be landmark jitter.
        # Without a usable ratio, the separate missing-evidence timer applies.
        return ratio is not None and ratio < OPTICAL_CLEAR_OPEN_RATIO
    if gap >= PINCH_CLEAR_OPEN_GAP_M:
        # A fingertip depth patch can momentarily land on another surface.
        return (ratio is not None and ratio <= OPTICAL_PINCH_HOLD_RATIO
                and 0 <= features.pinch_ratio <= OPTICAL_WORLD_HOLD_RATIO)
    gap_closed = gap < PINCH_END_GAP_M
    ratio_closed = features.pinch_ratio < PINCH_END_RATIO
    return gap_closed != ratio_closed


def supports_pinch_opening(
    features: GripFeatures, gap: float | None, optical_ratio: float | None, *, depth_handoff: bool = False,
) -> bool:
    if features.folded == 4:
        return False
    if depth_handoff:
        # Losing the measured gap must not promote a previously conflicting
        # single optical cue into a deliberate release.
        return (optical_ratio is not None and optical_ratio >= OPTICAL_PINCH_OPEN_RATIO
                and features.pinch_ratio > OPTICAL_WORLD_HOLD_RATIO and features.loosely_folded < 4)
    measured_open = gap is not None and gap >= PINCH_END_GAP_M
    model_open = features.pinch_ratio >= PINCH_END_RATIO
    if optical_ratio is None:
        return measured_open or model_open
    # Corroborate moderate separation when RGB is available. A large gap
    # alone can be a bad depth patch while the observed fingers stay closed.
    return (
        (optical_ratio >= OPTICAL_CLEAR_OPEN_RATIO and features.loosely_folded < 4)
        or (measured_open and model_open)
        or (optical_ratio >= OPTICAL_PINCH_OPEN_RATIO
            and (measured_open or features.pinch_ratio > OPTICAL_WORLD_HOLD_RATIO))
    )


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


def clear_pinch_opening(candidate: Candidate) -> bool:
    ratio = valid_optical_ratio(candidate)
    world_ratio = candidate.features.pinch_ratio
    return (ratio is not None and ratio >= OPTICAL_CLEAR_OPEN_RATIO
            and isfinite(world_ratio) and world_ratio >= OPTICAL_WORLD_CLEAR_OPEN_RATIO
            and candidate.features.loosely_folded < 4)


def blend_image(start: Vec2, end: Vec2, alpha: float) -> Vec2:
    return start[0] + (end[0] - start[0]) * alpha, start[1] + (end[1] - start[1]) * alpha


def tracking_alpha(elapsed: float, travel: float, still: float, moving: float) -> float:
    motion = min(1.0, max(0.0, (travel - still) / (moving - still)))
    seconds = TRACK_STILL_SECONDS + (TRACK_MOVING_SECONDS - TRACK_STILL_SECONDS) * motion
    return 1 - exp(-max(0.001, elapsed) / seconds)


def match_cost(sample: Candidate, track: Track) -> float | None:
    side_changed = sample.side is not None and track.side is not None and sample.side != track.side
    if sample.pose is not None and track.palm is not None:
        distance = dist(sample.pose.palm, track.palm)
        limit = REACQUIRE_DISTANCE_M if track.interrupted else MAX_HAND_TRAVEL
        if distance > limit or (track.interrupted and side_changed):
            return None
        return distance + (0.045 if side_changed else 0)
    # The cursor changes between fingertip and palm. Identity always compares
    # current palm anchors, with a tighter gate when the hand was absent.
    if side_changed or not valid_image(sample.image_palm) or not valid_image(track.image_palm):
        return None
    distance = dist(sample.image_palm, track.image_palm)
    limit = REACQUIRE_IMAGE_TRAVEL if track.interrupted else MAX_IMAGE_TRAVEL
    return distance + IMAGE_MATCH_PENALTY if distance <= limit else None


class HandTracker:
    def __init__(self) -> None:
        self.tracks: tuple[Track, ...] = ()
        self.next_id = 0
        self.last_at: float | None = None
        self.measured_pinch_pair: tuple[frozenset[str], float] | None = None

    def reset(self) -> None:
        self.tracks = ()
        self.last_at = None
        self.measured_pinch_pair = None

    def pause(self, at: float, tracking: Tracking) -> tuple[Tracking, tuple[Hand, ...]]:
        # Keep identity privately, never publish an old pose as a current one.
        # Repeated missing frames cannot extend the last valid observation.
        if self.tracks and self.last_at is not None and 0 <= at - self.last_at < TRACK_TIMEOUT:
            self.tracks = tuple(
                replace(track, interrupted=True, release_since=None) for track in self.tracks
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

    def continues_measured_pair(self, samples: list[Candidate], assignment: tuple[int, ...], at: float) -> bool:
        pair = self.measured_pinch_pair
        if pair is None or at - pair[1] > PINCH_EVIDENCE_GRACE or any(index < 0 for index in assignment):
            return False
        first, second = samples
        if (first.side is None or second.side is None or first.side == second.side
                or not valid_image(first.image_palm) or not valid_image(second.image_palm)
                or dist(first.image_palm, second.image_palm) < CLOSE_PAIR_IMAGE_SEPARATION):
            return False
        tracks = [self.tracks[index] for index in assignment]
        if frozenset(track.hand.id for track in tracks) != pair[0] or not all(
            track.hand.grip == 'pinch' and sample.side == track.side and valid_image(track.image_palm)
            and dist(sample.image_palm, track.image_palm) <= CLOSE_PAIR_IMAGE_TRAVEL
            for sample, track in zip(samples, tracks)
        ):
            return False
        # Near overlap, two plausible labels must not authorize an apparent
        # swap of the palms. Keep the normal ambiguity pause on a crossing.
        previous_a, previous_b = (track.image_palm for track in tracks)
        current_a, current_b = first.image_palm, second.image_palm
        return sum((b - a) * (d - c) for a, b, c, d in zip(previous_a, previous_b, current_a, current_b)) > 0

    def update(self, samples: list[Candidate | None], at: float) -> tuple[Tracking, tuple[Hand, ...]]:
        if not isfinite(at) or (self.last_at is not None and at < self.last_at):
            self.reset()
            return 'lost', ()
        if not samples:
            return self.pause(at, 'lost')
        if len(samples) > 2:
            self.reset()
            return 'ambiguous', ()
        observed = [sample for sample in samples if sample is not None and (
            sample.pose.valid() if sample.pose is not None else valid_image(sample.image_palm)
        )]
        if not observed:
            return self.pause(at, 'ambiguous' if len(samples) > 1 else 'lost')
        close_rgb_pair = False
        if len(observed) == 2:
            first, second = observed
            if first.pose is not None and second.pose is not None:
                if dist(first.pose.palm, second.pose.palm) < MIN_PALM_SEPARATION:
                    return self.pause(at, 'ambiguous')
            else:
                close_rgb_pair = not (valid_image(first.image_palm) and valid_image(second.image_palm)
                                      and dist(first.image_palm, second.image_palm) >= MIN_IMAGE_PALM_SEPARATION)
        if self.last_at is not None and at - self.last_at >= TRACK_TIMEOUT:
            self.reset()
        self.tracks = tuple(
            track for track in self.tracks
            if track.observed_at is not None and 0 <= at - track.observed_at < TRACK_TIMEOUT
        )

        assignment = self.assignments(observed)
        if assignment is None or (close_rgb_pair and not self.continues_measured_pair(observed, assignment, at)):
            return self.pause(at, 'ambiguous')
        next_tracks = []
        suspended_pinches = []
        for sample, index in zip(observed, assignment):
            previous = self.tracks[index] if index >= 0 else None
            resuming = previous is not None and previous.interrupted
            elapsed = at - previous.observed_at if previous is not None else 0
            previous_grip = previous.hand.grip if previous else 'open'
            grip = classify_grip(sample, previous_grip)
            features = sample.features
            measured = sample.pose
            pointing = measured is not None and sample.pointing
            gap = sample.pinch_gap if measured is not None else None
            thumbs_up = measured is not None and sample.thumbs_up and valid_image(sample.image_palm)
            optical_aim = valid_image(sample.image_index)
            optical_ratio = valid_optical_ratio(sample)
            optically_open = optical_ratio is not None and optical_ratio >= OPTICAL_PINCH_OPEN_RATIO
            optical_pinch = gap is None and grip == 'pinch'
            private_candidate = not previous.published if previous else optical_pinch
            pinch_since = None
            pinch_evidence_at = None
            release_since = None
            pinch_had_depth = previous is not None and previous.pinch_had_depth
            recent_pinch = (
                previous is not None and previous.pinch_evidence_at is not None
                and at - previous.pinch_evidence_at <= PINCH_EVIDENCE_GRACE
            )
            depth_handoff = previous_grip == 'pinch' and pinch_had_depth and gap is None and not thumbs_up
            opening_supported = supports_pinch_opening(features, gap, optical_ratio, depth_handoff=depth_handoff)
            pinch_without_gap = (
                previous_grip == 'pinch' and gap is None and optical_ratio is None
                and features.folded < 4 and not thumbs_up
                and (features.pinch_ratio < PINCH_END_RATIO if pointing
                     else optical_aim and features.pinch_ratio <= PINCH_START_RATIO)
            )
            if depth_handoff and grip == 'open' and not opening_supported:
                # A short depth dropout can carry motion using only the new RGB
                # aim. Weak evidence never extends the last confirmed pinch.
                fresh_aim = (optical_aim and valid_image(sample.image_palm)) or (
                    pointing and optical_ratio is None and features.pinch_ratio < PINCH_END_RATIO
                )
                if recent_pinch and fresh_aim and isfinite(features.pinch_ratio) and features.loosely_folded < 4:
                    grip = 'pinch'
                    pinch_evidence_at = previous.pinch_evidence_at
            elif pinch_without_gap:
                if not recent_pinch:
                    continue
                grip = 'pinch'
                pinch_evidence_at = previous.pinch_evidence_at
            elif grip == 'pinch':
                pinch_evidence_at = at
                pinch_had_depth = pinch_had_depth or gap is not None
                if (previous is not None or optical_pinch) and previous_grip != 'pinch':
                    pinch_since = at
                    if previous and not resuming and elapsed <= PINCH_CONFIRM_MAX_INTERVAL and previous.pinch_since is not None:
                        pinch_since = previous.pinch_since
                    if at - pinch_since < PINCH_CONFIRM_SECONDS:
                        grip = 'open'
            confirm_handoff_release = depth_handoff and opening_supported and not clear_pinch_opening(sample)
            if previous is not None and grip == 'open' and (
                confirm_handoff_release or (not depth_handoff and uncertain_release(sample, previous_grip))
            ):
                release_since = previous.release_since if previous.release_handoff == confirm_handoff_release else None
                if release_since is None:
                    release_since = at
                if at - release_since < GRIP_RELEASE_CONFIRM_SECONDS:
                    if previous_grip != 'pinch' or recent_pinch:
                        grip = previous_grip
                        pinch_evidence_at = previous.pinch_evidence_at
                    elif confirm_handoff_release:
                        # Opening can finish confirming without exposing motion
                        # from a hold whose evidence window has already expired.
                        suspended_pinches.append(replace(previous, interrupted=True, pinch_evidence_at=None,
                                                         release_since=release_since, release_handoff=True))
                        continue
            released_pinch = previous is not None and previous_grip == 'pinch' and grip == 'open'
            if released_pinch and not (thumbs_up or opening_supported):
                # The hold grace can expire without evidence that the fingers
                # opened. Keep the ID privately, requiring fresh grip evidence
                # to resume. Never publish or refresh the old pose.
                suspended_pinches.append(replace(previous, interrupted=True, pinch_evidence_at=None,
                                                 release_since=None))
                continue
            known_open = previous is not None and previous.published and previous_grip == 'open'
            clearly_open = grip == 'open' and (
                released_pinch or (known_open and optical_ratio is not None and opening_supported)
                or clear_pinch_opening(sample) or (features.loosely_folded < 3 and (
                    optically_open or features.pinch_ratio >= PINCH_END_RATIO
                    or gap is not None and gap >= PINCH_CLEAR_OPEN_GAP_M
                ))
            )
            # A hand first seen already closed has no neutral observation to
            # publish. Keep its identity private until it confirms or opens.
            published = not (private_candidate and grip == 'open' and not (clearly_open or thumbs_up))
            palm_anchor = grip == 'fist' or thumbs_up
            if published and not pointing and not palm_anchor and (
                not optical_aim or (grip == 'open' and not (clearly_open or optical_pinch))
            ):
                continue

            tip = measured.tip if measured is not None else None
            palm = measured.palm if measured is not None else None
            image = sample.image_palm if palm_anchor else sample.image_index
            if not valid_image(image):
                image = None
            if previous:
                old = previous.hand
                identifier = old.id
                old_palm_anchor = old.grip == 'fist' or old.thumbs_up
                # Keep the usual smoothing when aiming or changing grip.
                tracking_motion = old.grip == grip and grip != 'open'
                if palm is not None and old.pose is not None and not resuming:
                    travel = dist(old.pose.palm, palm) if tracking_motion else 0.0
                    metric_alpha = tracking_alpha(elapsed, travel, still=0.010, moving=0.034)
                    palm = blend(old.pose.palm, palm, metric_alpha)
                    if tip is not None and not palm_anchor and not old_palm_anchor and pointing and old.pointing:
                        tip = blend(old.pose.tip, tip, metric_alpha)
                if image is not None and old.image is not None and palm_anchor == old_palm_anchor and not resuming:
                    travel = dist(old.image, image) if tracking_motion else 0.0
                    image = blend_image(old.image, image, tracking_alpha(elapsed, travel, still=0.007, moving=0.025))
            else:
                self.next_id += 1
                identifier = f'hand-{self.next_id}'

            side = sample.side if sample.side is not None else previous.side if previous else None
            pose = None
            if palm is not None:
                pose = Pose(tip if tip is not None and not palm_anchor and pointing else palm, palm)
            hand = Hand(identifier, pose, grip, pointing, image, thumbs_up)
            next_tracks.append(Track(
                hand, measured.palm if measured is not None else None, side, pinch_since, pinch_evidence_at,
                release_since, published, at, image_palm=sample.image_palm,
                pinch_had_depth=grip == 'pinch' and pinch_had_depth,
                release_handoff=confirm_handoff_release,
            ))

        hands = tuple(track.hand for track in sorted(next_tracks, key=lambda track: track.hand.id) if track.published)
        if any(hand.grip != 'pinch' for hand in hands):
            self.measured_pinch_pair = None
        elif len(hands) == 2 and all(hand.pose is not None for hand in hands):
            self.measured_pinch_pair = frozenset(hand.id for hand in hands), at
        next_tracks.extend(suspended_pinches)
        # A missing hand retains identity for a bounded return, but its old
        # position never appears in this frame or refreshes its observation time.
        if all(index >= 0 for index in assignment):
            observed_ids = {track.hand.id for track in next_tracks}
            next_tracks.extend(
                replace(track, interrupted=True, release_since=None) for track in self.tracks
                if track.hand.id not in observed_ids
            )
        self.last_at = at
        self.tracks = tuple(sorted(next_tracks, key=lambda track: track.hand.id))
        return ('tracking', hands) if hands else ('lost', ())
