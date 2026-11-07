"""Associate depth-valid hand poses and classify mutually exclusive grips."""

from dataclasses import dataclass
from itertools import product
from math import dist, exp, isfinite
from typing import Literal

from calibration import Pose, Vec3

Tracking = Literal['tracking', 'lost', 'ambiguous']
Grip = Literal['open', 'pinch', 'fist']

MAX_HAND_TRAVEL = 0.15
AMBIGUOUS_MATCH_MARGIN = 0.028
MIN_PALM_SEPARATION = 0.045
TRACK_TIMEOUT = 0.24


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


@dataclass(frozen=True)
class Hand:
    id: str
    pose: Pose
    grip: Grip
    pointing: bool


@dataclass(frozen=True)
class Track:
    hand: Hand
    palm: Vec3
    side: str | None


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
    features = candidate.features
    gap = candidate.pinch_gap
    if previous == 'fist':
        return 'fist' if features.loosely_folded >= 3 else 'open'
    if previous == 'pinch':
        return 'pinch' if gap is not None and gap < 0.045 and features.pinch_ratio < 0.56 and features.folded < 4 else 'open'
    if features.folded == 4:
        return 'fist'
    if gap is not None and gap <= 0.028 and features.pinch_ratio <= 0.36:
        return 'pinch'
    return 'open'


def blend(start: Vec3, end: Vec3, alpha: float) -> Vec3:
    return (
        start[0] + (end[0] - start[0]) * alpha,
        start[1] + (end[1] - start[1]) * alpha,
        start[2] + (end[2] - start[2]) * alpha,
    )


def match_cost(sample: Candidate, track: Track) -> float | None:
    distance = dist(sample.pose.palm, track.palm)
    if distance > MAX_HAND_TRAVEL:
        return None
    side_changed = sample.side is not None and track.side is not None and sample.side != track.side
    return distance + (0.045 if side_changed else 0)


class HandTracker:
    def __init__(self) -> None:
        self.tracks: tuple[Track, ...] = ()
        self.next_id = 0
        self.last_at: float | None = None

    def reset(self) -> None:
        self.tracks = ()
        self.last_at = None

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
        if not samples:
            self.reset()
            return 'lost', ()
        if len(samples) > 2 or any(sample is None for sample in samples):
            self.reset()
            return ('ambiguous' if len(samples) > 1 else 'lost'), ()

        observed = [sample for sample in samples if sample is not None]
        if len(observed) == 2 and dist(observed[0].pose.palm, observed[1].pose.palm) < MIN_PALM_SEPARATION:
            self.reset()
            return 'ambiguous', ()
        if self.last_at is not None and at - self.last_at >= TRACK_TIMEOUT:
            self.reset()

        assignment = self.assignments(observed)
        if assignment is None:
            self.reset()
            return 'ambiguous', ()

        elapsed = at - self.last_at if self.last_at is not None else 0
        alpha = 1 - exp(-max(0.001, elapsed) / 0.045)
        next_tracks = []
        for sample, index in zip(observed, assignment):
            previous = self.tracks[index] if index >= 0 else None
            grip = classify_grip(sample, previous.hand.grip if previous else 'open')
            if previous and previous.hand.grip == 'pinch' and sample.pinch_gap is None:
                self.reset()
                return 'lost', ()
            if not sample.pointing and grip != 'fist':
                self.reset()
                return 'lost', ()

            tip, palm = sample.pose.tip, sample.pose.palm
            if previous:
                old = previous.hand
                identifier = old.id
                palm = blend(old.pose.palm, palm, alpha)
                if grip != 'fist' and old.grip != 'fist':
                    tip = blend(old.pose.tip, tip, alpha)
            else:
                self.next_id += 1
                identifier = f'hand-{self.next_id}'

            side = sample.side if sample.side is not None else previous.side if previous else None
            # A fist anchors at the palm whether or not the index tip is visible.
            pose = Pose(palm if grip == 'fist' else tip, palm)
            hand = Hand(identifier, pose, grip, sample.pointing)
            next_tracks.append(Track(hand, sample.pose.palm, side))

        self.last_at = at
        self.tracks = tuple(sorted(next_tracks, key=lambda track: track.hand.id))
        return 'tracking', tuple(track.hand for track in self.tracks)
