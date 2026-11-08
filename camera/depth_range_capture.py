"""Capture the thumb-tip reach while the operator confirms each end by gesture."""

from collections import deque
from dataclasses import dataclass
from statistics import median
from typing import Literal

from depth_range import MAX_DEPTH_M, MIN_DEPTH_M, DepthRange
from realsense import Frame, Snapshot


OPEN_WINDOW = 0.36
OPEN_MIN_SPAN = 0.14
OPEN_MIN_SAMPLES = 5
OPEN_TOLERANCE_M = 0.020
TRANSITION_TIMEOUT = 0.80
CONFIRM_HOLD = 0.60
CONFIRM_TOLERANCE_M = 0.05
CONFIRM_DEPTH_GRACE = 0.12
MAX_FRAME_GAP = 0.15


@dataclass(frozen=True)
class ThumbReading:
    at: float
    depth_m: float


class DepthRangeCapture:
    def __init__(self, started_at: float) -> None:
        self.step: Literal['near', 'far', 'complete'] = 'near'
        self.near_m: float | None = None
        self.capture_m: float | None = None
        self.confirming = False
        self.result: DepthRange | None = None
        self.hint = 'Open your fingers and hold your thumb steady at the closest distance.'
        self._last_at = started_at
        self._hand_id: str | None = None
        self._open_readings: deque[ThumbReading] = deque()
        self._latched: ThumbReading | None = None
        self._holding_since: float | None = None
        self._missing_since: float | None = None
        self._retry_hint: str | None = None

    def observe(self, snapshot: Snapshot) -> None:
        if snapshot.frame is None or self.result is not None:
            return
        # Process camera observations in order so a gap between setup polls
        # cannot hide a release, a second hand or a tracking dropout.
        for frame in (*snapshot.gesture_history, snapshot.frame):
            if frame.at > self._last_at:
                self._observe_frame(frame)
                self._last_at = frame.at
                if self.result is not None:
                    break

    def _clear_evidence(self) -> None:
        self._open_readings.clear()
        self._latched = None
        self.capture_m = None
        self._reset_hold()

    def _reset_hold(self) -> None:
        self._holding_since = None
        self._missing_since = None
        self.confirming = False

    def _observe_frame(self, frame: Frame) -> None:
        if frame.at - self._last_at > MAX_FRAME_GAP:
            self._clear_evidence()

        if frame.detected_hands != 1 or frame.tracking != 'tracking' or len(frame.hands) != 1:
            self._clear_evidence()
            self._hand_id = None
            self.hint = 'Keep just one hand in the camera’s view.'
            return

        hand, = frame.hands
        if hand.id != self._hand_id:
            self._clear_evidence()
            self._hand_id = hand.id

        depth_m = frame.thumb_depth_m
        if depth_m is not None and not MIN_DEPTH_M <= depth_m <= MAX_DEPTH_M:
            depth_m = None
        if hand.grip == 'open' and not hand.thumbs_up:
            self._observe_open(frame.at, depth_m)
        else:
            self._open_readings.clear()
            if hand.thumbs_up:
                self._observe_confirmation(frame.at, depth_m)
            else:
                self._reset_hold()

    def _observe_open(self, at: float, depth_m: float | None) -> None:
        if self._holding_since is not None:
            self._clear_evidence()
        if depth_m is None:
            self._open_readings.clear()
            self.capture_m = None
            self.hint = 'Keep your thumb visible to the camera until it can read the distance.'
            return

        readings = self._open_readings
        if self._latched is not None and abs(depth_m - self._latched.depth_m) > CONFIRM_TOLERANCE_M:
            self._latched = None
        while readings and at - readings[0].at > OPEN_WINDOW:
            readings.popleft()
        readings.append(ThumbReading(at, depth_m))

        center = median(sample.depth_m for sample in readings)
        inliers = [sample for sample in readings if abs(sample.depth_m - center) <= OPEN_TOLERANCE_M]
        if (len(inliers) < OPEN_MIN_SAMPLES or len(inliers) * 10 < len(readings) * 7
                or abs(depth_m - center) > OPEN_TOLERANCE_M or at - inliers[0].at < OPEN_MIN_SPAN):
            self.capture_m = None
            self.hint = self._retry_hint or 'Hold your thumb steady with your fingers open.'
            return

        self.capture_m = median(sample.depth_m for sample in inliers)
        self._latched = ThumbReading(at, self.capture_m)
        self._retry_hint = None
        self.hint = 'Keep your thumb in place and curl your fingers into a thumbs up.'

    def _observe_confirmation(self, at: float, depth_m: float | None) -> None:
        if self._missing_since is not None and at - self._missing_since > CONFIRM_DEPTH_GRACE:
            self._reset_hold()
        latch = self._latched
        if latch is None or (self._holding_since is None and at - latch.at > TRANSITION_TIMEOUT):
            self._clear_evidence()
            self.hint = self._retry_hint or 'Open your fingers to measure this distance first.'
            return
        if depth_m is None:
            if self._holding_since is not None and self._missing_since is None:
                self._missing_since = self._last_at
            self.confirming = False
            self.hint = 'Keep your thumb visible to the camera as you confirm.'
            return
        if abs(depth_m - latch.depth_m) > CONFIRM_TOLERANCE_M:
            self._clear_evidence()
            self.hint = 'Your thumb moved. Open your fingers to measure again.'
            return

        if self._holding_since is None:
            self._holding_since = at
        elif self._missing_since is not None:
            self._holding_since += at - self._missing_since
        self._missing_since = None
        self.capture_m = latch.depth_m
        self.confirming = True
        self.hint = 'Hold your thumbs up steady to confirm.'
        if at - self._holding_since >= CONFIRM_HOLD:
            self._accept(latch.depth_m)

    def _accept(self, depth_m: float) -> None:
        self._clear_evidence()
        if self.near_m is None:
            self.near_m = depth_m
            self.step = 'far'
            self._retry_hint = 'Near distance captured. Open your fingers and move to the farthest distance.'
            self.hint = self._retry_hint
            return

        try:
            self.result = DepthRange(self.near_m, depth_m)
        except ValueError:
            self._retry_hint = 'The far point must be at least 20 cm farther away. Open your fingers and try again.'
            self.hint = self._retry_hint
            return

        self.step = 'complete'
        self.hint = 'Both depth endpoints captured.'
