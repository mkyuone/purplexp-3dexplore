"""Map tracked hands into a forgiving interaction region without saved calibration."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import copysign, exp, hypot, isfinite
from statistics import median

from tracking import Grip, Hand


IMAGE_LEFT = 0.14
IMAGE_RIGHT = 0.86
IMAGE_TOP = 0.12
IMAGE_BOTTOM = 0.88
EDGE_EASING = 0.10
XY_DEADZONE = 0.003
DEPTH_DEADZONE = 0.005
DEPTH_WIDTH_M = 0.60
MAX_DEPTH = 0.50
NEUTRAL_SECONDS = 1.8
XY_STILL_SECONDS = 0.070
XY_MOVE_SECONDS = 0.018
DEPTH_STILL_SECONDS = 0.100
DEPTH_MOVE_SECONDS = 0.025


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def soft_axis(value: float, low: float, high: float) -> float:
    value = clamp((value - low) / (high - low), 0.0, 1.0)
    if value < EDGE_EASING:
        fraction = value / EDGE_EASING
        return EDGE_EASING * fraction * fraction * (2 - fraction)
    if value > 1 - EDGE_EASING:
        fraction = (1 - value) / EDGE_EASING
        return 1 - EDGE_EASING * fraction * fraction * (2 - fraction)
    return value


def depth_deadzone(previous: float, current: float) -> float:
    change = current - previous
    return previous + copysign(max(0.0, abs(change) - DEPTH_DEADZONE), change)


def smoothing_alpha(distance: float, elapsed: float, still_seconds: float, move_seconds: float, held: bool) -> float:
    speed = distance / max(elapsed, 0.001)
    # Both ranges keep the same response to small movements and noise.
    fast_speed = 0.50 if held else 1.0
    motion = clamp((speed - 0.08) / (fast_speed - 0.08), 0.0, 1.0)
    seconds = still_seconds + (move_seconds - still_seconds) * motion
    return 1 - exp(-elapsed / seconds)


@dataclass(frozen=True)
class MappedHand:
    id: str
    x: float
    y: float
    depth: float
    grip: Grip
    thumbs_up: bool = False
    depth_valid: bool = True


class FuzzyMapper:
    def __init__(self) -> None:
        self.neutral_depth_m: float | None = None
        self.last_at: float | None = None
        self.previous: dict[str, MappedHand] = {}
        self.output: tuple[MappedHand, ...] = ()

    def reset(self) -> None:
        self.neutral_depth_m = None
        self.last_at = None
        self.previous.clear()
        self.output = ()

    def update(self, hands: Sequence[Hand], at: float) -> tuple[MappedHand, ...]:
        if not isfinite(at) or (self.last_at is not None and at < self.last_at):
            return ()
        if at == self.last_at:
            return self.output
        elapsed = min(0.1, at - self.last_at) if self.last_at is not None else 0.0
        self.last_at = at
        valid = [hand for hand in hands if hand.image is not None and len(hand.image) == 2
                 and (hand.pose is None or hand.pose.valid())
                 and all(isfinite(value) and 0 <= value <= 1 for value in hand.image)]
        measured_depths = [hand.pose.palm[2] for hand in valid if hand.pose is not None]
        if measured_depths:
            observed_depth = median(measured_depths)
            if self.neutral_depth_m is None:
                self.neutral_depth_m = observed_depth
            elif all(hand.grip == 'open' and not hand.thumbs_up for hand in hands):
                alpha = 1 - exp(-elapsed / NEUTRAL_SECONDS)
                self.neutral_depth_m += (observed_depth - self.neutral_depth_m) * alpha

        targets = []
        for hand in valid:
            image_x, image_y = hand.image
            x = 1 - soft_axis(image_x, IMAGE_LEFT, IMAGE_RIGHT)
            y = soft_axis(image_y, IMAGE_TOP, IMAGE_BOTTOM)
            depth = 0.0
            if hand.pose is not None:
                assert self.neutral_depth_m is not None
                depth = clamp((hand.pose.palm[2] - self.neutral_depth_m) / DEPTH_WIDTH_M, -MAX_DEPTH, MAX_DEPTH)
            targets.append(MappedHand(hand.id, x, y, depth, hand.grip, hand.thumbs_up,
                                      depth_valid=hand.pose is not None))

        depth_change = 0.0
        for target in targets:
            previous = self.previous.get(target.id)
            if (previous is not None and previous.depth_valid and target.depth_valid
                    and previous.grip == target.grip and previous.thumbs_up == target.thumbs_up):
                change = abs(depth_deadzone(previous.depth, target.depth) - previous.depth)
                depth_change = max(depth_change, change)
        depth_held = any(hand.grip != 'open' for hand in targets)
        depth_alpha = smoothing_alpha(depth_change, elapsed, DEPTH_STILL_SECONDS, DEPTH_MOVE_SECONDS, depth_held)

        mapped = []
        for hand in targets:
            x, y, depth = hand.x, hand.y, hand.depth
            previous = self.previous.get(hand.id)
            if previous is not None and previous.grip == hand.grip and previous.thumbs_up == hand.thumbs_up:
                dx, dy = x - previous.x, y - previous.y
                distance = hypot(dx, dy)
                fraction = max(0.0, 1 - XY_DEADZONE / distance) if distance else 0.0
                xy_alpha = smoothing_alpha(distance * fraction, elapsed, XY_STILL_SECONDS, XY_MOVE_SECONDS,
                                           hand.grip != 'open')
                x = x if x in (0.0, 1.0) else previous.x + dx * fraction * xy_alpha
                y = y if y in (0.0, 1.0) else previous.y + dy * fraction * xy_alpha
                if previous.depth_valid and hand.depth_valid:
                    depth = previous.depth + (depth_deadzone(previous.depth, depth) - previous.depth) * depth_alpha
            mapped.append(MappedHand(hand.id, x, y, depth, hand.grip, hand.thumbs_up,
                                     depth_valid=hand.depth_valid))

        self.output = tuple(mapped)
        self.previous = {hand.id: hand for hand in mapped}
        return self.output
