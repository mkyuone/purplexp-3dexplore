"""Live hand observations and RGB overlay for the diagnostics page."""

from dataclasses import dataclass, replace
from math import isfinite

from estimators import EstimatedHand, visible_landmark
from tracking import Candidate, Track, Vec2


# MediaPipe's 21-point hand topology, in landmark index order.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
HAND_COLORS = {'Left': (193, 239, 111), 'Right': (255, 166, 204), None: (91, 207, 255)}
HandOverlay = tuple[str | None, tuple[Vec2 | None, ...]]


@dataclass(frozen=True)
class DiagnosticHand:
    side: str | None
    grip: str
    depth_mm: int | None
    gap_mm: int | None
    confidence: float | None

    def row(self) -> dict:
        data = {'side': self.side, 'grip': self.grip, 'depthMm': self.depth_mm, 'gapMm': self.gap_mm}
        if self.confidence is not None:
            data['confidence'] = self.confidence
        return data


def matching_track(sample: Candidate | None, tracks: tuple[Track, ...]) -> Track | None:
    if sample is None:
        return None
    if sample.pose is not None:
        return next((track for track in tracks if track.palm == sample.pose.palm), None)
    return next((track for track in tracks if track.palm is None and track.image_palm == sample.image_palm), None)


def diagnostic_hands(
    detections: list[EstimatedHand], samples: list[Candidate | None], tracks: tuple[Track, ...],
) -> tuple[DiagnosticHand, ...]:
    result = []
    assigned = {}
    for estimate, sample in zip(detections, samples):
        track = matching_track(sample, tracks)
        label = estimate.handedness[0] if estimate.handedness else None
        side = track.side if track is not None and track.side in ('Left', 'Right') else None
        confidence = None
        # Tasks receives unmirrored camera pixels, so labels are anatomical.
        if label is not None and label.category_name in ('Left', 'Right') and 0.65 <= label.score <= 1:
            side = side or label.category_name
            if side == label.category_name:
                confidence = round(label.score, 3)
        grip = 'untracked'
        if track is not None:
            hand = track.hand
            grip = 'thumbs up' if hand.thumbs_up else 'pinch pending' if hand.pinch_pending else hand.grip
        depth = sample.pose.palm[2] if sample is not None and sample.pose is not None else None
        gap = sample.pinch_gap if sample is not None else None
        row = DiagnosticHand(side, grip, to_mm(depth), to_mm(gap), confidence)
        if side is not None and side in assigned:
            previous = assigned[side]
            prior = result[previous]
            if (grip != 'untracked', confidence or 0) > (prior.grip != 'untracked', prior.confidence or 0):
                result[previous] = replace(prior, side=None, confidence=None)
                assigned[side] = len(result)
            else:
                row = replace(row, side=None, confidence=None)
        elif side is not None:
            assigned[side] = len(result)
        result.append(row)
    return tuple(result)


def to_mm(meters: float | None) -> int | None:
    return round(1000 * meters) if meters is not None and isfinite(meters) and meters >= 0 else None


def hand_overlay(detections: list[EstimatedHand], hands: tuple[DiagnosticHand, ...]) -> tuple[HandOverlay, ...]:
    return tuple(
        (hand.side, tuple((point.x, point.y) if visible_landmark(point) else None for point in estimate.landmarks))
        for estimate, hand in zip(detections, hands) if len(estimate.landmarks) == 21
    )


def encode_diagnostic_rgb(cv2, rgb, overlay: tuple[HandOverlay, ...]) -> bytes | None:
    height, width = rgb.shape[:2]
    scale = min(480 / width, 360 / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    frame = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    for side, points in overlay:
        color = HAND_COLORS[side]
        pixels = [None if point is None else (round(point[0] * (size[0] - 1)), round(point[1] * (size[1] - 1)))
                  for point in points]
        for start, end in HAND_CONNECTIONS:
            if pixels[start] is not None and pixels[end] is not None:
                cv2.line(frame, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
        for point in pixels:
            if point is not None:
                cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    ok, image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return image.tobytes() if ok else None
