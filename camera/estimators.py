"""Local landmark estimators with a shared, depth-independent image contract."""

import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from math import dist, isfinite
from pathlib import Path
from typing import Protocol, Sequence

from tracking import GripFeatures, grip_features


class GpuInferenceError(RuntimeError):
    """The native GPU hand tracker could not start or process a frame."""


class Landmark(Protocol):
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class ImageLandmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandSide:
    category_name: str
    score: float


@dataclass(frozen=True)
class EstimatedHand:
    landmarks: Sequence[Landmark]
    features: GripFeatures | None
    handedness: Sequence[HandSide]


def visible_landmark(point: Landmark) -> bool:
    return all(isfinite(value) and 0 <= value < 1 for value in (point.x, point.y))


def is_thumbs_up(landmarks: Sequence[Landmark], width: int, height: int, features: GripFeatures) -> bool:
    """Recognize a straight upward thumb with four visibly folded fingers."""
    if len(landmarks) != 21 or width <= 0 or height <= 0 or features.loosely_folded != 4:
        return False
    if not all(visible_landmark(point) for point in landmarks):
        return False
    points = [(point.x * width, point.y * height) for point in landmarks]
    palm_scale = max(dist(points[0], points[9]), dist(points[5], points[17]))
    if palm_scale < 8:
        return False
    mcp, ip, tip = points[2], points[3], points[4]
    first, second = dist(mcp, ip), dist(ip, tip)
    extension = dist(mcp, tip)
    if min(first, second) < palm_scale * 0.10 or extension < palm_scale * 0.52:
        return False
    # A thumb resting across the fist can clear the fingertips while still
    # sitting close to the highest knuckle.
    fist_top = min(y for _, y in points[5:])
    return (
        extension / (first + second) >= 0.92
        and (mcp[1] - tip[1]) / extension >= 0.70
        and tip[1] <= ip[1] - palm_scale * 0.10
        and tip[1] <= fist_top - palm_scale * 0.35
        and dist(tip, points[8]) >= palm_scale * 0.55
    )


class MediaPipeEstimator:
    def __init__(self, mp, detector, *, delegate: str = 'cpu') -> None:
        self.mp = mp
        self.detector = detector
        self.delegate = delegate
        self.diagnostics = {'rawEstimatedHands': 0, 'acceptedHands': 0, 'invalidFeatures': 0}

    def detect(self, rgb, timestamp_ms: int) -> list[EstimatedHand]:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        try:
            result = self.detector.detect_for_video(image, timestamp_ms)
        except (RuntimeError, TimeoutError) as error:
            if self.delegate == 'gpu':
                raise GpuInferenceError('MediaPipe could not process a hand frame on the GPU.') from error
            raise
        hands = []
        for index, landmarks in enumerate(result.hand_landmarks):
            world = result.hand_world_landmarks[index] if index < len(result.hand_world_landmarks) else []
            features = grip_features([(point.x, point.y, point.z) for point in world])
            categories = result.handedness[index] if index < len(result.handedness) else []
            handedness = tuple(HandSide(category.category_name, category.score) for category in categories)
            hands.append(EstimatedHand(landmarks, features, handedness))
        self.diagnostics = {
            'rawEstimatedHands': len(result.hand_landmarks),
            'acceptedHands': len(hands),
            'invalidFeatures': sum(hand.features is None for hand in hands),
        }
        return hands


def dependency_problem(names: tuple[str, ...]) -> str | None:
    missing = [name for name in names if find_spec(name) is None]
    if missing:
        return f'Install the camera bridge packages. Missing {", ".join(missing)}.'
    return None


def tracker_problem(tracker: str, model: Path) -> str | None:
    if tracker != 'mediapipe':
        return f'Unknown hand tracker {tracker!r}.'
    if problem := dependency_problem(('mediapipe', 'numpy')):
        return problem
    if not model.is_file():
        return f'MediaPipe hand model is missing at {model}. Run camera/download_rgb_models.py.'
    return None


def tracker_choices(model: Path) -> list[dict]:
    problem = tracker_problem('mediapipe', model)
    choice = {'value': 'mediapipe', 'label': 'MediaPipe Full', 'available': problem is None}
    if problem:
        choice['reason'] = problem
    return [choice]


@contextmanager
def create_estimator(tracker: str, model: Path, *, num_hands: int = 2, delegate: str = 'cpu'):
    if problem := tracker_problem(tracker, model):
        raise RuntimeError(problem)
    if delegate not in ('cpu', 'gpu'):
        raise ValueError(f'Unknown hand inference delegate {delegate!r}.')
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        base_options = (
            python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.GPU)
            if delegate == 'gpu' else python.BaseOptions(model_asset_path=str(model))
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except Exception as error:
        raise RuntimeError(f'MediaPipe could not load its hand model. {error}') from error

    try:
        detector = vision.HandLandmarker.create_from_options(options)
    except Exception as error:
        if delegate == 'gpu' and isinstance(error, (RuntimeError, TimeoutError)):
            raise GpuInferenceError('MediaPipe could not initialize the GPU hand tracker.') from error
        raise RuntimeError(f'MediaPipe could not load its hand model. {error}') from error
    try:
        yield MediaPipeEstimator(mp, detector, delegate=delegate)
    finally:
        recovering = delegate == 'gpu' and isinstance(sys.exc_info()[1], GpuInferenceError)
        try:
            detector.close()
        except RuntimeError:
            if not recovering:
                raise
            logging.exception('MediaPipe GPU hand tracker could not close after a GPU failure')
