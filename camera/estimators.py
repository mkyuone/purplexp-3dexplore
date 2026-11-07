"""Local landmark estimators with a shared, depth-independent image contract."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from math import dist, isfinite
from pathlib import Path
from statistics import median
from typing import Protocol, Sequence

from tracking import GripFeatures, grip_features


RTMPOSE_MODELS_DIR = Path(__file__).resolve().parents[1] / '.runtime' / 'camera' / 'rtmpose'


class Landmark(Protocol):
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


def image_features(landmarks: Sequence[Landmark], width: int, height: int) -> GripFeatures | None:
    """Classify only the visible image geometry, with no invented physical Z."""
    if len(landmarks) != 21 or width <= 0 or height <= 0:
        return None
    if not all(visible_landmark(landmarks[index]) for index in (0, 4, 5, 8, 9, 17)):
        return None
    points = [(point.x * width, point.y * height, 0.0) for point in landmarks]
    palm_width = dist(points[5], points[17])
    if palm_width < 6:
        return None
    folded = loosely_folded = 0
    wrist = points[0]
    for base in (5, 9, 13, 17):
        # An occluded or offscreen non-aim finger cannot invalidate a visible
        # palm, but it also supplies no evidence for a closed fist.
        if not all(visible_landmark(point) for point in landmarks[base:base + 4]):
            continue
        mcp, pip, dip, tip = points[base:base + 4]
        lengths = (dist(mcp, pip), dist(pip, dip), dist(dip, tip))
        wrist_span = dist(wrist, mcp)
        if min(lengths) < 1 or wrist_span < 2:
            continue
        first = (pip[0] - mcp[0], pip[1] - mcp[1])
        second = (dip[0] - pip[0], dip[1] - pip[1])
        bend = (first[0] * second[0] + first[1] * second[1]) / (lengths[0] * lengths[1])
        reach = dist(wrist, tip) / wrist_span
        folded += bend < 0.55 and reach < 1.25
        loosely_folded += bend < 0.78 and reach < 1.45
    return GripFeatures(dist(points[4], points[8]) / palm_width, folded, loosely_folded)


def visible_landmark(point: Landmark) -> bool:
    return all(isfinite(value) and 0 <= value < 1 for value in (point.x, point.y))


def is_thumbs_up(landmarks: Sequence[Landmark], width: int, height: int, features: GripFeatures) -> bool:
    """Recognize a straight upward thumb with four visibly folded fingers."""
    if len(landmarks) != 21 or width <= 0 or height <= 0 or features.loosely_folded != 4:
        return False
    required = (0, 1, 2, 3, 4, 5, 8, 9, 12, 13, 16, 17, 20)
    if not all(visible_landmark(landmarks[index]) for index in required):
        return False
    points = [(point.x * width, point.y * height) for point in landmarks]
    palm_scale = max(dist(points[0], points[9]), dist(points[5], points[17]))
    if palm_scale < 8:
        return False
    mcp, ip, tip = points[2], points[3], points[4]
    first, second = dist(mcp, ip), dist(ip, tip)
    extension = dist(mcp, tip)
    if min(first, second) < palm_scale * 0.10 or extension < palm_scale * 0.45:
        return False
    palm_y = median(points[index][1] for index in (5, 9, 13, 17))
    finger_y = min(points[index][1] for index in (8, 12, 16, 20))
    return (
        extension / (first + second) >= 0.85
        and (mcp[1] - tip[1]) / extension >= 0.70
        and tip[1] <= ip[1] - palm_scale * 0.10
        and tip[1] <= mcp[1] - palm_scale * 0.30
        and tip[1] <= palm_y - palm_scale * 0.35
        and tip[1] <= finger_y - palm_scale * 0.20
        and dist(tip, points[8]) >= palm_scale * 0.55
    )


class MediaPipeEstimator:
    def __init__(self, mp, detector) -> None:
        self.mp = mp
        self.detector = detector
        self.diagnostics = {'rawEstimatedHands': 0, 'acceptedHands': 0, 'invalidFeatures': 0}

    def set_camera(self, camera: dict) -> None:
        pass

    def detect(self, rgb, timestamp_ms: int) -> list[EstimatedHand]:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(image, timestamp_ms)
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


class RTMPoseAdapter:
    def __init__(self, detector) -> None:
        self.detector = detector

    def set_camera(self, camera: dict) -> None:
        self.detector.set_camera(camera)

    @property
    def diagnostics(self) -> dict:
        return self.detector.diagnostics

    def detect(self, rgb, timestamp_ms: int) -> list[EstimatedHand]:
        height, width = rgb.shape[:2]
        return [EstimatedHand(
            hand.landmarks,
            image_features(hand.landmarks, width, height),
            (HandSide(hand.side, hand.score),) if hand.side is not None else (),
        ) for hand in self.detector.detect(rgb, timestamp_ms)]


def dependency_problem(names: tuple[str, ...]) -> str | None:
    missing = [name for name in names if find_spec(name) is None]
    if missing:
        return f'Install the camera bridge packages. Missing {", ".join(missing)}.'
    return None


def tracker_problem(tracker: str, model: Path) -> str | None:
    if tracker == 'mediapipe':
        if problem := dependency_problem(('mediapipe', 'numpy')):
            return problem
        if not model.is_file():
            return f'MediaPipe hand model is missing at {model}. Run camera/download_rgb_models.py.'
        return None
    if tracker == 'rtmpose':
        if problem := dependency_problem(('onnxruntime', 'cv2', 'numpy')):
            return problem
        try:
            from rtmpose import model_problem
            return model_problem(RTMPOSE_MODELS_DIR)
        except (ImportError, OSError) as error:
            return f'RTMPose could not load its runtime. {error}'
    return f'Unknown hand tracker {tracker!r}.'


def tracker_choices(model: Path) -> list[dict]:
    choices = []
    for value, label in (
        ('mediapipe', 'MediaPipe Full'),
        ('rtmpose', 'RTMPose (independent model)'),
    ):
        problem = tracker_problem(value, model)
        choice = {'value': value, 'label': label, 'available': problem is None}
        if problem:
            choice['reason'] = problem
        choices.append(choice)
    return choices


@contextmanager
def create_estimator(tracker: str, model: Path):
    if problem := tracker_problem(tracker, model):
        raise RuntimeError(problem)
    if tracker == 'mediapipe':
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(model)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as error:
            raise RuntimeError(f'MediaPipe could not load its hand model. {error}') from error
        estimator = MediaPipeEstimator(mp, detector)
    else:
        try:
            from rtmpose import RTMPoseEstimator
            detector = RTMPoseEstimator(RTMPOSE_MODELS_DIR)
        except Exception as error:
            raise RuntimeError(f'RTMPose could not load its hand models. {error}') from error
        estimator = RTMPoseAdapter(detector)
    try:
        yield estimator
    finally:
        detector.close()
