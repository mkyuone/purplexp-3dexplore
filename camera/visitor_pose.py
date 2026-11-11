"""Asynchronous RGB body landmarks for assigning hands to visitors."""

from contextlib import contextmanager
from dataclasses import dataclass
from math import dist, isfinite
from pathlib import Path
from threading import Lock

from tracking import Vec2


POSE_MODEL_NAME = 'pose_landmarker_lite.task'
DEFAULT_POSE_MODEL = Path(__file__).resolve().parents[1] / '.runtime' / 'camera' / POSE_MODEL_NAME
POSE_INTERVAL = .08
MAX_POSE_AGE = .25
LANDMARK_CONFIDENCE = .55
LEFT_ARM = (11, 13, 15)
RIGHT_ARM = (12, 14, 16)


@dataclass(frozen=True, slots=True)
class BodyArm:
    shoulder: Vec2 | None
    elbow: Vec2 | None
    wrist: Vec2 | None
    wrist_depth_m: float | None = None


@dataclass(frozen=True, slots=True)
class BodyPose:
    center: Vec2
    left: BodyArm
    right: BodyArm
    scale: float
    depth_m: float | None = None


@dataclass(frozen=True, slots=True)
class PoseSample:
    at: float
    bodies: tuple[BodyPose, ...]


def image_point(landmark) -> Vec2 | None:
    if any(score is None or not isfinite(score) or score < LANDMARK_CONFIDENCE
           for score in (landmark.visibility, landmark.presence)):
        return None
    x, y = landmark.x, landmark.y
    if all(value is not None and isfinite(value) and 0 <= value < 1 for value in (x, y)):
        return x, y
    return None


def body_poses(poses) -> tuple[BodyPose, ...]:
    bodies = []
    for pose in poses:
        if len(pose) <= RIGHT_ARM[-1]:
            continue
        left = BodyArm(*(image_point(pose[index]) for index in LEFT_ARM))
        right = BodyArm(*(image_point(pose[index]) for index in RIGHT_ARM))
        if left.shoulder is None or right.shoulder is None:
            continue
        # Both shoulders remain the reference when the lower body leaves the image.
        lx, ly = left.shoulder
        rx, ry = right.shoulder
        bodies.append(BodyPose(((lx + rx) / 2, (ly + ry) / 2), left, right, dist(left.shoulder, right.shoulder)))
    return tuple(bodies)


class VisitorPoseEstimator:
    def __init__(self, mp) -> None:
        self.mp = mp
        self.detector = None
        self.last_submission = float('-inf')
        self.lock = Lock()
        self.latest: PoseSample | None = None

    def receive(self, result, _image, timestamp_ms: int) -> None:
        sample = PoseSample(timestamp_ms / 1000, body_poses(result.pose_landmarks))
        with self.lock:
            if self.latest is None or sample.at > self.latest.at:
                self.latest = sample

    def update(self, rgb, at: float) -> PoseSample | None:
        if self.detector is not None and at - self.last_submission >= POSE_INTERVAL:
            self.last_submission = at
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
            self.detector.detect_async(image, int(at * 1000))
        with self.lock:
            sample = self.latest
        return sample if sample is not None and 0 <= at - sample.at <= MAX_POSE_AGE else None


@contextmanager
def create_pose_estimator(model: Path = DEFAULT_POSE_MODEL):
    if not model.is_file():
        raise RuntimeError(f'MediaPipe pose model is missing at {model}. Run python camera/download_rgb_models.py.')
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        estimator = VisitorPoseEstimator(mp)
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=3,
            result_callback=estimator.receive,
        )
        detector = vision.PoseLandmarker.create_from_options(options)
    except Exception as error:
        raise RuntimeError(f'MediaPipe could not load its pose model. {error}') from error
    estimator.detector = detector
    try:
        yield estimator
    finally:
        detector.close()
