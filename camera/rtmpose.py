"""Independent RGB hand detection and 21-point RTMPose inference with ONNX.

Model conventions are documented by OpenMMLab's RTMPose hand configuration and
Tau-J/rtmlib's RTMDet, RTMPose and top-down preprocessing implementations.
https://github.com/Tau-J/rtmlib/tree/main/rtmlib/tools
https://github.com/open-mmlab/mmpose/tree/main/configs/hand_2d_keypoint/rtmpose
Only image coordinates are estimated here. RealSense supplies physical depth.
"""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import cv2
import numpy as np


MODEL_NAMES = ('rtmdet_nano_hand.onnx', 'rtmpose_m_hand.onnx')
DETECTION_SIZE = 320
POSE_SIZE = 256
MIN_DETECTION = 0.30
NMS_THRESHOLD = 0.45
RGB_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
RGB_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class HandEstimate:
    landmarks: tuple[Landmark, ...]
    score: float
    confidences: tuple[float, ...]
    side: None = None


def model_problem(folder: Path) -> str | None:
    for name in MODEL_NAMES:
        path = folder / name
        if not path.is_file():
            return f'RTMPose model is missing: {name}'
        with path.open('rb') as model:
            beginning = model.read(128)
        if beginning.startswith(b'version https://git-lfs.github.com/spec/'):
            return f'RTMPose model is a Git LFS pointer. Download the model data for {name}.'
        if path.stat().st_size < 1024:
            return f'RTMPose model is incomplete: {name}'
    return None


def detector_input(rgb):
    height, width = rgb.shape[:2]
    ratio = DETECTION_SIZE / max(height, width)
    resized = cv2.resize(rgb, (int(width * ratio), int(height * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((DETECTION_SIZE, DETECTION_SIZE, 3), 114, dtype=np.uint8)
    padded[:resized.shape[0], :resized.shape[1]] = resized
    # The published RTMDet deploy pipeline uses BGR with to_rgb=False.
    bgr = padded[..., ::-1].astype(np.float32)
    normalized = (bgr - RGB_MEAN[::-1]) / RGB_STD[::-1]
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), ratio


def hand_boxes(detections, ratio: float, width: int, height: int):
    detections = np.asarray(detections)
    if detections.ndim != 3 or detections.shape[0] != 1 or detections.shape[2] != 5:
        raise ValueError('The RTMDet hand model must return baked-NMS boxes shaped [1,N,5].')
    rows = detections[0]
    valid = np.isfinite(rows).all(axis=1) & (rows[:, 4] >= MIN_DETECTION)
    rows = rows[valid].copy()
    rows[:, :4] /= ratio
    valid = ((rows[:, 2] - rows[:, 0] >= 6) & (rows[:, 3] - rows[:, 1] >= 6)
             & (rows[:, 2] > 0) & (rows[:, 3] > 0) & (rows[:, 0] < width) & (rows[:, 1] < height))
    rows = rows[valid]
    order = np.argsort(-rows[:, 4], kind='stable')
    kept = []
    # Apply a final class-independent NMS before limiting inference to two hands.
    while len(order) and len(kept) < 2:
        index = int(order[0])
        kept.append(rows[index])
        remaining = order[1:]
        start = np.maximum(rows[index, :2], rows[remaining, :2])
        end = np.minimum(rows[index, 2:4], rows[remaining, 2:4])
        extent = np.maximum(end - start, 0)
        intersection = extent[:, 0] * extent[:, 1]
        area = np.prod(rows[index, 2:4] - rows[index, :2])
        other_area = np.prod(rows[remaining, 2:4] - rows[remaining, :2], axis=1)
        overlap = intersection / (area + other_area - intersection)
        order = remaining[overlap <= NMS_THRESHOLD]
    return kept, len(rows)


def pose_input(rgb, box):
    box = np.asarray(box, dtype=np.float64)
    center = (box[:2] + box[2:4]) / 2
    extent = float(np.max(box[2:4] - box[:2]) * 1.25)
    factor = POSE_SIZE / extent
    affine = np.array([[factor, 0, POSE_SIZE / 2 - center[0] * factor],
                       [0, factor, POSE_SIZE / 2 - center[1] * factor]], dtype=np.float32)
    crop = cv2.warpAffine(rgb, affine, (POSE_SIZE, POSE_SIZE), flags=cv2.INTER_LINEAR)
    # The pose model was trained with bgr_to_rgb=True, so our RGB input is ready.
    normalized = (crop.astype(np.float32) - RGB_MEAN) / RGB_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), center, extent


def decode_pose(simcc_x, simcc_y, center, extent: float, width: int, height: int):
    simcc_x, simcc_y = np.asarray(simcc_x), np.asarray(simcc_y)
    shape = (1, 21, POSE_SIZE * 2)
    if simcc_x.shape != shape or simcc_y.shape != shape:
        raise ValueError('RTMPose hand outputs must be two SimCC tensors shaped [1,21,512].')
    if not np.isfinite(simcc_x).all() or not np.isfinite(simcc_y).all():
        return None
    confidence = (simcc_x.max(axis=-1)[0] + simcc_y.max(axis=-1)[0]) / 2
    locations = np.stack((simcc_x.argmax(axis=-1)[0], simcc_y.argmax(axis=-1)[0]), axis=-1)
    pixels = locations / 2 / POSE_SIZE * extent + center - extent / 2
    normalized = pixels / np.array([width, height])
    # A nonpositive SimCC response has no valid location. Keep that joint missing
    # for the downstream per-landmark and measured-depth validation.
    normalized[confidence <= 0] = np.nan
    landmarks = tuple(Landmark(float(x), float(y)) for x, y in normalized)
    return landmarks, tuple(float(value) for value in confidence)


class RTMPoseEstimator:
    def __init__(self, models_dir: Path) -> None:
        if problem := model_problem(models_dir):
            raise ValueError(problem)
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError('RTMPose requires the onnxruntime Python package.') from error
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.detector = ort.InferenceSession(str(models_dir / MODEL_NAMES[0]), options,
                                            providers=['CPUExecutionProvider'])
        self.pose = ort.InferenceSession(str(models_dir / MODEL_NAMES[1]), options,
                                        providers=['CPUExecutionProvider'])
        for session, size in ((self.detector, DETECTION_SIZE), (self.pose, POSE_SIZE)):
            inputs = session.get_inputs()
            shape = inputs[0].shape if len(inputs) == 1 else []
            if len(shape) != 4 or shape[1:] != [3, size, size] or (
                isinstance(shape[0], int) and shape[0] != 1
            ):
                raise ValueError('RTMPose model inputs do not match the supported hand models.')
        if len(self.pose.get_outputs()) != 2:
            raise ValueError('RTMPose requires the two-output SimCC hand model.')
        self.last_timestamp = None
        self.last_result: tuple[HandEstimate, ...] = ()
        self.diagnostics: dict = {}

    def set_camera(self, intrinsics: dict) -> None:
        pass

    def close(self) -> None:
        self.detector = None
        self.pose = None
        self.last_result = ()

    def detect(self, rgb, timestamp_ms: int) -> tuple[HandEstimate, ...]:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) < 1:
            raise ValueError('RTMPose expects an 8-bit RGB image.')
        if not isfinite(timestamp_ms):
            raise ValueError('RTMPose frame timestamp must be finite.')
        if timestamp_ms == self.last_timestamp:
            return self.last_result
        self.last_timestamp = timestamp_ms
        height, width = rgb.shape[:2]
        tensor, ratio = detector_input(rgb)
        outputs = self.detector.run(None, {self.detector.get_inputs()[0].name: tensor})
        boxes, count = hand_boxes(outputs[0], ratio, width, height)
        self.diagnostics = {
            'detectorRegions': count, 'acceptedHands': 0, 'maxPoseConfidence': 0.0,
            'maxDetectionConfidence': max((float(box[4]) for box in boxes), default=0.0),
        }
        hands = []
        crops = [pose_input(rgb, box) for box in boxes]
        if crops:
            batch = np.concatenate([crop[0] for crop in crops], axis=0)
            simcc_x, simcc_y = self.pose.run(None, {self.pose.get_inputs()[0].name: batch})
            shape = (len(crops), 21, POSE_SIZE * 2)
            if simcc_x.shape != shape or simcc_y.shape != shape:
                raise ValueError('RTMPose must return one pair of SimCC vectors for each input hand.')
            for index, (box, (_, center, extent)) in enumerate(zip(boxes, crops)):
                decoded = decode_pose(simcc_x[index:index + 1], simcc_y[index:index + 1],
                                      center, extent, width, height)
                if decoded is None:
                    continue
                landmarks, confidences = decoded
                self.diagnostics['maxPoseConfidence'] = max(self.diagnostics['maxPoseConfidence'], max(confidences))
                if all(value <= 0 for value in confidences):
                    continue
                hands.append(HandEstimate(landmarks, float(box[4]), confidences))
        self.last_result = tuple(hands)
        self.diagnostics['acceptedHands'] = len(hands)
        return self.last_result
