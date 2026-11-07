"""Capture aligned D435i frames locally and expose only timestamped hand poses."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from math import dist
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4
import logging

from calibration import MAX_DEPTH_M, MIN_DEPTH_M, Pose, Vec3, median_point, valid_point
from tracking import Candidate, Hand, HandTracker, Tracking, grip_features

FRAME_MAX_AGE = 0.22
CALIBRATION_WINDOW = 0.45
LEASE_SECONDS = 6.0
RGB_SIZE = (640, 480)
FPS = 30


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class CameraInfo:
    profile: str
    session: str


@dataclass(frozen=True)
class Frame:
    sequence: int
    at: float
    tracking: Tracking
    hands: tuple[Hand, ...]


@dataclass(frozen=True)
class Snapshot:
    info: CameraInfo | None
    frame: Frame | None
    history: tuple[Frame, ...]
    error: str | None


def patch_depth(np, depth, x: int, y: int, units: float, near: float | None = None) -> float | None:
    height, width = depth.shape
    patch = depth[max(0, y - 3):min(height, y + 4), max(0, x - 3):min(width, x + 4)] * units
    valid = (patch >= MIN_DEPTH_M) & (patch <= MAX_DEPTH_M)
    if near is not None:
        valid &= np.abs(patch - near) <= 0.16
    samples = patch[valid]
    if samples.size < 9:
        return None
    center = float(np.median(samples))
    if np.count_nonzero(np.abs(samples - center) <= 0.022) < max(9, samples.size * 0.65):
        return None
    return center


def locate_depth(rs, np, depth, intrinsics, units: float, x: float, y: float, near: float | None) -> Vec3 | None:
    if not (0 <= x < 1 and 0 <= y < 1):
        return None
    height, width = depth.shape
    px, py = x * width, y * height
    z = patch_depth(np, depth, int(px), int(py), units, near)
    if z is None:
        return None
    a, b, c = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], z)
    point = (float(a), float(b), float(c))
    return point if valid_point(point) else None


def fuse_hand(rs, np, image_landmarks, world_landmarks, handedness, aligned_depth) -> Candidate | None:
    if len(image_landmarks) != 21:
        return None
    features = grip_features([(point.x, point.y, point.z) for point in world_landmarks])
    if features is None:
        return None
    depth = np.asanyarray(aligned_depth.get_data())
    intrinsics = aligned_depth.profile.as_video_stream_profile().intrinsics
    units = aligned_depth.get_units()

    def locate(x: float, y: float, near: float | None = None) -> Vec3 | None:
        return locate_depth(rs, np, depth, intrinsics, units, x, y, near)

    wrist = image_landmarks[0]
    palm_points = []
    for index in (5, 9, 17):
        base = image_landmarks[index]
        point = locate((wrist.x + base.x) / 2, (wrist.y + base.y) / 2)
        if point is not None:
            palm_points.append(point)
    if len(palm_points) < 2:
        return None
    palm = median_point(palm_points)
    if sum(abs(point[2] - palm[2]) <= 0.055 for point in palm_points) < 2:
        return None
    index = image_landmarks[8]
    thumb = image_landmarks[4]
    tip = locate(index.x, index.y, palm[2])
    if tip is None and features.loosely_folded < 3:
        return None
    thumb_tip = locate(thumb.x, thumb.y, palm[2])
    gap = dist(thumb_tip, tip) if tip is not None and thumb_tip is not None else None
    side = None
    if handedness and handedness[0].score >= 0.65:
        side = handedness[0].category_name
    return Candidate(Pose(tip or palm, palm), side, features, gap, tip is not None)


def open_streams(rs, context, serial: str):
    for width in (848, 640):
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, 480, rs.format.z16, FPS)
        config.enable_stream(rs.stream.color, *RGB_SIZE, rs.format.rgb8, FPS)
        try:
            profile = pipeline.start(config)
        except RuntimeError as error:
            logging.warning('D435i %dx480 depth stream could not start: %s', width, error)
            continue
        return pipeline, profile, width
    raise CameraError('Could not open D435i RGB and depth. Close RealSense Viewer and connect the camera directly over USB 3.')


class CameraFeed:
    def __init__(self, model: Path, serial: str | None = None) -> None:
        self.model = model
        self.serial = serial
        self.lock = Lock()
        self.stop = Event()
        self.wake = Event()
        self.last_contact = float('-inf')
        self.info: CameraInfo | None = None
        self.frame: Frame | None = None
        self.history: deque[Frame] = deque(maxlen=24)
        self.error: str | None = None
        self.thread = Thread(target=self.run, name='D435i capture', daemon=True)
        self.thread.start()

    def snapshot(self) -> Snapshot:
        with self.lock:
            self.last_contact = monotonic()
            snapshot = Snapshot(self.info, self.frame, tuple(self.history), self.error)
        self.wake.set()
        return snapshot

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=4)

    def wanted(self) -> bool:
        with self.lock:
            return not self.stop.is_set() and monotonic() - self.last_contact < LEASE_SECONDS

    def publish(self, frame: Frame) -> None:
        with self.lock:
            self.frame = frame
            hand = frame.hands[0] if frame.tracking == 'tracking' and len(frame.hands) == 1 else None
            if hand is None or not hand.pointing or hand.grip != 'open' or monotonic() - frame.at >= FRAME_MAX_AGE:
                self.history.clear()
                return
            if self.history and (
                self.history[-1].hands[0].id != hand.id or frame.at - self.history[-1].at >= FRAME_MAX_AGE
            ):
                self.history.clear()
            self.history.append(frame)
            while frame.at - self.history[0].at > CALIBRATION_WINDOW:
                self.history.popleft()

    def run(self) -> None:
        while not self.stop.is_set():
            if not self.wanted():
                self.wake.wait()
                self.wake.clear()
                continue
            message = None
            try:
                self.capture()
            except CameraError as error:
                message = str(error)
            except Exception:
                logging.exception('D435i capture stopped')
                message = 'D435i tracking stopped. Check the USB cable and the camera bridge terminal.'
            finally:
                with self.lock:
                    self.info = None
                    self.frame = None
                    self.history.clear()
                    self.error = message
            if message is not None:
                self.stop.wait(2)

    def capture(self) -> None:
        try:
            import mediapipe as mp
            import numpy as np
            import pyrealsense2 as rs
        except ImportError as error:
            raise CameraError('Install the packages from camera/requirements.txt in the camera bridge Python environment.') from error

        if not self.model.is_file():
            raise CameraError('MediaPipe model file is missing. Set --model to hand_landmarker.task.')
        context = rs.context()
        devices = [
            device for device in context.query_devices()
            if 'D435I' in device.get_info(rs.camera_info.name).upper()
            and (self.serial is None or device.get_info(rs.camera_info.serial_number) == self.serial)
        ]
        if not devices:
            raise CameraError('No D435i detected. Connect it over USB 3 and check camera access.')
        if len(devices) > 1:
            raise CameraError('Multiple D435i cameras detected. Choose one with --serial.')

        serial = devices[0].get_info(rs.camera_info.serial_number)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
        )
        with ExitStack() as resources:
            try:
                landmarker = resources.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(options))
            except (RuntimeError, ValueError) as error:
                raise CameraError('Hand Landmarker could not load. Check the model file and Python packages.') from error
            if not self.wanted():
                return
            pipeline, profile, depth_width = open_streams(rs, context, serial)
            resources.callback(pipeline.stop)
            color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            # Modified Brown-Conrady color intrinsics cannot be deprojected.
            to_depth = color_intrinsics.model == rs.distortion.modified_brown_conrady
            align = rs.align(rs.stream.depth if to_depth else rs.stream.color)
            alignment = 'depth' if to_depth else 'color'
            with self.lock:
                self.info = CameraInfo(f'{serial}:{RGB_SIZE[0]}x{RGB_SIZE[1]}:{depth_width}:{FPS}:{alignment}', uuid4().hex)
                self.error = None

            self.track_frames(rs, np, mp, landmarker, pipeline, align)

    def track_frames(self, rs, np, mp, landmarker, pipeline, align) -> None:
        tracker = HandTracker()
        sequence = 0
        previous_pair = (-1, -1)
        previous_timestamp = -1
        last_pair_at = monotonic()
        while self.wanted():
            found, frames = pipeline.try_wait_for_frames(250)
            at = monotonic()
            if at - last_pair_at > 2:
                raise CameraError('D435i synchronized RGB and depth frames stopped. Check the USB connection.')
            if not found:
                continue
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            pair = (color.get_frame_number(), depth.get_frame_number())
            if pair[0] == previous_pair[0] or pair[1] == previous_pair[1]:
                continue
            aligned = align.process(frames)
            color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color or not depth:
                continue
            previous_pair = pair
            last_pair_at = at
            timestamp = max(previous_timestamp + 1, int(at * 1000))
            previous_timestamp = timestamp
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(color.get_data()))
            result = landmarker.detect_for_video(image, timestamp)
            world = result.hand_world_landmarks
            sides = result.handedness
            samples = [
                fuse_hand(rs, np, landmarks, world[index], sides[index] if index < len(sides) else [], depth)
                if index < len(world) else None
                for index, landmarks in enumerate(result.hand_landmarks)
            ]
            tracking, hands = tracker.update(samples, at)
            sequence += 1
            self.publish(Frame(sequence, at, tracking, hands))
