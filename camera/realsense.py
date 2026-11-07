"""Capture aligned D435i frames locally and expose only timestamped hand poses."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from math import dist, hypot, isfinite
from pathlib import Path
from statistics import median
from threading import Event, Lock, Thread
from time import monotonic
from uuid import uuid4
import logging

from calibration import MAX_DEPTH_M, MIN_DEPTH_M, Pose, Vec3, median_point, valid_point
from estimators import is_thumbs_up
from options import CameraOptions
from tracking import Candidate, Hand, HandTracker, Track, Tracking, Vec2, grip_features, valid_image

FRAME_MAX_AGE = 0.22
CALIBRATION_WINDOW = 0.7
LEASE_SECONDS = 6.0
PREVIEW_LEASE = 0.60
PREVIEW_INTERVAL = 1 / 6
RGB_SIZE = (640, 480)


class CameraError(RuntimeError):
    pass


def enable_depth_projector(rs, profile) -> None:
    sensor = profile.get_device().first_depth_sensor()
    if not sensor.supports(rs.option.emitter_enabled):
        raise CameraError('This camera cannot enable the depth projector.')
    sensor.set_option(rs.option.emitter_enabled, 1)


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
    calibration_pose: Pose | None = None
    detected_hands: int = 0
    processing_ms: float = 0.0
    diagnostics: dict | None = None


@dataclass(frozen=True)
class Snapshot:
    info: CameraInfo | None
    frame: Frame | None
    history: tuple[Frame, ...]
    error: str | None


@dataclass(frozen=True)
class Preview:
    at: float
    jpeg: bytes


@dataclass(frozen=True)
class PreviewMarker:
    point: Vec2
    valid: bool
    fist: bool = False


def preview_markers(landmarks, samples: list[Candidate | None], tracks: tuple[Track, ...]) -> tuple[PreviewMarker, ...]:
    markers = []
    for index, hand in enumerate(landmarks):
        if len(hand) <= 8:
            continue
        sample = samples[index] if index < len(samples) else None
        track = None
        if sample is not None:
            if sample.pose is not None:
                track = next((track for track in tracks if track.palm == sample.pose.palm), None)
            else:
                track = next((track for track in tracks if track.palm is None and track.image_palm == sample.image_palm), None)
        thumbs_up = track.hand.thumbs_up if track else sample is not None and sample.thumbs_up
        fist = track.hand.grip == 'fist' if track else sample is not None and sample.features.folded == 4 and not thumbs_up
        point = sample.image_palm if (fist or thumbs_up) and sample else (hand[8].x, hand[8].y)
        if point is not None and all(0 <= value <= 1 for value in point):
            markers.append(PreviewMarker(point, track is not None and track.palm is not None, fist))
    return tuple(markers)


def encode_preview(cv2, rgb, markers: tuple[PreviewMarker, ...]) -> bytes | None:
    height, width = rgb.shape[:2]
    scale = min(360 / width, 240 / height, 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    try:
        frame = cv2.cvtColor(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        for marker in markers:
            ring = (204, 240, 126) if marker.valid else (128, 202, 255)
            x, y = marker.point
            point = (round(x * (size[0] - 1)), round(y * (size[1] - 1)))
            if marker.fist:
                start = (point[0] - 8, point[1] - 8)
                end = (point[0] + 8, point[1] + 8)
                cv2.rectangle(frame, start, end, (25, 17, 12), 4, cv2.LINE_AA)
                cv2.rectangle(frame, start, end, ring, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, point, 9, (25, 17, 12), 4, cv2.LINE_AA)
                cv2.circle(frame, point, 8, ring, 2, cv2.LINE_AA)
            cv2.circle(frame, point, 2, ring, -1, cv2.LINE_AA)
        ok, image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        return image.tobytes() if ok else None
    except cv2.error:
        # A diagnostic frame must never interrupt hand tracking.
        return None


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


def locate_depth(
    rs, np, depth, intrinsics, units: float, x: float, y: float, near: float | None,
    sample: tuple[float, float] | None = None,
) -> Vec3 | None:
    if not (0 <= x < 1 and 0 <= y < 1):
        return None
    sample_x, sample_y = sample if sample is not None else (x, y)
    if not (0 <= sample_x < 1 and 0 <= sample_y < 1):
        return None
    height, width = depth.shape
    z = patch_depth(np, depth, int(sample_x * width), int(sample_y * height), units, near)
    if z is None:
        return None
    a, b, c = rs.rs2_deproject_pixel_to_point(intrinsics, [x * width, y * height], z)
    point = (float(a), float(b), float(c))
    return point if valid_point(point) else None


def fuse_hand(rs, np, image_landmarks, world_landmarks, handedness, aligned_depth, features_override=None) -> Candidate | None:
    if len(image_landmarks) != 21:
        return None
    features = features_override or grip_features([(point.x, point.y, point.z) for point in world_landmarks])
    if features is None:
        return None
    depth = np.asanyarray(aligned_depth.get_data())
    height, width = depth.shape
    wrist = image_landmarks[0]
    palm_bases = [image_landmarks[index] for index in (5, 9, 17)]
    palm_rays = [((wrist.x + base.x) / 2, (wrist.y + base.y) / 2) for base in palm_bases]
    image_palm = (median(x for x, _ in palm_rays), median(y for _, y in palm_rays))
    index = image_landmarks[8]
    side = None
    if handedness and handedness[0].score >= 0.65:
        side = handedness[0].category_name
    # Use image pixels for the finger closure ratio, so image aspect ratio
    # cannot change which thumb/index gap counts as a pinch.
    optical_points = [(image_landmarks[i].x * width, image_landmarks[i].y * height) for i in (4, 8, 5, 17)]
    optical_ratio = None
    if all(0 <= image_landmarks[i].x < 1 and 0 <= image_landmarks[i].y < 1 for i in (4, 8, 5, 17)):
        thumb, finger, index_base, pinky_base = optical_points
        palm_width = hypot(index_base[0] - pinky_base[0], index_base[1] - pinky_base[1])
        if palm_width >= 8:
            ratio = hypot(thumb[0] - finger[0], thumb[1] - finger[1]) / palm_width
            optical_ratio = ratio if isfinite(ratio) else None
    sample = Candidate(
        None, side, features, None, False,
        image_index=(index.x, index.y),
        image_palm=image_palm,
        optical_pinch_ratio=optical_ratio,
    )
    visible_palm = (valid_image((wrist.x, wrist.y)) and valid_image(image_palm)
                    and sum(valid_image((base.x, base.y)) for base in palm_bases) >= 2)
    optical_sample = sample if visible_palm else None

    intrinsics = aligned_depth.profile.as_video_stream_profile().intrinsics
    units = aligned_depth.get_units()

    def locate(x: float, y: float, near: float | None = None) -> Vec3 | None:
        return locate_depth(rs, np, depth, intrinsics, units, x, y, near)

    palm_points = [point for x, y in palm_rays if (point := locate(x, y)) is not None]
    if len(palm_points) < 2:
        return optical_sample
    palm = median_point(palm_points)
    if sum(abs(point[2] - palm[2]) <= 0.055 for point in palm_points) < 2:
        return optical_sample

    def fingertip(tip_index: int, joint_index: int) -> Vec3 | None:
        target = image_landmarks[tip_index]
        point = locate(target.x, target.y, palm[2])
        if point is None and features.folded < 4:
            # Stereo depth is sparse at a fingertip's outline. Use measured
            # depth inside its distal segment on the same fingertip color ray.
            joint = image_landmarks[joint_index]
            inner = ((target.x + joint.x) / 2, (target.y + joint.y) / 2)
            point = locate_depth(rs, np, depth, intrinsics, units, target.x, target.y, palm[2], inner)
        return point

    tip = fingertip(8, 7)
    thumb_tip = fingertip(4, 3) if tip is not None else None
    gap = dist(thumb_tip, tip) if tip is not None and thumb_tip is not None else None
    return replace(
        sample, pose=Pose(tip or palm, palm), pinch_gap=gap, pointing=tip is not None,
        thumbs_up=is_thumbs_up(image_landmarks, width, height, features),
    )


def open_streams(rs, context, serial: str, options: CameraOptions = CameraOptions()):
    for width in (848, 640):
        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, 480, rs.format.z16, options.fps)
        config.enable_stream(rs.stream.color, *RGB_SIZE, rs.format.rgb8, options.fps)
        try:
            profile = pipeline.start(config)
        except RuntimeError as error:
            logging.warning('D435i %dx480 depth stream could not start: %s', width, error)
            continue
        return pipeline, profile, width
    raise CameraError('Could not open the selected D435i image and depth streams. Check the camera connection or choose another frame rate.')


class CameraFeed:
    def __init__(self, model: Path, serial: str | None = None, options: CameraOptions = CameraOptions()) -> None:
        self.model = model
        self.serial = serial
        self.lock = Lock()
        self.stop = Event()
        self.wake = Event()
        self.last_contact = float('-inf')
        self.info: CameraInfo | None = None
        self.frame: Frame | None = None
        self.history: deque[Frame] = deque(maxlen=24)
        self.preview: Preview | None = None
        self.preview_clients: dict[str, float] = {}
        self.preview_until = 0.0
        self.next_preview_at = 0.0
        self.error: str | None = None
        self.options = options
        self.active_options: CameraOptions | None = None
        self.revision = 0
        self.frame_times: deque[float] = deque(maxlen=60)
        self.processing_ms = 0.0
        self.thread = Thread(target=self.run, name='D435i capture', daemon=True)
        self.thread.start()

    def configure(self, options: CameraOptions) -> None:
        with self.lock:
            if options == self.options and self.error is None:
                return
            self.options = options
            self.revision += 1
            self.info = None
            self.frame = None
            self.preview = None
            self.history.clear()
            self.frame_times.clear()
            self.processing_ms = 0.0
            self.error = None
            self.last_contact = monotonic()
        self.wake.set()

    def settings(self) -> dict:
        from estimators import tracker_choices
        with self.lock:
            now = monotonic()
            frame = self.frame
            span = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0
            diagnostics = {
                'poseFps': (len(self.frame_times) - 1) / span if span > 0 and frame and now - frame.at < FRAME_MAX_AGE else 0,
                'processingMs': self.processing_ms,
                'frameAgeMs': max(0, (now - frame.at) * 1000) if frame else None,
                'estimatedHands': frame.detected_hands if frame else 0,
                'depthHands': frame.diagnostics.get('depthHands', 0) if frame and frame.diagnostics else 0,
                'trackedHands': len(frame.hands) if frame else 0,
                'model': frame.diagnostics.get('model', {}) if frame and frame.diagnostics else {},
            }
            return {
                **self.options.to_dict(),
                'sources': [{'value': 'rgb', 'label': 'RGB'}],
                'trackers': tracker_choices(self.model), 'frameRates': [30, 60],
                'applying': self.error is None and (self.active_options != self.options or self.info is None or frame is None),
                'error': self.error, 'diagnostics': diagnostics,
            }

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

    def request_preview(self, session: str, after: float, client: str = 'calibration') -> bytes | None:
        with self.lock:
            if self.info is None or self.info.session != session:
                return None
            now = monotonic()
            self.preview_clients[client] = now + PREVIEW_LEASE
            self.update_preview_expiry(now)
            preview = self.preview
            if preview is None or preview.at <= after or now - preview.at >= PREVIEW_LEASE:
                return None
            return preview.jpeg

    def update_preview_expiry(self, now: float) -> None:
        self.preview_clients = {client: until for client, until in self.preview_clients.items() if until > now}
        self.preview_until = max(self.preview_clients.values(), default=0.0)
        if not self.preview_clients:
            self.preview = None

    def end_preview(self, client: str) -> None:
        with self.lock:
            self.preview_clients.pop(client, None)
            self.update_preview_expiry(monotonic())

    def preview_due(self, at: float) -> bool:
        with self.lock:
            self.update_preview_expiry(monotonic())
            if monotonic() >= self.preview_until:
                self.preview = None
                return False
            if at < self.next_preview_at:
                return False
            self.next_preview_at = at + PREVIEW_INTERVAL
            return True

    def publish_preview(self, at: float, jpeg: bytes) -> None:
        with self.lock:
            if monotonic() < self.preview_until:
                self.preview = Preview(at, jpeg)

    def wanted(self, revision: int | None = None) -> bool:
        with self.lock:
            return (not self.stop.is_set() and monotonic() - self.last_contact < LEASE_SECONDS
                    and (revision is None or revision == self.revision))

    def publish(self, frame: Frame, revision: int | None = None) -> None:
        with self.lock:
            if revision is not None and revision != self.revision:
                return
            self.frame = frame
            self.frame_times.append(frame.at)
            self.processing_ms += (frame.processing_ms - self.processing_ms) * 0.2
            if frame.detected_hands > 1:
                self.history.clear()
                return
            while self.history and frame.at - self.history[0].at > CALIBRATION_WINDOW:
                self.history.popleft()
            if frame.calibration_pose is not None and monotonic() - frame.at < FRAME_MAX_AGE:
                self.history.append(frame)

    def run(self) -> None:
        while not self.stop.is_set():
            if not self.wanted():
                self.wake.wait()
                self.wake.clear()
                continue
            message = None
            with self.lock:
                revision = self.revision
            try:
                self.capture(revision)
            except CameraError as error:
                message = str(error)
            except Exception:
                logging.exception('D435i capture stopped')
                message = 'D435i tracking stopped. Check the USB cable and the camera bridge terminal.'
            finally:
                with self.lock:
                    if revision != self.revision:
                        message = None
                    self.info = None
                    self.active_options = None
                    self.frame = None
                    self.frame_times.clear()
                    self.processing_ms = 0.0
                    self.history.clear()
                    self.preview = None
                    self.preview_clients.clear()
                    self.preview_until = 0.0
                    self.next_preview_at = 0.0
                    self.error = message
            if message is not None:
                self.stop.wait(2)

    def capture(self, revision: int) -> None:
        try:
            import numpy as np
            import pyrealsense2 as rs
        except ImportError as error:
            raise CameraError('Install the packages from camera/requirements.txt in the camera bridge Python environment.') from error
        try:
            import cv2
        except ImportError:
            cv2 = None
            logging.warning('Camera preview requires OpenCV, installed with MediaPipe.')

        from estimators import create_estimator
        with self.lock:
            options = self.options
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
        with ExitStack() as resources:
            try:
                estimator = resources.enter_context(create_estimator(options.tracker, self.model))
            except (RuntimeError, ValueError) as error:
                raise CameraError(str(error)) from error
            if not self.wanted(revision):
                return
            pipeline, profile, depth_width = open_streams(rs, context, serial, options)
            resources.callback(pipeline.stop)
            enable_depth_projector(rs, profile)
            color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            # Modified Brown-Conrady color intrinsics cannot be deprojected.
            to_depth = color_intrinsics.model == rs.distortion.modified_brown_conrady
            align = rs.align(rs.stream.depth if to_depth else rs.stream.color)
            alignment = 'depth' if to_depth else 'color'
            with self.lock:
                if revision != self.revision:
                    return
                profile_key = f'{serial}:{RGB_SIZE[0]}x{RGB_SIZE[1]}:{depth_width}:{options.fps}:{alignment}'
                self.info = CameraInfo(profile_key, uuid4().hex)
                self.active_options = options
                self.error = None

            self.track_frames(rs, np, estimator, pipeline, align, options, revision, cv2)

    def track_frames(self, rs, np, estimator, pipeline, align, options, revision, cv2=None) -> None:
        tracker = HandTracker()
        sequence = 0
        previous_pair = (-1, -1)
        previous_timestamp = -1
        last_pair_at = monotonic()
        while self.wanted(revision):
            found, frames = pipeline.try_wait_for_frames(250)
            at = monotonic()
            if at - last_pair_at > 2:
                raise CameraError('D435i synchronized RGB and depth frames stopped. Check the USB connection.')
            if not found:
                continue
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue
            pair = (color.get_frame_number(), depth.get_frame_number())
            if pair[0] == previous_pair[0] or pair[1] == previous_pair[1]:
                continue
            if align is not None:
                aligned = align.process(frames)
                color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color or not depth:
                continue
            previous_pair = pair
            last_pair_at = at
            timestamp = max(previous_timestamp + 1, int(at * 1000))
            previous_timestamp = timestamp
            rgb = np.ascontiguousarray(color.get_data())
            detections = estimator.detect(rgb, timestamp)
            samples = [
                fuse_hand(rs, np, hand.landmarks, [], hand.handedness, depth, hand.features)
                for hand in detections
            ]
            if not self.wanted(revision):
                return
            tracking, hands = tracker.update(samples, at)
            # Calibration observes a single depth-valid pointing hand even
            # when the live gesture classifier briefly reads a grip.
            sample = samples[0] if len(samples) == 1 else None
            calibration_pose = sample.pose if sample and sample.pointing and sample.features.folded < 4 else None
            sequence += 1
            diagnostics = {
                'depthHands': sum(sample is not None and sample.pose is not None for sample in samples),
                'model': dict(getattr(estimator, 'diagnostics', {})),
            }
            self.publish(Frame(
                sequence, at, tracking, hands, calibration_pose, len(samples),
                (monotonic() - at) * 1000, diagnostics,
            ), revision)
            if cv2 is not None and self.preview_due(at):
                current_tracks = tuple(track for track in tracker.tracks if track.published and track.observed_at == at)
                markers = preview_markers([hand.landmarks for hand in detections], samples, current_tracks)
                jpeg = encode_preview(cv2, rgb, markers)
                if jpeg is not None:
                    self.publish_preview(at, jpeg)
