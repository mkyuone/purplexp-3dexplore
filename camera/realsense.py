"""Capture aligned D435i frames and provide the shared hand tracking pipeline."""

from collections import deque
from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from math import dist, hypot, isfinite
from pathlib import Path
from statistics import median
from threading import Condition, Event, Lock, Thread
from time import monotonic, time
from uuid import uuid4
import logging

from calibration import MAX_DEPTH_M, MIN_DEPTH_M, Pose, Vec3, median_point, valid_point
from diagnostics import DiagnosticHand, diagnostic_hands, encode_diagnostic_rgb, hand_overlay, matching_track
from estimators import is_thumbs_up
from infrared_preview import INFRARED_INDEXES, InfraredEncoder, InfraredPreview
from options import CameraOptions
from tracking import (
    OPTICAL_WORLD_START_RATIO, PINCH_DISAGREEMENT_GAP_M, PINCH_START_GAP_M,
    Candidate, Hand, HandTracker, Track, Tracking, Vec2,
    grip_features, valid_image,
)

FRAME_MAX_AGE = 0.22
CALIBRATION_WINDOW = 0.7
LEASE_SECONDS = 6.0
PREVIEW_LEASE = 0.60
PREVIEW_INTERVAL = 1 / 6
THUMB_DEPTH_LEASE = 2.0
RGB_SIZE = (640, 480)
DEPTH_QUEUE_SIZE = 4
STREAM_TIMEOUT = 2.0
DEPTH_RETRY_DELAY = 2.0
FINGERTIP_DEPTH_DISAGREEMENT_M = 0.035
FINGERTIP_DEPTH_CORROBORATION_M = 0.012
FINGERTIP_GAP_CHECK_OPTICAL_RATIO = 0.28
PALM_LENGTH_SCALE = 0.65
ImageProjector = Callable[[float, float, float], Vec3 | None]


class CameraError(RuntimeError):
    pass


def enable_depth_projector(rs, sensor) -> None:
    if not sensor.supports(rs.option.emitter_enabled):
        raise CameraError('This camera cannot enable the depth projector.')
    sensor.set_option(rs.option.emitter_enabled, 1)


@dataclass(frozen=True)
class CameraInfo:
    profile: str | None
    session: str
    project_image: ImageProjector | None = None
    serial: str | None = None


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
    thumb_depth_m: float | None = None
    hand_diagnostics: tuple[DiagnosticHand, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    info: CameraInfo | None
    frame: Frame | None
    history: tuple[Frame, ...]
    error: str | None
    gesture_history: tuple[Frame, ...] = ()


@dataclass(frozen=True)
class Preview:
    at: float
    jpeg: bytes


@dataclass(frozen=True)
class PreviewMarker:
    point: Vec2
    valid: bool
    fist: bool = False


def preview_markers(
    landmarks, samples: list[Candidate | None], tracks: tuple[Track, ...], *, mark_thumb: bool = False,
) -> tuple[PreviewMarker, ...]:
    markers = []
    for index, hand in enumerate(landmarks):
        if len(hand) <= 8:
            continue
        sample = samples[index] if index < len(samples) else None
        if mark_thumb:
            point = (hand[4].x, hand[4].y)
            if valid_image(point):
                markers.append(PreviewMarker(point, sample is not None and sample.thumb_depth_m is not None))
            continue
        track = matching_track(sample, tracks)
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
    sample: tuple[float, float] | None = None, *, deproject=None,
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
    if deproject is not None:
        return deproject(x * width, y * height, z)
    a, b, c = rs.rs2_deproject_pixel_to_point(intrinsics, [x * width, y * height], z)
    point = (float(a), float(b), float(c))
    return point if valid_point(point) else None


def fuse_hand(
    rs, np, image_landmarks, world_landmarks, handedness, aligned_depth,
    features_override=None, *, image_size=RGB_SIZE, deproject=None, measure_thumb_depth=False,
) -> Candidate | None:
    if len(image_landmarks) != 21:
        return None
    features = features_override or grip_features([(point.x, point.y, point.z) for point in world_landmarks])
    if features is None:
        return None
    depth = np.asanyarray(aligned_depth.get_data()) if aligned_depth else None
    width, height = image_size if depth is None else (depth.shape[1], depth.shape[0])
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
        palm_scale = hypot(index_base[0] - pinky_base[0], index_base[1] - pinky_base[1])
        middle_base = image_landmarks[9]
        if valid_image((wrist.x, wrist.y)) and valid_image((middle_base.x, middle_base.y)):
            # The palm's width can nearly vanish when the hand turns sideways.
            palm_length = hypot((middle_base.x - wrist.x) * width, (middle_base.y - wrist.y) * height)
            palm_scale = max(palm_scale, palm_length * PALM_LENGTH_SCALE)
        if palm_scale >= 8:
            ratio = hypot(thumb[0] - finger[0], thumb[1] - finger[1]) / palm_scale
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
    if depth is None:
        return optical_sample

    intrinsics = aligned_depth.profile.as_video_stream_profile().intrinsics if deproject is None else None
    units = aligned_depth.get_units()
    projection = {} if deproject is None else {'deproject': deproject}

    def locate(x: float, y: float, near: float | None = None) -> Vec3 | None:
        return locate_depth(rs, np, depth, intrinsics, units, x, y, near, **projection)

    palm_points = [point for x, y in palm_rays if (point := locate(x, y)) is not None]
    if len(palm_points) < 2:
        return optical_sample
    palm = median_point(palm_points)
    if sum(abs(point[2] - palm[2]) <= 0.055 for point in palm_points) < 2:
        return optical_sample

    def fingertip_pad(tip_index: int, joint_index: int) -> Vec3 | None:
        target, joint = image_landmarks[tip_index], image_landmarks[joint_index]
        inner = ((target.x + joint.x) / 2, (target.y + joint.y) / 2)
        return locate_depth(rs, np, depth, intrinsics, units, target.x, target.y, palm[2], inner, **projection)

    def fingertip(tip_index: int, joint_index: int) -> Vec3 | None:
        target = image_landmarks[tip_index]
        point = locate(target.x, target.y, palm[2])
        if point is None and features.folded < 4:
            # Stereo depth is sparse at a fingertip's outline. Use measured
            # depth inside its distal segment on the same fingertip color ray.
            return fingertip_pad(tip_index, joint_index)
        return point

    tip = fingertip(8, 7)
    thumb_tip = fingertip(4, 3) if tip is not None else None
    thumb_pad = None
    gap = dist(thumb_tip, tip) if tip is not None and thumb_tip is not None else None
    gap_corroborated = False
    if (gap is not None and gap > PINCH_START_GAP_M and features.loosely_folded < 4
            and optical_ratio is not None and optical_ratio <= FINGERTIP_GAP_CHECK_OPTICAL_RATIO
            and features.pinch_ratio <= OPTICAL_WORLD_START_RATIO):
        # Verify a gap that conflicts with a close image pose against samples
        # inside the fingers, where the patch is less likely to include background.
        index_pad = fingertip_pad(8, 7)
        thumb_pad = fingertip_pad(4, 3)
        if gap <= PINCH_DISAGREEMENT_GAP_M and index_pad is not None and thumb_pad is not None:
            gap_corroborated = (abs(tip[2] - index_pad[2]) <= FINGERTIP_DEPTH_CORROBORATION_M
                                and abs(thumb_tip[2] - thumb_pad[2]) <= FINGERTIP_DEPTH_CORROBORATION_M
                                and dist(index_pad, thumb_pad) > PINCH_START_GAP_M)
        if index_pad is not None and abs(tip[2] - index_pad[2]) > FINGERTIP_DEPTH_DISAGREEMENT_M:
            tip, gap = index_pad, None
        if thumb_pad is not None and abs(thumb_tip[2] - thumb_pad[2]) > FINGERTIP_DEPTH_DISAGREEMENT_M:
            gap = None
    thumbs_up = is_thumbs_up(image_landmarks, width, height, features)
    thumb_depth_m = None
    if measure_thumb_depth and (features.folded < 4 or thumbs_up):
        target = image_landmarks[4]
        # The index can be completely hidden as the other fingers curl.
        # Calibration needs a separate stereo sample at the extended thumb.
        measured_thumb = thumb_tip if tip is not None else locate(target.x, target.y, palm[2])
        thumb_pad = thumb_pad or fingertip_pad(4, 3)
        # A patch on the tip can straddle the background. Prefer the depth
        # inside the thumb's distal segment when it contradicts that patch.
        if thumb_pad is not None and (measured_thumb is None or
                abs(measured_thumb[2] - thumb_pad[2]) > FINGERTIP_DEPTH_DISAGREEMENT_M):
            measured_thumb = thumb_pad
        if measured_thumb is not None and valid_point(measured_thumb):
            thumb_depth_m = measured_thumb[2]
    return replace(
        sample, pose=Pose(tip or palm, palm), pinch_gap=gap, pointing=tip is not None,
        thumbs_up=thumbs_up, thumb_depth_m=thumb_depth_m, pinch_gap_corroborated=gap_corroborated,
    )


def video_profile(sensor, stream, fmt, size, fps, *, index=None):
    for profile in sensor.get_stream_profiles():
        if (profile.stream_type() == stream and profile.format() == fmt and profile.fps() == fps
                and (index is None or profile.stream_index() == index)):
            video = profile.as_video_stream_profile()
            if (video.width(), video.height()) == size:
                return video
    raise CameraError(f'D435i has no {stream} profile at {size[0]}×{size[1]} and {fps} fps.')


def close_sensor(action) -> None:
    try:
        action()
    except RuntimeError as error:
        # A disconnected sensor may already be stopped or closed. Still close
        # the other sensor and keep the original capture error available.
        logging.debug('D435i sensor cleanup: %s', error)


def start_sensor(rs, sensor, profile, receiver, resources: ExitStack) -> None:
    clock = rs.option.global_time_enabled
    try:
        if sensor.supports(clock):
            sensor.set_option(clock, 1)
    except RuntimeError as error:
        logging.debug('D435i global time unavailable: %s', error)
    sensor.open(profile)
    resources.callback(close_sensor, sensor.close)
    sensor.start(receiver)
    resources.callback(close_sensor, sensor.stop)


@contextmanager
def open_color(rs, device, fps: int):
    color = device.first_color_sensor()
    profile = video_profile(color, rs.stream.color, rs.format.rgb8, RGB_SIZE, fps)
    # Independent queues keep RGB flowing through a depth outage. The single
    # RGB slot discards superseded frames while inference is busy.
    queue = rs.frame_queue(1)
    with ExitStack() as resources:
        start_sensor(rs, color, profile, queue, resources)
        yield profile, queue


@contextmanager
def open_depth(rs, device, fps: int, widths, *, infrared=None):
    depth = device.first_depth_sensor()
    errors = []
    with ExitStack() as resources:
        for width in widths:
            try:
                size = (width, 480)
                profile = video_profile(depth, rs.stream.depth, rs.format.z16, size, fps)
                queue = rs.frame_queue(DEPTH_QUEUE_SIZE)
                profiles, receiver = profile, queue
                if infrared is not None:
                    profiles = [profile, *(
                        video_profile(depth, rs.stream.infrared, rs.format.y8, size, fps, index=index)
                        for index in INFRARED_INDEXES
                    )]

                    def receiver(frame, depth_queue=queue):
                        try:
                            frame_profile = frame.get_profile()
                            stream = frame_profile.stream_type()
                            if stream == rs.stream.depth:
                                depth_queue.enqueue(frame)
                            elif stream == rs.stream.infrared:
                                infrared.submit(frame_profile.stream_index(), frame)
                        except RuntimeError:
                            # SDK buffers may become invalid as a sensor disconnects.
                            logging.debug('D435i stereo callback dropped a frame', exc_info=True)
                with ExitStack() as attempt:
                    start_sensor(rs, depth, profiles, receiver, attempt)
                    enable_depth_projector(rs, depth)
                    resources.enter_context(attempt.pop_all())
                break
            except RuntimeError as error:
                errors.append(str(error))
        else:
            raise CameraError('Could not start D435i depth. ' + ' / '.join(errors))
        yield profile, queue


@dataclass(frozen=True)
class CameraSample:
    frame: object
    number: int
    timestamp: float
    domain: object
    arrival: float | None
    captured_at: float | None = None

    @classmethod
    def read(cls, rs, frame):
        field = rs.frame_metadata_value.time_of_arrival
        arrival = frame.get_frame_metadata(field) if frame.supports_frame_metadata(field) else None
        return cls(frame, frame.get_frame_number(), frame.get_timestamp(), frame.get_frame_timestamp_domain(), arrival)

    def distance(self, other) -> float:
        if self.domain == other.domain:
            distance = abs(self.timestamp - other.timestamp)
        elif self.arrival is not None and other.arrival is not None:
            distance = abs(self.arrival - other.arrival)
        else:
            return float('inf')
        return distance if isfinite(distance) else float('inf')


class DepthMatcher:
    def __init__(self, rs, queue, fps: int) -> None:
        self.rs = rs
        self.queue = queue
        self.window = 500 / fps
        self.pending: deque[CameraSample] = deque(maxlen=DEPTH_QUEUE_SIZE)
        self.last_number = None
        self.last_seen = monotonic()

    def match(self, color: CameraSample):
        for _ in range(DEPTH_QUEUE_SIZE):
            frame = self.queue.poll_for_frame()
            if not frame:
                break
            sample = CameraSample.read(self.rs, frame)
            if sample.number != self.last_number:
                self.pending.append(sample)
                self.last_number = sample.number
                self.last_seen = monotonic()
        if not self.pending:
            return None
        index = min(range(len(self.pending)), key=lambda index: color.distance(self.pending[index]))
        depth = self.pending[index]
        if color.distance(depth) >= self.window:
            return None
        for _ in range(index + 1):
            self.pending.popleft()
        return depth.frame


class DepthStream:
    def __init__(self, rs, device, color, fps: int, on_profile, *, infrared=None, cv2=None, np=None) -> None:
        self.rs = rs
        self.device = device
        self.color = color
        self.fps = fps
        self.on_profile = on_profile
        self.infrared = infrared
        self.cv2 = cv2
        self.np = np
        self.stop = Event()
        self.lock = Lock()
        self.source = None
        self.thread = Thread(target=self.run, name='D435i depth', daemon=True)
        self.thread.start()

    def run(self) -> None:
        widths = (848, 640)
        last_error = None
        infrared_failed = False

        def infrared_requested() -> bool:
            nonlocal infrared_failed
            if self.infrared is None or not self.infrared.wanted():
                infrared_failed = False
                return False
            if self.cv2 is None:
                self.infrared.fail('IR previews require OpenCV in the camera bridge environment.')
                return False
            return not infrared_failed

        while not self.stop.is_set():
            with_infrared = infrared_requested()
            try:
                with ExitStack() as resources:
                    if with_infrared:
                        encoder = InfraredEncoder(self.cv2, self.np, self.infrared.publish)
                        resources.callback(self.infrared.clear_frames)
                        resources.callback(encoder.close)
                        depth = open_depth(self.rs, self.device, self.fps, widths, infrared=encoder)
                    else:
                        depth = open_depth(self.rs, self.device, self.fps, widths)
                    profile, queue = resources.enter_context(depth)
                    if self.stop.is_set():
                        return
                    width = profile.width()
                    modified = self.color.intrinsics.model == self.rs.distortion.modified_brown_conrady
                    projector = depth_space_projector(self.rs, self.color, profile) if modified else None
                    self.on_profile(width, image_projector(self.rs, self.color.intrinsics, projector))
                    matcher = DepthMatcher(self.rs, queue, self.fps)
                    started_at = monotonic()
                    with self.lock:
                        self.source = (matcher, projector)
                    try:
                        while not self.stop.wait(.25):
                            now = monotonic()
                            if infrared_requested() != with_infrared:
                                break
                            if now - matcher.last_seen > STREAM_TIMEOUT:
                                if len(widths) > 1:
                                    widths = tuple(candidate for candidate in (640, 848) if candidate != width) + (width,)
                                raise CameraError('D435i depth frames stopped.')
                            if (with_infrared and now - started_at > STREAM_TIMEOUT
                                    and not self.infrared.fresh_pair(STREAM_TIMEOUT)):
                                raise CameraError('D435i infrared frames stopped.')
                            if matcher.last_number is not None:
                                # Once the stream works, retain the same profile across retries.
                                widths = (width,)
                                last_error = None
                    finally:
                        with self.lock:
                            self.source = None
            except RuntimeError as error:
                if with_infrared and not self.stop.is_set():
                    infrared_failed = True
                    self.infrared.fail('The D435i could not stream both IR previews with the current depth mode.')
                    logging.warning('%s Restoring depth without infrared previews.', error)
                    continue
                if not self.stop.is_set() and str(error) != last_error:
                    last_error = str(error)
                    logging.warning('%s Retrying depth while RGB tracking continues.', error)
                self.stop.wait(DEPTH_RETRY_DELAY)

    def match(self, color: CameraSample):
        with self.lock:
            source = self.source
        if source is None:
            return None, None
        matcher, projector = source
        return matcher.match(color), projector

    def close(self) -> None:
        self.stop.set()
        self.thread.join()


def depth_aligner(rs):
    queue = rs.frame_queue(1)
    pending_depth = []

    def combine(color, source):
        source.frame_ready(source.allocate_composite_frame([color, pending_depth[0]]))

    combiner = rs.processing_block(combine)
    combiner.start(queue)
    align = rs.align(rs.stream.color)

    def aligned_depth(color, depth):
        pending_depth.append(depth)
        try:
            combiner.invoke(color)
        finally:
            pending_depth.clear()
        frames = queue.poll_for_frame()
        return align.process(frames.as_frameset()).get_depth_frame() if frames else None

    return aligned_depth


def depth_space_projector(rs, color, depth):
    intrinsics = color.intrinsics
    extrinsics = color.get_extrinsics_to(depth)
    origin = extrinsics.translation

    def deproject(pixel_x: float, pixel_y: float, depth_z: float) -> Vec3 | None:
        # The SDK cannot deproject Modified Brown-Conrady RGB directly. Invert
        # its forward projection, then solve the color ray at the measured
        # depth-camera Z. Keep physical points in the original calibration
        # coordinate space without warping the RGB image used for gestures.
        x = (pixel_x - intrinsics.ppx) / intrinsics.fx
        y = (pixel_y - intrinsics.ppy) / intrinsics.fy
        for _ in range(8):
            px, py = rs.rs2_project_point_to_pixel(intrinsics, [x, y, 1])
            dx, dy = pixel_x - px, pixel_y - py
            if max(abs(dx), abs(dy)) < 0.25:
                endpoint = rs.rs2_transform_point_to_point(extrinsics, [x, y, 1])
                ray = tuple(b - a for a, b in zip(origin, endpoint))
                if ray[2] <= 0:
                    return None
                scale = (depth_z - origin[2]) / ray[2]
                point = tuple(a + b * scale for a, b in zip(origin, ray))
                return point if valid_point(point) else None
            x += dx / intrinsics.fx
            y += dy / intrinsics.fy
        return None

    return deproject


def image_projector(rs, intrinsics, depth_space=None) -> ImageProjector:
    def project(x: float, y: float, z: float) -> Vec3 | None:
        if not valid_image((x, y)) or not MIN_DEPTH_M <= z <= MAX_DEPTH_M:
            return None
        pixel_x, pixel_y = x * intrinsics.width, y * intrinsics.height
        if depth_space is not None:
            return depth_space(pixel_x, pixel_y, z)
        a, b, c = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], z)
        point = (float(a), float(b), float(c))
        return point if valid_point(point) else None

    return project


class PreviewEncoder:
    def __init__(self, cv2, publish, *, encode=None) -> None:
        self.cv2 = cv2
        self.publish = publish
        self.encode = encode or encode_preview
        self.ready = Condition()
        self.pending = None
        self.closed = False
        self.thread = Thread(target=self.run, name='D435i preview', daemon=True)
        self.thread.start()

    def submit(self, at: float, rgb, markers: tuple) -> None:
        # Detach the preview from the SDK buffer so slow encoding cannot hold
        # capture frames. Only the latest queued preview is retained.
        copy = rgb.copy()
        with self.ready:
            self.pending = (at, copy, markers)
            self.ready.notify()

    def run(self) -> None:
        while True:
            with self.ready:
                self.ready.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                at, rgb, markers = self.pending
                self.pending = None
            try:
                jpeg = self.encode(self.cv2, rgb, markers)
            except Exception:
                logging.exception('D435i preview could not encode a frame')
                continue
            if jpeg is not None:
                self.publish(at, jpeg)

    def close(self) -> None:
        with self.ready:
            self.closed = True
            self.pending = None
            self.ready.notify()
        self.thread.join(timeout=1)


class CameraFeed:
    camera_name = 'RealSense D435i'
    frame_rates = (30, 60)
    infrared_count = len(INFRARED_INDEXES)

    def __init__(self, model: Path, serial: str | None = None, options: CameraOptions = CameraOptions()) -> None:
        self.model = model
        self.serial = serial
        self.lock = Lock()
        self.changed = Condition(self.lock)
        self.stop = Event()
        self.wake = Event()
        self.last_contact = float('-inf')
        self.info: CameraInfo | None = None
        self.frame: Frame | None = None
        self.infrared = InfraredPreview()
        self.history: deque[Frame] = deque(maxlen=24)
        self.gesture_history: deque[Frame] = deque(maxlen=48)
        self.preview: Preview | None = None
        self.preview_clients: dict[str, float] = {}
        self.preview_until = 0.0
        self.next_preview_at = 0.0
        self.hand_diagnostics_until = 0.0
        self.diagnostic_rgb: Preview | None = None
        self.diagnostic_rgb_until = 0.0
        self.next_diagnostic_rgb_at = 0.0
        self.thumb_depth_until = 0.0
        self.error: str | None = None
        self.depth_error: str | None = None
        self.options = options
        self.active_options: CameraOptions | None = None
        self.revision = 0
        self.frame_times: deque[float] = deque(maxlen=60)
        self.processing_times: deque[float] = deque(maxlen=60)
        self.processing_ms = 0.0
        self.thread = Thread(target=self.run, name=f'{self.camera_name} capture', daemon=True)
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
            self.diagnostic_rgb = None
            self.next_diagnostic_rgb_at = 0.0
            self.thumb_depth_until = 0.0
            self.history.clear()
            self.gesture_history.clear()
            self.frame_times.clear()
            self.processing_times.clear()
            self.processing_ms = 0.0
            self.error = None
            self.depth_error = None
            self.last_contact = monotonic()
            self.changed.notify_all()
        self.wake.set()

    def settings(self) -> dict:
        from estimators import tracker_choices
        with self.lock:
            now = monotonic()
            frame = self.frame
            last = (frame.diagnostics or {}) if frame else {}
            span = self.frame_times[-1] - self.frame_times[0] if len(self.frame_times) > 1 else 0
            processing = sorted(self.processing_times)
            diagnostics = {
                'poseFps': (len(self.frame_times) - 1) / span if span > 0 and frame and now - frame.at < FRAME_MAX_AGE else 0,
                'processingMs': self.processing_ms,
                'processingP95Ms': processing[(len(processing) * 95 + 99) // 100 - 1] if processing else None,
                'frameAgeMs': max(0, (now - frame.at) * 1000) if frame else None,
                'estimatedHands': frame.detected_hands if frame else 0,
                'depthHands': last.get('depthHands', 0),
                'trackedHands': len(frame.hands) if frame else 0,
                'model': last.get('model', {}),
            }
            for key in ('queueMs', 'inferenceMs', 'alignmentMs', 'fusionMs',
                        'captureDroppedFrames', 'depthFrameAvailable', 'frameAgeSource'):
                if key in last:
                    diagnostics[key] = last[key]
            return {
                **self.options.to_dict(),
                'cameraName': self.camera_name,
                'sources': [{'value': 'rgb', 'label': 'RGB'}],
                'trackers': tracker_choices(self.model), 'frameRates': list(self.frame_rates),
                'applying': self.error is None and (self.active_options != self.options or self.info is None or frame is None),
                'error': self.error, 'depthError': self.depth_error, 'diagnostics': diagnostics,
            }

    def snapshot(self, *, include_gestures: bool = False, watch_hand_diagnostics: bool = False) -> Snapshot:
        with self.lock:
            self.last_contact = monotonic()
            if watch_hand_diagnostics:
                self.hand_diagnostics_until = self.last_contact + PREVIEW_LEASE
            gestures = tuple(self.gesture_history) if include_gestures else ()
            snapshot = Snapshot(self.info, self.frame, tuple(self.history), self.error, gestures)
        self.wake.set()
        return snapshot

    def wait_for_pose(self, session: str, after: int, timeout: float) -> None:
        with self.changed:
            self.changed.wait_for(
                lambda: self.stop.is_set() or self.info is None or self.info.session != session
                or self.frame is None or self.frame.sequence > after,
                timeout=timeout,
            )

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        with self.changed:
            self.changed.notify_all()
        self.thread.join(timeout=4)

    def request_infrared(self, index: int) -> bytes | None:
        if not 1 <= index <= self.infrared_count:
            return None
        image = self.infrared.request(index)
        with self.lock:
            self.last_contact = monotonic()
        self.wake.set()
        return image

    def infrared_status(self) -> str | None:
        return self.infrared.status()

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

    def request_diagnostic_rgb(self) -> bytes | None:
        with self.lock:
            now = monotonic()
            self.last_contact = now
            self.diagnostic_rgb_until = now + PREVIEW_LEASE
            preview = self.diagnostic_rgb
            if preview is None or now - preview.at >= PREVIEW_LEASE:
                image = None
            else:
                image = preview.jpeg
        self.wake.set()
        return image

    def diagnostic_rgb_due(self, at: float) -> bool:
        with self.lock:
            if monotonic() >= self.diagnostic_rgb_until:
                self.diagnostic_rgb = None
                return False
            if at < self.next_diagnostic_rgb_at:
                return False
            self.next_diagnostic_rgb_at = at + PREVIEW_INTERVAL
            return True

    def publish_diagnostic_rgb(self, at: float, jpeg: bytes, session: str) -> None:
        with self.lock:
            now = monotonic()
            if (self.info is not None and self.info.session == session and now < self.diagnostic_rgb_until
                    and now - at < PREVIEW_LEASE and (self.diagnostic_rgb is None or at > self.diagnostic_rgb.at)):
                self.diagnostic_rgb = Preview(at, jpeg)

    def update_preview_expiry(self, now: float) -> None:
        self.preview_clients = {client: until for client, until in self.preview_clients.items() if until > now}
        self.preview_until = max(self.preview_clients.values(), default=0.0)
        if not self.preview_clients:
            self.preview = None

    def end_preview(self, client: str) -> None:
        with self.lock:
            self.preview_clients.pop(client, None)
            self.update_preview_expiry(monotonic())

    def request_thumb_depth(self) -> None:
        with self.lock:
            self.thumb_depth_until = monotonic() + THUMB_DEPTH_LEASE

    def thumb_depth_requested(self) -> bool:
        with self.lock:
            return monotonic() < self.thumb_depth_until

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

    def publish_preview(self, at: float, jpeg: bytes, session: str | None = None) -> None:
        with self.lock:
            now = monotonic()
            if session is not None and (self.info is None or self.info.session != session):
                return
            if (now < self.preview_until and now - at < PREVIEW_LEASE
                    and (self.preview is None or at > self.preview.at)):
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
            if self.thumb_depth_until:
                if monotonic() < self.thumb_depth_until:
                    self.gesture_history.append(frame)
                else:
                    self.thumb_depth_until = 0.0
                    self.gesture_history.clear()
            self.frame_times.append(frame.at)
            self.processing_times.append(frame.processing_ms)
            self.processing_ms += (frame.processing_ms - self.processing_ms) * 0.2
            self.changed.notify_all()
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
                logging.exception('%s capture stopped', self.camera_name)
                message = f'{self.camera_name} tracking stopped. Check the USB cable and the camera bridge terminal.'
            finally:
                with self.lock:
                    if revision != self.revision:
                        message = None
                    self.info = None
                    self.active_options = None
                    self.frame = None
                    self.frame_times.clear()
                    self.processing_times.clear()
                    self.processing_ms = 0.0
                    self.history.clear()
                    self.gesture_history.clear()
                    self.preview = None
                    self.preview_clients.clear()
                    self.preview_until = 0.0
                    self.next_preview_at = 0.0
                    self.diagnostic_rgb = None
                    self.next_diagnostic_rgb_at = 0.0
                    self.thumb_depth_until = 0.0
                    self.error = message
                    self.depth_error = None
                    self.changed.notify_all()
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
            device = devices[0]
            color, color_queue = resources.enter_context(open_color(rs, device, options.fps))
            alignment = 'depth' if color.intrinsics.model == rs.distortion.modified_brown_conrady else 'color'
            align = depth_aligner(rs)
            def profile_key(width):
                return f'{serial}:{RGB_SIZE[0]}x{RGB_SIZE[1]}:{width}:{options.fps}:{alignment}'

            session = self.begin_session(revision, options, serial)
            if session is None:
                return

            def set_depth_profile(width: int, project_image: ImageProjector):
                self.set_depth_profile(session, profile_key(width), project_image)

            depth = DepthStream(rs, device, color, options.fps, set_depth_profile,
                                infrared=self.infrared, cv2=cv2, np=np)
            resources.callback(depth.close)
            preview, diagnostic_rgb = self.open_previews(resources, cv2, session)
            self.track_frames(rs, np, estimator, color_queue, depth, align, revision, preview, diagnostic_rgb)

    def begin_session(self, revision: int, options: CameraOptions, serial: str) -> str | None:
        with self.lock:
            if revision != self.revision:
                return None
            session = uuid4().hex
            self.info = CameraInfo(None, session, serial=serial)
            self.active_options = options
            self.error = self.depth_error = None
            return session

    def set_depth_profile(self, session: str, profile: str, project_image: ImageProjector) -> None:
        with self.lock:
            if self.info is not None and self.info.session == session and self.info.profile != profile:
                self.info = replace(self.info, profile=profile, project_image=project_image)

    def open_previews(self, resources: ExitStack, cv2, session: str):
        if cv2 is None:
            return None, None
        preview = PreviewEncoder(cv2, lambda at, jpeg: self.publish_preview(at, jpeg, session))
        resources.callback(preview.close)
        diagnostic = PreviewEncoder(cv2, lambda at, jpeg: self.publish_diagnostic_rgb(at, jpeg, session),
                                    encode=encode_diagnostic_rgb)
        resources.callback(diagnostic.close)
        return preview, diagnostic

    def track_frames(self, rs, np, estimator, color_queue, depth_stream, align, revision,
                     preview=None, diagnostic_rgb=None, *, read_color=None) -> None:
        if read_color is None:
            def read_color(frame):
                return CameraSample.read(rs, frame)
        tracker = HandTracker()
        sequence = 0
        previous_number = None
        previous_timestamp = -1
        last_rgb_at = monotonic()
        last_pose_at = float('-inf')
        dropped = 0
        while self.wanted(revision):
            found, frame = color_queue.try_wait_for_frame(100)
            dequeued_at = monotonic()
            if dequeued_at - last_rgb_at > STREAM_TIMEOUT:
                raise CameraError('RGB frames stopped. Check the camera USB connection.')
            if not found or not frame:
                continue
            color = read_color(frame)
            if color.number == previous_number:
                continue
            if previous_number is not None:
                dropped += max(0, color.number - previous_number - 1)
            previous_number = color.number
            last_rgb_at = dequeued_at
            if color.captured_at is not None:
                queue_ms = (dequeued_at - color.captured_at) * 1000
            else:
                queue_ms = time() * 1000 - color.arrival if color.arrival is not None else None
            if queue_ms is None or not isfinite(queue_ms) or abs(queue_ms) > 3_600_000:
                # Unsupported clock origins cannot be compared with wall time.
                queue_ms = None
            elif queue_ms < -2 or queue_ms >= FRAME_MAX_AGE * 1000:
                dropped += 1
                continue
            else:
                queue_ms = max(0, queue_ms)
            at = dequeued_at if queue_ms is None else dequeued_at - queue_ms / 1000
            if at < last_pose_at:
                dropped += 1
                continue
            last_pose_at = at
            timestamp = max(previous_timestamp + 1, int(at * 1000))
            previous_timestamp = timestamp
            rgb = np.ascontiguousarray(frame.get_data())
            before_inference = monotonic()
            detections = estimator.detect(rgb, timestamp)
            after_inference = monotonic()
            try:
                paired_depth, project = depth_stream.match(color)
                depth = align(frame, paired_depth) if paired_depth is not None and detections else None
            except RuntimeError:
                # The depth sensor may have disconnected or restarted since
                # the frame was queued. Keep publishing the fresh RGB pose.
                depth = paired_depth = project = None
            after_alignment = monotonic()
            measure_thumb_depth = len(detections) == 1 and self.thumb_depth_requested()
            samples = [
                fuse_hand(rs, np, hand.landmarks, [], hand.handedness, depth, hand.features,
                          deproject=project, measure_thumb_depth=measure_thumb_depth)
                for hand in detections
            ]
            if not self.wanted(revision):
                return
            tracking, hands = tracker.update(samples, at)
            after_tracking = monotonic()
            preview_due = preview is not None and self.preview_due(at)
            diagnostic_rgb_due = diagnostic_rgb is not None and self.diagnostic_rgb_due(at)
            diagnostics_due = after_tracking < self.hand_diagnostics_until or diagnostic_rgb_due
            current_tracks = ()
            if preview_due or diagnostics_due:
                current_tracks = tuple(track for track in tracker.tracks if track.published and track.observed_at == at)
            observed = diagnostic_hands(detections, samples, current_tracks) if diagnostics_due else ()
            # Calibration observes a single depth-valid pointing hand even
            # when the live gesture classifier briefly reads a grip.
            sample = samples[0] if len(samples) == 1 else None
            calibration_pose = sample.pose if sample and sample.pointing and sample.features.folded < 4 else None
            sequence += 1
            diagnostics = {
                'depthHands': sum(sample is not None and sample.pose is not None for sample in samples),
                'model': dict(getattr(estimator, 'diagnostics', {})),
                'queueMs': queue_ms,
                'inferenceMs': (after_inference - before_inference) * 1000,
                'alignmentMs': (after_alignment - after_inference) * 1000,
                'fusionMs': (after_tracking - after_alignment) * 1000,
                'captureDroppedFrames': dropped,
                'depthFrameAvailable': paired_depth is not None,
                'frameAgeSource': 'hostArrival' if color.captured_at is None and queue_ms is not None else 'dequeue',
            }
            self.publish(Frame(
                sequence, at, tracking, hands, calibration_pose, len(samples),
                (monotonic() - dequeued_at) * 1000, diagnostics,
                thumb_depth_m=sample.thumb_depth_m if sample is not None else None,
                hand_diagnostics=observed,
            ), revision)
            if preview_due:
                markers = preview_markers([hand.landmarks for hand in detections], samples, current_tracks,
                                          mark_thumb=measure_thumb_depth)
                preview.submit(at, rgb, markers)
            if diagnostic_rgb_due:
                diagnostic_rgb.submit(at, rgb, hand_overlay(detections, observed))
