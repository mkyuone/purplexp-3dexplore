"""Loopback bridge for the Spatial CAD Explorer's D435i pose and calibration API."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import dist
from pathlib import Path
from secrets import compare_digest, token_urlsafe
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic
from urllib.parse import parse_qs, urlsplit
import json
import logging
import os
import re
import signal

from calibration import CENTER, Calibration, CalibrationFitError, Pose, TARGETS, fit, fit_corners, median_point
from navigation import FuzzyMapper
from options import CameraOptions, DEFAULT_SETTINGS, load_options, save_options
from realsense import CameraFeed, CameraInfo, FRAME_MAX_AGE, Snapshot

PORT = 4182
POSE_RECOVERY_TIMEOUT = 0.65
WAITING_FOR_FRAMES = 'Waiting for fresh D435i frames. Check the camera connection if this persists.'
CALIBRATION_TIMEOUT = 120.0
OPERATOR_PREVIEW_TIMEOUT = 6.0
MAX_OPERATOR_PREVIEWS = 8
CAPTURE_WINDOW = 0.65
CAPTURE_MIN_SPAN = 0.10
CAPTURE_MIN_SAMPLES = 5
CAPTURE_STABILITY = 0.018
ORIGINS = {'http://localhost:4180', 'http://127.0.0.1:4180'}
DEFAULT_PROFILES = Path(__file__).resolve().parents[1] / '.runtime' / 'camera' / 'calibration.json'
DISPLAY = re.compile(
    r'[1-9][0-9]{0,4}x[1-9][0-9]{0,4}@[0-9]{1,2}(?:\.[0-9]{1,12})?'
    r'(?::[1-9][0-9]{0,4}x[1-9][0-9]{0,4}:-?[0-9]{1,6},-?[0-9]{1,6})?'
)
ACTIONS = ('begin', 'capture', 'cancel')


def valid_display(value: object) -> bool:
    return isinstance(value, str) and DISPLAY.fullmatch(value) is not None


def offline(message: str) -> dict:
    return {'status': 'offline', 'message': message}


def calibrating(step: int = 0, message: str = 'Start five-point calibration in Setup.') -> dict:
    return {'status': 'calibration', 'step': step, 'message': message}


def camera_problem(snapshot: Snapshot) -> str | None:
    if snapshot.error:
        return snapshot.error
    if snapshot.info is None or snapshot.frame is None or monotonic() - snapshot.frame.at >= FRAME_MAX_AGE:
        return WAITING_FOR_FRAMES
    return None


def stale_pose(age: float) -> dict | None:
    if age < FRAME_MAX_AGE:
        return None
    if age < POSE_RECOVERY_TIMEOUT:
        return {'status': 'recovering'}
    return offline(WAITING_FOR_FRAMES)


class Profiles:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, Calibration] = {}
        try:
            stored = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(stored, dict):
                raise ValueError('Expected a profile map.')
            for key, value in stored.items():
                try:
                    self.records[key] = Calibration.from_dict(value)
                except ValueError:
                    logging.warning('Ignoring an invalid display calibration in %s', path)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            logging.warning('Could not read camera calibration profiles from %s', path)

    def save(self, key: str, profile: Calibration) -> None:
        records = self.records | {key: profile}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({key: value.to_dict() for key, value in records.items()}, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)
            self.records = records
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


@dataclass
class Attempt:
    session: str
    after: float
    touched: float
    points: list[Pose | None] = field(default_factory=lambda: [None] * len(TARGETS))
    retries: list[int] = field(default_factory=lambda: [0] * len(TARGETS))
    preview_token: str = field(default_factory=lambda: token_urlsafe(24))
    message: str = 'Hold one open hand at the highlighted point, then capture.'

    @property
    def step(self) -> int:
        return self.points.index(None)


@dataclass
class OperatorPreview:
    display: str
    started: float
    touched: float


@dataclass(frozen=True)
class CaptureReadiness:
    pose: Pose | None
    hint: str


def capture_readiness(snapshot: Snapshot, attempt: Attempt) -> CaptureReadiness:
    frame = snapshot.frame
    assert frame is not None
    if frame.detected_hands == 0:
        return CaptureReadiness(None, 'Bring one hand into the camera’s view.')
    if frame.detected_hands > 1:
        return CaptureReadiness(None, 'Keep only your pointing hand in the camera’s view.')
    latest = frame.calibration_pose
    if latest is None or not latest.valid():
        return CaptureReadiness(None, 'The camera needs depth for your index finger and palm. Turn your palm toward it.')

    recent = [
        (sample.at, sample.calibration_pose) for sample in snapshot.history
        if attempt.after < sample.at <= frame.at
        and frame.at - sample.at <= CAPTURE_WINDOW
        and sample.calibration_pose is not None and sample.calibration_pose.valid()
    ]
    if len(recent) < CAPTURE_MIN_SAMPLES:
        return CaptureReadiness(None, 'Hold your fingertip at this point while the camera collects steady samples.')

    tip = median_point([pose.tip for _, pose in recent])
    palm = median_point([pose.palm for _, pose in recent])

    def close_to_median(pose: Pose) -> bool:
        return dist(pose.tip, tip) <= CAPTURE_STABILITY and dist(pose.palm, palm) <= CAPTURE_STABILITY

    inliers = [(at, pose) for at, pose in recent if close_to_median(pose)]
    if (len(inliers) < CAPTURE_MIN_SAMPLES or len(inliers) * 10 < len(recent) * 7
            or not close_to_median(latest) or frame.at - inliers[0][0] < CAPTURE_MIN_SPAN):
        return CaptureReadiness(None, 'Hold your fingertip and palm still at this point.')

    pose = Pose(
        median_point([sample.tip for _, sample in inliers]),
        median_point([sample.palm for _, sample in inliers]),
    )
    step = attempt.step
    if step != CENTER and not attempt.retries[step] and any(
        point is not None and dist(pose.tip, point.tip) < 0.045 for point in attempt.points
    ):
        return CaptureReadiness(None, 'Move your hand to the next highlighted point.')
    return CaptureReadiness(pose, 'Steady. Keep your hand in place and capture.')


def attempt_status(attempt: Attempt, readiness: CaptureReadiness) -> dict:
    return {
        **calibrating(attempt.step, attempt.message),
        'captureReady': readiness.pose is not None,
        'captureHint': readiness.hint,
    }


def retry_capture(attempt: Attempt, snapshot: Snapshot, error: CalibrationFitError) -> dict:
    retry = min(error.targets, key=attempt.retries.__getitem__)
    attempt.retries[retry] += 1
    attempt.points[retry] = None
    attempt.message = f'{error} Retake the highlighted point. Your other captures were kept.'
    logging.info('Calibration requested a retake of point %d: %s', retry + 1, error)
    return attempt_status(attempt, capture_readiness(snapshot, attempt))


class CameraService:
    def __init__(self, feed: CameraFeed, profiles: Profiles, settings_path: Path = DEFAULT_SETTINGS) -> None:
        self.feed = feed
        self.profiles = profiles
        self.lock = Lock()
        self.attempts: dict[str, Attempt] = {}
        self.operator_previews: dict[str, OperatorPreview] = {}
        self.fuzzy = FuzzyMapper()
        self.fuzzy_session: str | None = None
        self.settings_path = settings_path

    def settings(self, value: dict | None = None) -> dict:
        with self.lock:
            if value is not None:
                options = CameraOptions.from_dict(value)
                available = next(
                    (choice for choice in self.feed.settings()['trackers'] if choice['value'] == options.tracker),
                    None,
                )
                if available is None or not available['available']:
                    raise ValueError(available.get('reason', 'Hand tracker unavailable.') if available else 'Hand tracker unavailable.')
                save_options(self.settings_path, options)
                self.feed.configure(options)
                self.attempts.clear()
            self.feed.snapshot()
            return self.feed.settings()

    def active_attempt(self, info: CameraInfo, display: str) -> Attempt | None:
        attempt = self.attempts.get(display)
        if attempt and (attempt.session != info.session or monotonic() - attempt.touched >= CALIBRATION_TIMEOUT):
            del self.attempts[display]
            return None
        return attempt

    def poses(self, display: str, mode: str = 'fuzzy') -> dict:
        with self.lock:
            snapshot = self.feed.snapshot()
            info, frame = snapshot.info, snapshot.frame
            if snapshot.error or info is None or frame is None:
                return offline(snapshot.error or WAITING_FOR_FRAMES)
            if stale := stale_pose(monotonic() - frame.at):
                return stale
            if mode == 'fuzzy':
                if self.fuzzy_session != info.session:
                    self.fuzzy.reset()
                    self.fuzzy_session = info.session
                hands = [{
                    'id': hand.id, 'x': hand.x, 'y': hand.y, 'depth': hand.depth, 'depthValid': hand.depth_valid,
                    'pinched': hand.grip == 'pinch' and not hand.thumbs_up,
                    'fist': hand.grip == 'fist' and not hand.thumbs_up, 'thumbsUp': hand.thumbs_up,
                } for hand in self.fuzzy.update(frame.hands, frame.at)]
            else:
                if attempt := self.active_attempt(info, display):
                    return attempt_status(attempt, capture_readiness(snapshot, attempt))
                calibration = self.profiles.records.get(f'{info.profile}|{display}')
                if calibration is None:
                    return calibrating()
                hands = []
                for hand in frame.hands:
                    if hand.pose is None or (not hand.pointing and hand.grip != 'fist' and not hand.thumbs_up):
                        continue
                    projected = calibration.project(hand.pose)
                    if projected is not None:
                        x, y, depth = projected
                        hands.append({
                            'id': hand.id, 'x': x, 'y': y, 'depth': depth, 'depthValid': True,
                            'pinched': hand.grip == 'pinch' and not hand.thumbs_up,
                            'fist': hand.grip == 'fist' and not hand.thumbs_up, 'thumbsUp': hand.thumbs_up,
                        })
            age = monotonic() - frame.at
            if stale := stale_pose(age):
                return stale
            tracking = frame.tracking if hands or frame.tracking == 'ambiguous' else 'lost'
            return {
                'status': 'ready', 'session': f'{info.session}:{mode}:{display}', 'sequence': frame.sequence,
                'ageMs': max(0, age * 1000), 'tracking': tracking, 'hands': hands,
            }

    def preview(self, display: str, token: str | None) -> bytes | None:
        with self.lock:
            self.expire_previews()
            if token is None:
                return None
            operator = self.operator_previews.get(token)
            if operator is not None:
                if operator.display != display:
                    return None
                operator.touched = monotonic()
                snapshot = self.feed.snapshot()
                return (
                    self.feed.request_preview(snapshot.info.session, operator.started, token)
                    if snapshot.info is not None else None
                )
            attempt = self.attempts.get(display)
            if attempt is None or not token.isascii() or not compare_digest(token, attempt.preview_token):
                return None
            if monotonic() - attempt.touched >= CALIBRATION_TIMEOUT:
                del self.attempts[display]
                return None
            snapshot = self.feed.snapshot()
            if snapshot.info is None or self.active_attempt(snapshot.info, display) is None:
                return None
            return self.feed.request_preview(snapshot.info.session, attempt.after, token)

    def expire_previews(self) -> None:
        expired = [
            token for token, preview in self.operator_previews.items()
            if monotonic() - preview.touched >= OPERATOR_PREVIEW_TIMEOUT
        ]
        for token in expired:
            del self.operator_previews[token]
            self.feed.end_preview(token)

    def preview_session(self, display: str, action: str, token: str | None) -> dict:
        with self.lock:
            self.expire_previews()
            if action == 'end':
                preview = self.operator_previews.get(token)
                if preview is not None and preview.display == display:
                    del self.operator_previews[token]
                    self.feed.end_preview(token)
                return {'status': 'ready'}
            if len(self.operator_previews) >= MAX_OPERATOR_PREVIEWS:
                oldest = min(self.operator_previews, key=lambda token: self.operator_previews[token].touched)
                del self.operator_previews[oldest]
                self.feed.end_preview(oldest)
            token = token_urlsafe(24)
            now = monotonic()
            self.operator_previews[token] = OperatorPreview(display, now, now)
            self.feed.snapshot()
            return {'status': 'ready', 'previewToken': token}

    def calibrate(self, display: str, action: str) -> dict:
        with self.lock:
            if action == 'cancel':
                self.attempts.pop(display, None)
            snapshot = self.feed.snapshot()
            if message := camera_problem(snapshot):
                return offline(message)
            info, frame = snapshot.info, snapshot.frame
            assert info is not None and frame is not None
            key = f'{info.profile}|{display}'
            if action == 'cancel':
                return (
                    {'status': 'ready', 'message': 'Using the previous calibration.'}
                    if key in self.profiles.records else calibrating(message='Calibration canceled.')
                )
            if action == 'begin':
                attempt = Attempt(info.session, frame.at, monotonic())
                self.attempts[display] = attempt
                return {
                    **attempt_status(attempt, capture_readiness(snapshot, attempt)),
                    'previewToken': attempt.preview_token,
                }

            attempt = self.active_attempt(info, display)
            if attempt is None:
                return calibrating(message='This calibration session ended. Cancel and restart it in Setup.')
            attempt.touched = monotonic()
            readiness = capture_readiness(snapshot, attempt)
            if readiness.pose is None:
                return attempt_status(attempt, readiness)
            captured = attempt.step
            attempt.points[captured] = readiness.pose
            attempt.after = frame.at
            if None in attempt.points:
                if attempt.step == CENTER:
                    try:
                        fit_corners([point for point in attempt.points[:CENTER] if point is not None])
                    except CalibrationFitError as error:
                        return retry_capture(attempt, snapshot, error)
                    attempt.message = 'Corners accepted. The center helps set a comfortable depth.'
                else:
                    attempt.message = 'Move your hand to the next ring, then capture.'
                return attempt_status(attempt, capture_readiness(snapshot, attempt))
            try:
                profile = fit([point for point in attempt.points if point is not None])
            except CalibrationFitError as error:
                return retry_capture(attempt, snapshot, error)
            try:
                self.profiles.save(key, profile)
            except OSError:
                logging.exception('Could not save camera calibration')
                attempt.points[captured] = None
                attempt.message = 'Could not save calibration. Check the bridge profile path, then capture again.'
                return attempt_status(attempt, capture_readiness(snapshot, attempt))
            del self.attempts[display]
            logging.info('Hand camera calibration saved.')
            return {'status': 'ready', 'message': 'Calibration saved.'}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *_args):
        pass

    def allowed(self, require_origin: bool = False) -> bool:
        host = self.headers.get('Host', '').lower()
        origin = self.headers.get('Origin')
        port = self.server.server_port
        if host not in {f'localhost:{port}', f'127.0.0.1:{port}'}:
            return False
        if origin is not None:
            return origin in ORIGINS
        return not require_origin and self.headers.get('Sec-Fetch-Site') != 'cross-site'

    def send_json(self, status: int, data: dict | None = None) -> None:
        body = b'' if data is None else json.dumps(data, allow_nan=False, separators=(',', ':')).encode('utf-8')
        self.send_body(status, body, 'application/json; charset=utf-8')

    def send_body(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            origin = self.headers.get('Origin')
            if origin in ORIGINS and self.allowed():
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
                if self.command == 'OPTIONS':
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    if self.path.split('?', 1)[0] == '/preview':
                        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Calibration-Preview')
                        self.send_header('Access-Control-Max-Age', '300')
                    else:
                        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            if body:
                self.send_header('Content-Type', content_type)
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_OPTIONS(self) -> None:
        if not self.allowed(require_origin=True):
            self.send_json(403, {'message': 'Only the local Explorer can access the camera bridge.'})
        elif self.path.split('?', 1)[0] not in {'/poses', '/calibration', '/preview', '/settings'}:
            self.send_json(404, {'message': 'Unknown camera bridge endpoint.'})
        else:
            self.send_json(204)

    def do_GET(self) -> None:
        if not self.allowed():
            self.send_json(403, {'message': 'Only the local Explorer can access the camera bridge.'})
            return
        try:
            url = urlsplit(self.path)
            query = parse_qs(url.query, max_num_fields=2, keep_blank_values=True)
        except ValueError:
            self.send_json(400, {'message': 'Invalid camera bridge request.'})
            return
        if url.path == '/settings' and not query:
            self.send_json(200, self.server.service.settings())
            return
        if url.path not in {'/poses', '/preview'}:
            self.send_json(404, {'message': 'Unknown camera bridge endpoint.'})
            return
        displays = query.get('display', [])
        if len(displays) != 1 or not valid_display(displays[0]):
            self.send_json(400, {'message': 'A display signature is required.'})
            return
        display = displays[0]
        if url.path == '/preview':
            if not self.allowed(require_origin=True):
                self.send_json(403, {'message': 'Only the local Explorer can access the calibration preview.'})
                return
            image = self.server.service.preview(display, self.headers.get('X-Calibration-Preview'))
            self.send_body(200 if image else 204, image or b'', 'image/jpeg')
        else:
            modes = query.get('mode', ['fuzzy'])
            if len(modes) != 1 or modes[0] not in {'fuzzy', 'calibrated'}:
                self.send_json(400, {'message': 'Choose fuzzy or calibrated hand navigation.'})
                return
            self.send_json(200, self.server.service.poses(display, modes[0]))

    def do_POST(self) -> None:
        if not self.allowed(require_origin=True):
            self.send_json(403, {'message': 'Only the local Explorer can access the camera bridge.'})
            return
        if self.path not in {'/calibration', '/preview', '/settings'}:
            self.send_json(404, {'message': 'Unknown camera bridge endpoint.'})
            return
        if self.headers.get_content_type() != 'application/json':
            self.send_json(415, {'message': 'Calibration requests must be JSON.'})
            return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 2048:
                raise ValueError('Invalid request size.')
            request = json.loads(self.rfile.read(size))
        except (ValueError, TimeoutError):
            self.send_json(400, {'message': 'Invalid calibration request.'})
            return
        if self.path == '/settings':
            if not isinstance(request, dict):
                self.send_json(400, {'message': 'Camera settings must be a JSON object.'})
                return
            try:
                self.send_json(200, self.server.service.settings(request))
            except ValueError as error:
                self.send_json(400, {'message': str(error)})
            except OSError:
                self.send_json(500, {'message': 'Camera settings could not be saved.'})
            return
        actions = ('begin', 'end') if self.path == '/preview' else ACTIONS
        if not isinstance(request, dict) or not valid_display(request.get('display')) or request.get('action') not in actions:
            self.send_json(400, {'message': 'A display signature and calibration action are required.'})
            return
        if self.path == '/preview':
            token = request.get('previewToken')
            if request['action'] == 'end' and not isinstance(token, str):
                self.send_json(400, {'message': 'A preview token is required to end the preview.'})
                return
            self.send_json(200, self.server.service.preview_session(request['display'], request['action'], token))
        else:
            self.send_json(200, self.server.service.calibrate(request['display'], request['action']))


class BridgeServer(ThreadingHTTPServer):
    def __init__(self, model: Path, serial: str | None, profiles: Path, port: int = PORT, settings: Path = DEFAULT_SETTINGS) -> None:
        super().__init__(('127.0.0.1', port), Handler)
        self.feed = CameraFeed(model, serial, load_options(settings))
        self.service = CameraService(self.feed, Profiles(profiles), settings)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(3)
        return connection, address

    def server_close(self) -> None:
        super().server_close()
        if hasattr(self, 'feed'):
            self.feed.close()


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='MediaPipe Hand Landmarker task file.')
    parser.add_argument('--serial', help='RealSense D435i serial number if more than one is connected.')
    parser.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES, help='Path for saved calibration profiles.')
    parser.add_argument('--settings', type=Path, default=DEFAULT_SETTINGS, help='Path for operator camera settings.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    with BridgeServer(args.model.expanduser(), args.serial, args.profiles.expanduser(), settings=args.settings.expanduser()) as server:
        logging.info('D435i pose bridge on http://127.0.0.1:%d', PORT)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
