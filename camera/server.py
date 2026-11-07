"""Loopback bridge for the Spatial CAD Explorer's D435i pose and calibration API."""

from argparse import ArgumentParser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import dist
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic
from urllib.parse import parse_qs, urlsplit
import json
import logging
import os
import re

from calibration import Calibration, Pose, TARGETS, fit, steady_pose
from realsense import CameraFeed, CameraInfo, FRAME_MAX_AGE, Snapshot

PORT = 4182
CALIBRATION_TIMEOUT = 120.0
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
        return 'Waiting for fresh D435i frames. Check the camera connection if this persists.'
    return None


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
    points: list[Pose] = field(default_factory=list)
    message: str = 'Hold one open hand at the highlighted point, then capture.'


class CameraService:
    def __init__(self, feed: CameraFeed, profiles: Profiles) -> None:
        self.feed = feed
        self.profiles = profiles
        self.lock = Lock()
        self.attempts: dict[str, Attempt] = {}

    def active_attempt(self, info: CameraInfo, display: str) -> Attempt | None:
        attempt = self.attempts.get(display)
        if attempt and (attempt.session != info.session or monotonic() - attempt.touched >= CALIBRATION_TIMEOUT):
            del self.attempts[display]
            return None
        return attempt

    def poses(self, display: str) -> dict:
        with self.lock:
            snapshot = self.feed.snapshot()
            if message := camera_problem(snapshot):
                return offline(message)
            info, frame = snapshot.info, snapshot.frame
            assert info is not None and frame is not None
            if attempt := self.active_attempt(info, display):
                return calibrating(len(attempt.points), attempt.message)
            calibration = self.profiles.records.get(f'{info.profile}|{display}')
            if calibration is None:
                return calibrating()

            hands = []
            for hand in frame.hands:
                projected = calibration.project(hand.pose)
                if projected is not None:
                    x, y, depth = projected
                    hands.append({
                        'id': hand.id, 'x': x, 'y': y, 'depth': depth, 'depthValid': True,
                        'pinched': hand.grip == 'pinch', 'fist': hand.grip == 'fist',
                    })
            age_ms = (monotonic() - frame.at) * 1000
            if age_ms >= FRAME_MAX_AGE * 1000:
                return offline('Waiting for fresh D435i frames.')
            tracking = frame.tracking if hands or frame.tracking == 'ambiguous' else 'lost'
            return {
                'status': 'ready', 'session': f'{info.session}:{display}', 'sequence': frame.sequence,
                'ageMs': max(0, age_ms), 'tracking': tracking, 'hands': hands,
            }

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
                return calibrating(message=attempt.message)

            attempt = self.active_attempt(info, display)
            if attempt is None:
                return calibrating(message='This calibration session ended. Cancel and restart it in Setup.')
            attempt.touched = monotonic()
            samples = [sample for sample in snapshot.history if sample.at > attempt.after]
            pose = steady_pose([sample.hands[0].pose for sample in samples])
            if pose is None:
                attempt.message = 'Hold one open hand still with your index finger visible, then try again.'
                return calibrating(len(attempt.points), attempt.message)
            if attempt.points and dist(pose.tip, attempt.points[-1].tip) < 0.045:
                attempt.message = 'Move your hand to the next point before capturing.'
                return calibrating(len(attempt.points), attempt.message)
            attempt.points.append(pose)
            attempt.after = frame.at
            if len(attempt.points) < len(TARGETS):
                attempt.message = 'Move to the next point at the same depth, then capture.'
                return calibrating(len(attempt.points), attempt.message)
            try:
                profile = fit(attempt.points)
                self.profiles.save(key, profile)
            except ValueError as error:
                attempt.points.clear()
                attempt.message = str(error)
                return calibrating(message=attempt.message)
            except OSError:
                logging.exception('Could not save camera calibration')
                attempt.points.pop()
                attempt.message = 'Could not save calibration. Check the bridge profile path, then capture again.'
                return calibrating(len(attempt.points), attempt.message)
            del self.attempts[display]
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
        try:
            self.send_response(status)
            origin = self.headers.get('Origin')
            if origin in ORIGINS and self.allowed():
                self.send_header('Access-Control-Allow-Origin', origin)
                self.send_header('Vary', 'Origin')
                if self.command == 'OPTIONS':
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            if body:
                self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_OPTIONS(self) -> None:
        if not self.allowed(require_origin=True):
            self.send_json(403, {'message': 'Only the local Explorer can access the camera bridge.'})
        elif self.path.split('?', 1)[0] not in {'/poses', '/calibration'}:
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
        if url.path != '/poses':
            self.send_json(404, {'message': 'Unknown camera bridge endpoint.'})
            return
        displays = query.get('display', [])
        if len(displays) != 1 or not valid_display(displays[0]):
            self.send_json(400, {'message': 'A display signature is required.'})
            return
        self.send_json(200, self.server.service.poses(displays[0]))

    def do_POST(self) -> None:
        if not self.allowed(require_origin=True):
            self.send_json(403, {'message': 'Only the local Explorer can access the camera bridge.'})
            return
        if self.path != '/calibration':
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
        if not isinstance(request, dict) or not valid_display(request.get('display')) or request.get('action') not in ACTIONS:
            self.send_json(400, {'message': 'A display signature and calibration action are required.'})
            return
        self.send_json(200, self.server.service.calibrate(request['display'], request['action']))


class BridgeServer(ThreadingHTTPServer):
    def __init__(self, model: Path, serial: str | None, profiles: Path, port: int = PORT) -> None:
        super().__init__(('127.0.0.1', port), Handler)
        self.feed = CameraFeed(model, serial)
        self.service = CameraService(self.feed, Profiles(profiles))

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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    with BridgeServer(args.model.expanduser(), args.serial, args.profiles.expanduser()) as server:
        logging.info('D435i pose bridge on http://127.0.0.1:%d', PORT)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
