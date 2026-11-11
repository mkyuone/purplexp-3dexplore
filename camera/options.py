"""Operator-selected camera acquisition and hand estimator settings."""

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
import json
import logging
import os

DEFAULT_SETTINGS = Path(__file__).resolve().parents[1] / '.runtime' / 'camera' / 'settings.json'


@dataclass(frozen=True)
class CameraOptions:
    source: str = 'rgb'
    tracker: str = 'mediapipe'
    fps: int = 30
    interaction: str = 'visitors'

    @classmethod
    def from_dict(cls, value: object) -> 'CameraOptions':
        required = {'source', 'tracker', 'fps'}
        if not isinstance(value, dict) or set(value) not in (required, required | {'interaction'}):
            raise ValueError('Choose a hand tracker and frame rate.')
        if value['source'] != 'rgb' or value['tracker'] != 'mediapipe':
            raise ValueError('Choose RGB and a supported hand tracker.')
        if type(value['fps']) is not int or value['fps'] not in (30, 60):
            raise ValueError('Choose 30 or 60 frames per second.')
        if value.get('interaction', 'visitors') not in ('visitors', 'single-user'):
            raise ValueError('Choose visitor turns or single-user interaction.')
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


def load_options(path: Path) -> CameraOptions:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(value, dict):
            if value.get('source') == 'ir1' or value.get('tracker') == 'mercury':
                value = value | {'source': 'rgb', 'tracker': 'mediapipe'}
            elif value.get('tracker') in ('mediapipe-lite', 'rtmpose'):
                value = value | {'tracker': 'mediapipe'}
        return CameraOptions.from_dict(value)
    except FileNotFoundError:
        return CameraOptions()
    except (OSError, ValueError):
        logging.warning('Ignoring invalid camera settings in %s', path)
        return CameraOptions()


def save_options(path: Path, options: CameraOptions) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(options.to_dict(), handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
