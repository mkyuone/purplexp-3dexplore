"""Install the checksum-verified MediaPipe hand and pose models locally."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from visitor_pose import POSE_MODEL_NAME

DIRECTORY = Path(__file__).resolve().parents[1] / '.runtime' / 'camera'
FULL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
FULL_SHA = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
POSE_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
POSE_SHA = '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a'


def verified_file(path: Path, digest: str) -> bool:
    return path.is_file() and sha256(path.read_bytes()).hexdigest() == digest


def fetch(url: str, digest: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        data = response.read(64 * 1024 * 1024)
    if sha256(data).hexdigest() != digest:
        raise ValueError(f'Model download checksum failed for {url}.')
    return data


def save(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        output.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download(directory: Path) -> None:
    full = directory / 'hand_landmarker.task'
    if not verified_file(full, FULL_SHA):
        save(full, fetch(FULL_URL, FULL_SHA))
    pose = directory / POSE_MODEL_NAME
    if not verified_file(pose, POSE_SHA):
        save(pose, fetch(POSE_URL, POSE_SHA))
    print('Verified MediaPipe hand and pose bundles.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    download(parser.parse_args().directory)
