"""Install the checksum-verified MediaPipe Full hand model locally."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

DIRECTORY = Path(__file__).resolve().parents[1] / '.runtime' / 'camera'
FULL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
FULL_SHA = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'


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
    print('Verified MediaPipe Full bundle.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    download(parser.parse_args().directory)
