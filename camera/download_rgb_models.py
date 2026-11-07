"""Install checksum-verified RGB hand model comparison assets locally."""

from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile

DIRECTORY = Path(__file__).resolve().parents[1] / '.runtime' / 'camera'
FULL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
FULL_SHA = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
RTM_MODELS = (
    ('rtmdet_nano_hand', 'rtmdet_nano_8xb32-300e_hand-267f9c8f.zip',
     '9c0370a43c02b2fe42b4382aba7383d97cfa3ed35623b655cac4f0c25cfde402',
     '568d3ea97a5b142488366b67e036b6a5cb0a1fef9087a710cb8e66b6979fbac2'),
    ('rtmpose_m_hand', 'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip',
     '45c9e1aa21b8fe33859973c49bce4647645d312e7a226e398a42f9137d44a56d',
     '39e858936bca0f94c09847d4e70b68a51d6c0adac61f36b457fcadb54621cd29'),
)


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
    for name, archive_name, archive_digest, weight_digest in RTM_MODELS:
        path = directory / 'rtmpose' / (name + '.onnx')
        if not verified_file(path, weight_digest):
            data = fetch('https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/' + archive_name, archive_digest)
            with ZipFile(BytesIO(data)) as archive:
                models = [entry for entry in archive.namelist() if entry.endswith('.onnx')]
                if len(models) != 1:
                    raise ValueError(f'Expected one ONNX model in {archive_name}.')
                weights = archive.read(models[0])
                if sha256(weights).hexdigest() != weight_digest:
                    raise ValueError(f'ONNX checksum failed for {name}.')
                save(path, weights)
                for entry in archive.namelist():
                    if Path(entry).name in ('pipeline.json', 'deploy.json'):
                        save(path.parent / (name + '_' + Path(entry).name), archive.read(entry))
        print(f'Verified {name}.')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DIRECTORY)
    download(parser.parse_args().directory)
