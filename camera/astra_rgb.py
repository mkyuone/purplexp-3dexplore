"""Find an Astra Pro's paired USB devices and read its Linux UVC color stream."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread
from time import monotonic
import logging
import os


RGB_SIZE = (640, 480)
RGB_FPS = 30
RGB_USB_ID = ('2bc5', '0501')
DEPTH_USB_ID = ('2bc5', '0403')


class AstraRgbError(RuntimeError):
    pass


@dataclass(frozen=True)
class AstraDevice:
    rgb_device: Path
    depth_bus: int
    depth_address: int
    usb_parent: str
    identity: str


def _attribute(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ''


def _usb_id(path: Path) -> tuple[str, str]:
    return (_attribute(path / 'idVendor').lower(), _attribute(path / 'idProduct').lower())


def discover_astra(
    usb_parent: str | None = None, *,
    sysfs: Path = Path('/sys'), dev_root: Path = Path('/dev'),
) -> AstraDevice:
    """Pair the original Astra Pro's color and depth devices by their built-in USB hub."""
    depths = {}
    for entry in (sysfs / 'bus/usb/devices').glob('*'):
        if _usb_id(entry) != DEPTH_USB_ID:
            continue
        try:
            bus = int(_attribute(entry / 'busnum'))
            address = int(_attribute(entry / 'devnum'))
        except ValueError:
            continue  # Unplugged during discovery.
        depths.setdefault(entry.resolve().parent, []).append((entry, bus, address))

    pairs = []
    for video in (sysfs / 'class/video4linux').glob('video*'):
        path = dev_root / video.name
        if not path.exists():
            continue
        # uvcvideo registers its image node before the metadata node.
        if _attribute(video / 'index') != '0':
            continue
        usb = next((node for node in video.resolve().parents if _usb_id(node) == RGB_USB_ID), None)
        if usb is None or (usb_parent is not None and usb.parent.name != usb_parent):
            continue
        for depth, bus, address in depths.get(usb.parent, ()):
            serial = _attribute(depth / 'serial')
            pairs.append(AstraDevice(path, bus, address, usb.parent.name, f'astra:{serial or usb.parent.name}'))

    if not pairs:
        raise AstraRgbError('No paired Astra Pro RGB (2bc5:0501) and depth (2bc5:0403) devices found. Check the USB connection.')
    if len(pairs) != 1:
        hubs = ', '.join(sorted({pair.usb_parent for pair in pairs}))
        raise AstraRgbError(f'Multiple Astra Pro pairs found. Select a USB hub with --astra-usb-port ({hubs}).')
    return pairs[0]


@dataclass(frozen=True)
class RgbFrame:
    data: object
    number: int
    at: float

    def get_data(self):
        return self.data


class _RgbQueue:
    def __init__(self, cv2, capture) -> None:
        self.changed = Condition()
        self.frame: RgbFrame | None = None
        self.error: AstraRgbError | None = None
        self.closed = False
        self.thread = Thread(target=self._run, args=(cv2, capture), name='Astra Pro RGB', daemon=True)
        self.thread.start()

    def _run(self, cv2, capture) -> None:
        number = 0
        try:
            while not self.closed:
                received, bgr = capture.read()
                at = monotonic()
                if not received or bgr is None:
                    raise AstraRgbError('Astra Pro RGB frames stopped. Check the USB connection.')
                if bgr.shape != (RGB_SIZE[1], RGB_SIZE[0], 3):
                    raise AstraRgbError('Astra Pro RGB returned an unexpected frame format.')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # This UVC feed is already mirrored. Normalize it to the
                # registered depth and factory RGB rays before tracking.
                cv2.flip(rgb, 1, rgb)
                number += 1
                with self.changed:
                    if self.closed:
                        return
                    self.frame = RgbFrame(rgb, number, at)
                    self.changed.notify_all()
        except (cv2.error, AstraRgbError) as error:
            with self.changed:
                self.error = AstraRgbError(str(error))
                self.changed.notify_all()
        finally:
            capture.release()

    def try_wait_for_frame(self, timeout_ms: int) -> tuple[bool, RgbFrame | None]:
        with self.changed:
            self.changed.wait_for(lambda: self.frame is not None or self.error is not None or self.closed,
                                  timeout=timeout_ms / 1000)
            if self.error is not None and not self.closed:
                raise self.error
            if self.frame is None or self.closed:
                return False, None
            frame, self.frame = self.frame, None
            return True, frame

    def close(self) -> None:
        with self.changed:
            self.closed = True
            self.frame = None
            self.changed.notify_all()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            logging.warning('Astra Pro RGB is waiting for the V4L2 driver to release its last frame.')


@contextmanager
def open_rgb(cv2, device: AstraDevice | str | Path):
    path = device.rgb_device if isinstance(device, AstraDevice) else device
    # OpenCV's V4L2 backend has its own timeout, separate from our queue timeout.
    # Only the reader thread releases VideoCapture, so a read and release cannot race.
    os.environ.setdefault('OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT', '1')
    capture = cv2.VideoCapture()
    try:
        if not capture.open(str(path), cv2.CAP_V4L2):
            raise AstraRgbError(f'Could not open Astra Pro RGB at {path}. Check access and close other camera apps.')
        mjpeg = cv2.VideoWriter_fourcc(*'MJPG')
        mode = (
            (cv2.CAP_PROP_FOURCC, mjpeg),
            (cv2.CAP_PROP_FRAME_WIDTH, RGB_SIZE[0]),
            (cv2.CAP_PROP_FRAME_HEIGHT, RGB_SIZE[1]),
            (cv2.CAP_PROP_FPS, RGB_FPS),
        )
        for prop, value in mode:
            capture.set(prop, value)
        if not all(abs(capture.get(prop) - value) < .1 for prop, value in mode):
            raise AstraRgbError(f'Astra Pro RGB at {path} could not start 640×480 MJPEG at 30 fps.')
        queue = _RgbQueue(cv2, capture)
    except BaseException as error:
        capture.release()
        if isinstance(error, cv2.error):
            raise AstraRgbError(f'Could not configure Astra Pro RGB at {path}: {error}') from error
        raise
    try:
        yield queue
    finally:
        queue.close()
