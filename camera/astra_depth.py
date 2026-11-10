"""Astra Pro depth and infrared capture through Orbbec's OpenNI2 Linux SDK."""

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from ctypes import (
    CDLL, POINTER, Structure, byref, c_char, c_char_p, c_float, c_int, c_uint8,
    c_uint16, c_uint64, c_void_p, sizeof,
)
from math import isfinite
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from os import R_OK, W_OK, access
from pathlib import Path
from struct import Struct
from subprocess import DEVNULL, Popen, TimeoutExpired
from time import monotonic
import re
import signal
import sys

from calibration import Vec3, valid_point
from infrared_preview import PREVIEW_INTERVAL

WIDTH, HEIGHT, FPS = 640, 480, 30
API_VERSION = 2003
SDK_VERSION = (2, 3, 0, 86)
DEPTH_SENSOR = 3
DEPTH_1_MM = 100
IR_SENSOR = 1
GRAY16 = 203
TIMEOUT = 102
VIDEO_MODE = 3
MIRRORING = 7
IMAGE_REGISTRATION = 5
DEPTH_TO_COLOR = 1
FACTORY_PARAMETERS = 14
DEPTH_START_SECONDS = 3.0
DEPTH_STOP_SECONDS = .7
IR_START_SECONDS = 2.0
FRAME_TIME = Struct('=d')
FRAME_BYTES = FRAME_TIME.size + WIDTH * HEIGHT * 2


class AstraDepthError(RuntimeError):
    pass


class AstraInfraredError(AstraDepthError):
    pass


# Layouts come from OniCTypes.h in the pinned Orbbec SDK. In particular,
# stock OpenNI2 lacks the serial field in the device list structure.
class Version(Structure):
    _fields_ = [(name, c_int) for name in ('major', 'minor', 'maintenance', 'build')]


class DeviceInfo(Structure):
    _fields_ = [(name, c_char * 256) for name in ('uri', 'vendor', 'name', 'serial')] + [
        ('vendor_id', c_uint16), ('product_id', c_uint16),
    ]


class VideoMode(Structure):
    _fields_ = [(name, c_int) for name in ('format', 'width', 'height', 'fps')]


class Frame(Structure):
    _fields_ = [
        ('size', c_int), ('data', c_void_p), ('sensor', c_int), ('timestamp', c_uint64),
        ('index', c_int), ('width', c_int), ('height', c_int), ('mode', VideoMode),
        ('cropping', c_int), ('crop_x', c_int), ('crop_y', c_int), ('stride', c_int),
    ]


class FactoryParameters(Structure):
    _fields_ = [
        ('depth_intrinsics', c_float * 4), ('color_intrinsics', c_float * 4),
        ('rotation', c_float * 9), ('translation', c_float * 3),
        ('depth_distortion', c_float * 5), ('color_distortion', c_float * 5),
    ]


def load_sdk(sdk_dir: Path):
    lib = Path(sdk_dir).expanduser().resolve() / 'libOpenNI2.so'
    driver = lib.parent / 'OpenNI2' / 'Drivers' / 'liborbbec.so'
    if not lib.is_file() or not driver.is_file():
        raise AstraDepthError('Point --openni2-dir at Orbbec OpenNI2 2.3.0.86 sdk/libs for Astra Pro depth.')
    try:
        sdk = CDLL(str(lib))
    except OSError as error:
        raise AstraDepthError(f'Orbbec OpenNI2 could not load: {error}') from error
    signatures = {
        'oniGetVersion': (Version,),
        'oniInitialize': (c_int, c_int),
        'oniShutdown': (None,),
        'oniGetExtendedError': (c_char_p,),
        'oniGetDeviceList': (c_int, POINTER(POINTER(DeviceInfo)), POINTER(c_int)),
        'oniReleaseDeviceList': (c_int, POINTER(DeviceInfo)),
        'oniDeviceOpen': (c_int, c_char_p, POINTER(c_void_p)),
        'oniDeviceClose': (c_int, c_void_p),
        'oniDeviceCreateStream': (c_int, c_void_p, c_int, POINTER(c_void_p)),
        'oniDeviceGetProperty': (c_int, c_void_p, c_int, c_void_p, POINTER(c_int)),
        'oniDeviceSetProperty': (c_int, c_void_p, c_int, c_void_p, c_int),
        'oniDeviceIsImageRegistrationModeSupported': (c_int, c_void_p, c_int),
        'oniStreamGetProperty': (c_int, c_void_p, c_int, c_void_p, POINTER(c_int)),
        'oniStreamSetProperty': (c_int, c_void_p, c_int, c_void_p, c_int),
        'oniStreamStart': (c_int, c_void_p),
        'oniStreamStop': (None, c_void_p),
        'oniStreamDestroy': (None, c_void_p),
        'oniWaitForAnyStream': (c_int, POINTER(c_void_p), c_int, POINTER(c_int), c_int),
        'oniStreamReadFrame': (c_int, c_void_p, POINTER(POINTER(Frame))),
        'oniFrameRelease': (None, POINTER(Frame)),
    }
    try:
        for name, (result, *args) in signatures.items():
            function = getattr(sdk, name)
            function.restype = result
            function.argtypes = args
    except AttributeError as error:
        raise AstraDepthError('Astra Pro depth requires Orbbec OpenNI2 2.3.0.86.') from error
    version = sdk.oniGetVersion()
    if (version.major, version.minor, version.maintenance, version.build) != SDK_VERSION:
        raise AstraDepthError('Astra Pro depth requires Orbbec OpenNI2 2.3.0.86.')
    return sdk


def check(sdk, status: int, message: str) -> None:
    if status != 0:
        detail = (sdk.oniGetExtendedError() or b'').decode(errors='replace').strip()
        raise AstraDepthError(f'{message} ({detail or status})')


def depth_uri(sdk, bus: int, address: int) -> bytes:
    devices = POINTER(DeviceInfo)()
    count = c_int()
    check(sdk, sdk.oniGetDeviceList(byref(devices), byref(count)), 'OpenNI2 could not list USB depth devices.')
    try:
        for i in range(count.value):
            device = devices[i]
            match = re.fullmatch(rb'2bc5/0403@(\d+)/(\d+)(?:@[^\s]+)?', device.uri, flags=re.IGNORECASE)
            if match and (int(match[1]), int(match[2])) == (bus, address):
                return device.uri
    finally:
        sdk.oniReleaseDeviceList(devices)
    raise AstraDepthError('OpenNI2 could not find the depth device paired with this Astra Pro RGB camera.')


class ColorProjection:
    def __init__(self, np, cv2, parameters: FactoryParameters) -> None:
        fx, fy, cx, cy = parameters.color_intrinsics
        distortion = tuple(parameters.color_distortion)
        if (not all(map(isfinite, (fx, fy, cx, cy, *distortion)))
                or not (fx > 0 and fy > 0 and 0 <= cx < WIDTH and 0 <= cy < HEIGHT)):
            raise AstraDepthError('OpenNI2 did not provide valid Astra Pro factory RGB intrinsics.')
        # Astra factory coefficients put the third radial term before the
        # tangential pair, as in Orbbec's OBCameraParamsToCameraInfo wrapper.
        k1, k2, k3, p1, p2 = distortion
        self.np = np
        self.cv2 = cv2
        self.intrinsics = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)
        self.distortion = np.array((k1, k2, p1, p2, k3), dtype=np.float64)

    def deproject(self, x: float, y: float, z: float) -> Vec3 | None:
        if not (0 <= x < WIDTH and 0 <= y < HEIGHT and isfinite(z)):
            return None
        pixel = self.np.array([[[x, y]]], dtype=self.np.float64)
        ray = self.cv2.undistortPoints(pixel, self.intrinsics, self.distortion)[0, 0]
        point = (float(ray[0] * z), float(ray[1] * z), z)
        return point if valid_point(point) else None


def frame_pixels(np, frame: Frame, sensor: int, pixel_format: int):
    row_bytes = WIDTH * 2
    if (not frame.data or frame.sensor != sensor or frame.mode.format != pixel_format
            or (frame.width, frame.height, frame.mode.fps) != (WIDTH, HEIGHT, FPS) or frame.cropping
            or frame.stride < row_bytes or frame.size < (HEIGHT - 1) * frame.stride + row_bytes):
        if sensor == IR_SENSOR:
            raise AstraInfraredError('Astra Pro returned an unexpected OpenNI2 infrared frame.')
        raise AstraDepthError('Astra Pro returned an unexpected OpenNI2 depth frame.')
    data = (c_uint8 * frame.size).from_address(frame.data)
    return np.ndarray((HEIGHT, WIDTH), dtype=np.uint16, buffer=data, strides=(frame.stride, 2))


def copy_depth(np, frame: Frame, output=None):
    pixels = frame_pixels(np, frame, DEPTH_SENSOR, DEPTH_1_MM)
    if output is None:
        return pixels.copy()
    np.copyto(output, pixels)
    return output


class NativeDepthCapture:
    def __init__(self, np, sdk, device, stream, parameters: FactoryParameters) -> None:
        self.np = np
        self.sdk = sdk
        self.device = device
        self.stream = stream
        self.streams = (c_void_p * 1)(stream)
        self.parameters = parameters
        self.at = 0.0

    def set_infrared(self, stream) -> None:
        self.streams = (c_void_p * 1)(self.stream) if stream is None else (c_void_p * 2)(self.stream, stream)

    def read(self, timeout_ms: int, output=None, *, on_infrared=None):
        index = c_int()
        status = self.sdk.oniWaitForAnyStream(self.streams, len(self.streams), byref(index), timeout_ms)
        if status == TIMEOUT:
            return None
        if status != 0 and len(self.streams) == 2:
            raise AstraInfraredError('OpenNI2 could not wait for the Astra Pro infrared stream.')
        check(self.sdk, status, 'Astra Pro depth stream stopped.')
        frame = POINTER(Frame)()
        status = self.sdk.oniStreamReadFrame(self.streams[index.value], byref(frame))
        if index.value == 1 and status != 0:
            raise AstraInfraredError('Astra Pro infrared frame could not be read.')
        check(self.sdk, status, 'Astra Pro depth frame could not be read.')
        if not frame:
            return None
        try:
            if index.value == 1:
                if on_infrared is not None:
                    on_infrared(monotonic(), frame.contents)
                return None
            self.at = monotonic()
            return copy_depth(self.np, frame.contents, output)
        finally:
            self.sdk.oniFrameRelease(frame)


def check_depth_access(bus: int, address: int) -> None:
    usb = Path(f'/dev/bus/usb/{bus:03d}/{address:03d}')
    if not usb.exists():
        raise AstraDepthError('The Astra Pro USB depth device disconnected.')
    if not access(usb, R_OK | W_OK):
        raise AstraDepthError(f'Astra Pro depth needs read/write access to {usb}. Check the Orbbec USB permissions.')


@contextmanager
def open_native_depth(np, sdk_dir: Path, bus: int, address: int):
    sdk = load_sdk(sdk_dir)
    with ExitStack() as cleanup:
        check(sdk, sdk.oniInitialize(API_VERSION), 'Orbbec OpenNI2 could not initialize.')
        cleanup.callback(sdk.oniShutdown)
        handle = c_void_p()
        check(sdk, sdk.oniDeviceOpen(depth_uri(sdk, bus, address), byref(handle)), 'OpenNI2 could not open Astra Pro depth.')
        cleanup.callback(sdk.oniDeviceClose, handle)

        stream = c_void_p()
        check(sdk, sdk.oniDeviceCreateStream(handle, DEPTH_SENSOR, byref(stream)), 'OpenNI2 could not create the Astra Pro depth stream.')
        cleanup.callback(sdk.oniStreamDestroy, stream)
        mode = VideoMode(DEPTH_1_MM, WIDTH, HEIGHT, FPS)
        check(sdk, sdk.oniStreamSetProperty(stream, VIDEO_MODE, byref(mode), sizeof(mode)), 'Astra Pro depth cannot stream at 640×480 and 30 fps.')
        mirror = c_int(0)
        check(sdk, sdk.oniStreamSetProperty(stream, MIRRORING, byref(mirror), sizeof(mirror)), 'OpenNI2 could not set the Astra Pro depth orientation.')

        if not sdk.oniDeviceIsImageRegistrationModeSupported(handle, DEPTH_TO_COLOR):
            raise AstraDepthError('This Astra Pro does not expose OpenNI2 depth-to-RGB registration.')
        registration = c_int(DEPTH_TO_COLOR)
        check(sdk, sdk.oniDeviceSetProperty(handle, IMAGE_REGISTRATION, byref(registration), sizeof(registration)), 'OpenNI2 could not set the Astra Pro depth registration mode.')
        parameters = FactoryParameters()
        size = c_int(sizeof(parameters))
        check(sdk, sdk.oniDeviceGetProperty(handle, FACTORY_PARAMETERS, byref(parameters), byref(size)), 'OpenNI2 could not read Astra Pro factory calibration.')
        if size.value != sizeof(parameters):
            raise AstraDepthError('OpenNI2 returned an unexpected Astra Pro factory calibration format.')
        check(sdk, sdk.oniStreamStart(stream), 'OpenNI2 could not start Astra Pro depth.')
        cleanup.callback(sdk.oniStreamStop, stream)

        # Stream startup can apply SDK defaults. Check the actual mode before
        # trusting these pixels against the independently captured UVC image.
        mirror.value = -1
        size = c_int(sizeof(mirror))
        check(sdk, sdk.oniStreamGetProperty(stream, MIRRORING, byref(mirror), byref(size)), 'OpenNI2 could not verify the Astra Pro depth orientation.')
        if mirror.value:
            raise AstraDepthError('Astra Pro depth remained mirrored after startup.')
        registration.value = -1
        size = c_int(sizeof(registration))
        check(sdk, sdk.oniDeviceGetProperty(handle, IMAGE_REGISTRATION, byref(registration), byref(size)), 'OpenNI2 could not verify Astra Pro depth registration.')
        if registration.value != DEPTH_TO_COLOR:
            raise AstraDepthError('Astra Pro depth registration was not active after startup.')
        yield NativeDepthCapture(np, sdk, handle, stream, parameters)


@contextmanager
def open_native_infrared(sdk, device):
    with ExitStack() as cleanup:
        stream = c_void_p()
        check(sdk, sdk.oniDeviceCreateStream(device, IR_SENSOR, byref(stream)), 'OpenNI2 could not create the Astra Pro infrared stream.')
        cleanup.callback(sdk.oniStreamDestroy, stream)
        mode = VideoMode(GRAY16, WIDTH, HEIGHT, FPS)
        check(sdk, sdk.oniStreamSetProperty(stream, VIDEO_MODE, byref(mode), sizeof(mode)), 'Astra Pro infrared cannot stream at 640×480 and 30 fps.')
        mirror = c_int(0)
        check(sdk, sdk.oniStreamSetProperty(stream, MIRRORING, byref(mirror), sizeof(mirror)), 'OpenNI2 could not set the Astra Pro infrared orientation.')
        check(sdk, sdk.oniStreamStart(stream), 'OpenNI2 could not start Astra Pro infrared.')
        cleanup.callback(sdk.oniStreamStop, stream)
        yield stream


def receive_packet(receiver: Connection, worker: Popen, timeout: float) -> bytes | None:
    try:
        if not receiver.poll(timeout):
            return None
        packet = receiver.recv_bytes(FRAME_BYTES)
    except (EOFError, OSError) as error:
        status = worker.poll()
        detail = f' (exit status {status})' if status is not None else ''
        raise AstraDepthError(f'The Astra Pro OpenNI2 process stopped{detail}.') from error
    if len(packet) != FRAME_BYTES and packet.startswith(b'E'):
        raise AstraDepthError(packet[1:].decode(errors='replace'))
    return packet


class DepthCapture:
    def __init__(self, np, projection: ColorProjection, receiver: Connection, worker: Popen,
                 on_infrared: Callable[[float, bytes], None] | None,
                 on_infrared_error: Callable[[str], None] | None) -> None:
        self.np = np
        self.deproject = projection.deproject
        self.receiver = receiver
        self.worker = worker
        self.at = 0.0
        self.infrared_enabled = False
        self.on_infrared = on_infrared
        self.on_infrared_error = on_infrared_error

    def set_infrared(self, enabled: bool) -> None:
        if self.infrared_enabled != enabled:
            try:
                self.receiver.send_bytes(b'1' if enabled else b'0')
            except OSError as error:
                raise AstraDepthError('The Astra Pro OpenNI2 process stopped.') from error
            self.infrared_enabled = enabled

    def read(self, timeout_ms: int):
        deadline = monotonic() + timeout_ms / 1000
        while True:
            packet = receive_packet(self.receiver, self.worker, max(0.0, deadline - monotonic()))
            if packet is None:
                return None
            if len(packet) == FRAME_BYTES:
                self.at, = FRAME_TIME.unpack_from(packet)
                return self.np.ndarray((HEIGHT, WIDTH), dtype=self.np.uint16, buffer=packet, offset=FRAME_TIME.size)
            if packet.startswith(b'I') and len(packet) > 1 + FRAME_TIME.size:
                if self.infrared_enabled and self.on_infrared is not None:
                    at, = FRAME_TIME.unpack_from(packet, 1)
                    self.on_infrared(at, packet[1 + FRAME_TIME.size:])
            elif packet.startswith(b'W'):
                if self.infrared_enabled and self.on_infrared_error is not None:
                    self.on_infrared_error(packet[1:].decode(errors='replace'))
            else:
                raise AstraDepthError('The Astra Pro OpenNI2 process sent an invalid depth frame.')
            if monotonic() >= deadline:
                return None


def reap_worker(worker: Popen) -> None:
    try:
        worker.wait(timeout=.15)
    except TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=DEPTH_STOP_SECONDS)
        except TimeoutExpired:
            worker.kill()
            worker.wait()


@contextmanager
def open_depth(np, cv2, sdk_dir: Path, device, *, on_infrared=None, on_infrared_error=None):
    bus, address = device.depth_bus, device.depth_address
    check_depth_access(bus, address)
    # The vendor driver can abort in its USB event thread during shutdown.
    # A dedicated process keeps RGB inference and the HTTP bridge running.
    receiver, sender = Pipe()
    with receiver, sender:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), str(sender.fileno()),
                   str(sdk_dir), str(bus), str(address)]
        try:
            worker = Popen(command, stdin=DEVNULL, pass_fds=(sender.fileno(),))
        except OSError as error:
            raise AstraDepthError(f'Could not start the Astra Pro OpenNI2 process: {error}') from error
        sender.close()
        try:
            packet = receive_packet(receiver, worker, DEPTH_START_SECONDS)
            if packet is None:
                raise AstraDepthError('The Astra Pro OpenNI2 process did not start in time.')
            if len(packet) != 1 + sizeof(FactoryParameters) or not packet.startswith(b'C'):
                raise AstraDepthError('The Astra Pro OpenNI2 process sent invalid factory calibration.')
            parameters = FactoryParameters.from_buffer_copy(packet, 1)
            yield DepthCapture(np, ColorProjection(np, cv2, parameters), receiver, worker, on_infrared, on_infrared_error)
        finally:
            receiver.close()
            reap_worker(worker)


def run_depth_worker(sender: Connection, sdk_dir: Path, bus: int, address: int) -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    with sender:
        try:
            import numpy as np

            with open_native_depth(np, sdk_dir, bus, address) as stream, ExitStack() as infrared:
                sender.send_bytes(b'C' + bytes(stream.parameters))
                packet = bytearray(FRAME_BYTES)
                pixels = np.ndarray((HEIGHT, WIDTH), dtype=np.uint16, buffer=packet, offset=FRAME_TIME.size)
                infrared_requested = False
                last_infrared_at = None
                next_preview_at = 0.0

                def close_infrared():
                    nonlocal last_infrared_at
                    stream.set_infrared(None)
                    infrared.close()
                    last_infrared_at = None

                def warn_infrared(error):
                    close_infrared()
                    sender.send_bytes(b'W' + str(error).encode())

                def preview_infrared(at, frame):
                    nonlocal last_infrared_at, next_preview_at
                    last_infrared_at = at
                    if at < next_preview_at:
                        return
                    next_preview_at = at + PREVIEW_INTERVAL
                    image = frame_pixels(np, frame, IR_SENSOR, GRAY16)
                    try:
                        gray = cv2.convertScaleAbs(image, alpha=255 / 1023)
                        cv2.LUT(gray, infrared_tone, gray)
                        ok, encoded = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    except cv2.error as error:
                        raise AstraInfraredError('Could not encode the Astra Pro infrared preview.') from error
                    if not ok or encoded.nbytes >= FRAME_BYTES - 1 - FRAME_TIME.size:
                        raise AstraInfraredError('Could not encode the Astra Pro infrared preview.')
                    sender.send_bytes(b'I' + FRAME_TIME.pack(at) + encoded.tobytes())

                while not stopping:
                    if sender.poll():
                        enabled = sender.recv_bytes(1) == b'1'
                        if infrared_requested != enabled:
                            infrared_requested = enabled
                            if enabled:
                                try:
                                    import cv2

                                    # Lift dim returns with a fixed curve so
                                    # brightness stays steady across frames.
                                    infrared_tone = np.rint(np.sqrt(np.arange(256) / 255) * 255).astype(np.uint8)
                                    ir_stream = infrared.enter_context(open_native_infrared(stream.sdk, stream.device))
                                    stream.set_infrared(ir_stream)
                                    last_infrared_at = monotonic()
                                    next_preview_at = 0.0
                                except (AstraDepthError, ImportError) as error:
                                    warn_infrared(error)
                            else:
                                close_infrared()
                    try:
                        if stream.read(100, pixels, on_infrared=preview_infrared) is not None:
                            FRAME_TIME.pack_into(packet, 0, stream.at)
                            sender.send_bytes(packet)
                        if last_infrared_at is not None and monotonic() - last_infrared_at > IR_START_SECONDS:
                            warn_infrared('Astra Pro infrared frames stopped.')
                    except AstraInfraredError as error:
                        warn_infrared(error)
        except AstraDepthError as error:
            if not stopping:
                try:
                    sender.send_bytes(b'E' + str(error).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass
        except (BrokenPipeError, ConnectionResetError, EOFError):
            pass


if __name__ == '__main__':
    fd, sdk_dir, bus, address = sys.argv[1:]
    run_depth_worker(Connection(int(fd)), Path(sdk_dir), int(bus), int(address))
