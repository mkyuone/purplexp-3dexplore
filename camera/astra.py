"""Use the Astra Pro's independent UVC color and OpenNI2 depth streams."""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
import logging

from astra_depth import AstraDepthError, open_depth
from astra_rgb import AstraRgbError, discover_astra, open_rgb
from estimators import create_estimator
from infrared_preview import InfraredPreview
from options import CameraOptions
from realsense import CameraError, CameraFeed, CameraSample, DEPTH_QUEUE_SIZE, DEPTH_RETRY_DELAY, ImageProjector, RGB_SIZE, STREAM_TIMEOUT

DEFAULT_OPENNI2_DIR = Path(__file__).resolve().parents[1] / '.runtime' / 'camera' / 'openni2' / 'sdk' / 'libs'
PAIR_WINDOW_SECONDS = 0.025


@dataclass(frozen=True)
class DepthSample:
    at: float
    pixels: object
    deproject: ImageProjector


@dataclass(frozen=True)
class RegisteredDepth:
    pixels: object

    def get_data(self):
        return self.pixels

    def get_units(self) -> float:
        return .001


class AstraDepthStream:
    def __init__(self, np, cv2, sdk_dir, device, on_profile, on_error, infrared: InfraredPreview) -> None:
        self.stop = Event()
        self.lock = Lock()
        self.pending: deque[DepthSample] = deque(maxlen=DEPTH_QUEUE_SIZE)
        self.thread = Thread(target=self.run, args=(np, cv2, sdk_dir, device, on_profile, on_error, infrared),
                             name='Astra Pro depth', daemon=True)
        self.thread.start()

    def run(self, np, cv2, sdk_dir, device, on_profile, on_error, infrared: InfraredPreview) -> None:
        last_error = None
        usb_parent = device.usb_parent

        def publish_infrared(at: float, jpeg: bytes) -> None:
            infrared.publish(1, at, jpeg)

        while not self.stop.is_set():
            try:
                with open_depth(np, cv2, sdk_dir, device, on_infrared=publish_infrared,
                                on_infrared_error=infrared.fail) as stream:
                    last_frame_at = monotonic()
                    have_frame = False
                    infrared_enabled = False
                    next_infrared_check = 0.0
                    try:
                        while not self.stop.is_set():
                            now = monotonic()
                            if now >= next_infrared_check:
                                next_infrared_check = now + .25
                                requested = infrared.wanted()
                                if requested != infrared_enabled:
                                    stream.set_infrared(requested)
                                    infrared_enabled = requested
                            pixels = stream.read(100)
                            now = monotonic()
                            if pixels is None:
                                if now - last_frame_at > STREAM_TIMEOUT:
                                    raise AstraDepthError('Astra Pro depth frames stopped.')
                                continue
                            last_frame_at = now
                            if not have_frame:
                                on_profile(stream.deproject)
                                on_error(None)
                                have_frame = True
                                last_error = None
                            with self.lock:
                                self.pending.append(DepthSample(stream.at, pixels, stream.deproject))
                    finally:
                        with self.lock:
                            self.pending.clear()
                        infrared.clear_frames()
            except Exception as error:
                if self.stop.is_set():
                    return
                expected = isinstance(error, (AstraDepthError, AstraRgbError))
                message = str(error) if expected else 'Astra Pro depth stopped. Check the camera bridge terminal.'
                if message != last_error:
                    if expected:
                        logging.warning('%s RGB tracking continues.', message)
                    else:
                        logging.exception('Astra Pro depth stopped; RGB tracking continues.')
                    on_error(message)
                    last_error = message
                self.stop.wait(DEPTH_RETRY_DELAY)
                if self.stop.is_set():
                    return
                try:
                    device = discover_astra(usb_parent=usb_parent)
                except AstraRgbError:
                    # The USB depth child can re-enumerate while RGB stays open.
                    # Keep retrying the same hub until its depth device returns.
                    pass

    def match(self, color: CameraSample):
        # Both streams use host dequeue times. Their USB clocks are independent.
        at = color.captured_at
        if at is None:
            return None, None
        with self.lock:
            while self.pending and self.pending[0].at < at - PAIR_WINDOW_SECONDS:
                self.pending.popleft()
            if self.pending:
                index = min(range(len(self.pending)), key=lambda i: abs(self.pending[i].at - at))
                if abs(self.pending[index].at - at) <= PAIR_WINDOW_SECONDS:
                    for _ in range(index):
                        self.pending.popleft()
                    depth = self.pending.popleft()
                    return depth, depth.deproject
        return None, None

    def close(self) -> None:
        self.stop.set()
        self.thread.join()


def read_color(frame) -> CameraSample:
    return CameraSample(frame, frame.number, frame.at * 1000, 'hostMonotonic', None, captured_at=frame.at)


def align_depth(_color, depth: DepthSample):
    # OpenNI registers depth in the same orientation as the corrected RGB.
    return RegisteredDepth(depth.pixels)


class AstraFeed(CameraFeed):
    camera_name = 'Orbbec Astra Pro'
    frame_rates = (30,)
    infrared_count = 1

    def __init__(self, model: Path, options: CameraOptions, sdk_dir: Path | None, usb_port: str | None) -> None:
        self.sdk_dir = sdk_dir.expanduser() if sdk_dir else DEFAULT_OPENNI2_DIR
        self.usb_port = usb_port
        super().__init__(model, options=replace(options, fps=30))

    def capture(self, revision: int) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as error:
            raise CameraError('Install the packages from camera/requirements.txt in the camera bridge Python environment.') from error
        with self.lock:
            options = self.options
        try:
            device = discover_astra(usb_parent=self.usb_port)
            with ExitStack() as resources:
                try:
                    estimator = resources.enter_context(create_estimator(options.tracker, self.model))
                except (RuntimeError, ValueError) as error:
                    raise CameraError(str(error)) from error
                if not self.wanted(revision):
                    return
                color = resources.enter_context(open_rgb(cv2, device))
                session = self.begin_session(revision, options, device.identity)
                if session is None:
                    return

                def on_profile(deproject):
                    width, height = RGB_SIZE
                    self.set_depth_profile(
                        session, f'{device.identity}:{width}x{height}:30:factory-color-unmirrored',
                        lambda x, y, z: deproject(x * width, y * height, z),
                    )

                def on_error(message):
                    with self.lock:
                        if self.info is not None and self.info.session == session:
                            self.depth_error = message

                depth = AstraDepthStream(np, cv2, self.sdk_dir, device, on_profile, on_error, self.infrared)
                resources.callback(depth.close)
                preview, diagnostic_rgb = self.open_previews(resources, cv2, session)
                self.track_frames(None, np, estimator, color, depth, align_depth, revision, preview, diagnostic_rgb,
                                  read_color=read_color)
        except AstraRgbError as error:
            raise CameraError(str(error)) from error

    def infrared_status(self) -> str | None:
        preview_error = super().infrared_status()
        with self.lock:
            return self.depth_error or preview_error
