"""Short-lived grayscale previews from the two RealSense infrared imagers."""

from threading import Condition, Lock, Thread
from time import monotonic
import logging


INFRARED_INDEXES = (1, 2)
VIEWER_LEASE = 1.5
FRAME_MAX_AGE = 0.65
PREVIEW_INTERVAL = 1 / 8


class InfraredPreview:
    def __init__(self) -> None:
        self.lock = Lock()
        self.until = float('-inf')
        self.frames: dict[int, tuple[float, bytes]] = {}
        self.error: str | None = None

    def request(self, index: int) -> bytes | None:
        with self.lock:
            now = monotonic()
            self.until = now + VIEWER_LEASE
            frame = self.frames.get(index)
            return frame[1] if frame and now - frame[0] < FRAME_MAX_AGE else None

    def wanted(self) -> bool:
        with self.lock:
            if monotonic() < self.until:
                return True
            self.frames.clear()
            self.error = None
            return False

    def status(self) -> str | None:
        with self.lock:
            return self.error if monotonic() < self.until else None

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message
            self.frames.clear()

    def clear_frames(self) -> None:
        with self.lock:
            self.frames.clear()

    def fresh_pair(self, max_age: float) -> bool:
        with self.lock:
            now = monotonic()
            return all(index in self.frames and now - self.frames[index][0] < max_age
                       for index in INFRARED_INDEXES)

    def publish(self, index: int, at: float, jpeg: bytes) -> None:
        with self.lock:
            if monotonic() < self.until and not self.error:
                self.frames[index] = (at, jpeg)


class InfraredEncoder:
    def __init__(self, cv2, np, publish) -> None:
        self.cv2 = cv2
        self.np = np
        self.publish = publish
        self.ready = Condition()
        self.pending = {}
        self.next_frame_at: dict[int, float] = {}
        self.closed = False
        self.thread = Thread(target=self.run, name='D435i infrared preview', daemon=True)
        self.thread.start()

    def submit(self, index: int, frame) -> None:
        at = monotonic()
        with self.ready:
            if self.closed or at < self.next_frame_at.get(index, 0):
                return
            self.next_frame_at[index] = at + PREVIEW_INTERVAL
        # The SDK releases its frame buffer when the sensor callback returns.
        image = self.np.asanyarray(frame.get_data()).copy()
        with self.ready:
            if not self.closed:
                self.pending[index] = (at, image)
                self.ready.notify()

    def run(self) -> None:
        cv2 = self.cv2
        while True:
            with self.ready:
                self.ready.wait_for(lambda: self.closed or self.pending)
                if self.closed:
                    return
                pending, self.pending = self.pending, {}
            for index, (at, image) in pending.items():
                try:
                    height, width = image.shape[:2]
                    if width > 640:
                        image = cv2.resize(image, (640, round(height * 640 / width)), interpolation=cv2.INTER_AREA)
                    ok, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if ok:
                        with self.ready:
                            if not self.closed:
                                self.publish(index, at, jpeg.tobytes())
                except Exception:
                    logging.debug('D435i could not encode infrared preview %d', index, exc_info=True)

    def close(self) -> None:
        with self.ready:
            self.closed = True
            self.pending.clear()
            self.ready.notify()
        self.thread.join(timeout=1)
