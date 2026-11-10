from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from astra_rgb import AstraRgbError, discover_astra, open_rgb


class DiscoveryTests(TestCase):
    def setUp(self):
        directory = self.enterContext(TemporaryDirectory())
        self.sysfs = Path(directory) / 'sys'
        self.dev = Path(directory) / 'dev'
        self.usb = self.sysfs / 'bus/usb/devices'
        self.video = self.sysfs / 'class/video4linux'
        self.usb.mkdir(parents=True)
        self.video.mkdir(parents=True)
        self.dev.mkdir()

    def add_usb(self, hub, suffix, product, address, serial=None):
        name = f'{hub}.{suffix}'
        device = self.sysfs / 'devices/usb3' / hub / name
        device.mkdir(parents=True)
        (self.usb / name).symlink_to(device)
        for key, value in [('idVendor', '2bc5'), ('idProduct', product), ('busnum', '3'), ('devnum', address)]:
            (device / key).write_text(f'{value}\n')
        if serial is not None:
            (device / 'serial').write_text(f'{serial}\n')
        return device

    def add_video(self, usb, number, index):
        name = f'video{number}'
        entry = usb / f'{usb.name}:1.0/video4linux' / name
        entry.mkdir(parents=True)
        (entry / 'index').write_text(f'{index}\n')
        (self.video / name).symlink_to(entry)
        path = self.dev / name
        path.touch()
        return path

    def discover(self, **options):
        return discover_astra(sysfs=self.sysfs, dev_root=self.dev, **options)

    def test_pairs_image_node_only_to_depth_on_the_same_hub(self):
        color = self.add_usb('3-1', 1, '0501', 5)
        depth = self.add_usb('3-1', 2, '0403', 6, 'serial-1')
        self.add_usb('3-2', 1, '0403', 9)
        image = self.add_video(color, 5, 0)
        self.add_video(color, 4, 1)
        result = self.discover()
        self.assertEqual((result.rgb_device, result.depth_bus, result.depth_address), (image, 3, 6))
        self.assertEqual((result.usb_parent, result.identity), ('3-1', 'astra:serial-1'))
        (depth / 'devnum').write_text('14\n')
        self.assertEqual(self.discover().identity, result.identity)
        self.assertEqual(self.discover().depth_address, 14)

    def test_multiple_units_require_an_exact_hub(self):
        first = self.add_usb('3-1', 1, '0501', 5)
        second = self.add_usb('3-2', 1, '0501', 8)
        self.add_usb('3-1', 2, '0403', 6)
        self.add_usb('3-2', 2, '0403', 9)
        self.add_video(first, 4, 0)
        video = self.add_video(second, 7, 0)
        with self.assertRaisesRegex(AstraRgbError, 'Multiple Astra Pro'):
            self.discover()
        result = self.discover(usb_parent='3-2')
        self.assertEqual((result.rgb_device, result.depth_address, result.identity), (video, 9, 'astra:3-2'))
        with self.assertRaisesRegex(AstraRgbError, 'No paired Astra Pro'):
            self.discover(usb_parent='3-3')


class CaptureTests(TestCase):
    def setUp(self):
        self.enterContext(patch.dict('os.environ'))
        self.capture = Mock()
        self.cv2 = SimpleNamespace(
            error=type('CvError', (RuntimeError,), {}), CAP_V4L2=1, CAP_PROP_FOURCC=2,
            CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, CAP_PROP_FPS=5, COLOR_BGR2RGB=6,
            VideoCapture=Mock(return_value=self.capture), VideoWriter_fourcc=Mock(return_value=99),
            cvtColor=Mock(side_effect=lambda image, _: image.name), flip=Mock(),
        )
        self.capture.get.side_effect = {2: 99, 3: 640, 4: 480, 5: 30}.__getitem__
        self.frames = Queue()

        def read():
            try:
                return self.frames.get(timeout=1)
            except Empty:
                return False, None

        self.capture.read.side_effect = read

    def test_latest_frame_only_and_release(self):
        for name in ('first', 'second', 'third'):
            self.frames.put((True, SimpleNamespace(name=name, shape=(480, 640, 3))))
        before = monotonic()
        # Wait until a fourth read starts, which means all three frames have been published.
        read = self.capture.read.side_effect
        final_read = Event()

        def after_three_frames():
            if self.capture.read.call_count == 4:
                final_read.set()
            return read()

        self.capture.read.side_effect = after_three_frames
        with open_rgb(self.cv2, '/dev/video4') as queue:
            self.assertTrue(final_read.wait(1))
            received, frame = queue.try_wait_for_frame(100)
            self.assertTrue(received)
            self.assertEqual((frame.get_data(), frame.number), ('third', 3))
            self.assertLessEqual(before, frame.at)
            self.assertLessEqual(frame.at, monotonic())
            self.assertEqual(queue.try_wait_for_frame(1), (False, None))
        self.capture.open.assert_called_once_with('/dev/video4', self.cv2.CAP_V4L2)
        self.capture.release.assert_called_once()
        self.assertEqual(queue.try_wait_for_frame(0), (False, None))

    def test_read_failure_is_reported_and_capture_released(self):
        self.frames.put((False, None))
        with open_rgb(self.cv2, '/dev/video4') as queue:
            with self.assertRaisesRegex(AstraRgbError, 'RGB frames stopped'):
                queue.try_wait_for_frame(1000)
        self.capture.release.assert_called_once()

    def test_color_is_unmirrored_before_publishing(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('OpenCV and NumPy are required to check RGB orientation')
        self.cv2.cvtColor.side_effect = cv2.cvtColor
        self.cv2.flip.side_effect = cv2.flip
        self.cv2.COLOR_BGR2RGB = cv2.COLOR_BGR2RGB
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr[100, 40] = (30, 20, 10)
        self.frames.put((True, bgr))
        with open_rgb(self.cv2, '/dev/video4') as queue:
            received, frame = queue.try_wait_for_frame(1000)
            self.frames.put((False, None))
            self.assertTrue(received)
            self.assertEqual(tuple(frame.data[100, 599]), (10, 20, 30))
            self.assertEqual(tuple(frame.data[100, 40]), (0, 0, 0))

    def test_wrong_negotiated_mode_is_rejected_before_reading(self):
        self.capture.get.side_effect = {2: 99, 3: 640, 4: 480, 5: 15}.__getitem__
        with self.assertRaisesRegex(AstraRgbError, '640×480 MJPEG at 30 fps'):
            with open_rgb(self.cv2, '/dev/video4'):
                self.fail('Opened unsupported capture mode')
        self.capture.read.assert_not_called()
        self.capture.release.assert_called_once()
