from ctypes import POINTER, addressof, c_int, c_uint16, cast
from pathlib import Path
from subprocess import Popen
from time import monotonic
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from astra_depth import (
    AstraDepthError, ColorProjection, DEPTH_1_MM, DEPTH_SENSOR, DeviceInfo,
    FPS, FRAME_BYTES, FactoryParameters, Frame, HEIGHT, VideoMode, WIDTH,
    copy_depth, depth_uri, open_depth,
)


class AstraDepthTests(TestCase):
    def test_device_list_uses_exact_paired_usb_address(self):
        devices = (DeviceInfo * 3)()
        devices[0].uri = b'2bc5/0403@3/12'
        devices[1].uri = b'2bc5/0501@3/14'
        devices[2].uri = b'2bc5/0403@3/14'
        sdk = Mock()

        def list_devices(output, count):
            cast(output, POINTER(POINTER(DeviceInfo)))[0] = cast(devices, POINTER(DeviceInfo))
            cast(count, POINTER(c_int))[0] = len(devices)
            return 0

        sdk.oniGetDeviceList.side_effect = list_devices
        self.assertEqual(depth_uri(sdk, 3, 14), b'2bc5/0403@3/14')
        with self.assertRaisesRegex(AstraDepthError, 'paired'):
            depth_uri(sdk, 4, 14)
        self.assertEqual(sdk.oniReleaseDeviceList.call_count, 2)

    def test_copy_depth_uses_stride_and_detaches_from_sdk_buffer(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest('NumPy is required to read depth frames')
        stride = WIDTH + 2
        data = (c_uint16 * (stride * HEIGHT))()
        data[WIDTH - 1] = 710
        data[stride] = 920
        frame = Frame(len(data) * 2, addressof(data), DEPTH_SENSOR, 0, 1,
                      WIDTH, HEIGHT, VideoMode(DEPTH_1_MM, WIDTH, HEIGHT, FPS),
                      0, 0, 0, stride * 2)
        depth = copy_depth(np, frame)
        data[stride] = 0
        self.assertEqual(depth.dtype, np.uint16)
        self.assertEqual(depth[0, WIDTH - 1], 710)
        self.assertEqual(depth[1, 0], 920)
        frame.size = frame.stride
        with self.assertRaisesRegex(AstraDepthError, 'unexpected'):
            copy_depth(np, frame)

    def test_projection_uses_factory_camera_center_and_focal_length(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('OpenCV and NumPy are required for projection')
        parameters = FactoryParameters()
        with self.assertRaisesRegex(AstraDepthError, 'factory RGB intrinsics'):
            ColorProjection(np, cv2, parameters)
        parameters.color_intrinsics[:] = 500, 400, 300, 200
        project = ColorProjection(np, cv2, parameters).deproject
        self.assertEqual(project(300, 200, 1.0), (0.0, 0.0, 1.0))
        self.assertEqual(project(400, 280, 1.0), (0.2, 0.2, 1.0))
        self.assertIsNone(project(400, 280, 0.1))

        parameters.color_distortion[:] = .04, .39, -1.36, .003, -.001
        projection = ColorProjection(np, cv2, parameters)
        np.testing.assert_allclose(projection.distortion, (.04, .39, .003, -.001, -1.36))

    def test_worker_passes_depth_and_timestamp_and_can_fail_without_the_bridge(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('OpenCV and NumPy are required for Astra Pro depth')
        parameters = FactoryParameters()
        parameters.color_intrinsics[:] = 500, 400, 300, 200
        worker = f'''
from multiprocessing.connection import Connection
from struct import pack_into
from time import monotonic
import os, sys
sender = Connection(int(sys.argv[1]), readable=False)
sender.send_bytes({b'C' + bytes(parameters)!r})
frame = bytearray({FRAME_BYTES})
pack_into('=dH', frame, 0, monotonic(), 700)
sender.send_bytes(frame)
os._exit(17)
'''
        processes = []

        def start(command, **options):
            process = Popen([command[0], '-c', worker, *command[3:]], **options)
            processes.append(process)
            return process

        device = SimpleNamespace(depth_bus=3, depth_address=14)
        started = monotonic()
        with patch('astra_depth.check_depth_access'), patch('astra_depth.Popen', side_effect=start):
            with open_depth(np, cv2, Path('.'), device) as capture:
                depth = capture.read(1000)
                self.assertEqual(depth[0, 0], 700)
                self.assertEqual(depth[0, WIDTH - 1], 0)
                self.assertGreaterEqual(capture.at, started)
                self.assertLessEqual(capture.at, monotonic())
                self.assertEqual(capture.deproject(300, 200, 1.0), (0, 0, 1.0))
                with self.assertRaisesRegex(AstraDepthError, 'OpenNI2 process stopped'):
                    capture.read(1000)
        self.assertEqual(processes[0].returncode, 17)

    def test_infrared_worker_messages_and_toggles_preserve_depth_delivery(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('OpenCV and NumPy are required for Astra Pro depth')
        parameters = FactoryParameters()
        parameters.color_intrinsics[:] = 500, 400, 300, 200
        worker = f'''
from multiprocessing.connection import Connection
from struct import pack, pack_into
from time import monotonic
import sys
socket = Connection(int(sys.argv[1]))
socket.send_bytes({b'C' + bytes(parameters)!r})
frame = bytearray({FRAME_BYTES})
assert socket.recv_bytes() == b'1'
socket.send_bytes(b'WInfrared temporarily unavailable')
pack_into('=dH', frame, 0, monotonic(), 700)
socket.send_bytes(frame)
assert socket.recv_bytes() == b'0'
assert socket.recv_bytes() == b'1'
socket.send_bytes(b'I' + pack('=d', monotonic()) + b'preview')
pack_into('=dH', frame, 0, monotonic(), 800)
socket.send_bytes(frame)
'''

        def start(command, **options):
            return Popen([command[0], '-c', worker, *command[3:]], **options)

        device = SimpleNamespace(depth_bus=3, depth_address=14)
        preview = Mock()
        error = Mock()
        with patch('astra_depth.check_depth_access'), patch('astra_depth.Popen', side_effect=start):
            with open_depth(np, cv2, Path('.'), device, on_infrared=preview, on_infrared_error=error) as capture:
                capture.set_infrared(True)
                capture.set_infrared(True)
                self.assertEqual(capture.read(1000)[0, 0], 700)
                error.assert_called_once_with('Infrared temporarily unavailable')
                preview.assert_not_called()

                capture.set_infrared(False)
                capture.set_infrared(True)
                self.assertEqual(capture.read(1000)[0, 0], 800)
                preview.assert_called_once()
                at, image = preview.call_args.args
                self.assertLessEqual(at, capture.at)
                self.assertEqual(image, b'preview')
