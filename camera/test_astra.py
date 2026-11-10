from collections import deque
from contextlib import contextmanager
from threading import Event, Lock
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

from astra import AstraDepthStream, DepthSample, align_depth, read_color


class AstraStreamTests(TestCase):
    def test_only_requests_infrared_during_the_viewer_lease(self):
        stream = object.__new__(AstraDepthStream)
        stream.stop = Event()
        stream.lock = Lock()
        stream.pending = deque()
        clock = [1.0]
        infrared = Mock(wanted=Mock(side_effect=[False, True, True, False]))
        worker = Mock()

        def read(_timeout):
            if clock[0] >= 1.75:
                stream.stop.set()
            clock[0] += .25
            return None

        worker.read.side_effect = read

        @contextmanager
        def open_capture(*_args, on_infrared, on_infrared_error):
            self.assertEqual(on_infrared_error, infrared.fail)
            on_infrared(1.5, b'jpeg')
            yield worker

        with patch('astra.open_depth', open_capture), patch('astra.monotonic', side_effect=lambda: clock[0]):
            stream.run(None, None, None, SimpleNamespace(usb_parent='3-1'), Mock(), Mock(), infrared)
        self.assertEqual([call.args for call in worker.set_infrared.call_args_list], [(True,), (False,)])
        infrared.publish.assert_called_once_with(1, 1.5, b'jpeg')
        infrared.clear_frames.assert_called_once()

    def test_pairs_the_nearest_depth_once_and_ignores_stale_depth(self):
        stream = object.__new__(AstraDepthStream)
        stream.lock = Lock()
        project = lambda x, y, z: (x, y, z)

        def depth(at, pixels):
            return DepthSample(at, pixels, project)

        def color(at):
            return read_color(SimpleNamespace(at=at, number=1))

        stream.pending = deque((depth(.90, 1), depth(.98, 2), depth(1.01, 3), depth(1.06, 4)))
        paired, deproject = stream.match(color(1.0))
        self.assertIs(deproject, project)
        self.assertEqual(align_depth(None, paired).get_data(), 3)
        self.assertEqual(stream.match(color(1.02)), (None, None))
        self.assertEqual(stream.match(color(1.05))[0].pixels, 4)
        self.assertEqual(stream.match(color(1.06)), (None, None))

        stream.pending.append(depth(1.50, 5))
        self.assertEqual(stream.match(color(1.52))[0].pixels, 5)
