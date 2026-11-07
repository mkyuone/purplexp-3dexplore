from pathlib import Path
from json import loads
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from options import CameraOptions, load_options, save_options
from realsense import CameraFeed, Frame, enable_depth_projector
from server import CameraService, Profiles
from test_preview import StubFeed, request


class SettingsFeed(StubFeed):
    def __init__(self):
        super().__init__()
        self.options = CameraOptions()
        self.changes = 0
        self.model_available = True

    def configure(self, options):
        self.options = options
        self.changes += 1

    def settings(self):
        tracker = {'value': 'mediapipe', 'label': 'MediaPipe Full', 'available': self.model_available}
        if not self.model_available:
            tracker['reason'] = 'Install the MediaPipe Full model.'
        return {
            **self.options.to_dict(), 'applying': self.changes > 0,
            'sources': [{'value': 'rgb', 'label': 'RGB'}],
            'trackers': [tracker], 'frameRates': [30, 60],
        }


class CameraSettingsTests(TestCase):
    def test_depth_projector_is_enabled_even_if_it_was_previously_off(self):
        for initial in (0, 1):
            with self.subTest(initial=initial):
                value = [initial]
                sensor = SimpleNamespace(
                    supports=lambda option: True,
                    get_option=lambda option: value[0],
                    set_option=lambda option, selected: value.__setitem__(0, selected),
                )
                profile = SimpleNamespace(get_device=lambda: SimpleNamespace(first_depth_sensor=lambda: sensor))
                rs = SimpleNamespace(option=SimpleNamespace(emitter_enabled='emitter'))
                enable_depth_projector(rs, profile)
                self.assertEqual(value[0], 1)

    def test_settings_are_saved_and_invalid_old_records_fall_back(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            self.assertEqual(load_options(path), CameraOptions())
            options = CameraOptions('rgb', 'mediapipe', 60)
            save_options(path, options)
            self.assertEqual(load_options(path), options)
            path.write_text('{"source":"ir1","tracker":"mercury","fps":60}')
            self.assertEqual(load_options(path), CameraOptions('rgb', 'mediapipe', 60))
            path.write_text('{"source":"rgb","tracker":"mediapipe-lite","fps":60}')
            self.assertEqual(load_options(path), CameraOptions('rgb', 'mediapipe', 60))
            path.write_text('{"source":"rgb","tracker":"rtmpose","fps":60}')
            self.assertEqual(load_options(path), CameraOptions('rgb', 'mediapipe', 60))
            path.write_text('{"source":"ir1","tracker":"mercury","fps":true}')
            self.assertEqual(load_options(path), CameraOptions())

    def test_unavailable_tracker_cannot_replace_the_current_saved_configuration(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            service = CameraService(SettingsFeed(), Profiles(Path(directory) / 'profiles.json'), path)
            selected = CameraOptions('rgb', 'mediapipe', 60)
            service.settings(selected.to_dict())
            service.feed.model_available = False
            with self.assertRaisesRegex(ValueError, 'Install the MediaPipe Full'):
                service.settings(CameraOptions('rgb', 'mediapipe', 30).to_dict())
            self.assertEqual(service.feed.changes, 1)
            self.assertEqual(load_options(path), selected)
            self.assertEqual(service.settings()['source'], 'rgb')
            with self.assertRaisesRegex(ValueError, 'Choose RGB'):
                service.settings({'source': 'ir1', 'tracker': 'mediapipe', 'fps': 60})
            self.assertEqual(service.feed.changes, 1)

    def test_settings_change_keeps_its_origin_boundary_and_rejects_nonobject_json(self):
        with TemporaryDirectory() as directory:
            service = CameraService(SettingsFeed(), Profiles(Path(directory) / 'profiles.json'), Path(directory) / 'settings.json')
            status, _, result = request(service, 'GET', '/settings')
            result = loads(result)
            self.assertEqual((status, result['source']), (200, 'rgb'))
            status, _, _ = request(service, 'POST', '/settings', payload=CameraOptions('rgb', 'mediapipe', 60).to_dict(), origin='https://example.com')
            self.assertEqual(status, 403)
            self.assertEqual(service.feed.changes, 0)
            status, _, _ = request(service, 'POST', '/settings', payload=['ir1'])
            self.assertEqual(status, 400)
            for tracker in ('mediapipe-lite', 'rtmpose'):
                payload = {'source': 'rgb', 'tracker': tracker, 'fps': 60}
                status, _, result = request(service, 'POST', '/settings', payload=payload)
                self.assertEqual(status, 400)
                self.assertIn('supported hand tracker', loads(result)['message'])
            self.assertEqual(service.feed.changes, 0)
            status, _, result = request(service, 'POST', '/settings', payload=CameraOptions('rgb', 'mediapipe', 60).to_dict())
            result = loads(result)
            self.assertEqual((status, result['source'], result['applying']), (200, 'rgb', True))

    def test_an_old_capture_cannot_publish_after_a_source_change(self):
        with patch('realsense.Thread.start'):
            feed = CameraFeed(Path('unused-model'))
        feed.configure(CameraOptions('rgb', 'mediapipe', 60))
        feed.publish(Frame(1, 10, 'lost', ()), revision=0)
        self.assertIsNone(feed.frame)
        feed.publish(Frame(2, 10.1, 'lost', (), processing_ms=8), revision=1)
        self.assertEqual(feed.frame.sequence, 2)
        self.assertEqual(len(feed.frame_times), 1)
