"""Offline camera-tool tests: cv2.VideoCapture is mocked, no hardware needed."""

import numpy as np
import pytest
from pydantic_ai import BinaryContent

from robot_agent import tools


class FakeCapture:
    def __init__(self, index, backend=None, *, opened=True, read_ok=True):
        self.index = index
        self.opened = opened
        self.read_ok = read_ok
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.read_ok:
            return False, None
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.fixture
def fake_camera(monkeypatch):
    """Patch VideoCapture; returns (created_caps, options) so tests can
    control behavior via options, e.g. options["opened"] = False."""
    created = []
    options = {}

    def factory(index, backend=None):
        cap = FakeCapture(index, backend, **options)
        created.append(cap)
        return cap

    monkeypatch.setattr(tools.cv2, "VideoCapture", factory)
    return created, options


def test_capture_returns_jpeg_image(fake_camera):
    created, _ = fake_camera
    result = tools.capture_photo()
    assert isinstance(result, BinaryContent)
    assert result.media_type == "image/jpeg"
    assert result.data.startswith(b"\xff\xd8\xff")  # JPEG magic bytes
    assert created[0].released


def test_camera_not_opened(fake_camera):
    created, options = fake_camera
    options["opened"] = False
    result = tools.capture_photo(1)
    assert isinstance(result, str)
    assert "could not be opened" in result
    assert created[0].released


def test_capture_failure_releases_camera(fake_camera):
    created, options = fake_camera
    options["read_ok"] = False
    result = tools.capture_photo()
    assert isinstance(result, str)
    assert "failed to capture" in result
    assert created[0].released
