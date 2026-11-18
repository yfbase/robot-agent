"""Tools the model can call. Currently: camera capture.

The camera tool returns the photo as ``BinaryContent``; pydantic-ai sends it
back to the model as the tool result, so a vision-capable model can see the
image. Requires a vision model (e.g. deepseek-v4-flash) and opencv-python.
"""

import sys

import cv2
from pydantic_ai import BinaryContent


def capture_photo(camera_index: int = 0) -> BinaryContent | str:
    """Take a photo with a local camera and return the image.

    Args:
        camera_index: Which camera to use; 0 is the default camera.
    """
    # DirectShow opens much faster than the default backend on Windows.
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_index, backend)
    try:
        if not cap.isOpened():
            return f"Camera {camera_index} could not be opened. Is it connected and free?"
        # Skip a few warm-up frames so auto-exposure settles.
        for _ in range(5):
            cap.read()
        ok, frame = cap.read()
        if not ok:
            return f"Camera {camera_index} opened but failed to capture a frame."
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return "Failed to encode the photo as JPEG."
        return BinaryContent(data=buffer.tobytes(), media_type="image/jpeg")
    finally:
        cap.release()
