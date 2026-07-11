import logging
import threading
import time
from typing import Optional, Union, Tuple
import cv2
import numpy as np

from core.exceptions import CameraError
from services.cv_processor import OpenCVProcessor
from services.insightface_service import InsightFaceService

logger = logging.getLogger(__name__)


class CameraManager:
    """
    Facade and Adapter for OpenCV Camera operations.
    Runs frame capture in a background thread to prevent blocking the async event loop.
    """

    def __init__(self, source: Union[int, str] = 0, target_size: Tuple[int, int] = (640, 480)) -> None:
        """
        Initializes the Camera Manager.

        Args:
            source: Integer camera index or string path to video file.
            target_size: Desired resolution as a (width, height) tuple.
        """
        # If source is a string representing a number, convert to int
        if isinstance(source, str) and source.isdigit():
            self._source: Union[int, str] = int(source)
        else:
            self._source = source

        self._target_width, self._target_height = target_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None

        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """
        Opens the camera stream and starts the background thread.
        Raises CameraError if the camera cannot be opened.
        """
        logger.info("Opening camera source: %s", self._source)

        with self._lock:
            if self._running:
                logger.warning("Camera already running.")
                return

            self._cap = cv2.VideoCapture(self._source)
            if not self._cap or not self._cap.isOpened():
                self._cap = None
                raise CameraError(f"Failed to open camera source: {self._source}")

            # Try setting resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_height)

            # Confirm dimensions
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("Camera opened successfully. Resolution configured to %dx%d (requested %dx%d)",
                        actual_w, actual_h, self._target_width, self._target_height)

            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, name="CameraCaptureThread", daemon=True)
            self._thread.start()

    def close(self) -> None:
        """Closes the camera and stops the background thread."""
        logger.info("Closing camera stream.")
        self._running = False

        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._cap:
                self._cap.release()
                self._cap = None
            self._latest_frame = None

        logger.info("Camera stream closed.")

        def register_callback(self, callback_func, interval_seconds: float = 0.0) -> None:
            with self._lock:
                self._frame_callback = callback_func
                self._callback_interval = interval_seconds
                self._last_callback_time = 0.0  # מאפס את הטיימר כדי שהפריים הבא יתקבל מיד
                logger.info("New camera callback registered (interval: %s sec)", interval_seconds)

        def unregister_callback(self) -> None:
            with self._lock:
                self._frame_callback = None
                logger.info("Camera callback unregistered.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the latest frame retrieved by the background thread.
        Returns a tuple: (success, frame) where frame is a NumPy array in BGR format.
        """
        with self._lock:
            if not self._running or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def _capture_loop(self) -> None:
        """Background thread target to constantly fetch frames from OpenCV."""
        consecutive_failures = 0
        max_failures = 30  # Allow some tolerance, especially for remote streams or device lag

        while self._running:
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if ret:
                consecutive_failures = 0
                with self._lock:
                    self._latest_frame = frame



            else:
                consecutive_failures += 1
                logger.warning("Camera failed to read frame. Consecutive failures: %d", consecutive_failures)
                if consecutive_failures >= max_failures:
                    logger.error("Camera connection lost permanently after %d retries.", consecutive_failures)
                    self._running = False
                    break
                time.sleep(0.01)  # Brief wait before retry

            # Throttle slightly to not peg CPU if read is instant (e.g. static video files)
            time.sleep(0.005)
