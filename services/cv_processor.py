import logging
from typing import Tuple, List, Dict, Any, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpenCVProcessor:
    """
    Facade for OpenCV image manipulations and visualization utilities.
    Ensures consistent rendering of bounding boxes, text labels, landmarks, and skeleton joints.
    """

    @staticmethod
    def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
        """Converts BGR frame (OpenCV format) to RGB."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
        """Converts RGB frame to BGR (OpenCV format)."""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resizes an image to specified width and height."""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def to_jpeg(image: np.ndarray, quality: int = 80) -> bytes:
        """Encodes an image to a JPEG byte string, suitable for UI/Websocket streaming."""
        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise ValueError("Failed to encode frame to JPEG.")
        return encoded.tobytes()

    def draw_bounding_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        label: Optional[str] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draws a bounding box and optional text label on the image.

        Args:
            image: Frame (NumPy array).
            box: Bounding box as (x_min, y_min, x_max, y_max).
            label: Text to draw above the box.
            color: RGB/BGR color tuple.
            thickness: Line thickness.
        """
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
            # Draw label background
            cv2.rectangle(
                image,
                (x_min, y_min - text_size[1] - 5),
                (x_min + text_size[0], y_min),
                color,
                cv2.FILLED
            )
            # Draw text
            cv2.putText(
                image,
                label,
                (x_min, y_min - 4),
                font,
                font_scale,
                (0, 0, 0),
                text_thickness,
                lineType=cv2.LINE_AA
            )
        return image

    def draw_landmarks(
        self,
        image: np.ndarray,
        landmarks: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 2
    ) -> np.ndarray:
        """
        Draws normalized landmarks (such as face mesh points) on the image.

        Args:
            image: Frame (NumPy array).
            landmarks: List of landmarks containing normalized 'x' and 'y' (0.0 to 1.0) values.
            color: Dot color.
            radius: Dot radius.
        """
        h, w = image.shape[:2]
        for lm in landmarks:
            # Re-scale normalized landmark back to screen space
            cx = int(lm["x"] * w)
            cy = int(lm["y"] * h)
            cv2.circle(image, (cx, cy), radius, color, -1)
        return image

    def draw_skeleton(
        self,
        image: np.ndarray,
        skeleton_landmarks: List[Dict[str, Any]],
        connections: Optional[List[Tuple[int, int]]] = None,
        joint_color: Tuple[int, int, int] = (255, 0, 0),
        line_color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draws skeleton joints and connecting lines.

        Args:
            image: Frame (NumPy array).
            skeleton_landmarks: List of body joints with normalized 'x' and 'y'.
            connections: Index pairs representing body segment connections.
            joint_color: Dot color for joints.
            line_color: Line color for segments.
            thickness: Segment line thickness.
        """
        h, w = image.shape[:2]

        # Draw connection segments first
        if connections:
            for start_idx, end_idx in connections:
                if start_idx < len(skeleton_landmarks) and end_idx < len(skeleton_landmarks):
                    pt1 = skeleton_landmarks[start_idx]
                    pt2 = skeleton_landmarks[end_idx]

                    # Only draw if landmarks have reasonable coordinates/confidence
                    c1 = (int(pt1["x"] * w), int(pt1["y"] * h))
                    c2 = (int(pt2["x"] * w), int(pt2["y"] * h))

                    cv2.line(image, c1, c2, line_color, thickness)

        # Draw individual joint dots
        for lm in skeleton_landmarks:
            cx = int(lm["x"] * w)
            cy = int(lm["y"] * h)
            cv2.circle(image, (cx, cy), thickness + 1, joint_color, -1)

        return image
