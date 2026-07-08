from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np


@dataclass(frozen=True)
class CameraFrameReceived:
    """Fired when a raw frame is read from the camera device."""
    frame: np.ndarray


@dataclass(frozen=True)
class FaceDetected:
    """Fired when faces are detected in the active frame."""
    # Bounding boxes as (x_min, y_min, x_max, y_max)
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # Face mesh landmarks
    landmarks: List[List[Dict[str, Any]]] = field(default_factory=list)


@dataclass(frozen=True)
class PoseDetected:
    """Fired when body skeletons are detected in the active frame."""
    landmarks: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[Tuple[int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class RecognitionUpdated:
    """Fired when face verification results are updated."""
    similarity_score: float = 0.0
    is_match: bool = False
    faces_detected: int = 0
