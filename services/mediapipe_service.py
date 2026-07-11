import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# MediaPipe is listed in requirements.txt. We import it safely.
try:
    import mediapipe as mp
except ImportError:
    mp = None

from services.interfaces import FaceLandmarkDetector, PoseDetector

logger = logging.getLogger(__name__)


class MediaPipeService(FaceLandmarkDetector, PoseDetector):
    """
    Facade and Adapter for MediaPipe solutions.
    Implements FaceLandmarkDetector and PoseDetector.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        enable_face_mesh: bool = True,
        enable_pose: bool = True
    ) -> None:
        """
        Initializes the MediaPipe solutions based on requested configuration.
        """
        self._enable_face_mesh = enable_face_mesh
        self._enable_pose = enable_pose

        self._mp_face_mesh = None
        self._mp_pose = None

        if mp is None:
            logger.error("MediaPipe library is not installed or import failed. Using mock fallbacks.")
            return

        # Initialize Face Mesh
        if self._enable_face_mesh:
            try:
                self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                logger.info("MediaPipe Face Mesh initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize MediaPipe Face Mesh: %s", e)

        # Initialize Pose
        if self._enable_pose:
            try:
                self._mp_pose = mp.solutions.pose.Pose(
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                logger.info("MediaPipe Pose initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize MediaPipe Pose: %s", e)

    def detect_landmarks(self, rgb_image: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        Detects facial landmarks on an RGB image.
        """
        # TODO: Add custom preprocessing or data validation if required for face detection assignments.
        
        if not self._enable_face_mesh or not self._mp_face_mesh:
            # Fallback mock if disabled or not imported
            return []

        try:
            results = self._mp_face_mesh.process(rgb_image)
            if not results or not results.multi_face_landmarks:
                return []

            all_faces = []
            for face_landmarks in results.multi_face_landmarks:
                face = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    # Store as structured DTO mapping to interface requirements
                    face.append({
                        "id": idx,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z)
                    })
                all_faces.append(face)
            return all_faces

        except Exception as e:
            logger.error("Error in MediaPipe face landmark detection: %s", e)
            return []

    def detect_pose(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects body joints (skeleton landmarks) on an RGB image.
        """
        # TODO: Add custom filtering or coordinate normalization for pose tracking assignments.

        if not self._enable_pose or not self._mp_pose:
            # Fallback mock if disabled or not imported
            return []

        try:
            results = self._mp_pose.process(rgb_image)
            if not results or not results.pose_landmarks:
                return []

            skeleton = []
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                # Retrieve standard visibility/presence flags from MediaPipe
                visibility = getattr(lm, "visibility", 1.0)
                skeleton.append({
                    "id": idx,
                    "name": f"joint_{idx}",
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "confidence": float(visibility)
                })
            return skeleton

        except Exception as e:
            logger.error("Error in MediaPipe pose detection: %s", e)
            return []

    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """
        Returns the connections mapping between joints defined by MediaPipe solutions.
        """
        if mp is not None and self._enable_pose:
            try:
                connections = mp.solutions.pose.POSE_CONNECTIONS
                return list(connections)
            except AttributeError:
                pass
        
        # Static fallback of standard MediaPipe pose connection lines if mp is not loaded
        return [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28), (27, 29), (28, 30), (29, 31),
            (30, 32), (27, 31), (28, 32)
        ]

    def close(self) -> None:
        """Releases the MediaPipe pipeline resources."""
        if self._mp_face_mesh:
            self._mp_face_mesh.close()
        if self._mp_pose:
            self._mp_pose.close()
        logger.info("MediaPipe pipeline resources released.")
