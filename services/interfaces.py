from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np


class FaceLandmarkDetector(ABC):
    """
    Interface for face landmark detection engines.
    """

    @abstractmethod
    def detect_landmarks(self, rgb_image: np.ndarray) -> List[List[Dict[str, Any]]]:
        """
        Detects facial landmarks (mesh points) on the image.

        Args:
            rgb_image: Input image in RGB format (NumPy array).

        Returns:
            A list of detected faces, where each face is represented by a list
            of landmark dictionaries with normalized keys: 'x', 'y', 'z', and optional 'confidence'.
        """
        pass


class PoseDetector(ABC):
    """
    Interface for body pose/skeleton detection engines.
    """

    @abstractmethod
    def detect_pose(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects body joints (skeleton landmarks) on the image.

        Args:
            rgb_image: Input image in RGB format (NumPy array).

        Returns:
            A list of joint dictionaries representing the skeleton.
            Each joint dictionary contains keys: 'name', 'x', 'y', 'z', 'confidence'.
        """
        pass

    @abstractmethod
    def get_pose_connections(self) -> List[Tuple[int, int]]:
        """
        Returns the connections between skeleton joints.

        Returns:
            List of connection index pairs.
        """
        pass


class FaceEmbeddingGenerator(ABC):
    """
    Interface for face recognition embedding engines.
    """

    @abstractmethod
    def extract_face_embeddings(self, rgb_image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Detects faces and extracts their corresponding embedding vectors.

        Args:
            rgb_image: Input image in RGB format (NumPy array).

        Returns:
            A list of tuples containing:
            - The bounding box as (x_min, y_min, x_max, y_max)
            - The face embedding vector as a 1D NumPy array.
        """
        pass
