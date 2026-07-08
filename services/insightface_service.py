import logging
from typing import List, Tuple, Optional
import numpy as np

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

from services.interfaces import FaceEmbeddingGenerator

logger = logging.getLogger(__name__)


class InsightFaceService(FaceEmbeddingGenerator):
    """
    Facade and Adapter for InsightFace (ArcFace) Face Analysis framework.
    Implements FaceEmbeddingGenerator.
    """

    def __init__(
        self,
        model_name: str = "buffalo_l",
        model_dir: Optional[str] = None,
        ctx_id: int = -1,
        det_thresh: float = 0.6,
        det_size: Tuple[int, int] = (640, 640)
    ) -> None:
        """
        Initializes the InsightFace model app.
        Falls back to a Mock generator if models are missing, ensuring the project remains operational.
        """
        self._model_name = model_name
        self._model_dir = model_dir
        self._ctx_id = ctx_id
        self._det_thresh = det_thresh
        self._det_size = det_size

        self._app = None
        self._is_mock = False

        if FaceAnalysis is None:
            logger.warning("InsightFace is not installed. Falling back to Mock Embeddings.")
            self._is_mock = True
            return

        try:
            # Initialize FaceAnalysis application
            self._app = FaceAnalysis(
                name=self._model_name,
                root=self._model_dir or "./models"
            )
            # Prepare context (CPU/GPU) and image size
            self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
            logger.info("InsightFace FaceAnalysis initialized successfully (using ArcFace).")
        except Exception as e:
            logger.warning(
                "Failed to initialize real InsightFace models (%s). "
                "Falling back to Mock Embeddings for skeleton execution.", e
            )
            self._is_mock = True

    def extract_face_embeddings(self, rgb_image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Extracts face bounding boxes and 512-dimensional embeddings from the frame.

        Returns:
            A list of tuples: (bounding_box_tuple, embedding_vector)
            Where bounding_box_tuple is (x_min, y_min, x_max, y_max)
            And embedding_vector is a 512-dimensional NumPy float array.
        """
        # TODO: Add face normalization, alignment, or embedding validation if required.

        if self._is_mock or not self._app:
            return self._generate_mock_embeddings(rgb_image)

        try:
            # Detect faces and generate embeddings
            faces = self._app.get(rgb_image)
            if not faces:
                return []

            results = []
            for face in faces:
                # Bounding box coordinates from detection
                box = face.bbox.astype(int)  # [x_min, y_min, x_max, y_max]
                bbox_tuple = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                
                # Retrieve the normalized ArcFace embedding vector (512-dim)
                embedding = face.normed_embedding
                results.append((bbox_tuple, embedding))

            return results

        except Exception as e:
            logger.error("Error in InsightFace embedding extraction: %s", e)
            return []

    def _generate_mock_embeddings(self, rgb_image: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Generates mock face embeddings to keep the demo runnable without weights files.
        Simulates detecting a single face in the middle of the frame.
        """
        h, w = rgb_image.shape[:2]
        
        # Simulate a face box in the center of the frame
        cx, cy = w // 2, h // 2
        box_w, box_h = int(w * 0.35), int(h * 0.45)
        
        x_min = max(0, cx - box_w // 2)
        y_min = max(0, cy - box_h // 2)
        x_max = min(w, cx + box_w // 2)
        y_max = min(h, cy + box_h // 2)
        
        bbox = (x_min, y_min, x_max, y_max)
        
        # Generate a repeatable, simulated 512-dimensional embedding
        # We can seed it based on frame coordinates or make it pseudo-random
        np.random.seed(42)
        mock_embedding = np.random.randn(512).astype(np.float32)
        # Normalize to unit length (ArcFace style)
        mock_embedding /= np.linalg.norm(mock_embedding)
        
        return [(bbox, mock_embedding)]
