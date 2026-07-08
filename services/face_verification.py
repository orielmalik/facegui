import logging
import numpy as np
from typing import Tuple

from services.similarity import SimilarityService

logger = logging.getLogger(__name__)


class FaceVerificationService:
    """
    Adapter providing a simple interface to verify a face embedding against a stored/reference embedding.
    Follows SOLID principles by relying on a separate SimilarityService.
    Does not implement any user database, identity mapping, or business logic.
    """

    def __init__(self, similarity_service: SimilarityService, threshold: float = 0.75) -> None:
        """
        Initializes the Verification Service.

        Args:
            similarity_service: Service used to compute similarity scores.
            threshold: Numerical threshold above which embeddings are considered a match.
        """
        self._similarity_service = similarity_service
        self._threshold = threshold
        logger.info("FaceVerificationService initialized with threshold: %.2f", self._threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        logger.info("Verification threshold updated from %.2f to %.2f", self._threshold, value)
        self._threshold = value

    def verify(self, face_embedding: np.ndarray, stored_embedding: np.ndarray) -> Tuple[float, bool]:
        """
        Verifies if two embeddings represent the same face.

        Args:
            face_embedding: Current detected face embedding (NumPy array).
            stored_embedding: Reference face embedding to compare against.

        Returns:
            A tuple containing:
            - similarity_score (float): The calculated similarity score (0.0 to 1.0).
            - match (bool): True if score >= threshold, False otherwise.
        """
        # 1. Calculate the similarity score
        score = self._similarity_service.calculate_similarity(face_embedding, stored_embedding)

        # 2. Check match based on the threshold
        # TODO: Implement complex business matching rules here (e.g. multi-step verification,
        # logging mismatch anomalies, checking user blacklists, etc.).
        is_match = score >= self._threshold

        logger.debug("Verification check: similarity=%.4f (threshold=%.2f) -> match=%s",
                     score, self._threshold, is_match)

        return score, is_match
