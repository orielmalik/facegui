import logging
import numpy as np

from patterns.strategy import SimilarityStrategy, CosineSimilarityStrategy

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Service responsible for calculating similarity between embeddings.
    Uses the Strategy pattern to allow choosing between Cosine, Euclidean, or other methods.
    """

    def __init__(self, strategy: SimilarityStrategy | None = None) -> None:
        """
        Initializes the Similarity Service with a comparison strategy.
        Defaults to CosineSimilarityStrategy.
        """
        self._strategy = strategy or CosineSimilarityStrategy()
        logger.info("SimilarityService initialized with strategy: %s", self._strategy.__class__.__name__)

    def set_strategy(self, strategy: SimilarityStrategy) -> None:
        """Dynamically updates the comparison strategy."""
        logger.info("Switching similarity strategy to: %s", strategy.__class__.__name__)
        self._strategy = strategy

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculates similarity using the configured strategy.

        Returns:
            A similarity score normalized between 0.0 and 1.0.
        """
        # TODO: Add custom checks or vector length assertions here.
        return self._strategy.compare(embedding1, embedding2)

    def explain_score_meaning(self, score: float) -> str:
        """
        Provides a written explanation of what the similarity score represents.

        IMPORTANT:
        This is a mathematical similarity score (e.g. normalized angle difference or inverse distance),
        NOT a probability. It indicates how close the two embeddings are in high-dimensional vector space.
        A score of 0.85 means high vector similarity, but does not mean an 85% probability of a match.
        """
        explanation = (
            f"Similarity Score: {score:.4f}.\n"
            "This value is a geometric similarity score in high-dimensional embedding space, "
            "not a classification probability. A higher value indicates that the face features are "
            "closer together geometrically. It should be thresholded to make match/non-match decisions "
            "rather than interpreted as a probability percentage."
        )
        return explanation
