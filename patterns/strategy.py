from abc import ABC, abstractmethod
import numpy as np


class SimilarityStrategy(ABC):
    """
    Abstract Base Class defining the strategy for comparing two feature vectors (embeddings).
    """

    @abstractmethod
    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compares two face embeddings and returns a normalized similarity score.

        Args:
            embedding1: First feature vector (1D NumPy array).
            embedding2: Second feature vector (1D NumPy array).

        Returns:
            A similarity score normalized between 0.0 (no similarity) and 1.0 (perfect match).
        """
        pass


class CosineSimilarityStrategy(SimilarityStrategy):
    """
    Computes Cosine Similarity between two embeddings.
    Cosine similarity is defined as: dot(A, B) / (norm(A) * norm(B)).
    The raw value range is [-1.0, 1.0], normalized here to [0.0, 1.0].
    """

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        dot_product = np.dot(embedding1, embedding2)
        raw_cosine = dot_product / (norm1 * norm2)

        # Normalize [-1.0, 1.0] to [0.0, 1.0]
        normalized_score = (raw_cosine + 1.0) / 2.0
        return float(np.clip(normalized_score, 0.0, 1.0))


class EuclideanSimilarityStrategy(SimilarityStrategy):
    """
    Computes similarity based on Normalized Euclidean Distance.
    Euclidean distance is in range [0, inf). We convert it to a similarity score
    in [0.0, 1.0] using 1.0 / (1.0 + distance).
    """

    def compare(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        distance = np.linalg.norm(embedding1 - embedding2)
        # Normalize distance to [0, 1] similarity
        similarity = 1.0 / (1.0 + distance)
        return float(similarity)
