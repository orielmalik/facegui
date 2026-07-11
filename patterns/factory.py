import logging
from typing import Dict, Type
from patterns.strategy import SimilarityStrategy, CosineSimilarityStrategy, EuclideanSimilarityStrategy
from config.config import AppConfig

logger = logging.getLogger(__name__)


class SimilarityStrategyFactory:
    """
    Factory to instantiate SimilarityStrategy objects based on strategy name.
    Simplifies selection and addition of comparison algorithms.
    """

    _strategies: Dict[str, Type[SimilarityStrategy]] = {
        "cosine": CosineSimilarityStrategy,
        "euclidean": EuclideanSimilarityStrategy,
    }

    @classmethod
    def create(cls, strategy_name: str) -> SimilarityStrategy:
        """
        Creates and returns a concrete SimilarityStrategy.
        """
        name = strategy_name.lower().strip()
        strategy_class = cls._strategies.get(name)

        if not strategy_class:
            logger.warning("Unknown strategy name '%s'. Falling back to 'cosine'.", strategy_name)
            return CosineSimilarityStrategy()

        logger.debug("Instantiated similarity strategy: %s", strategy_class.__name__)
        return strategy_class()


class VisionServiceFactory:
    """
    Factory that instantiates services based on AppConfig,
    separating instantiation logic from domain usage.
    """

    @staticmethod
    def create_similarity_service(strategy_name: str):
        """
        Creates SimilarityService with the desired similarity strategy.
        Note: The actual SimilarityService class is imported inside to avoid circular dependency
        with the services package.
        """
        from services.similarity import SimilarityService
        strategy = SimilarityStrategyFactory.create(strategy_name)
        return SimilarityService(strategy=strategy)

    @staticmethod
    def create_face_verification_service():

            from services.face_verification import FaceVerificationService
            return FaceVerificationService(
            )