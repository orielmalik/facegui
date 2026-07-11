import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class FaceVerificationService:
    def __init__(self, api_client):
        self._api_client = api_client
        self._last_identify_time = 0
        self._identify_interval = 30  # שניות

    def identify(self, face_embedding: np.ndarray):
        """זיהוי - רק פעם אחת כל 30 שניות"""
        current_time = time.time()

        # בדיקה אם עברו 30 שניות
        if current_time - self._last_identify_time < self._identify_interval:
            return False, None

        self._last_identify_time = current_time

        try:

            response = self._api_client.post(
                "/faces/identify",
                data={"vector": face_embedding.tolist()}
            )
            return True, response
        except Exception as e:
            logger.error(f"Identify failed: {e}")
            return False, None

    def register(self, name: str, face_embedding: np.ndarray, metadata=None):
        """רישום אדם"""
        try:
            response = self._api_client.post(
                "/api/faces",
                {
                    "name": name,
                    "faceVector": face_embedding.tolist(),
                    "metadata": metadata or {}
                }
            )
            logger.info(f"REGISTER success for {name}")
            return True, response
        except Exception as e:
            logger.error(f"REGISTER failed: {e}")
            return False, None