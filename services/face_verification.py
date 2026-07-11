from typing import Tuple
import numpy as np
import logging

from web.ApiClient import ApiClient

logger = logging.getLogger(__name__)


class FaceVerificationService:

    def __init__(self):
        self._api_client = ApiClient("http://localhost:8082")

    async def identify(self, face_embedding: np.ndarray) -> Tuple[bool, dict | None]:

        try:
            response = await self._api_client.get(
                "/faces/identify",
                {
                    "vector": face_embedding.tolist()
                }
            )

            logger.info(
                "Face identified successfully: %s",
                response.get("name")
            )

            return True, response

        except Exception as e:
            logger.debug(
                "Face not found: %s",
                str(e)
            )

            return False, None

        async def register(
                self,
                name: str,
                face_embedding: np.ndarray,
                metadata: dict | None = None
        ):
            try:
                response = await self._api_client.post(
                    "/faces",
                    {
                        "name": name,
                        "faceVector": face_embedding.tolist(),
                        "metadata": metadata
                    }
                )

                logger.info(
                    "Face registered successfully: %s",
                    response.get("name")
                )

                return True, response

            except Exception as e:
                logger.debug(
                    "Face registration failed: %s",
                    str(e)
                )

                return False, None