import numpy as np


class FaceVerificationService:

    def __init__(self, api_client):
        self._api_client = api_client


    def identify(
            self,
            face_embedding: np.ndarray
    ):

        response = self._api_client.get(
            "/faces/identify",
            {
                "vector": face_embedding.tolist()
            }
        )

        return True, response



    def register(
            self,
            name: str,
            face_embedding: np.ndarray,
            metadata=None
    ):

        response = self._api_client.post(
            "/faces",
            {
                "name": name,
                "faceVector": face_embedding.tolist(),
                "metadata": metadata
            }
        )

        return True, response