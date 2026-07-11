import logging
from typing import List, Tuple
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class InsightFaceService:

    def __init__(
            self,
            model_name="buffalo_l",
            ctx_id=0,
            det_size=(640, 640)
    ):
        logger.info("Initializing InsightFace...")

        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        logger.info("InsightFace initialized")

        # נוסיף כאן reference פשוט (לשימוש זמני)
        self._reference_embedding = None  # נמלא אחר כך ב-Enroll

    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        מחזיר רשימת פנים עם bbox + embedding
        ללא verify כרגע (כדי שיעבוד מיד)
        """
        if frame is None:
            return []

        try:
            faces = self.app.get(frame)  # InsightFace עושה detection + embedding
            results = []

            for face in faces:
                bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
                embedding = face.normed_embedding

                results.append({
                    "bbox": bbox,
                    "embedding": embedding,
                    "landmarks": face.landmark_2d_106 if hasattr(face, 'landmark_2d_106') else None,
                    "score": 0.0,  # נשאיר 0.0 כרגע
                    "is_match": False  # נשאיר False עד שנוסיף verify
                })

            return results

        except Exception as e:
            logger.error(f"Error in detect_faces: {e}")
            return []
