import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from services.camera import CameraManager
from services.cv_processor import OpenCVProcessor
from services.mediapipe_service import MediaPipeService
from services.insightface_service import InsightFaceService
from services.face_verification import FaceVerificationService
from patterns.observer import EventHub

logger = logging.getLogger(__name__)


class ComputerVisionPipeline:
    """
    Implements the Pipeline Pattern.
    Sequentially executes:
      Camera Frame -> Preprocessing -> Face Detection -> Face Landmarks ->
      Pose Detection -> Embedding Gen -> Similarity Calculation -> Results -> UI Event
    """

    def __init__(
        self,
        camera_manager: CameraManager,
        cv_processor: OpenCVProcessor,
        mediapipe_service: MediaPipeService,
        insightface_service: InsightFaceService,
        verification_service: FaceVerificationService,
        event_hub: EventHub
    ) -> None:
        self._camera = camera_manager
        self._cv_processor = cv_processor
        self._mediapipe = mediapipe_service
        self._insightface = insightface_service
        self._verification = verification_service
        self._event_hub = event_hub

        # Dummy stored face embedding representing a registered user reference vector.
        # Generated as standard unit-length 512 float array.
        np.random.seed(99)
        self._reference_embedding = np.random.randn(512).astype(np.float32)
        self._reference_embedding /= np.linalg.norm(self._reference_embedding)

    def set_reference_embedding(self, embedding: np.ndarray) -> None:
        """Sets the reference face embedding to verify real-time faces against."""
        self._reference_embedding = embedding / np.linalg.norm(embedding)
        logger.info("New reference face embedding set in pipeline.")

    def run_step(self) -> Optional[Dict[str, Any]]:
        """
        Executes one complete run (iteration) of the pipeline.
        Fetches the latest frame, processes it, overlays drawings, and publishes events.
        """
        try:
            # 1. Capture Camera Frame
            success, raw_frame = self._camera.read_frame()
            if not success or raw_frame is None:
                return None

            # 2. Image Preprocessing
            processed_frame = self._preprocess(raw_frame)

            # Convert to RGB once for ML models (MediaPipe and InsightFace require RGB)
            rgb_frame = self._cv_processor.bgr_to_rgb(processed_frame)

            # 3. Face Detection & 6. Embedding Generation (Combined in InsightFace)
            # In real InsightFace, detection and embedding happen in one analysis pass.
            faces = self._insightface.extract_face_embeddings(rgb_frame)

            # 4. Face Landmarks (MediaPipe Face Mesh)
            face_meshes = self._mediapipe.detect_landmarks(rgb_frame)

            # 5. Body/Pose Detection (MediaPipe Pose skeleton)
            pose_skeleton = self._mediapipe.detect_pose(rgb_frame)

            # 7. Similarity & 8. Verification Results
            verification_results = []
            best_match_score = 0.0
            best_match_status = False

            for bbox, embedding in faces:
                score, is_match = self._verification.verify(embedding, self._reference_embedding)
                verification_results.append({
                    "bbox": bbox,
                    "similarity_score": score,
                    "is_match": is_match
                })
                # Keep track of highest similarity for telemetry display
                if score > best_match_score:
                    best_match_score = score
                    best_match_status = is_match

            # Draw visual guides onto BGR image frame
            annotated_frame = processed_frame.copy()

            # Render face bounding boxes with matching results
            for res in verification_results:
                label = f"Match: {res['similarity_score']:.2f}" if res["is_match"] else f"Unknown: {res['similarity_score']:.2f}"
                color = (0, 255, 0) if res["is_match"] else (0, 0, 255)
                self._cv_processor.draw_bounding_box(annotated_frame, res["bbox"], label=label, color=color)

            # Render landmarks (Face Mesh)
            for mesh in face_meshes:
                self._cv_processor.draw_landmarks(annotated_frame, mesh, color=(0, 255, 255), radius=1)

            # Render body skeleton
            if pose_skeleton:
                connections = self._mediapipe.get_pose_connections()
                self._cv_processor.draw_skeleton(
                    annotated_frame,
                    pose_skeleton,
                    connections=connections,
                    joint_color=(0, 165, 255),
                    line_color=(0, 255, 0),
                    thickness=2
                )

            # Convert processed frame to JPEG bytes for streaming over WebSocket
            jpeg_bytes = self._cv_processor.to_jpeg(annotated_frame, quality=80)

            # Prepare telemetry payload
            payload = {
                "frame_bytes": jpeg_bytes,
                "telemetry": {
                    "faces_detected": len(faces),
                    "landmarks_detected": len(face_meshes) > 0,
                    "pose_detected": len(pose_skeleton) > 0,
                    "best_similarity_score": best_match_score,
                    "is_match": best_match_status,
                }
            }

            # 9. UI Update - Publish completed pipeline data to observers
            self._event_hub.publish("frame_processed", payload)
            
            return payload

        except Exception as e:
            logger.error("Pipeline iteration failure: %s", e, exc_info=True)
            return None

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Executes frame scaling or adjustments.
        """
        # TODO: Add custom pre-processing steps (contrast adjustments, denoising, crop, etc.)
        # as needed for vision assignments.
        return frame
