import logging
import time
from typing import Dict, Any, Optional

import numpy as np

from services.camera import CameraManager
from services.cv_processor import OpenCVProcessor
from services.mediapipe_service import MediaPipeService
from services.insightface_service import InsightFaceService
from services.face_verification import FaceVerificationService
from patterns.observer import EventHub


logger = logging.getLogger(__name__)


class ComputerVisionPipeline:

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


        self.frame_counter = 0
        self.ai_interval = 4


        self.last_faces = []
        self.last_face_meshes = []
        self.last_pose = None


        # IDENTIFY CONTROL
        self.last_identify_time = 0
        self.identify_interval = 30
        self.last_identity = None


        # REGISTER CONTROL
        self.enroll_requested = False
        self.enroll_name = None


    def request_enroll(self, name: str):

        self.enroll_requested = True
        self.enroll_name = name

        logger.info(
            "Enroll requested for %s",
            name
        )

    def run_step(self) -> Optional[Dict[str, Any]]:

        try:

            success, raw_frame = self._camera.read_frame()

            if not success or raw_frame is None:
                return None

            processed_frame = self._preprocess(raw_frame)
            rgb_frame = self._cv_processor.bgr_to_rgb(processed_frame)

            self.frame_counter += 1

            if self.frame_counter % self.ai_interval == 0:
                self.last_faces = self._insightface.detect_faces(rgb_frame)
                self.last_face_meshes = self._mediapipe.detect_landmarks(rgb_frame)
                self.last_pose = self._mediapipe.detect_pose(rgb_frame)

            faces = self.last_faces
            face_meshes = self.last_face_meshes
            pose_skeleton = self.last_pose

            verification_results = []
            best_match_score = 0.0
            best_match_status = False

            for face in faces:
                bbox = face.get("bbox")
                embedding = face.get("embedding")

                if bbox is None or embedding is None:
                    continue

                # ==================== REGISTER ====================
                if self.enroll_requested and self.enroll_name:
                    logger.info(f"Sending REGISTER for {self.enroll_name}")
                    try:
                        success, response = self._verification.register(
                            self.enroll_name,
                            embedding
                        )
                        if success:
                            logger.info(f"✅ Registered: {self.enroll_name}")
                            # שלח event ל-UI
                            self._event_hub.publish("enroll_success", {"name": self.enroll_name})
                        else:
                            self._event_hub.publish("enroll_failed", {"name": self.enroll_name})
                    except Exception as e:
                        logger.error(f"Register failed: {e}")
                        self._event_hub.publish("enroll_failed", {"name": self.enroll_name})

                    self.enroll_requested = False
                    self.enroll_name = None
                # ==================== IDENTIFY (כל 30 שניות) ====================
                now = time.time()
                if now - self.last_identify_time >= self.identify_interval:
                    self.last_identify_time = now
                    logger.info("Sending IDENTIFY to API")
                    try:
                        success, identity = self._verification.identify(embedding)
                        if success and identity:
                            self.last_identity = identity
                            self._event_hub.publish("person_identified", {
                                "name": identity.get("name", "Unknown"),
                                "similarity": identity.get("similarityScore", 0.0)
                            })

                    except Exception as e:
                        logger.error(f"Identify failed: {e}")

                verification_results.append({
                    "bbox": bbox,
                    "similarity_score": 0.0,
                    "is_match": False
                })

            # ====================== ציור ======================
            annotated_frame = processed_frame.copy()

            for result in verification_results:
                label = "Known" if result["is_match"] else "Unknown"
                self._cv_processor.draw_bounding_box(
                    annotated_frame, result["bbox"], label, (0,255,0)
                )

            for mesh in face_meshes:
                self._cv_processor.draw_landmarks(annotated_frame, mesh, color=(0,255,255))

            if pose_skeleton:
                connections = self._mediapipe.get_pose_connections()
                self._cv_processor.draw_skeleton(annotated_frame, pose_skeleton, connections)

            jpeg_bytes = self._cv_processor.to_jpeg(annotated_frame, quality=60)

            payload = {
                "frame_bytes": jpeg_bytes,
                "telemetry": {
                    "faces_detected": len(faces),
                    "identified": self.last_identity
                }
            }

            self._event_hub.publish("frame_processed", payload)

            return payload

        except Exception as e:
            logger.error("Pipeline failure %s", e, exc_info=True)
            return None



    def _preprocess(
        self,
        frame: np.ndarray
    ) -> np.ndarray:


        import cv2


        frame = cv2.flip(
            frame,
            1
        )


        frame = cv2.resize(
            frame,
            (640,480),
            interpolation=cv2.INTER_AREA
        )


        return frame