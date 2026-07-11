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


            rgb_frame = self._cv_processor.bgr_to_rgb(
                processed_frame
            )


            self.frame_counter += 1


            if self.frame_counter % self.ai_interval == 0:


                self.last_faces = (
                    self._insightface.detect_faces(
                        rgb_frame
                    )
                )


                self.last_face_meshes = (
                    self._mediapipe.detect_landmarks(
                        rgb_frame
                    )
                )


                self.last_pose = (
                    self._mediapipe.detect_pose(
                        rgb_frame
                    )
                )


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



                #
                # REGISTER
                #

                if self.enroll_requested:


                    logger.info(
                        "Sending REGISTER API request"
                    )


                    try:

                        success, response = (
                            self._verification.register(
                                self.enroll_name,
                                embedding
                            )
                        )


                        logger.info(
                            "REGISTER response=%s",
                            response
                        )


                        self.enroll_requested = False


                    except Exception as e:

                        logger.error(
                            "REGISTER failed %s",
                            e
                        )



                #
                # IDENTIFY TIMER
                #

                now = time.time()


                if (
                    now - self.last_identify_time
                    >= self.identify_interval
                ):


                    self.last_identify_time = now


                    logger.info(
                        "Sending IDENTIFY API request"
                    )


                    try:


                        success, identity = (
                            self._verification.identify(
                                embedding
                            )
                        )


                        if success:

                            self.last_identity = identity


                            logger.info(
                                "IDENTIFIED %s",
                                identity
                            )


                    except Exception as e:

                        logger.error(
                            "IDENTIFY failed %s",
                            e
                        )




                verification_results.append(
                    {
                        "bbox": bbox,
                        "similarity_score": best_match_score,
                        "is_match": best_match_status
                    }
                )



            annotated_frame = processed_frame.copy()



            for result in verification_results:


                label = (
                    "Known"
                    if result["is_match"]
                    else
                    "Unknown"
                )


                self._cv_processor.draw_bounding_box(
                    annotated_frame,
                    result["bbox"],
                    label,
                    (0,255,0)
                )



            for mesh in face_meshes:

                self._cv_processor.draw_landmarks(
                    annotated_frame,
                    mesh,
                    color=(0,255,255),
                    radius=1
                )



            if pose_skeleton:


                connections = (
                    self._mediapipe
                    .get_pose_connections()
                )


                self._cv_processor.draw_skeleton(
                    annotated_frame,
                    pose_skeleton,
                    connections,
                    joint_color=(0,165,255),
                    line_color=(0,255,0),
                    thickness=2
                )



            jpeg_bytes = (
                self._cv_processor.to_jpeg(
                    annotated_frame,
                    quality=60
                )
            )



            payload = {

                "frame_bytes": jpeg_bytes,

                "telemetry":
                {
                    "faces_detected": len(faces),

                    "identified":
                        self.last_identity
                }
            }



            self._event_hub.publish(
                "frame_processed",
                payload
            )


            return payload



        except Exception as e:

            logger.error(
                "Pipeline failure %s",
                e,
                exc_info=True
            )

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