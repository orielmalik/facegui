import logging
from typing import Optional

from config.config import AppConfig
from patterns.observer import EventHub
from patterns.factory import VisionServiceFactory
from services.camera import CameraManager
from services.cv_processor import OpenCVProcessor
from services.mediapipe_service import MediaPipeService
from services.insightface_service import InsightFaceService
from services.face_verification import FaceVerificationService
from pipeline.cv_pipeline import ComputerVisionPipeline
from web.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class DependencyContainer:
    """
    Lightweight dependency injection registry.
    Manages service lifecycles and instances.
    """

    def __init__(self) -> None:
        self.config: Optional[AppConfig] = None
        self.event_hub: Optional[EventHub] = None
        self.camera_manager: Optional[CameraManager] = None
        self.cv_processor: Optional[OpenCVProcessor] = None
        self.mediapipe_service: Optional[MediaPipeService] = None
        self.insightface_service: Optional[InsightFaceService] = None
        self.verification_service: Optional[FaceVerificationService] = None
        self.pipeline: Optional[ComputerVisionPipeline] = None
        self.websocket_manager: Optional[WebSocketManager] = None

    def initialize(self, config: AppConfig) -> None:
        """Instantiates all platform services based on settings."""
        self.config = config
        
        # 1. Core utilities
        self.event_hub = EventHub()
        self.websocket_manager = WebSocketManager()
        self.cv_processor = OpenCVProcessor()
        
        # 2. Camera Manager
        self.camera_manager = CameraManager(
            source=config.camera.source,
            target_size=(config.camera.width, config.camera.height)
        )
        
        # 3. MediaPipe and InsightFace adapters
        self.mediapipe_service = MediaPipeService(
            min_detection_confidence=config.mediapipe.min_detection_confidence,
            min_tracking_confidence=config.mediapipe.min_tracking_confidence,
            max_num_faces=config.mediapipe.max_num_faces,
            enable_face_mesh=config.mediapipe.enable_face_mesh,
            enable_pose=config.mediapipe.enable_pose
        )
        
        self.insightface_service = InsightFaceService(
            model_name=config.insightface.model_name,
            ctx_id=config.insightface.ctx_id,
            det_size=config.insightface.det_size
        )
        
        # 4. Verification and Pipeline
        self.verification_service = VisionServiceFactory.create_face_verification_service(
        )
        
        self.pipeline = ComputerVisionPipeline(
            camera_manager=self.camera_manager,
            cv_processor=self.cv_processor,
            mediapipe_service=self.mediapipe_service,
            insightface_service=self.insightface_service,
            verification_service=self.verification_service,
            event_hub=self.event_hub
        )
        
        logger.info("DependencyContainer initialized successfully.")


# Singleton registry instance
deps = DependencyContainer()
