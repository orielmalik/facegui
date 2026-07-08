import os
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class CameraConfig(BaseModel):
    """Configuration settings for camera connection and properties."""
    source: str = Field(default="0", description="Camera source index or video file path")
    width: int = Field(default=640, description="Target frame width")
    height: int = Field(default=480, description="Target frame height")
    fps: int = Field(default=30, description="Target frames per second")
    frame_skip: int = Field(default=0, description="Number of frames to skip to reduce CPU load")


class MediaPipeConfig(BaseModel):
    """Configuration settings for MediaPipe Face & Pose models."""
    min_detection_confidence: float = Field(default=0.5, description="Min confidence value for detection")
    min_tracking_confidence: float = Field(default=0.5, description="Min confidence value for landmarks tracking")
    max_num_faces: int = Field(default=1, description="Maximum number of faces to detect")
    enable_pose: bool = Field(default=True, description="Enable or disable body pose detection")
    enable_face_mesh: bool = Field(default=True, description="Enable or disable face landmarks mesh")


class InsightFaceConfig(BaseModel):
    """Configuration settings for InsightFace / ArcFace embedding models."""
    model_name: str = Field(default="buffalo_l", description="InsightFace model pack name")
    model_dir: Optional[str] = Field(default=None, description="Directory containing local model files")
    ctx_id: int = Field(default=-1, description="Context ID: -1 for CPU, >=0 for GPU ID")
    det_thresh: float = Field(default=0.6, description="Face detection confidence threshold")
    det_size: tuple[int, int] = Field(default=(640, 640), description="Resize size for face detection preprocessing")


class ServerConfig(BaseModel):
    """Configuration settings for the NiceGUI/Uvicorn web server."""
    host: str = Field(default="0.0.0.0", description="IP address to bind the web server to")
    port: int = Field(default=8080, description="Port to run the web server on")
    reload: bool = Field(default=False, description="Enable Uvicorn auto-reload")
    debug: bool = Field(default=False, description="Enable debug mode")
    title: str = Field(
        default="Real Time CV Platform",
        description="Application title displayed in NiceGUI browser and dashboard"
    )


class AppConfig(BaseModel):
    """Root configuration holding all settings for the application."""
    camera: CameraConfig = Field(default_factory=CameraConfig)
    mediapipe: MediaPipeConfig = Field(default_factory=MediaPipeConfig)
    insightface: InsightFaceConfig = Field(default_factory=InsightFaceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    log_level: str = Field(default="INFO", description="Global logging level")


def load_config_from_file(config_path: str) -> AppConfig:
    """
    Loads configuration from a YAML file and validates it with AppConfig Pydantic model.
    If the file does not exist, returns AppConfig with default values.
    """
    if not os.path.exists(config_path):
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return AppConfig.model_validate(data)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}. Using defaults.")
        return AppConfig()
