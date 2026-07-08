class AppException(Exception):
    """Base exception for all application-specific errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CameraError(AppException):
    """Exception raised when camera operations fail (opening, reading, or connection loss)."""
    pass


class ModelLoadError(AppException):
    """Exception raised when an AI model (MediaPipe, InsightFace, ONNX) fails to load."""
    pass


class PipelineError(AppException):
    """Exception raised during real-time computer vision pipeline execution."""
    pass
