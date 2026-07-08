import asyncio
import base64
import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from nicegui import app, ui

from config.config import load_config_from_file
from core.logging import configure_logging
from core.exceptions import CameraError
from services.camera import CameraManager
from services.cv_processor import OpenCVProcessor
from services.mediapipe_service import MediaPipeService
from services.insightface_service import InsightFaceService
from patterns.factory import VisionServiceFactory
from patterns.observer import EventHub
from pipeline.cv_pipeline import ComputerVisionPipeline
from web.websocket_manager import WebSocketManager
from web.dashboard import build_dashboard

# We use standard library logging
logger = logging.getLogger("NiceGuiRecoApp")

# Global instances (to be initialized on startup)
camera: Optional[CameraManager] = None
pipeline: Optional[ComputerVisionPipeline] = None
ws_manager = WebSocketManager()
event_hub = EventHub()
pipeline_task: Optional[asyncio.Task] = None


async def frame_processing_loop(camera_device: CameraManager, cv_pipeline: ComputerVisionPipeline, fps: int) -> None:
    """
    Asynchronous task runner that constantly triggers the Computer Vision Pipeline steps
    at the configured frame rate. Ensures non-blocking behavior of the main thread.
    """
    delay = 1.0 / max(1, fps)
    logger.info("Starting CV frame processing loop task at target delay of %.3fs (Max FPS: %d)", delay, fps)

    while True:
        try:
            # 1. Trigger the sequential processing pipeline
            cv_pipeline.run_step()
            # 2. Yield control back to asyncio event loop to allow other tasks (like WebSocket IO) to run
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            logger.info("CV processing task was cancelled.")
            break
        except Exception as e:
            logger.error("Error in background frame processing loop: %s", e)
            await asyncio.sleep(1.0)  # Cool-down wait on critical errors


async def handle_pipeline_event(payload: dict) -> None:
    """
    Subscriber callback registered to the EventHub.
    Translates raw processed frame bytes to Base64 and broadcasts to WebSockets.
    """
    try:
        frame_bytes = payload["frame_bytes"]
        telemetry = payload["telemetry"]

        # Base64 encode for simple inline drawing in HTML/canvas elements
        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")

        # Broadcast to all connected clients asynchronously
        await ws_manager.broadcast_json({
            "frame": frame_b64,
            "telemetry": telemetry
        })
    except Exception as e:
        logger.error("Failed to broadcast pipeline updates: %s", e)


@app.on_startup
async def startup() -> None:
    """Triggered by NiceGUI on web server startup."""
    global camera, pipeline, pipeline_task

    # 1. Load Configurations & Setup logging
    config = load_config_from_file("config/settings.yaml")
    configure_logging(config.log_level)
    logger.info("Initializing Real-Time Computer Vision Platform infrastructure...")

    # 2. Instantiate and configure adapters & services
    camera = CameraManager(
        source=config.camera.source,
        target_size=(config.camera.width, config.camera.height)
    )
    cv_processor = OpenCVProcessor()

    mediapipe_service = MediaPipeService(
        min_detection_confidence=config.mediapipe.min_detection_confidence,
        min_tracking_confidence=config.mediapipe.min_tracking_confidence,
        max_num_faces=config.mediapipe.max_num_faces,
        enable_face_mesh=config.mediapipe.enable_face_mesh,
        enable_pose=config.mediapipe.enable_pose
    )

    insightface_service = InsightFaceService(
        model_name=config.insightface.model_name,
        model_dir=config.insightface.model_dir,
        ctx_id=config.insightface.ctx_id,
        det_thresh=config.insightface.det_thresh,
        det_size=config.insightface.det_size
    )

    verification_service = VisionServiceFactory.create_face_verification_service(
        strategy_name="cosine",
        verification_threshold=0.75
    )

    # 3. Instantiate the Pipeline pattern
    pipeline = ComputerVisionPipeline(
        camera_manager=camera,
        cv_processor=cv_processor,
        mediapipe_service=mediapipe_service,
        insightface_service=insightface_service,
        verification_service=verification_service,
        event_hub=event_hub
    )

    # 4. Subscribe the WebSocket broadcaster to EventHub notifications
    event_hub.subscribe_async("frame_processed", handle_pipeline_event)

    # 5. Open camera resources
    try:
        camera.open()
    except CameraError as ce:
        logger.critical("Initialization aborted: %s", ce)
        return

    # 6. Start async pipeline execution task
    pipeline_task = asyncio.create_task(
        frame_processing_loop(camera, pipeline, config.camera.fps)
    )


@app.on_shutdown
async def shutdown() -> None:
    """Triggered by NiceGUI on web server termination."""
    global pipeline_task, camera
    logger.info("Stopping real-time services and releasing resources...")

    # Stop background processing
    if pipeline_task:
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass
        pipeline_task = None

    # Release camera handle
    if camera:
        camera.close()


# ==========================================
# CUSTOM WEBSOCKET ENDPOINT (FastAPI Integration)
# ==========================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    FastAPI WebSocket handler injected into the NiceGUI/Starlette application stack.
    Handles multi-client connections, handshakes, and cleanup on disconnect.
    """
    await ws_manager.connect(websocket)
    try:
        # Keep client connection open to receive incoming signals or check status
        while True:
            # We discard inputs in this demo skeleton, but wait for connection close
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Client disconnected gracefully.")
    except Exception as e:
        logger.debug("WebSocket exception occurred: %s", e)
    finally:
        await ws_manager.disconnect(websocket)


# ==========================================
# PAGE DEFINITIONS & RUNNER
# ==========================================
# Bind Dashboard route on default path
@ui.page("/")
def index_dashboard() -> None:
    config = load_config_from_file("config/settings.yaml")

    # We pass helper control triggers to UI dashboard
    def start_camera():
        try:
            if camera:
                camera.open()
        except CameraError as ce:
            ui.notify(f"Camera error: {ce}", type="negative")

    def stop_camera():
        if camera:
            camera.close()

    # Decorate ui elements in custom page builder
    build_dashboard(config, config.server.port)

    # Register user controls
    app.storage.user.update(cmd="")

    # Background timer to poll UI control updates
    async def storage_poll():
        cmd = app.storage.user.get("cmd")
        if cmd == "start":
            start_camera()
            app.storage.user.update(cmd="")
        elif cmd == "stop":
            stop_camera()
            app.storage.user.update(cmd="")

    ui.timer(0.2, storage_poll)


if __name__ in {"__main__", "__mp_main__"}:
    app_config = load_config_from_file("config/settings.yaml")

    ui.run(
        host=app_config.server.host,
        port=app_config.server.port,
        title="Real Time CV Platform",
        show=False,
        storage_secret="cv_skeleton_secret_token_1928"
    )
