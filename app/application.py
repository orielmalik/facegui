import asyncio
import base64
import logging
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from nicegui import app, ui

from app.dependencies import deps
from core.exceptions import CameraError

logger = logging.getLogger(__name__)


class Application:
    """
    Manages the core lifecycle of the computer vision pipeline task,
    Websocket connections, and integration with the web framework.
    """

    def __init__(self) -> None:
        self._pipeline_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Starts background workers and opens camera resources."""
        if self._running:
            return
        
        logger.info("Starting Application services...")
        self._running = True

        # Open camera stream
        try:
            deps.camera_manager.open()
        except CameraError as ce:
            logger.critical("Failed to open camera on startup: %s", ce)
            self._running = False
            return

        # Start the background pipeline runner
        fps = deps.config.camera.fps
        self._pipeline_task = asyncio.create_task(self._run_loop(fps))

    async def stop(self) -> None:
        """Stops background workers and releases resources."""
        if not self._running:
            return
        
        logger.info("Stopping Application services...")
        self._running = False

        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass
            self._pipeline_task = None

        if deps.camera_manager:
            deps.camera_manager.close()

    async def _run_loop(self, fps: int) -> None:
        """Background coroutine pulling frame processing at target FPS."""
        delay = 1.0 / max(1, fps)
        while self._running:
            try:
                deps.pipeline.run_step()
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in application CV loop: %s", e)
                await asyncio.sleep(1.0)

    async def handle_ws_route(self, websocket: WebSocket) -> None:
        """Handles FastAPI WebSocket connections."""
        await deps.websocket_manager.connect(websocket)
        try:
            while True:
                # Discard incoming messages, waiting for close signal
                await websocket.receive_text()
        except WebSocketDisconnect:
            logger.info("WS connection disconnected gracefully.")
        except Exception as e:
            logger.debug("WS socket exception: %s", e)
        finally:
            await deps.websocket_manager.disconnect(websocket)

    async def handle_event(self, payload: dict) -> None:
        """Handles pipeline outputs from EventHub, broadcasting them to WS clients."""
        try:
            frame_bytes = payload["frame_bytes"]
            telemetry = payload["telemetry"]
            
            # Encode frame to base64 for easy frontend rendering
            frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
            
            await deps.websocket_manager.broadcast_json({
                "frame": frame_b64,
                "telemetry": telemetry
            })
        except Exception as e:
            logger.error("Broadcasting pipeline data error: %s", e)
