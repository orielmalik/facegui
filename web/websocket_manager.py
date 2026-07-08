import asyncio
import logging
from typing import Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages active WebSocket client connections for real-time video frame and telemetry broadcasting.
    Ensures safe async connection, disconnection, and event loop friendly broadcasting.
    """

    def __init__(self) -> None:
        self._active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._active_connections.add(websocket)
        logger.info("New WebSocket connection registered. Total clients: %d", len(self._active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Unregisters and removes a WebSocket connection."""
        async with self._lock:
            if websocket in self._active_connections:
                self._active_connections.remove(websocket)
        logger.info("WebSocket connection removed. Total clients: %d", len(self._active_connections))

    async def broadcast_json(self, data: dict) -> None:
        """
        Broadcasts JSON data asynchronously to all active client connections.
        Prunes dead or broken connections on the fly without blocking.
        """
        async with self._lock:
            # Take a snapshot to iterate over safely without holding the lock during I/O
            connections = list(self._active_connections)

        if not connections:
            return

        # Prepare send tasks to run concurrently
        tasks = [self._send_to_client(websocket, data) for websocket in connections]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_to_client(self, websocket: WebSocket, data: dict) -> None:
        """Sends JSON payload to a single WebSocket client. Removes it if disconnected."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.debug("Failed to send WebSocket payload, client probably disconnected: %s", e)
            await self.disconnect(websocket)
