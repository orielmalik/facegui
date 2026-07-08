import asyncio
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventHub:
    """
    Central event broker implementing the Observer pattern.
    Allows decoupling of processing pipelines from UI and notifications.
    Supports both synchronous callbacks and asynchronous event listeners.
    """

    def __init__(self) -> None:
        self._sync_subscribers: Dict[str, List[Callable[[Any], None]]] = {}
        self._async_subscribers: Dict[str, List[Callable[[Any], Any]]] = {}

    def subscribe(self, event_name: str, callback: Callable[[Any], None]) -> None:
        """Subscribes a synchronous callback to an event."""
        if event_name not in self._sync_subscribers:
            self._sync_subscribers[event_name] = []
        self._sync_subscribers[event_name].append(callback)
        logger.debug("Synchronous subscriber added for event: %s", event_name)

    def subscribe_async(self, event_name: str, callback: Callable[[Any], Any]) -> None:
        """Subscribes an asynchronous coroutine callback to an event."""
        if event_name not in self._async_subscribers:
            self._async_subscribers[event_name] = []
        self._async_subscribers[event_name].append(callback)
        logger.debug("Asynchronous subscriber added for event: %s", event_name)

    def publish(self, event_name: str, data: Any) -> None:
        """
        Publishes an event to all registered subscribers.
        Synchronous callbacks are executed directly.
        Asynchronous callbacks are scheduled in the active asyncio event loop.
        """
        # Run synchronous callbacks
        if event_name in self._sync_subscribers:
            for callback in self._sync_subscribers[event_name]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error("Error in sync callback for event %s: %s", event_name, e, exc_info=True)

        # Run asynchronous callbacks
        if event_name in self._async_subscribers:
            try:
                loop = asyncio.get_running_loop()
                for callback in self._async_subscribers[event_name]:
                    loop.create_task(self._safe_async_call(callback, data, event_name))
            except RuntimeError:
                # No running event loop, log a warning
                logger.warning("Event loop is not running. Async subscribers for %s were skipped.", event_name)

    async def _safe_async_call(self, callback: Callable[[Any], Any], data: Any, event_name: str) -> None:
        """Helper to invoke async coroutine callbacks safely within the event loop."""
        try:
            await callback(data)
        except Exception as e:
            logger.error("Error in async callback for event %s: %s", event_name, e, exc_info=True)
