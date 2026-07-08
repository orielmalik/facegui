import logging
from nicegui import app, ui
from fastapi import WebSocket

from config.config import load_config_from_file
from core.logging import configure_logging
from app.dependencies import deps
from app.application import Application
from ui.page_manager import create_dashboard

logger = logging.getLogger(__name__)


def bootstrap(config_path: str = "config/settings.yaml") -> Application:
    """
    Bootstraps the computer vision application.
    Loads configurations, configures logs, binds observers, sets up
    page routing templates, and returns the lifecycled Application instance.
    """
    # 1. Load config
    config = load_config_from_file(config_path)

    # 2. Configure standard logging
    configure_logging(config.log_level)
    logger.info("Bootstrapping real-time CV platform application layers...")

    # 3. Initialize dependency registry singletons
    deps.initialize(config)

    # 4. Instantiate Application manager
    application = Application()

    # 5. Wire pipeline events from EventHub to Application handler
    deps.event_hub.subscribe_async("frame_processed", application.handle_event)

    # 6. Map startup/shutdown lifecycles
    app.on_startup(application.start)
    app.on_shutdown(application.stop)

    # 7. Map custom WebSocket route
    @app.websocket("/ws")
    async def ws_route(websocket: WebSocket) -> None:
        await application.handle_ws_route(websocket)

    # 8. Define UI page layout using page management builders
    @ui.page("/")
    def render_main_dashboard() -> None:
        title = getattr(config.web, "title", "Real-Time CV Sandbox") if hasattr(config, "web") else "Real-Time CV Sandbox"
        page = create_dashboard(title=title)

        # Build pages procedurally using abstraction interfaces
        page.add_camera_view(resolution=f"{config.camera.width}x{config.camera.height}")

        # Control panel callback actions mapping
        def start_feed():
            try:
                deps.camera_manager.open()
            except Exception as e:
                ui.notify(f"Camera start error: {e}", type="negative")

        def stop_feed():
            deps.camera_manager.close()

        page.add_controls(on_start=start_feed, on_stop=stop_feed)
        page.add_status()

        # Connect WebSocket Javascript hooks
        page.add_client_script()

    return application
