import logging
from nicegui import app

from config.config import load_config_from_file
from core.logging import configure_logging
from app.dependencies import deps
from app.application import Application

logger = logging.getLogger(__name__)


def bootstrap(config_path="config/settings.yaml") -> Application:

    config = load_config_from_file(config_path)

    configure_logging(config.log_level)

    logger.info("Starting application bootstrap")

    deps.initialize(config)

    application = Application()

    deps.event_hub.subscribe_async(
        "frame_processed",
        application.handle_event
    )

    app.on_startup(application.start)
    app.on_shutdown(application.stop)

    logger.info("Bootstrap completed")

    return application