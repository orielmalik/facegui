import threading
from datetime import time

from nicegui import ui
import base64

from  pipeline import cv_pipeline
from app.bootstrap import bootstrap
from app.dependencies import deps
from services import camera, cv_processor
from services.camera import CameraManager
from services.cv_processor import OpenCVProcessor
from services.insightface_service import InsightFaceService
from web.dashboard import build_modern_dashboard

pipeline = deps.insightface_service  # או הדרך שבה שלפת את ה-ComputerVisionPipeline


def main():

    # Load config + initialize services
    bootstrap()


    target=build_modern_dashboard(deps)


    ui.run(
        host=deps.config.server.host,
        port=deps.config.server.port,
        reload=False
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()