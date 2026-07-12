from nicegui import ui
from app.bootstrap import bootstrap
from app.dependencies import deps
from web.dashboard import build_modern_dashboard

pipeline = deps.insightface_service


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