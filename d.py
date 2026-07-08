from nicegui import ui
from ui.page_manager import create_dashboard


@ui.page('/')
def index():

    page = create_dashboard()

    page.add_camera_view()
    page.add_controls()
    page.add_status()
    page.add_client_script()


ui.run()