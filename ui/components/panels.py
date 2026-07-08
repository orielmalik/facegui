from typing import Tuple, Dict, Any, Callable, Optional
from nicegui import ui

from ui.components.buttons import create_button


def create_status_panel() -> Tuple[ui.card, Dict[str, Any]]:
    """
    Creates a styled face verification and computer vision telemetry panel.
    Returns a tuple: (ui.card, elements_dict) where elements_dict exposes
    the HTML/Label elements for runtime updating.
    """
    elements = {}

    card = ui.card().classes("w-full bg-slate-900/60 backdrop-blur-md border border-slate-800 p-4 rounded-2xl shadow-xl flex flex-col gap-4")
    with card:
        ui.label("Detection & Verification Telemetry").classes("text-lg font-bold text-indigo-200 border-b border-slate-800 pb-2")

        # Live counters list
        with ui.column().classes("w-full gap-3"):
            with ui.row().classes("w-full justify-between items-center"):
                ui.label("Faces Detected").classes("text-slate-300 font-medium")
                elements["faces_detected"] = ui.add_body_html('<span id="val_faces_detected" class="status-indicator status-fail">0</span>')

            with ui.row().classes("w-full justify-between items-center"):
                ui.label("Face Landmarks Mesh").classes("text-slate-300 font-medium")
                elements["landmarks_mesh"] = ui.add_body_html('<span id="val_landmarks_mesh" class="status-indicator status-fail">OFFLINE</span>')

            with ui.row().classes("w-full justify-between items-center"):
                ui.label("Pose Skeleton Tracker").classes("text-slate-300 font-medium")
                elements["pose_skeleton"] = ui.add_body_html('<span id="val_pose_skeleton" class="status-indicator status-fail">OFFLINE</span>')

        # Similarity score presentation card
        with ui.column().classes("w-full bg-slate-950 p-4 rounded-xl border border-slate-900 gap-3 items-center justify-center mt-2"):
            ui.label("Cosine Similarity Score").classes("text-xs text-slate-400 mb-1")
            elements["similarity_score"] = ui.html('<span id="val_similarity_score" class="text-3xl font-extrabold text-indigo-400">0.0000</span>')
            elements["similarity_desc"] = ui.html('<span id="val_similarity_desc" class="text-[10px] text-center text-slate-500 max-w-[200px] leading-tight">No faces checked yet</span>')

        with ui.row().classes("w-full justify-between items-center px-1 mt-2"):
            ui.label("Verification Match Decision").classes("text-slate-300 text-sm font-medium")
            elements["match_status"] = ui.html('<span id="val_match_status" class="status-indicator status-fail">NO MATCH</span>')

    return card, elements


def create_control_panel(
    on_start: Optional[Callable] = None,
    on_stop: Optional[Callable] = None
) -> ui.card:
    """
    Creates a camera flow control panel containing Start/Stop buttons.
    Exposes underlying components through action parameters.
    """
    card = ui.card().classes("w-full bg-slate-900/60 border border-slate-800 p-4 rounded-2xl flex flex-col gap-3")
    with card:
        ui.label("Camera Feed Operations").classes("text-sm font-bold text-slate-400 uppercase tracking-wider mb-1")
        with ui.row().classes("w-full gap-3 justify-start"):
            btn_start = create_button("Start Feed", on_click=on_start, color="emerald", icon="play_arrow")
            btn_stop = create_button("Stop Feed", on_click=on_stop, color="red", icon="stop")
            
            # Save references on card for easy customization
            card.default_slot.btn_start = btn_start
            card.default_slot.btn_stop = btn_stop
            
    return card
