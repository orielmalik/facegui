from typing import Optional, Tuple
from nicegui import ui


def create_camera_panel(
    title: str = "Live Camera Stream",
    resolution: str = "640x480"
) -> Tuple[ui.card, ui.html]:
    """
    Creates a card wrapper enclosing the live camera video stream canvas.
    Returns (ui.card, img_html_component) to allow further customizations.
    """
    card = ui.card().classes("w-full bg-slate-900/60 backdrop-blur-md border border-slate-800 p-4 rounded-2xl shadow-xl flex flex-col justify-between")
    
    with card:
        with ui.row().classes("w-full justify-between items-center mb-3"):
            ui.label(title).classes("text-lg font-bold text-indigo-200")
            ui.label(f"Resolution: {resolution}").classes("text-xs text-slate-400 bg-slate-800 px-2 py-1 rounded")
        
        # Stream box container
        stream_container = ui.column().classes("w-full items-center justify-center bg-slate-950 rounded-xl overflow-hidden border border-slate-800 relative min-h-[360px]")
        with stream_container:
            img_html = ui.html("""
                <img id="video_stream" 
                     src="https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=640&auto=format&fit=crop" 
                     alt="Camera Stream" 
                     style="width: 100%; height: auto; max-width: 640px; aspect-ratio: 4/3; object-fit: cover; display: block;" />
                <div id="stream_offline_overlay" 
                     style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(15, 23, 42, 0.85); display: flex; flex-direction: column; align-items: center; justify-content: center; transition: opacity 0.5s;">
                    <svg style="width: 48px; height: 48px; color: #64748b; margin-bottom: 12px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                       <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                    </svg>
                    <span style="color: #94a3b8; font-weight: 500;">Connecting to camera stream...</span>
                </div>
            """)

    return card, img_html
