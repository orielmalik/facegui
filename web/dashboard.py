import logging
from nicegui import ui, app
from config.config import AppConfig

logger = logging.getLogger(__name__)


def build_dashboard(config: AppConfig, ws_port: int) -> None:
    """
    Builds the main dashboard page using NiceGUI.
    Uses custom HTML and a dedicated client-side WebSocket listener to stream frames
    and update telemetry without blocking NiceGUI's main Python server thread.
    """
    # Force dark mode for a modern premium feel
    dark = ui.dark_mode()
    dark.enable()

    # Dynamic styling for headers and elements
    ui.add_body_html("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
            body {
                font-family: 'Outfit', sans-serif;
                background-color: #0f172a;
            }
            .dashboard-card {
                background: rgba(30, 41, 59, 0.7);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            }
            .status-indicator {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.875rem;
                font-weight: 600;
            }
            .status-ok {
                background: rgba(16, 185, 129, 0.15);
                color: #34d399;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }
            .status-fail {
                background: rgba(239, 68, 68, 0.15);
                color: #f87171;
                border: 1px solid rgba(239, 68, 68, 0.3);
            }
        </style>
    """)

    # Main dashboard header container
    with ui.header().classes("bg-slate-900 border-b border-slate-800 text-white p-4 flex items-center justify-between"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("videocam", size="md").classes("text-indigo-400 animate-pulse")
            ui.label(config.server.title).classes("text-xl font-bold tracking-wide")
        with ui.row().classes("items-center gap-2"):
            ui.label("System Status:").classes("text-slate-400 text-sm")
            ui.badge("ACTIVE", color="emerald").classes("px-2 py-1 text-xs font-semibold")

    # Responsive Grid Layout
    with ui.grid(columns="1fr").classes("w-full max-w-7xl mx-auto p-4 gap-6 lg:grid-cols-3"):
        
        # Left side: Live Video Stream Card
        with ui.card().classes("dashboard-card p-4 lg:col-span-2 flex flex-col justify-between"):
            with ui.row().classes("w-full justify-between items-center mb-3"):
                ui.label("Live Camera Stream").classes("text-lg font-bold text-indigo-200")
                ui.label("Resolution: 640x480").classes("text-xs text-slate-400 bg-slate-800 px-2 py-1 rounded")
            
            # Streaming Canvas Image Wrapper
            with ui.column().classes("w-full items-center justify-center bg-slate-950 rounded-xl overflow-hidden border border-slate-800 relative min-h-[360px]"):
                # Using custom HTML image that will be fed by JavaScript client-side WebSocket
                ui.add_body_html("""
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

            # Control buttons bar
            with ui.row().classes("w-full gap-3 mt-4 justify-start"):
                ui.button("Start Camera", on_click=lambda: app.storage.user.update(cmd="start")).props("icon=play_arrow color=emerald").classes("px-4 rounded-lg font-semibold")
                ui.button("Stop Camera", on_click=lambda: app.storage.user.update(cmd="stop")).props("icon=stop color=red").classes("px-4 rounded-lg font-semibold")

        # Right side: Metrics and Verification Telemetry Card
        with ui.card().classes("dashboard-card p-4 flex flex-col gap-5"):
            ui.label("Detection & Verification Telemetry").classes("text-lg font-bold text-indigo-200 border-b border-slate-800 pb-2")

            # Pipeline Telemetry indicators
            with ui.column().classes("w-full gap-4"):
                # Detections statuses
                with ui.row().classes("w-full justify-between items-center"):
                    ui.label("Faces Detected").classes("text-slate-300 font-medium")
                    ui.add_body_html('<span id="val_faces_detected" class="status-indicator status-fail">0</span>')

                with ui.row().classes("w-full justify-between items-center"):
                    ui.label("Face Landmarks Mesh").classes("text-slate-300 font-medium")
                    ui.add_body_html('<span id="val_landmarks_mesh" class="status-indicator status-fail">OFFLINE</span>')

                with ui.row().classes("w-full justify-between items-center"):
                    ui.label("Pose Skeleton Tracker").classes("text-slate-300 font-medium")
                    ui.add_body_html('<span id="val_pose_skeleton" class="status-indicator status-fail">OFFLINE</span>')

            # Similarity Score Placeholders
            with ui.column().classes("w-full bg-slate-900/60 p-4 rounded-xl border border-slate-800/80 gap-3 mt-2"):
                ui.label("Face Verification Engine").classes("text-xs font-bold text-slate-400 uppercase tracking-wider")

                with ui.column().classes("w-full items-center justify-center p-3 rounded-lg bg-slate-950 border border-slate-900"):
                    ui.label("Cosine Similarity Score").classes("text-xs text-slate-400 mb-1")
                    ui.add_body_html('<span id="val_similarity_score" class="text-3xl font-extrabold text-indigo-400">0.0000</span>')
                    ui.add_body_html('<span id="val_similarity_desc" class="text-[10px] text-center text-slate-500 mt-2 max-w-[200px] leading-tight">No faces checked yet</span>')

                with ui.row().classes("w-full justify-between items-center px-1 mt-2"):
                    ui.label("Verification Match Decision").classes("text-slate-300 text-sm font-medium")
                    ui.add_body_html('<span id="val_match_status" class="status-indicator status-fail">NO MATCH</span>')

            # Info box
            with ui.row().classes("w-full bg-indigo-950/20 border border-indigo-900/30 p-3 rounded-lg text-xs text-indigo-300 gap-2 items-start mt-2"):
                ui.icon("info", size="xs").classes("mt-0.5")
                ui.add_body_html("""
                    <div>
                        <strong>Architecture Sandbox:</strong> This represents a pipeline wrapper. Match decisions and dummy embeddings simulate verification flows. No database queries or real user identity checks are performed.
                    </div>
                """)

    # Embedded Javascript client-side WebSocket communication layer.
    # Connects to backend WS path, pulls processed frames & telemetry, and dynamically updates elements.
    ui.add_body_html(f"""
        <script>
            (function() {{
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/ws';
                let ws;
                let reconnectInterval = 2000;

                function connect() {{
                    ws = new WebSocket(wsUrl);

                    ws.onopen = () => {{
                        console.log('WS Connection established for CV frame pipeline.');
                        const overlay = document.getElementById('stream_offline_overlay');
                        if (overlay) overlay.style.opacity = 0;
                        setTimeout(() => {{ if (overlay) overlay.style.display = 'none'; }}, 500);
                    }};

                    ws.onmessage = (event) => {{
                        const data = JSON.parse(event.data);

                        // 1. Draw frame if present
                        if (data.frame) {{
                            document.getElementById('video_stream').src = 'data:image/jpeg;base64,' + data.frame;
                        }}

                        // 2. Update telemetry labels
                        if (data.telemetry) {{
                            const t = data.telemetry;
                            
                            // Faces Detected Indicator
                            const fCount = document.getElementById('val_faces_detected');
                            if (fCount) {{
                                fCount.textContent = t.faces_detected;
                                if (t.faces_detected > 0) {{
                                    fCount.className = 'status-indicator status-ok';
                                }} else {{
                                    fCount.className = 'status-indicator status-fail';
                                }}
                            }}

                            // Face Mesh Indicator
                            const faceMesh = document.getElementById('val_landmarks_mesh');
                            if (faceMesh) {{
                                if (t.landmarks_detected) {{
                                    faceMesh.textContent = 'DETECTED';
                                    faceMesh.className = 'status-indicator status-ok';
                                }} else {{
                                    faceMesh.textContent = 'OFFLINE';
                                    faceMesh.className = 'status-indicator status-fail';
                                }}
                            }}

                            // Pose Skeleton Indicator
                            const poseSkeleton = document.getElementById('val_pose_skeleton');
                            if (poseSkeleton) {{
                                if (t.pose_detected) {{
                                    poseSkeleton.textContent = 'TRACKING';
                                    poseSkeleton.className = 'status-indicator status-ok';
                                }} else {{
                                    poseSkeleton.textContent = 'OFFLINE';
                                    poseSkeleton.className = 'status-indicator status-fail';
                                }}
                            }}

                            // Similarity Score
                            const scoreLabel = document.getElementById('val_similarity_score');
                            if (scoreLabel) {{
                                scoreLabel.textContent = t.best_similarity_score.toFixed(4);
                            }}

                            // Similarity Description
                            const scoreDesc = document.getElementById('val_similarity_desc');
                            if (scoreDesc && t.faces_detected > 0) {{
                                scoreDesc.textContent = 'Similarity score represents high-dimensional cosine proximity, not probability.';
                            }}

                            // Match Status
                            const matchStatus = document.getElementById('val_match_status');
                            if (matchStatus) {{
                                if (t.is_match) {{
                                    matchStatus.textContent = 'MATCH FOUND';
                                    matchStatus.className = 'status-indicator status-ok';
                                }} else {{
                                    matchStatus.textContent = t.faces_detected > 0 ? 'UNKNOWN' : 'NO FACE';
                                    matchStatus.className = 'status-indicator status-fail';
                                }}
                            }}
                        }}
                    }};

                    ws.onclose = () => {{
                        console.log('WS Connection lost. Retrying in ' + reconnectInterval + 'ms');
                        const overlay = document.getElementById('stream_offline_overlay');
                        if (overlay) {{
                            overlay.style.display = 'flex';
                            overlay.style.opacity = 1;
                        }}
                        setTimeout(connect, reconnectInterval);
                    }};

                    ws.onerror = (err) => {{
                        console.error('WS Connection error:', err);
                        ws.close();
                    }};
                }}

                window.addEventListener('DOMContentLoaded', connect);
            }})();
        </script>
    """)
