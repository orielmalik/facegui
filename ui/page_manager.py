import logging
from typing import Optional, Dict, Any, Callable
from nicegui import ui

from ui.components.camera_view import create_camera_panel
from ui.components.panels import create_status_panel, create_control_panel
from ui.components.layouts import create_column, create_row

logger = logging.getLogger(__name__)


class DashboardPage:
    """
    Page manager abstraction that structures and builds the Web UI layout
    without requiring raw NiceGUI code everywhere.
    """

    def __init__(self, title: str) -> None:
        """Initializes dark mode, global page styling, and page grid containers."""
        # Enable Dark Mode
        dark = ui.dark_mode()
        dark.enable()

        # Modern Fonts & Styles injection
        ui.add_head_html("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
                body {
                    font-family: 'Outfit', sans-serif;
                    background-color: #0f172a;
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

        # 1. Header Navigation
        with ui.header().classes("bg-slate-900 border-b border-slate-800 text-white p-4 flex items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("videocam", size="md").classes("text-indigo-400 animate-pulse")
                ui.label(title).classes("text-xl font-bold tracking-wide")
            with ui.row().classes("items-center gap-2"):
                ui.label("System Status:").classes("text-slate-400 text-sm")
                ui.badge("ACTIVE", color="emerald").classes("px-2 py-1 text-xs font-semibold")

        # 2. Main Page Grid (responsive layout: 3-columns on desktop, stacked on mobile)
        self._grid = ui.grid(columns="1fr").classes("w-full max-w-7xl mx-auto p-4 gap-6 lg:grid-cols-3")
        
        # Grid slots for panels
        self.camera_card: Optional[ui.card] = None
        self.controls_card: Optional[ui.card] = None
        self.status_card: Optional[ui.card] = None
        
        # Storage for sub-elements for reference
        self.ui_elements: Dict[str, Any] = {}

        # Right sidebar column container (groups controls & status panels)
        with self._grid:
            self._left_col = create_column(gap="6").classes("lg:col-span-2")
            self._right_col = create_column(gap="6").classes("lg:col-span-1")

    def add_camera_view(self, title: str = "Live Camera Stream", resolution: str = "640x480") -> "DashboardPage":
        """Adds the video streaming canvas card to the left side column."""
        with self._left_col:
            card, img_html = create_camera_panel(title, resolution)
            self.camera_card = card
            self.ui_elements["video_image"] = img_html
        return self

    def add_controls(self, on_start: Optional[Callable] = None, on_stop: Optional[Callable] = None) -> "DashboardPage":
        """Adds control actions (Start/Stop feed buttons) to the right side column."""
        with self._right_col:
            card = create_control_panel(on_start=on_start, on_stop=on_stop)
            self.controls_card = card
        return self

    def add_status(self) -> "DashboardPage":
        """Adds telemetry indicator items to the right side column."""
        with self._right_col:
            card, elements = create_status_panel()
            self.status_card = card
            self.ui_elements.update(elements)
        return self

    def add_client_script(self) -> None:
        """Mounts the client-side JavaScript that receives WebSocket frames and telemetry."""
        ui.add_body_html("""
            <script>
                (function() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = protocol + '//' + window.location.host + '/ws';
                    let ws;
                    let reconnectInterval = 2000;

                    function connect() {
                        ws = new WebSocket(wsUrl);

                        ws.onopen = () => {
                            console.log('WS Connection established for CV frame pipeline.');
                            const overlay = document.getElementById('stream_offline_overlay');
                            if (overlay) overlay.style.opacity = 0;
                            setTimeout(() => { if (overlay) overlay.style.display = 'none'; }, 500);
                        };

                        ws.onmessage = (event) => {
                            const data = JSON.parse(event.data);

                            if (data.frame) {
                                document.getElementById('video_stream').src = 'data:image/jpeg;base64,' + data.frame;
                            }

                            if (data.telemetry) {
                                const t = data.telemetry;
                                
                                const fCount = document.getElementById('val_faces_detected');
                                if (fCount) {
                                    fCount.textContent = t.faces_detected;
                                    fCount.className = t.faces_detected > 0 ? 'status-indicator status-ok' : 'status-indicator status-fail';
                                }

                                const faceMesh = document.getElementById('val_landmarks_mesh');
                                if (faceMesh) {
                                    if (t.landmarks_detected) {
                                        faceMesh.textContent = 'DETECTED';
                                        faceMesh.className = 'status-indicator status-ok';
                                    } else {
                                        faceMesh.textContent = 'OFFLINE';
                                        faceMesh.className = 'status-indicator status-fail';
                                    }
                                }

                                const poseSkeleton = document.getElementById('val_pose_skeleton');
                                if (poseSkeleton) {
                                    if (t.pose_detected) {
                                        poseSkeleton.textContent = 'TRACKING';
                                        poseSkeleton.className = 'status-indicator status-ok';
                                    } else {
                                        poseSkeleton.textContent = 'OFFLINE';
                                        poseSkeleton.className = 'status-indicator status-fail';
                                    }
                                }

                                const scoreLabel = document.getElementById('val_similarity_score');
                                if (scoreLabel) {
                                    scoreLabel.textContent = t.best_similarity_score.toFixed(4);
                                }

                                const scoreDesc = document.getElementById('val_similarity_desc');
                                if (scoreDesc && t.faces_detected > 0) {
                                    scoreDesc.textContent = 'Similarity score represents high-dimensional cosine proximity, not probability.';
                                }

                                const matchStatus = document.getElementById('val_match_status');
                                if (matchStatus) {
                                    if (t.is_match) {
                                        matchStatus.textContent = 'MATCH FOUND';
                                        matchStatus.className = 'status-indicator status-ok';
                                    } else {
                                        matchStatus.textContent = t.faces_detected > 0 ? 'UNKNOWN' : 'NO FACE';
                                        matchStatus.className = 'status-indicator status-fail';
                                    }
                                }
                            }
                        };

                        ws.onclose = () => {
                            console.log('WS Connection lost. Retrying in ' + reconnectInterval + 'ms');
                            const overlay = document.getElementById('stream_offline_overlay');
                            if (overlay) {
                                overlay.style.display = 'flex';
                                overlay.style.opacity = 1;
                            }
                            setTimeout(connect, reconnectInterval);
                        };

                        ws.onerror = (err) => {
                            console.error('WS Connection error:', err);
                            ws.close();
                        };
                    }

                    window.addEventListener('DOMContentLoaded', connect);
                })();
            </script>
        """)


def create_dashboard(title: str = "Real-Time CV Sandbox") -> DashboardPage:
    """
    Builder function to initialize a DashboardPage instance.
    """
    return DashboardPage(title=title)
