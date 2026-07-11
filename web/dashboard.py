import asyncio
import base64
import logging

from nicegui import ui

logger = logging.getLogger(__name__)


def build_modern_dashboard(deps):
    enroll_mode = False
    processing_task = None

    ui.dark_mode().enable()

    # תמונה חיה
    video = ui.image().classes("rounded-xl border-4 border-slate-700 shadow-2xl").style(
        "width:640px; height:480px; object-fit:contain;")

    name_input = ui.input("Person Name for Enrollment").classes("w-80 mt-4")

    with ui.row().classes("gap-4 my-6 justify-center"):
        start_btn = ui.button("▶ Start Camera", color="green", icon="videocam").props("size=lg")
        stop_btn = ui.button("⏹ Stop Camera", color="red", icon="stop").props("size=lg")
        enroll_btn = ui.button("📸 CAPTURE & ENROLL", color="amber", icon="person_add").props("size=lg")

    status_face = ui.label("No Face").classes("text-lg font-medium")
    status_system = ui.label("Camera stopped").classes("text-lg font-medium")

    # ====================== לולאה ראשית ======================
    async def update_frame_loop():
        nonlocal enroll_mode

        while True:
            success, frame = deps.camera_manager.read_frame()
            if not success or frame is None:
                await asyncio.sleep(0.05)
                continue

            rgb = deps.cv_processor.bgr_to_rgb(frame)
            annotated = frame.copy()

            # InsightFace
            faces = deps.insightface_service.detect_faces(rgb)

            if faces:
                status_face.set_text(f"Face detected ({len(faces)})")
                status_face.classes("text-green-400")

                for face in faces:
                    bbox = face["bbox"]
                    embedding = face["embedding"]

                    deps.cv_processor.draw_bounding_box(annotated, bbox, "Face", (0, 255, 0))

                    # ==================== ENROLLMENT ====================
                    if enroll_mode and name_input.value.strip():
                        name = name_input.value.strip()
                        success, _ = deps.verification_service.register(name, embedding)

                        if success:
                            ui.notify(f"✅ {name} registered successfully!", type="positive")
                        else:
                            ui.notify("Registration failed", type="negative")

                        enroll_mode = False
                        name_input.value = ""

            else:
                status_face.set_text("No Face")
                status_face.classes("text-red-400")

            # MediaPipe Landmarks + Pose
            landmarks = deps.mediapipe_service.detect_landmarks(rgb)
            for lm in landmarks:
                deps.cv_processor.draw_landmarks(annotated, lm)

            pose = deps.mediapipe_service.detect_pose(rgb)
            if pose:
                deps.cv_processor.draw_skeleton(
                    annotated, pose, deps.mediapipe_service.get_pose_connections()
                )

            # עדכון UI
            jpeg = deps.cv_processor.to_jpeg(annotated, quality=78)
            b64 = base64.b64encode(jpeg).decode("utf-8")
            video.set_source(f"data:image/jpeg;base64,{b64}")

            await asyncio.sleep(0.033)  # ~30 FPS

    # ====================== כפתורים ======================
    async def start_camera():
        nonlocal processing_task
        try:
            deps.camera_manager.open()
            status_system.set_text("Camera running")
            status_system.classes("text-green-400")
            processing_task = asyncio.create_task(update_frame_loop())
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")

    def stop_camera():
        nonlocal processing_task
        if processing_task:
            processing_task.cancel()
        deps.camera_manager.close()
        status_system.set_text("Camera stopped")
        status_system.classes("text-red-400")

    def enroll_person():
        if not name_input.value.strip():
            ui.notify("Enter name first", type="warning")
            return
        deps.pipeline.request_enroll(name_input.value.strip())
        ui.notify(f"Enrollment requested for {name_input.value}", type="positive")
        name_input.value = ""


    start_btn.on_click(start_camera)
    stop_btn.on_click(stop_camera)
    enroll_btn.on_click(enroll_person)