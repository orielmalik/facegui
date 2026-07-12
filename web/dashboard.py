import asyncio
import base64
import logging

from nicegui import ui

logger = logging.getLogger(__name__)
def build_modern_dashboard(deps):
    enroll_mode = False
    processing_task = None

    ui.dark_mode().enable()

    video = ui.image().classes("rounded-xl border-4 border-slate-700 shadow-2xl").style(
        "width:640px; height:480px; object-fit:contain;")

    name_input = ui.input("Person Name").classes("w-80 mt-4")

    with ui.row().classes("gap-4 my-6 justify-center"):
        start_btn = ui.button("Start Camera", color="green", icon="videocam").props("size=lg")
        stop_btn = ui.button("Stop Camera", color="red", icon="stop").props("size=lg")
        enroll_btn = ui.button("Capture & Enroll", color="amber", icon="person_add").props("size=lg")

    status_face = ui.label("No Face").classes("text-lg font-medium")
    identified_name = ui.label("Unknown").classes("text-2xl font-bold text-green-400 mt-4")
    similarity_score = ui.label("Similarity: 0.00").classes("text-lg text-slate-300")

    deps.event_hub.subscribe("person_identified", lambda data:
    identified_name.set_text(f"✅ {data.get('name', 'Unknown')}")
                             )

    deps.event_hub.subscribe("person_identified", lambda data:
    similarity_score.set_text(f"Similarity: {data.get('similarity', 0.0):.3f}")
                             )
    # ====================== לולאה ======================
    async def update_frame_loop():
        nonlocal enroll_mode
        frame_counter = 0

        while True:
            success, frame = deps.camera_manager.read_frame()
            if not success or frame is None:
                await asyncio.sleep(0.05)
                continue

            frame_counter += 1
            do_ai = (frame_counter % 4 == 0)  # AI רק כל 4 פריימים (~7-8 FPS)

            rgb = deps.cv_processor.bgr_to_rgb(frame)
            annotated = frame.copy()

            if do_ai:
                faces = deps.insightface_service.detect_faces(rgb)
            else:
                faces = []

            if faces:
                status_face.set_text(f"Face detected ({len(faces)})")
                for face in faces:
                    bbox = face["bbox"]
                    embedding = face["embedding"]

                    deps.cv_processor.draw_bounding_box(annotated, bbox, "Face", (0, 255, 0))

                    if enroll_mode and name_input.value.strip():
                        name = name_input.value.strip()
                        success, _ = deps.verification_service.register(name, embedding)
                        if success:
                            ui.notify(f"Registered: {name}", type="positive")
                        enroll_mode = False
                        name_input.value = ""

            # MediaPipe - רק כשצריך
            if do_ai:
                landmarks = deps.mediapipe_service.detect_landmarks(rgb)
                for lm in landmarks:
                    deps.cv_processor.draw_landmarks(annotated, lm)

                pose = deps.mediapipe_service.detect_pose(rgb)
                if pose:
                    deps.cv_processor.draw_skeleton(annotated, pose, deps.mediapipe_service.get_pose_connections())

            jpeg = deps.cv_processor.to_jpeg(annotated, quality=60)
            b64 = base64.b64encode(jpeg).decode("utf-8")
            video.set_source(f"data:image/jpeg;base64,{b64}")

            await asyncio.sleep(0.04)  # 25 FPS max - פחות עומס
    # ====================== כפתורים ======================
    async def start_camera():
        nonlocal processing_task
        try:
            deps.camera_manager.open()
            processing_task = asyncio.create_task(update_frame_loop())
        except Exception as e:
            ui.notify(f"Error: {e}", type="negative")

    def stop_camera():
        nonlocal processing_task
        if processing_task:
            processing_task.cancel()
        deps.camera_manager.close()

    def enroll_person():
        if not name_input.value.strip():
            ui.notify("Enter name first", type="warning")
            return
        enroll_mode = True
        ui.notify("Look at camera...", type="positive")

    start_btn.on_click(start_camera)
    stop_btn.on_click(stop_camera)
    enroll_btn.on_click(enroll_person)