import asyncio
import base64
import logging

from nicegui import ui

from app.dependencies import DependencyContainer

logger = logging.getLogger(__name__)


def build_modern_dashboard(deps: DependencyContainer):

    enroll_mode = False
    processing_task = None
    current_embedding = None

    ui.dark_mode().enable()

    video = ui.image().classes(
        "rounded-xl border shadow-lg"
    ).style(
        """
        width:640px;
        height:480px;
        object-fit:contain;
        """
    )

    status_face = ui.label("No Face")
    status_system = ui.label("Camera stopped")

    name_input = ui.input(
        "Person name"
    ).classes("w-full")


    with ui.row():

        start_btn = ui.button(
            "Start Camera",
            color="green"
        )

        stop_btn = ui.button(
            "Stop Camera",
            color="red"
        )

        enroll_btn = ui.button(
            "CAPTURE & ENROLL",
            color="amber"
        )


    async def update_frame_loop():

        nonlocal current_embedding

        while True:

            success, frame = deps.camera_manager.read_frame()

            if not success or frame is None:
                await asyncio.sleep(0.05)
                continue


            rgb = deps.cv_processor.bgr_to_rgb(frame)

            annotated = frame.copy()


            # ======================
            # INSIGHTFACE
            # ======================

            faces = deps.insightface_service.detect_faces(rgb)


            if faces:

                status_face.set_text(
                    "Face detected"
                )


                for face in faces:

                    bbox = face["bbox"]

                    current_embedding = face["embedding"]


                    deps.cv_processor.draw_bounding_box(
                        annotated,
                        bbox,
                        "Face",
                        (0,255,0)
                    )


            else:

                status_face.set_text(
                    "No Face"
                )


            # ======================
            # MEDIAPIPE LANDMARKS
            # ======================

            landmarks = deps.mediapipe_service.detect_landmarks(rgb)


            for lm in landmarks:

                deps.cv_processor.draw_landmarks(
                    annotated,
                    lm
                )


            # ======================
            # POSE
            # ======================

            pose = deps.mediapipe_service.detect_pose(rgb)


            if pose:

                deps.cv_processor.draw_skeleton(
                    annotated,
                    pose,
                    deps.mediapipe_service.get_pose_connections()
                )


            # ======================
            # SEND IMAGE TO UI
            # ======================

            jpeg = deps.cv_processor.to_jpeg(
                annotated,
                quality=75
            )


            encoded = base64.b64encode(
                jpeg
            ).decode()


            video.set_source(
                f"data:image/jpeg;base64,{encoded}"
            )


            await asyncio.sleep(0.033)



    async def start_camera():

        nonlocal processing_task


        try:

            deps.camera_manager.open()


            status_system.set_text(
                "Camera running"
            )


            processing_task = asyncio.create_task(
                update_frame_loop()
            )


        except Exception as e:

            logger.exception(e)

            ui.notify(
                str(e),
                type="negative"
            )



    def stop_camera():

        nonlocal processing_task


        if processing_task:

            processing_task.cancel()
            processing_task = None


        deps.camera_manager.close()


        status_system.set_text(
            "Camera stopped"
        )



    async def enable_enroll():

        nonlocal enroll_mode


        if not name_input.value:

            ui.notify(
                "Enter name first",
                type="warning"
            )

            return


        enroll_mode = True


        ui.notify(
            "Ready to capture face",
            type="positive"
        )



    start_btn.on_click(
        start_camera
    )


    stop_btn.on_click(
        stop_camera
    )


    enroll_btn.on_click(
        enable_enroll
    )