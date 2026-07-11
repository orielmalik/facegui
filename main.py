from nicegui import ui
import cv2
import asyncio
import base64

image = ui.image()

cap = cv2.VideoCapture(0)

async def stream():

    while True:

        ret, frame = cap.read()

        if ret:

            success, encoded = cv2.imencode(
                '.jpg',
                frame
            )

            if success:

                image.source = (
                    'data:image/jpeg;base64,' +
                    base64.b64encode(encoded).decode()
                )

        await asyncio.sleep(0.03)

ui.timer(
    0.1,
    lambda: asyncio.create_task(stream()),
    once=True
)

ui.run()