from nicegui import ui


def show_notification(
    text: str,
    type: str = "info",
    position: str = "bottom-right",
    duration: float = 2.0
) -> None:
    """
    Shows a temporary pop-up notification on the browser.
    Valid types: 'info', 'ongoing', 'positive', 'negative', 'warning'.
    """
    ui.notify(
        text,
        type=type,
        position=position,
        close_button=True,
        timeout=int(duration * 1000)
    )
