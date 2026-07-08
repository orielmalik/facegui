from typing import Callable, Optional
from nicegui import ui


def create_button(
    text: str,
    on_click: Optional[Callable] = None,
    color: str = "primary",
    icon: Optional[str] = None
) -> ui.button:
    """
    Creates a customizable button element.
    Exposes the underlying NiceGUI ui.button component directly.
    """
    btn = ui.button(text, on_click=on_click)
    if color:
        btn.props(f"color={color}")
    if icon:
        btn.props(f"icon={icon}")
    
    # Custom baseline class styling
    btn.classes("px-4 py-2 rounded-lg font-semibold transition-all hover:scale-105 active:scale-95")
    return btn
