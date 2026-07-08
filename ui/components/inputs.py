from typing import Callable, Any, Dict, List, Optional, Union
from nicegui import ui


def create_text_input(
    label: str,
    value: str = "",
    on_change: Optional[Callable[[Any], None]] = None
) -> ui.input:
    """
    Creates a styled NiceGUI text input field.
    """
    ipt = ui.input(label=label, value=value, on_change=on_change)
    ipt.classes("w-full bg-slate-800 rounded-lg text-white border-slate-700")
    return ipt


def create_checkbox(
    label: str,
    value: bool = False,
    callback: Optional[Callable[[Any], None]] = None
) -> ui.checkbox:
    """
    Creates a styled NiceGUI checkbox.
    """
    cb = ui.checkbox(label, value=value, on_change=callback)
    cb.classes("text-slate-300 font-medium")
    return cb


def create_slider(
    label: str,
    min_value: float = 0.0,
    max_value: float = 1.0,
    value: float = 0.5,
    step: float = 0.05,
    callback: Optional[Callable[[Any], None]] = None
) -> ui.slider:
    """
    Creates a styled NiceGUI slider control with value labels.
    """
    with ui.column().classes("w-full gap-1"):
        ui.label(label).classes("text-xs text-slate-400 font-semibold")
        sld = ui.slider(min=min_value, max=max_value, step=step, value=value, on_change=callback)
        sld.props("label label-always")
        sld.classes("w-full")
    return sld


def create_select(
    options: Union[List[Any], Dict[Any, Any]],
    label: Optional[str] = None,
    value: Optional[Any] = None,
    callback: Optional[Callable[[Any], None]] = None
) -> ui.select:
    """
    Creates a styled NiceGUI selection drop-down.
    """
    sel = ui.select(options=options, label=label, value=value, on_change=callback)
    sel.classes("w-full bg-slate-800 rounded-lg text-white")
    return sel
