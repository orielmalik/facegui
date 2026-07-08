from typing import Optional
from nicegui import ui


def create_row(
    align_items: str = "start",
    gap: str = "4"
) -> ui.row:
    """
    Creates a flex row layout wrapper.
    Returns the underlying NiceGUI row for nesting children.
    """
    row = ui.row()
    row.classes(f"items-{align_items} gap-{gap} w-full")
    return row


def create_column(
    align_items: str = "start",
    gap: str = "4"
) -> ui.column:
    """
    Creates a flex column layout wrapper.
    Returns the underlying NiceGUI column for nesting children.
    """
    col = ui.column()
    col.classes(f"items-{align_items} gap-{gap} w-full")
    return col


def create_card(
    background_class: str = "bg-slate-900/60",
    border_class: str = "border-slate-800"
) -> ui.card:
    """
    Creates a stylized card container.
    """
    card = ui.card()
    card.classes(f"w-full p-4 rounded-2xl border shadow-lg {background_class} {border_class}")
    return card


def create_tabs() -> ui.tabs:
    """
    Creates a tabs selector wrapper.
    """
    return ui.tabs().classes("w-full border-b border-slate-800 text-slate-400 active:text-white")


def create_dialog() -> ui.dialog:
    """
    Creates a pop-up dialog box container.
    """
    dialog = ui.dialog()
    # Add default styling wrappers to layout internally
    dialog.classes("bg-slate-900 border border-slate-800 p-6 rounded-2xl shadow-2xl")
    return dialog
