"""Native menus (pywebview ``Menu`` objects): the macOS menubar, or the in-window menu bar on Windows / Linux.

Lumen has its full Fiji-style menubar inside the page (File, Edit, Image, Process, Analyze, Plugins …), so the native
menus only hold what belongs to the application around it: opening files through the system dialog, the Lumen
folder, the browser escape hatch, view zoom and the About window.  Actions are plain callables from ``app.py``."""
from __future__ import annotations

import sys
from typing import Any, Callable


def build_menus(actions: dict[str, Callable[[], Any]], *, include_view: bool = True) -> list:
    from webview.menu import Menu, MenuAction, MenuSeparator

    file_items = [
        MenuAction("Open Image…", actions["open_images"]),
        MenuAction("Open Folder as Stack…", actions["open_folder"]),
        MenuSeparator(),
        MenuAction("Open Lumen Folder (plugins, macros, logs)", actions["open_home"]),
        MenuAction("Open in Browser", actions["open_in_browser"]),
    ]
    if sys.platform != "darwin":  # macOS has Quit in the application menu already
        file_items += [MenuSeparator(), MenuAction("Quit", actions["quit"])]
    menus = [Menu("File", file_items)]
    if include_view:
        menus.append(
            Menu(
                "View",
                [
                    MenuAction("Reload", actions["reload"]),
                    MenuSeparator(),
                    MenuAction("Zoom In", actions["zoom_in"]),
                    MenuAction("Zoom Out", actions["zoom_out"]),
                    MenuAction("Actual Size", actions["zoom_reset"]),
                    MenuSeparator(),
                    MenuAction("Toggle Full Screen", actions["fullscreen"]),
                ],
            )
        )
    menus.append(
        Menu(
            "Help",
            [
                MenuAction("Lumen Documentation", actions["docs"]),
                MenuAction("Keyboard Shortcuts", actions["shortcuts"]),
                MenuAction("Show Log File", actions["show_log"]),
                MenuSeparator(),
                MenuAction("About Lumen", actions["about"]),
            ],
        )
    )
    return menus
