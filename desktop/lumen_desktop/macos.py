"""macOS extras through pyobjc - every function is best-effort and only called on Darwin.

pywebview builds the menubar as *App | Edit | View* (Edit is what makes Cmd+C / Cmd+V reach
WebKit) and appends our custom menus after it.  Here we (1) put our Reload / Zoom items with proper
key equivalents into that View menu instead of adding a second "View", (2) reorder the bar to
*App | File | Edit | View | Help*, (3) raise the window for the single-instance guard, (4) use
WKWebView's native ``pageZoom`` instead of CSS zoom, (5) receive the files Finder asks us to open
("Open With", a double-click on a document, a drop on the Dock icon) while the app runs, and (6) in debug
builds synthesise key presses through ``NSApplication.sendEvent:`` so a test can check that the page still
gets ⌘Z / ⌘C with the native Edit menu present."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger("lumen.desktop.macos")

_retained: list[Any] = []  # keep pyobjc objects referenced by AppKit alive


_HANDLER_CLASS: Any = None


def _handler_class():
    """An NSObject subclass (created once - Objective-C classes cannot be redefined) whose
    ``handleLumenAction:`` runs the Python callable registered for the menu item."""
    global _HANDLER_CLASS
    if _HANDLER_CLASS is not None:
        return _HANDLER_CLASS
    import AppKit  # noqa: WPS433
    import objc

    class LumenMenuHandler(AppKit.NSObject):
        def init(self):
            self = objc.super(LumenMenuHandler, self).init()
            if self is None:
                return None
            self.actions = {}
            return self

        def handleLumenAction_(self, sender):
            fn = self.actions.get(str(sender.representedObject()))
            if fn is not None:
                threading.Thread(target=fn, name="lumen-menu", daemon=True).start()

    _HANDLER_CLASS = LumenMenuHandler
    return _HANDLER_CLASS


def polish_menubar(view_items: list[tuple[str, Callable[[], Any], str, int] | None]) -> None:
    """``view_items``: (title, fn, key equivalent, extra modifier mask) tuples, ``None`` = separator.
    Inserted at the top of pywebview's View menu; then the menubar is reordered."""
    import AppKit
    from PyObjCTools import AppHelper

    def _do() -> None:
        try:
            app = AppKit.NSApplication.sharedApplication()
            main = app.mainMenu()
            if main is None:
                return
            def _title(item: Any) -> str:  # pywebview's top-level items carry the title on their submenu
                sub = item.submenu()
                return str(sub.title()) if sub is not None and str(sub.title()) else str(item.title())

            by_title = {_title(main.itemAtIndex_(i)): main.itemAtIndex_(i) for i in range(main.numberOfItems())}
            view = by_title.get("View")
            if view is not None and view.submenu() is not None and view_items:
                sub = view.submenu()
                if any(str(sub.itemAtIndex_(i).title()) == str(view_items[0][0]) for i in range(sub.numberOfItems()) if view_items[0]):
                    return  # already done (pywebview rebuilt nothing since)
                handler = _handler_class().alloc().init()
                _retained.append(handler)
                index = 0
                for entry in view_items:
                    if entry is None:
                        item = AppKit.NSMenuItem.separatorItem()
                    else:
                        title, fn, key, extra_mask = entry
                        name = f"lumen.{index}"
                        handler.actions[name] = fn
                        item = AppKit.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, "handleLumenAction:", key)
                        item.setTarget_(handler)
                        item.setRepresentedObject_(name)
                        if extra_mask:
                            item.setKeyEquivalentModifierMask_(AppKit.NSEventModifierFlagCommand | extra_mask)
                    sub.insertItem_atIndex_(item, index)
                    index += 1
            # App | File | Edit | View | Help
            order = ["File", "Edit", "View", "Help"]
            position = 1
            for title in order:
                item = by_title.get(title)
                if item is None:
                    continue
                main.removeItem_(item)
                main.insertItem_atIndex_(item, min(position, main.numberOfItems()))
                position += 1
        except Exception:  # noqa: BLE001
            log.warning("menubar polish failed", exc_info=True)

    AppHelper.callAfter(_do)


def set_dock_icon(path: str) -> None:
    """When running from source the process has no bundle icon; give the Dock ours."""
    import AppKit
    from PyObjCTools import AppHelper

    def _do() -> None:
        try:
            image = AppKit.NSImage.alloc().initWithContentsOfFile_(path)
            if image is not None:
                AppKit.NSApplication.sharedApplication().setApplicationIconImage_(image)
        except Exception:  # noqa: BLE001
            log.debug("dock icon failed", exc_info=True)

    AppHelper.callAfter(_do)


def set_page_zoom(window: Any, factor: float) -> bool:
    """WKWebView.pageZoom (macOS 11+) on the main thread; False when the native view is unavailable."""
    try:
        from PyObjCTools import AppHelper
        from webview.platforms.cocoa import BrowserView

        instance = BrowserView.instances.get(window.uid)
        view = getattr(instance, "webview", None) or getattr(instance, "webkit", None)  # the WKWebView (attribute name varies by version)
        if instance is None or view is None or not hasattr(view, "setPageZoom_"):
            log.debug("native zoom unavailable (instance=%s, view=%s)", instance is not None, type(view).__name__ if view is not None else None)
            return False

        def _do() -> None:
            try:
                view.setPageZoom_(float(factor))
            except Exception:  # noqa: BLE001
                log.debug("setPageZoom failed", exc_info=True)

        AppHelper.callAfter(_do)
        return True
    except Exception:  # noqa: BLE001
        log.debug("native zoom unavailable", exc_info=True)
        return False


def activate() -> None:
    try:
        import AppKit
        from PyObjCTools import AppHelper

        def _do() -> None:
            AppKit.NSRunningApplication.currentApplication().activateWithOptions_(AppKit.NSApplicationActivateIgnoringOtherApps)

        AppHelper.callAfter(_do)
    except Exception:  # noqa: BLE001
        log.debug("activate failed", exc_info=True)


def start_keepalive_tick(interval: float = 1.0) -> None:
    """A no-op Python callback on the main thread once a second: the Cocoa run loop otherwise runs no
    Python code while idle, so Python signal handlers (SIGTERM) would wait for the next UI event."""
    try:
        from PyObjCTools import AppHelper

        def _tick() -> None:
            AppHelper.callLater(interval, _tick)

        AppHelper.callAfter(lambda: AppHelper.callLater(interval, _tick))
    except Exception:  # noqa: BLE001
        log.debug("keepalive tick unavailable", exc_info=True)


def _fourcc(s: str) -> int:
    return int.from_bytes(s.encode("ascii"), "big")


_OPEN_CLASS: Any = None


def _open_handler_class():
    """The NSObject subclass receiving ``odoc`` (created once - Objective-C classes cannot be redefined); the Python
    callback sits on the class so a re-install only swaps the callable."""
    global _OPEN_CLASS
    if _OPEN_CLASS is not None:
        return _OPEN_CLASS
    import Foundation

    class LumenOpenHandler(Foundation.NSObject):
        callback = None

        def handleOpen_withReply_(self, event, reply):  # noqa: N802 - Objective-C selector
            paths: list[str] = []
            try:
                lst = event.paramDescriptorForKeyword_(_fourcc("----"))
                n = lst.numberOfItems() if lst is not None else 0
                for i in range(1, n + 1):
                    item = lst.descriptorAtIndex_(i).coerceToDescriptorType_(_fourcc("furl"))
                    if item is None:
                        continue
                    raw = Foundation.NSString.alloc().initWithData_encoding_(item.data(), Foundation.NSUTF8StringEncoding)
                    url = Foundation.NSURL.URLWithString_(raw) if raw else None
                    if url is not None and url.path():
                        paths.append(str(url.path()))
            except Exception:  # noqa: BLE001
                log.warning("could not read the open-documents event", exc_info=True)
            fn = type(self).callback
            if paths and fn is not None:
                log.info("Finder asks to open %s", paths)
                threading.Thread(target=fn, args=(paths,), name="lumen-odoc", daemon=True).start()

    _OPEN_CLASS = LumenOpenHandler
    return _OPEN_CLASS


# --------------------------------------------------------------------------- documents from Finder

_OPEN_HANDLER: Any = None


def install_launch_open_handler(on_open: Callable[[list[str]], None]) -> bool:
    """Catch the ``odoc`` event that *launches* Lumen (double-click on an image, Open With, a drop on the Dock icon
    while Lumen is not running).  The handler must be installed when AppKit announces
    ``NSApplicationWillFinishLaunchingNotification`` – earlier, AppKit's own default handlers replace it; later, the
    launching event has already been delivered.  Call before ``webview.start``.  (This replaces PyInstaller's
    argv emulation, which made every ``Lumen --mcp`` helper process check in with LaunchServices as "the running
    Lumen", so clicking the Dock icon activated a windowless helper instead of opening a window.)"""
    try:
        import AppKit
        import Foundation
    except Exception:  # noqa: BLE001
        log.debug("pyobjc unavailable - no launch open handler", exc_info=True)
        return False
    cls = _open_handler_class()
    if cls is None:
        return False
    cls.callback = staticmethod(on_open)

    class _LaunchObserver(Foundation.NSObject):
        def willFinish_(self, _note):  # noqa: N802 - Objective-C selector
            try:
                handler = cls.alloc().init()
                _retained.append(handler)
                Foundation.NSAppleEventManager.sharedAppleEventManager().setEventHandler_andSelector_forEventClass_andEventID_(
                    handler, "handleOpen:withReply:", _fourcc("aevt"), _fourcc("odoc"))
                log.info("launch open-documents handler installed")
            except Exception:  # noqa: BLE001
                log.warning("installing the launch open handler failed", exc_info=True)

    obs = _LaunchObserver.alloc().init()
    _retained.append(obs)
    Foundation.NSNotificationCenter.defaultCenter().addObserver_selector_name_object_(
        obs, "willFinish:", AppKit.NSApplicationWillFinishLaunchingNotification, None)
    return True


def install_open_documents_handler(on_open: Callable[[list[str]], None]) -> bool:
    """Route the ``odoc`` Apple event (Finder "Open With", double-click on an image whose default app is Lumen, a
    drop on the Dock icon) to ``on_open(paths)``.  Installed after launch: the files of the launching event itself
    arrive through PyInstaller's argv emulation (``sys.argv``); everything later comes through here."""
    global _OPEN_HANDLER
    try:
        import AppKit
        import Foundation
        import objc
        from PyObjCTools import AppHelper
    except Exception:  # noqa: BLE001
        log.debug("pyobjc unavailable - no Finder open handler", exc_info=True)
        return False

    cls = _open_handler_class()
    if cls is None:
        return False
    cls.callback = staticmethod(on_open)

    def _do() -> None:
        global _OPEN_HANDLER
        try:
            handler = cls.alloc().init()
            _retained.append(handler)
            Foundation.NSAppleEventManager.sharedAppleEventManager().setEventHandler_andSelector_forEventClass_andEventID_(
                handler, "handleOpen:withReply:", _fourcc("aevt"), _fourcc("odoc"))
            _OPEN_HANDLER = handler
            log.info("Finder open-documents handler installed")
        except Exception:  # noqa: BLE001
            log.warning("installing the open-documents handler failed", exc_info=True)

    AppHelper.callAfter(_do)
    return True


# --------------------------------------------------------------------------- debug: synthetic key presses

_KEYCODES = {"a": 0, "s": 1, "z": 6, "x": 7, "c": 8, "v": 9, "o": 31, "k": 40, "w": 13, "r": 15, "q": 12, "0": 29, "=": 24, "-": 27}


def send_key(combo: str) -> bool:
    """``"cmd+z"`` / ``"cmd+shift+z"``: an NSEvent key-down + key-up delivered with ``NSApplication.sendEvent:`` to the
    key window, i.e. through the same key-equivalent path a real key press takes (menus first, or the web view).
    In-process, so no accessibility permission is needed. Debug builds only (``--debug``)."""
    import AppKit
    from PyObjCTools import AppHelper

    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    key = parts[-1] if parts else ""
    if key not in _KEYCODES:
        return False
    mask = 0
    if "cmd" in parts:
        mask |= AppKit.NSEventModifierFlagCommand
    if "shift" in parts:
        mask |= AppKit.NSEventModifierFlagShift
    if "alt" in parts or "option" in parts:
        mask |= AppKit.NSEventModifierFlagOption
    if "ctrl" in parts:
        mask |= AppKit.NSEventModifierFlagControl
    chars = key.upper() if "shift" in parts and key.isalpha() else key

    def _do() -> None:
        app = AppKit.NSApplication.sharedApplication()
        win = app.keyWindow() or app.mainWindow() or (app.windows()[0] if app.windows() else None)
        if win is None:
            log.warning("send_key: no window")
            return
        for etype in (AppKit.NSEventTypeKeyDown, AppKit.NSEventTypeKeyUp):
            ev = AppKit.NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                etype, AppKit.NSMakePoint(0, 0), mask, AppKit.NSProcessInfo.processInfo().systemUptime(), win.windowNumber(), None, chars, key, False, _KEYCODES[key])
            app.sendEvent_(ev)

    AppHelper.callAfter(_do)
    return True


# --------------------------------------------------------------------------- debug: answer modal alerts and file panels


def _on_main(fn: Callable[[], Any], timeout: float = 10.0) -> Any:
    """Run ``fn`` on the main thread and wait for its result.  ``AppHelper.callAfter`` posts in the run loop's common
    modes, which include the modal-panel mode, so this also runs while an alert or a file panel is up."""
    from PyObjCTools import AppHelper

    box: dict[str, Any] = {}
    done = threading.Event()

    def _wrap() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = repr(exc)
        finally:
            done.set()

    AppHelper.callAfter(_wrap)
    if not done.wait(timeout):
        return "timeout"
    return box.get("value", box.get("error"))


def _buttons(view: Any) -> list[Any]:
    import AppKit

    out = []
    if view is None:
        return out
    if isinstance(view, AppKit.NSButton):
        out.append(view)
    for sub in view.subviews() or []:
        out += _buttons(sub)
    return out


def modal_click(title: str, timeout: float = 10.0) -> Any:
    """Debug builds: click the button titled ``title`` in the visible alert (the quit confirmation's "Quit" / "Cancel")."""
    import AppKit

    def _do() -> Any:
        app = AppKit.NSApplication.sharedApplication()
        for win in [app.modalWindow(), *list(app.windows())]:
            if win is None or not win.isVisible():
                continue
            for b in _buttons(win.contentView()):
                if str(b.title()).strip().lower() == title.strip().lower():
                    b.performClick_(None)
                    return f"clicked {b.title()!s} in {type(win).__name__}"
        return f"no visible button titled {title!r}"

    return _on_main(_do, timeout)


def panel_choose(path: str, timeout: float = 10.0) -> Any:
    """Debug builds: answer the visible Open / Save panel with ``path``.  The panel's UI runs in a separate system
    process (NSRemoteView), which no in-process event reaches; its public API does, though: the folder and the file name
    are set on the panel and its modal session is ended with the OK code - pywebview then reads the panel's own result
    (``URL`` / ``URLs``), as after a person's click.  Returns what the panel reported."""
    import os

    import AppKit
    import Foundation

    def _do() -> Any:
        app = AppKit.NSApplication.sharedApplication()
        win = app.modalWindow()
        if win is None or not isinstance(win, AppKit.NSSavePanel):
            return "no modal file panel"
        if isinstance(win, AppKit.NSOpenPanel):
            win.setDirectoryURL_(Foundation.NSURL.fileURLWithPath_(path))
        else:
            win.setDirectoryURL_(Foundation.NSURL.fileURLWithPath_(os.path.dirname(path)))
            win.setNameFieldStringValue_(os.path.basename(path))
        return type(win).__name__

    kind = _on_main(_do, timeout)
    if "Panel" not in str(kind):
        return kind
    time.sleep(1.0)

    def _ok() -> Any:
        app = AppKit.NSApplication.sharedApplication()
        win = app.modalWindow()
        if win is None:
            return "panel gone"
        sel = [str(u.path()) for u in (win.URLs() or [])] if isinstance(win, AppKit.NSOpenPanel) else ([str(win.URL().path())] if win.URL() else [])
        app.stopModalWithCode_(AppKit.NSModalResponseOK)
        return f"{kind}: OK with {sel}"

    return _on_main(_ok, timeout)
