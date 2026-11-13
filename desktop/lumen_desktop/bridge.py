"""The JavaScript side of the shell.

``DesktopApi`` is exposed as ``window.pywebview.api`` (pywebview ``js_api``).  Besides the shell actions it gives
the page **native file dialogs**: the frontend (``frontend/src/native.ts``) sends every File ▸ Open / Save / Browse…
through ``native_open_images`` / ``native_pick_save`` / ``native_pick_open`` / ``native_pick_folder`` when it runs
inside the desktop app, and keeps its in-app server file browser for the plain browser.  ``INJECTED_JS`` runs after
every page load and (a) opens external ``http(s)`` links in the system browser, (b) routes ``<a download>`` clicks
through a native save dialog when the engine's own download support is not available, (c) provides keyboard
shortcuts for the in-window menu on Windows / Linux, (d) a tiny toast and the CSS zoom fallback.  Everything is
idempotent (``window.__lumen`` guard)."""
from __future__ import annotations

import base64
import logging
import os
import re
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover
    from lumen_desktop.state import DesktopState

log = logging.getLogger("lumen.desktop.bridge")

_SAFE_NAME = re.compile(r"[^\w.\-() ]+")


def safe_filename(name: str, fallback: str = "download") -> str:
    name = _SAFE_NAME.sub("_", (name or "").strip()).strip("._ ")
    return name[:150] or fallback


def downloads_dir() -> Path:
    home = Path.home()
    for candidate in (home / "Downloads", home):
        if candidate.is_dir():
            return candidate
    return home


def open_in_file_manager(path: Path) -> None:
    """Reveal a folder in Finder / Explorer / the desktop's file manager."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        log.warning("could not open %s", path, exc_info=True)


def open_url(url: str) -> None:
    try:
        webbrowser.open(url, new=2)
    except Exception:  # noqa: BLE001
        log.warning("could not open %s", url, exc_info=True)


#: the open dialog's filter (pywebview syntax): everything Lumen reads natively; "All files" for the optional readers
#: (Bio-Formats, the [formats] extras) and folders of images
IMAGE_PATTERNS = ("*.tif;*.tiff;*.ome.tif;*.ome.tiff;*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.webp;*.jp2;*.pgm;*.ppm;*.tga;*.npy;"
                  "*.nd2;*.czi;*.lif;*.lsm;*.stk;*.oif;*.oib;*.ims;*.h5;*.hdf5;*.zvi;*.lei;*.dm3;*.dm4;"
                  "*.dcm;*.dicom;*.nii;*.nii.gz;*.mrc;*.mrcs;*.rec;*.map;*.fits;*.fit;*.fts;*.mha;*.mhd;*.nrrd;*.nhdr;"
                  "*.avi;*.mp4;*.mov;*.mkv;*.webm;*.m4v")
OPEN_FILE_TYPES = (f"Images ({IMAGE_PATTERNS})", "All files (*.*)")


def _file_types(extensions: list[str] | None) -> tuple[str, ...]:
    exts = [e if e.startswith(".") else "." + e for e in (extensions or []) if isinstance(e, str) and e.strip()]
    if not exts:
        return ()
    return (f"Files ({';'.join('*' + e for e in exts)})", "All files (*.*)")


def _initial_dir(*candidates: str | None) -> str:
    for c in candidates:
        if c:
            p = Path(c).expanduser()
            if p.is_file():
                p = p.parent
            if p.is_dir():
                return str(p)
    return str(Path.home())


class DesktopApi:
    """Methods callable from the page as ``window.pywebview.api.<name>(...)`` (each runs in its own thread)."""

    def __init__(self, state: DesktopState, server_url: str, actions: dict[str, Callable[[], Any]]):
        self._state = state
        self._server_url = server_url
        self._actions = actions  # reload / zoom_in / zoom_out / zoom_reset / fullscreen (filled in by app.py)
        self.window: Any = None  # set once the window exists
        self._on_unsaved: Callable[[int], Any] | None = None  # app.Shell.set_unsaved

    # ----------------------------------------------------------------- navigation
    def open_external(self, url: str) -> bool:
        if not isinstance(url, str) or not url.lower().startswith(("http://", "https://", "mailto:")):
            return False
        log.info("opening externally: %s", url)
        open_url(url)
        return True

    # ----------------------------------------------------------------- downloads (fallback path)
    def save_download(self, filename: str, mime: str, data_b64: str) -> str | None:
        """Native save dialog + write; returns the path or None when cancelled."""
        import webview

        if self.window is None:
            return None
        name = safe_filename(filename or "")
        initial = self._state.last_save_dir or str(downloads_dir())
        result = self.window.create_file_dialog(webview.SAVE_DIALOG, directory=initial, save_filename=name)
        target = _dialog_path(result)
        if not target:
            return None
        try:
            data = base64.b64decode(data_b64 or "")
            Path(target).write_bytes(data)
        except Exception as exc:  # noqa: BLE001
            log.exception("saving %s failed", target)
            self.toast(f"Could not save {name}: {exc}")
            return None
        self._state.last_save_dir = str(Path(target).parent)
        log.info("saved download %s (%d bytes, %s)", target, len(data), mime or "?")
        self.toast(f"Saved {Path(target).name}")
        return target

    # ----------------------------------------------------------------- native file dialogs (frontend/src/native.ts)
    def native_open_images(self, initial: str | None = None, multiple: bool = True) -> list[str] | None:
        """File ▸ Open: the system dialog with an image filter; returns the chosen paths or None (cancelled)."""
        import webview

        if self.window is None:
            return None
        result = self.window.create_file_dialog(webview.OPEN_DIALOG, directory=_initial_dir(initial, self._state.last_open_dir),
                                                allow_multiple=bool(multiple), file_types=OPEN_FILE_TYPES)
        paths = _dialog_paths(result)
        if paths:
            self._state.last_open_dir = str(Path(paths[0]).parent)
            self._state.save()
        return paths or None

    def native_pick_open(self, title: str | None = None, extensions: list[str] | None = None, initial: str | None = None) -> str | None:
        """A "Browse…" for an input file (a script, a table, a classifier model): one path or None."""
        import webview

        if self.window is None:
            return None
        kwargs: dict[str, Any] = {"directory": _initial_dir(initial, self._state.last_open_dir), "allow_multiple": False}
        types = _file_types(extensions)
        if types:
            kwargs["file_types"] = types
        path = _dialog_path(self.window.create_file_dialog(webview.OPEN_DIALOG, **kwargs))
        if path:
            self._state.last_open_dir = str(Path(path).parent)
            self._state.save()
        return path

    def native_pick_save(self, title: str | None = None, filename: str | None = None, initial: str | None = None) -> str | None:
        """File ▸ Save As… / an output-file "Browse…": the system save dialog; the chosen path or None."""
        import webview

        if self.window is None:
            return None
        name = safe_filename(filename or "", "untitled")
        path = _dialog_path(self.window.create_file_dialog(webview.SAVE_DIALOG, directory=_initial_dir(initial, self._state.last_save_dir), save_filename=name))
        if path:
            self._state.last_save_dir = str(Path(path).parent)
            self._state.save()
        return path

    def native_pick_folder(self, title: str | None = None, initial: str | None = None) -> str | None:
        """A folder: batch input / output, Open Folder as Stack."""
        import webview

        if self.window is None:
            return None
        path = _dialog_path(self.window.create_file_dialog(webview.FOLDER_DIALOG, directory=_initial_dir(initial, self._state.last_open_dir)))
        if path:
            self._state.last_open_dir = path
            self._state.save()
        return path

    # ----------------------------------------------------------------- shell actions (keyboard shortcuts)
    def action(self, name: str) -> bool:
        fn = self._actions.get(name)
        if fn is None:
            return False
        try:
            fn()
        except Exception:  # noqa: BLE001
            log.exception("action %s failed", name)
            return False
        return True

    def set_unsaved(self, count: int) -> bool:
        """The page reports how many images have unsaved changes (quit confirmation, see app.Shell.set_unsaved)."""
        fn = self._on_unsaved
        return bool(fn(count)) if fn is not None else False

    def info(self) -> dict[str, Any]:
        return {"url": self._server_url, "zoom": self._state.zoom, "platform": sys.platform, "native_dialogs": True, "app": "lumen-desktop"}

    # ----------------------------------------------------------------- helpers
    def toast(self, message: str) -> None:
        if self.window is None:
            return
        try:
            self.window.evaluate_js(f"window.__lumen && window.__lumen.toast({_js_str(message)})")
        except Exception:  # noqa: BLE001
            log.debug("toast failed", exc_info=True)


def _dialog_path(result: Any) -> str | None:
    """``create_file_dialog`` returns a str, a tuple of str or None depending on platform / version."""
    if not result:
        return None
    if isinstance(result, (tuple, list)):
        result = result[0] if result else None
    return str(result) if result else None


def _dialog_paths(result: Any) -> list[str]:
    if not result:
        return []
    if isinstance(result, (tuple, list)):
        return [str(r) for r in result if r]
    return [str(result)]


def _js_str(value: str) -> str:
    import json

    return json.dumps(value)


# ------------------------------------------------------------------------- injected script

INJECTED_JS = r"""
(function () {
  if (window.__lumen) { window.__lumen.configure(__CONFIG__); return; }
  var M = {
    cfg: __CONFIG__,
    configure: function (cfg) { M.cfg = cfg || M.cfg; M.applyZoom(M.cfg.zoom); },
    api: function () { return (window.pywebview && window.pywebview.api) || null; },
    toast: function (msg) {
      try {
        var el = document.getElementById('__lumen_toast');
        if (!el) {
          el = document.createElement('div');
          el.id = '__lumen_toast';
          el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:2147483647;' +
            'background:#1a1d25;color:#e8eaf0;border:1px solid #2a2f3b;font:13px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;' +
            'padding:9px 14px;border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.25);opacity:0;transition:opacity .2s;pointer-events:none;max-width:70vw';
          document.body.appendChild(el);
        }
        el.textContent = msg;
        el.style.opacity = '1';
        clearTimeout(M._toastTimer);
        M._toastTimer = setTimeout(function () { el.style.opacity = '0'; }, 3200);
      } catch (e) {}
    },
    applyZoom: function (z) {
      if (M.cfg.nativeZoom) return;
      try { document.documentElement.style.zoom = (z && z !== 1) ? String(z) : ''; } catch (e) {}
    },
    isExternal: function (href) {
      try { var u = new URL(href, location.href); return (u.protocol === 'http:' || u.protocol === 'https:') && u.origin !== location.origin; }
      catch (e) { return false; }
    },
    blobToBase64: function (blob) {
      return new Promise(function (resolve, reject) {
        var r = new FileReader();
        r.onload = function () { var s = String(r.result || ''); resolve(s.substring(s.indexOf(',') + 1)); };
        r.onerror = function () { reject(r.error); };
        r.readAsDataURL(blob);
      });
    },
    filenameFrom: function (a, resp) {
      var n = a.getAttribute('download');
      if (n) return n;
      try {
        var cd = resp && resp.headers && resp.headers.get('content-disposition');
        if (cd) { var m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(cd); if (m) return decodeURIComponent(m[1]); }
      } catch (e) {}
      try { var p = new URL(a.href, location.href).pathname.split('/').pop(); if (p) return p; } catch (e) {}
      return 'download';
    },
    saveAnchor: function (a) {
      var api = M.api();
      if (!api || !api.save_download) return false;
      fetch(a.href).then(function (resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return resp.blob().then(function (blob) {
          return M.blobToBase64(blob).then(function (b64) { return api.save_download(M.filenameFrom(a, resp), blob.type || '', b64); });
        });
      }).catch(function (err) { M.toast('Download failed: ' + (err && err.message ? err.message : err)); });
      return true;
    },
    onClick: function (ev) {
      if (ev.defaultPrevented || ev.button !== 0) return;
      var t = ev.target;
      var a = t && t.closest ? t.closest('a[href]') : null;
      if (!a) return;
      var href = a.getAttribute('href') || '';
      if (!href || href.charAt(0) === '#' || href.indexOf('javascript:') === 0) return;
      if (a.hasAttribute('download')) {
        if (M.cfg.nativeDownloads) return;           // the engine's own download handler opens the save dialog
        if (M.saveAnchor(a)) { ev.preventDefault(); ev.stopImmediatePropagation(); }
        return;
      }
      if (M.isExternal(a.href)) {
        var api = M.api();
        if (api && api.open_external) { ev.preventDefault(); ev.stopImmediatePropagation(); api.open_external(a.href); }
      }
    },
    onKey: function (ev) {
      if (M.cfg.nativeShortcuts) return;             // macOS: the menubar key equivalents handle these
      var mod = ev.ctrlKey || ev.metaKey;
      var api = M.api();
      if (!api || !api.action) return;
      var name = null;
      if (ev.key === 'F11') name = 'fullscreen';
      else if (mod && !ev.shiftKey && (ev.key === 'r' || ev.key === 'R')) name = 'reload';
      else if (mod && (ev.key === '=' || ev.key === '+')) name = 'zoom_in';
      else if (mod && ev.key === '-') name = 'zoom_out';
      else if (mod && ev.key === '0') name = 'zoom_reset';
      if (name) { ev.preventDefault(); api.action(name); }
    }
  };
  window.__lumen = M;
  document.addEventListener('click', M.onClick, true);
  document.addEventListener('keydown', M.onKey, true);
  M.applyZoom(M.cfg.zoom);
})();
"""


def injected_js(config: dict[str, Any]) -> str:
    import json

    return INJECTED_JS.replace("__CONFIG__", json.dumps(config))
