# Lumen as a desktop app

Lumen runs in two ways from the same code: as a local web app (`lumen` opens it in your browser) and as a
**desktop app** - a native window with its own Dock / taskbar icon, native Open and Save dialogs, and images that open
from Finder or Explorer. Both use the same engine, the same interface and the same per-user folder (`~/.lumen`:
plugins, macros, recent files, startup scripts), so a plugin installed in one shows up in the other.

## Install

| System | File | How |
|---|---|---|
| macOS (Apple silicon) | `Lumen-<version>-macos-arm64.dmg` | open the DMG, drag **Lumen** onto **Applications** |
| macOS (Intel) | `Lumen-<version>-macos-x86_64.dmg` | the same |
| Windows 10 / 11 | `Lumen-<version>-windows-x86_64.zip` | unzip anywhere, run `Lumen\Lumen.exe` (uses the WebView2 runtime that ships with Windows) |
| Linux (x86_64) | `Lumen-<version>-linux-x86_64.AppImage` or `.tar.gz` | make the AppImage executable and run it, or unpack the tar and run `Lumen/Lumen` |

**macOS security prompt.** The app is ad-hoc signed, not notarized by Apple (that needs a paid developer account). A copy
you **build yourself** on your Mac opens normally - nothing marks it as downloaded. A copy **downloaded** from the
internet makes macOS say it "cannot verify" the app: right-click **Lumen** in Applications ▸ **Open** ▸ **Open** once, or
allow it under System Settings ▸ Privacy & Security. The app contains no other executables (no bundled browser, no
ffmpeg): the window is the system's own WebKit.

## What the desktop app adds

- **A native window**: the full Lumen interface without a browser tab, remembering its size, position and zoom.
- **Native file dialogs**: File ▸ Open, Save As, and every *Browse…* of a path parameter (batch folders, scripts,
  classifier models) use the system's Open / Save / folder dialogs. The in-app file browser is still there in the
  browser version.
- **Images from the system**: double-click an image whose default app is Lumen, choose *Open With ▸ Lumen*, drop files
  on the Dock icon, or run `Lumen file.tif` - the image opens in the running window (TIFF, PNG, JPEG and the microscopy
  formats CZI, ND2, LIF, LSM, OIB, IMS, ZVI, … are declared to Finder).
- **One instance**: launching Lumen again brings the existing window to the front and hands it the files.
- **Quit protection**: quitting while an image has unsaved changes asks first (as Fiji does).
- **Native menus** around the in-app Fiji menus: *File* (Open Image…, Open Folder as Stack…, Open Lumen Folder, Open in
  Browser), *View* (zoom ⌘= / ⌘− / ⌘0, full screen) and *Help* (documentation, shortcuts, log file, About). Every Lumen
  shortcut (⌘K, ⌘O, ⌘S, ⌘Z, ⌘C / ⌘V, ⌘R in the script editor …) reaches the interface as in the browser.
- **Bundled readers**: the optional format readers (`[formats]`: ND2, CZI, LIF, DICOM, NIfTI, MRC, HDF5, Zarr, Olympus,
  Excel) are inside the app; Bio-Formats and the JVM script languages work as in the browser version when a Fiji
  installation and a JDK are present (docs/PLATFORMS.md).
- **An MCP server for AI assistants**: `Lumen --mcp` (macOS: `/Applications/Lumen.app/Contents/MacOS/Lumen --mcp`) runs
  Lumen's MCP server over stdio from the app itself, so Claude Code, Claude Desktop or Codex can work in the open window.
  `Lumen --mcp-config` prints the configuration (docs/MCP.md).

The app runs the Lumen server inside itself on a private loopback port (`127.0.0.1`, 8770 by default); nothing is
reachable from the network. *File ▸ Open in Browser* opens the same session in your browser if you want both.

## Files

| What | Where |
|---|---|
| Plugins, macros, recent files, startup scripts | `~/.lumen` (`%USERPROFILE%\.lumen`), or `LUMEN_HOME` |
| Desktop log, window state, web view storage | `~/.lumen/desktop/` (`desktop.log`, `desktop-state.json`, `webview/`) |

Command-line options of the app (`Lumen.app/Contents/MacOS/Lumen`, `Lumen.exe`, `Lumen`): `files…`, `--home DIR`,
`--port N`, `--debug` (web inspector), `--smoke` (start, check, exit - what the release workflow runs), `--version`.

## Build it yourself

Needs Python 3.11+ (3.14 by default, through uv when installed), Node.js 20+ for the interface and about 1 GB of disk.

```bash
python3 desktop/build.py                  # builds the UI, the build venv (desktop/.venv-build), the app and its package
python3 desktop/build.py --skip-frontend  # reuse the existing lumen/static build
python3 desktop/run_dev.py [files…]       # run the desktop window from source (no packaging)
python3 desktop/run_dev.py --smoke        # the same check the packaged app runs, from source
```

Output in `desktop/dist/`: `Lumen.app` + `Lumen-<version>-macos-<arch>.dmg` on macOS, `Lumen/` + a `.zip` on Windows,
`Lumen/` + `.tar.gz` (+ `.AppImage` when `appimagetool` is installed) on Linux. On this Mac a build takes about 90 s and
gives a 220 MB app (118 MB DMG). `.github/workflows/desktop-release.yml` builds all four targets on a `v*` tag, runs the
smoke check on each and attaches the packages to a GitHub release.

How it is put together: `desktop/lumen_desktop/` is the shell (pywebview window, in-process uvicorn server, single
instance, native menus, Finder open handler, splash and About pages); `frontend/src/native.ts` routes the interface's
file dialogs to the shell when it runs inside it; `desktop/build/*.spec` are the PyInstaller specs (the `lumen` package is
bundled as source files, because its commands are discovered at runtime). Tests: `tests/test_desktop_shell.py`
(instance hand-over, window state, dialog helpers, smoke mode); `desktop/dev_eval.py` drives a window started with
`--debug` (JavaScript in the page, synthetic key presses, files to open).

## What was verified, and what was not

Verified on macOS 27 (Apple silicon) with the packaged app: the smoke check (499 commands, an image round trip, a TIFF
opened, the Blobs sample, median / Gaussian / threshold / particles / FFT / histogram, the ImageJ macro layer, the bundled
docs, scikit-learn and all eleven bundled readers); launching with a file the way Finder does; a file sent from Finder to
the running app; a second launch handing its file over; native ⌘Z, ⌘C, ⌘V, ⌘A and ⌘K reaching the interface; SIGTERM
quitting cleanly.

Verified with the real system panels and alert (from source with `--debug`, through `desktop/dev_eval.py`):
- **Save As** (⌘⇧S) opened the real NSSavePanel with Lumen's suggested folder and name (the image's own folder, its name
  with the default format's extension); confirming it wrote the TIFF there (`ImageJ=1.54p`, Software *Lumen 0.1.0*, the
  processing log in Info).
- **Open** (⌘O) opened the real NSOpenPanel, modal, in the last folder; cancelling it changes nothing.
- **Quit with an unsaved image** (⌘Q) showed the alert *"Some images have unsaved changes. Quit Lumen anyway?"* with
  *Quit* / *Cancel*; **Cancel** kept the app, its server and the image; **Quit** ended it. This test found and fixed a bug:
  the shell used to stop its server in the window's *closing* event, which pywebview fires *before* the question - so
  Cancel would have left a window without an engine. The server now stops only once closing is certain.

Verified on GitHub's runners (`.github/workflows/desktop-release.yml`, run of 13 Nov 2026,
https://github.com/Jetzu/Lumen/actions/runs/43462244777): the **Windows**, **Linux** and **macOS (Intel)** builds were
built from the repository and their packaged apps passed the same smoke check (503 commands, the image round trip, the
lazily imported libraries, the bundled readers and docs) - ready after 702 ms (Windows), 277 ms (Linux) and 1134 ms (Intel
Mac). The Apple-silicon job of that run waited for a runner and was re-run. The re-run passed too: ready after 570 ms with
503 commands. All four builds are now verified on GitHub's runners.

Not verified: choosing a file *inside* the Open panel by automation - macOS draws file panels in a separate system
process that only real input reaches (scripting it needs the Accessibility permission, which would prompt you); the
Windows and Linux *windows* themselves (the smoke check runs the packaged app without a window: the server, the commands,
the libraries and the bundled UI, not WebView2 / Qt WebEngine on a screen).

Debug hooks for these checks (only in a shell started with `--debug`, behind the control socket's token):
`python desktop/dev_eval.py "<javascript>"`, `--key cmd+o`, `--open file…`, and the raw commands `__FOCUS__`,
`__MODAL__ <button title>` (click a button of the visible alert), `__PANEL__ <path>` (answer the visible file panel through
its public API), `__PY__ <expression>` (evaluate on the main thread).
