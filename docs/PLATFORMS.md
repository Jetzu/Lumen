# Platforms: Windows, Linux, macOS

Lumen is pure Python plus pre-built wheels and a static web UI. It runs the same way on all three platforms:
a local server on `127.0.0.1` and your browser as the window. This page collects what differs per platform.

| | Windows 10/11 | Linux (x86_64, aarch64) | macOS (Apple Silicon, Intel) |
|---|---|---|---|
| Python | 3.11 – 3.14 (`py -3`) | 3.11 – 3.14 | 3.11 – 3.14 (Homebrew or python.org) |
| Dependencies | all wheels, no compiler | all wheels (manylinux), no system packages | all wheels |
| Browser | Edge or Chrome (Firefox works too) | Chromium/Chrome or Firefox | Safari, Chrome or Firefox |
| Event loop | asyncio (uvloop is Unix-only and simply not installed) | uvloop | uvloop |
| Per-user folder | `C:\Users\<you>\.lumen` | `~/.lumen` | `~/.lumen` |
| Native window (`--window`) | pywebview + Edge WebView2 (preinstalled) | pywebview + GTK or Qt | pywebview + WebKit |

Every dependency in `pyproject.toml` (base, `[dev]`, `[formats]`, `[video]`, `[window]`) resolves to binary
wheels for `x86_64-pc-windows-msvc`, `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` and
`aarch64-apple-darwin` on Python 3.11 – 3.14 (checked with `uv pip install --dry-run --only-binary :all:`).
`uvicorn[standard]` lists `uvloop` with a `sys_platform != "win32"` marker, so on Windows uvicorn falls back
to the standard asyncio loop automatically; `httptools`, `watchfiles` and `websockets` ship Windows wheels.

## Windows

```bat
cd Lumen
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e .
lumen                       :: opens http://127.0.0.1:8765 in your default browser
lumen C:\data\cells.tif     :: open files at startup; forward slashes (C:/data/cells.tif) work too
```

`scripts\run_lumen.bat` does all of the above in one step (creates `.venv` on first use, then starts Lumen).

- Use the `py` launcher if `python` is not on `PATH`, or tick "Add python.exe to PATH" in the python.org installer.
  The Microsoft Store build of Python works as well.
- The first start may show a Windows Defender Firewall prompt for `python.exe`. Lumen only talks to your own
  browser over `127.0.0.1`; you can safely choose "Cancel" (the prompt is about *incoming* connections from
  other machines, which Lumen does not accept unless you pass `--host 0.0.0.0`).
- The file browser inside Lumen shows native paths (`C:\Users\...`). Type another drive (`D:\`) or any folder
  into the path box and press Enter to go there; the quick links point at `Desktop`, `Documents`, `Downloads`
  under your profile.
- Paths longer than 260 characters need the Windows long-path setting
  (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`) – deep OME-Zarr trees can hit this.
- Antivirus scanners can make the first `import` of numpy/scipy/scikit-image noticeably slow; later starts are fast.
- Text that Lumen writes (CSV/TSV exports, MetaImage headers, recent-files list) is always UTF-8 with explicit
  newlines, so a `µm` unit or a non-ASCII file name never depends on the console code page.
- `--window` needs `pip install -e ".[window]"`; pywebview uses the Edge WebView2 runtime that ships with Windows 10/11.

## Linux

```bash
cd Lumen
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
lumen                       # opens http://127.0.0.1:8765 with xdg-open
```

or `scripts/run_lumen.sh`, which creates `.venv` on first use and then starts Lumen.

- No system packages are needed: numpy, scipy, scikit-image, imagecodecs, tifffile and matplotlib all ship
  manylinux wheels. Rendering happens in the browser (2D canvas and WebGL for the 3-D viewer), so there is no
  OpenGL, Qt or X11 requirement on the Python side and Lumen works over SSH with `--no-browser` plus a port forward.
- Headless machines: `lumen --no-browser --host 127.0.0.1 --port 8765`, then `ssh -L 8765:127.0.0.1:8765 host`
  and open the URL locally. `webbrowser.open` simply returns `False` when there is no display.
- Wayland and X11 both work (it is just a browser tab). `--window` additionally needs pywebview with its
  GTK (`python3-gi`, `gir1.2-webkit2-4.1`) or Qt backend from your distribution.
- Uploads, URL downloads and zip exports go through `tempfile` (`/tmp` by default); set `TMPDIR` to move them
  if that file system is small.

## macOS

See the [security and macOS notes in the README](../README.md#security-and-macos-notes): Gatekeeper and the
`[video]` extra's unsigned `ffmpeg`, the Desktop/Documents/Downloads access prompt, and the application firewall.
Install with the same commands as on Linux (`python3 -m venv .venv`, `pip install -e .`) or `scripts/run_lumen.sh`;
Homebrew's `python3` and the python.org installer both work, on Apple Silicon and Intel.

## Java Bio-Formats bridge (optional, all platforms)

Gatan DM3/DM4, Zeiss ZVI and Leica LEI have pure-Python readers; every other proprietary format that only the
Java Bio-Formats library reads (Olympus VSI, SlideBook, Andor IQ, Hamamatsu VMS, Amira, FEI SER, PicoQuant PTU,
Volocity, DeltaVision, ICS, …) goes through the optional bridge (`lumen.core.readers.bioformats_bridge`), which has
two modes (`LUMEN_BIOFORMATS_MODE=auto|bioio|fiji-jars`, default `auto` = `bioio` when it is installed, else
`fiji-jars`):

* **fiji-jars – no Python package**: a JDK plus a **Fiji installation** (its `jars/bio-formats` folder is used
  read-only; `LUMEN_FIJI_JARS` when Fiji is not in `/Applications/Fiji`, `~/Fiji.app`, `/opt/fiji`, `C:\Fiji.app`, …).
  Each listing or read is one `java` subprocess (~1 s) running a small Groovy (or BeanShell) runner over
  `loci.formats.ImageReader`: it reports the series (sizes, dimension order, pixel type, resolution levels, physical
  sizes, channel names, metadata, OME-XML) and converts the requested series / level to a temp OME-TIFF
  (`OMETiffWriter`) that Lumen reads. Verified here with Fiji's Bio-Formats 8.1.1 and OpenJDK 21 (multi-series
  OME-TIFF, ICS, PNG).
* **bioio**: `pip install "lumen-imaging[bioformats]"` (`bioio` + `bioio-bioformats`, i.e. scyjava and JPype) **plus
  a JDK 11 or newer that you install yourself**; the first file you open downloads the Bio-Formats jars through
  Maven into `~/.m2` / `~/.jgo` (one-time, needs internet), later opens are offline. `jpype1` ships binary wheels
  per Python version – if none exists yet for a brand-new Python, create the venv with the previous release.

The JDK: Eclipse Temurin or another OpenJDK build (Windows: `winget install EclipseAdoptium.Temurin.21.JDK` or the
MSI; Linux: `apt install openjdk-21-jdk` / `dnf install java-21-openjdk`; macOS: `brew install openjdk` or
`brew install --cask temurin@21` – Temurin's builds are signed and notarized, so Gatekeeper raises no "unverified
developer" prompt). Lumen finds it through `JAVA_HOME`, `java` on `PATH`, Homebrew's `openjdk` kegs, the JDKs under
`/Library/Java/JavaVirtualMachines`, `/usr/lib/jvm` or `C:\Program Files\Eclipse Adoptium` / `Java` / `Microsoft` /
`Zulu`, and – on Linux / Windows – the JDK bundled with Fiji (not on macOS: Gatekeeper). A candidate counts once
`java -version` runs (macOS's `/usr/bin/java` stub without a JDK does not). Without a JDK or jars everything else
keeps working; `Plugins ▸ Bio-Formats ▸ Plugins Configuration` (`file.bioformats_config`) shows what is missing and
the open error names the install steps.

## JVM script bridge (optional, all platforms)

Fiji's JVM script languages – Groovy, BeanShell, Rhino JavaScript, JRuby and (with extra jars) Clojure – and compiled ImageJ-1 plugin jars named on `LUMEN_JVM_CLASSPATH` (`IJ.runPlugIn` from a script) run with
the real ImageJ 1.x through a second bridge with two modes (`LUMEN_JVM_MODE=auto|subprocess|inprocess`):

* **subprocess – no Python package** (`lumen.core.ijshim.jvm_subprocess`): one `java` process per script over the
  same Fiji installation's jars (`ij.jar`, `groovy`, `bsh`, `rhino`, `jruby-core` + `jruby-stdlib`; or `ij.jar` plus
  the language jar on `LUMEN_JVM_CLASSPATH` without Fiji) and the same JDK as above, in ImageJ's batch mode with no
  windows (a hidden ROI Manager, `-Dapple.awt.UIElement=true`, ImageJ's prefs isolated). `auto` picks it on macOS –
  where an in-process JVM cannot run ImageJ commands because AWT wants the process's main thread – and wherever
  `jpype` is not installed. Verified on this Mac (Fiji 2.16 / ImageJ 1.54p, OpenJDK 21): Groovy, BeanShell, Rhino and
  JRuby scripts blur / measure the Blobs sample bit-exactly with Lumen's own filters, ~0.7–1.5 s per run.
* **inprocess**: `pip install "lumen-imaging[jvm]"` (`jpype1` + `scyjava`) plus the JDK; the first script downloads
  `ij.jar` (`net.imagej:ij:1.54p`) and the language jars through Maven unless `LUMEN_IJ_JAR` / `LUMEN_JVM_CLASSPATH`
  point at local jars. The JVM is headless (`-Djava.awt.headless=true`); a non-headless AWT (`LUMEN_JVM_HEADLESS=0`,
  needed for the ROI Manager exchange there) works on Linux / Windows only. One JVM per process: the first bridge to
  start it (this one or the `bioio` Bio-Formats mode) decides the classpath.

Without a JDK / jars everything else keeps working: the Script panel and `POST /api/script/run` answer with the
install steps, and `Plugins ▸ Scripting ▸ JVM Script Bridge Status` (`script.jvm_bridge_status`) shows both modes
and what is found. Exchanged with the session per script: the active image (`imp`, back in place with undo), shown
images, output / `IJ.log`, the Results table and the ROI Manager; see docs/SCRIPTING.md ▸ JVM languages.

## Lumen's per-user folder and environment variables

| Variable | Meaning | Default |
|---|---|---|
| `LUMEN_HOME` | Where Lumen keeps `recent.json`, `plugins/`, `macros/`, `samples/` and the startup scripts (`~` is expanded) | `~/.lumen` (`C:\Users\<you>\.lumen`) |
| `LUMEN_PLUGINS` | Extra plugin folders, separated by `os.pathsep` (`;` on Windows, `:` elsewhere) | – |
| `LUMEN_SAMPLES_DIR` | Cache for the ImageJ sample images | `$LUMEN_HOME/samples` |
| `HTTP_PROXY` / `HTTPS_PROXY` / `NO_PROXY` | Proxy for the sample-image and URL downloads (urllib, pooch) | – |
| `MPLBACKEND` | Matplotlib backend (Lumen only renders plots to files; `Agg` is fine everywhere) | auto |
| `LUMEN_JVM_MODE`, `LUMEN_FIJI_JARS`, `LUMEN_JVM_TIMEOUT`, `LUMEN_JVM_KEEP_TEMP`, `LUMEN_BIOFORMATS_MODE`, `LUMEN_BIOFORMATS_TIMEOUT` | The subprocess bridges (above): `auto` / `subprocess` / `inprocess`; a Fiji `jars` folder (or Fiji folder) when not in a standard place; seconds a script may run; `1` keeps the run's temp folder; `auto` / `bioio` / `fiji-jars`; seconds a Bio-Formats run may take | auto, auto-detected, 600, –, auto, 900 |
| `LUMEN_IJ_JAR`, `LUMEN_JVM_CLASSPATH`, `LUMEN_JVM_LANGUAGES`, `LUMEN_JVM_EXTRA_ENDPOINTS`, `LUMEN_JVM_HEAP`, `LUMEN_JVM_HEADLESS` | The JVM script bridge (above): a local `ij.jar`; extra jars / folders / globs (`os.pathsep`-separated, both modes); the opt-in languages (`jruby,clojure` or `all`, in-process); more Maven coordinates (in-process); the heap (`4g`); `0` for a non-headless AWT (in-process) | Maven `ij` 1.54p, –, core languages, –, JVM default / 1 GiB+, headless |

The folder holds `recent.json`, `plugins/` (Python command modules), `macros/`, and the optional `startup.py` /
`startup.ijm` that run once when `lumen` starts. `lumen.core.io.lumen_home()` (and its twin in `lumen.core.ops`)
resolve `LUMEN_HOME`. Proxies can also be set inside Lumen (Edit ▸ Options ▸ Proxy Settings…, command
`edit.options_proxy`), which mirrors them into the variables above for the running process.

## File names and text files

- CSV/TSV exports use the `csv` module with `newline=""`, so line endings are identical on every platform and
  never become `\r\r\n` on Windows. Table exports (`file.export_results`), coordinates, tracks and the Results
  download are all UTF-8.
- ImageJ ROI zips: entry names are sanitised (`\ / : * ? " < > |` → `_`) and made unique case-insensitively, so a
  zip saved on Linux extracts cleanly on NTFS and APFS; the ROI's own name inside the `.roi` record is kept verbatim.
- MetaImage `.mhd` headers are written as UTF-8 with `\n`; non-ASCII data-file names round-trip.
- TIFF channel names are stored as UTF-16 in the ImageJ metadata and HDF5 names as UTF-8 attributes, so non-ASCII
  names (`α-tubulin`, `日本語`) survive on every platform.
- Windows cannot create files named `CON`, `PRN`, `AUX`, `NUL`, `COM1`…`COM9`, `LPT1`…`LPT9` (with any extension).
  Lumen does not rename these for you; pick another name if a save fails with "Permission denied".
- Case-insensitive file systems (NTFS, APFS by default): `Cells.tif` and `cells.tif` are the same file. Lumen's
  image names stay distinct in the session (`unique_name`), but saving both to one folder overwrites.

## Desktop app

The desktop app (docs/DESKTOP.md) is the same engine and interface in a native window: WebKit on macOS, the WebView2
control on Windows 10 / 11 (part of the system; otherwise a free download from Microsoft), Qt WebEngine (inside the
app) on Linux. It shares `~/.lumen` with the `lumen` command, runs its server on a loopback port only, and is built per
platform by `desktop/build.py` or the release workflow. The macOS build is ad-hoc signed, not notarized.

## One server, one user: what is shared on a server

Lumen is a single-user application served over HTTP. Everything below lives in the server process and is
seen by every browser tab and every machine that connects to it:

- the open images and their undo histories, the ROI Manager, the Results table, text windows, tables and plots;
- threshold state, the Memory Monitor and the job list;
- preferences (`Edit ▸ Options ▸ Preferences…` is saved on the server, not in the browser);
- the file dialog, which browses the *server's* file system (Home, Desktop, Documents …), and Save, which writes there;
- Python, ImageJ-macro and JavaScript scripts, plugin modules and pixel-classifier models, which execute in the
  server process with its permissions.

Only the macro recorder is per client: the page sends an `X-Lumen-Client` id with every request and each id gets
its own recording with its own start marker (`GET /api/recorder`); requests without the header share a default
recorder. That id is a bookkeeping key, not an authentication token.

For one person on a laptop this is the intended design. On a shared machine or a facility server, run one
server process per user (`lumen --port <n>` under each account, or a reverse proxy that assigns a port per login)
and bind it to `127.0.0.1` (the default) or to a network interface behind the facility's own access control;
Lumen has no login, no user separation and no sandbox for scripts.

## Known per-platform differences

- `help.properties` shows "Process RSS: n/a" on Windows unless `psutil` is installed (the Unix `resource` module
  is used as a fallback elsewhere and is imported only inside a `try`).
- ImageJ-macro compatibility functions return platform-native separators, exactly as ImageJ does:
  `File.separator` is `\` on Windows, `getDirectory("image")` ends in `\`, `getFileList()` marks folders with `\`.
- Plugin folders: `LUMEN_PLUGINS` uses `;` on Windows and `:` elsewhere (`os.pathsep`).
- Zip imports flatten entries and skip `__MACOSX` resource forks and absolute/`..` paths on every platform.

## Continuous integration

`.github/workflows/ci.yml` runs `pytest` on ubuntu-latest, windows-latest and macos-latest with Python 3.11 and 3.13
(pip cached, `LUMEN_HOME` pointed at a temporary folder, `PYTHONUTF8` deliberately unset so the Windows job runs with
cp1252 and catches text files written without an encoding), and builds the frontend once on Ubuntu (`npm ci && npm run
build`, uploaded as the `lumen-static` artifact). `.github/workflows/release.yml` builds the frontend, then a wheel
and an sdist with `python -m build` on `v*` tags, smoke-tests the wheel on a clean interpreter and uploads `dist/`
as an artifact; nothing is published automatically.

`tests/test_cross_platform.py` exercises the Windows-sensitive paths on every OS: separator-correct file listings,
uniform CSV line endings, ROI zip entry names, non-ASCII HDF5 channel names, `LUMEN_HOME`, `os.pathsep` plugin
lists, UTF-8 MetaImage headers, and that the whole package imports with `resource`, `fcntl`, `termios`, `tty`
and `pty` unavailable.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'lumen' is not recognized` / `command not found` | The venv is not activated: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`, or run `python -m lumen`. |
| `Address already in use` | Lumen tries the next 50 ports automatically; pass `--port` to choose one. |
| Browser does not open | Copy the printed `http://127.0.0.1:<port>` into Edge/Chrome/Firefox; use `--no-browser` on servers. |
| `UnicodeEncodeError` on a console message (Windows) | Fixed in `lumen` (stdout/stderr use `errors="replace"`); for scripts of your own set `PYTHONUTF8=1`. |
| Sample download fails behind a proxy | Set `HTTPS_PROXY` (and `NO_PROXY`) or use Edit ▸ Options ▸ Proxy Settings…. |
| `pip install` wants a compiler | Your Python is too new or too old for the wheels; use Python 3.11 – 3.14 (`py -3.12 -m venv .venv`). |
