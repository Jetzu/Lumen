# End-to-end smoke tests (Playwright + Google Chrome)

Browser tests that drive the real Lumen UI against a real server: open samples, run a filter with
live preview and undo it, threshold → mask → Analyze Particles, draw a selection and measure it,
render the 3-D view, toggle the theme, open the help overlay – and check that the browser console
stays free of errors.

## Running

```bash
pip install -e ".[e2e]"          # playwright + pytest (Lumen itself must be importable too)
python -m pytest tests/e2e -v    # or: pytest -m e2e
```

Requirements

- **Google Chrome** installed (the tests launch it headless through Playwright's `chrome` channel,
  so no browser download is needed). Without Chrome – or without Playwright – the suite **skips**
  instead of failing. `LUMEN_E2E_CHANNEL=chromium` uses Playwright's own download instead
  (`playwright install chromium` first), `LUMEN_E2E_CHANNEL=msedge` works too.
- A Python with Lumen installed to run the server. The fixture uses `$LUMEN_PYTHON` if set, else
  `<repo>/.venv/bin/python` (`.venv\Scripts\python.exe` on Windows), else the interpreter running
  pytest. That makes it possible to keep Playwright out of the project venv and run the suite from a
  throwaway environment: `uv venv /tmp/e2e && uv pip install --python /tmp/e2e/bin/python playwright pytest && /tmp/e2e/bin/python -m pytest tests/e2e`.

The whole suite is skipped (`-m "not e2e"` deselects it explicitly) in the ordinary unit-test run
when Playwright is missing, so `pytest -q` in the project venv stays green and fast.

## What is tested, and against which build

The server serves the **built** frontend in `lumen/static`. If that build is older than
`frontend/src`, tests for newer features skip with a message (e.g. the accessibility tests check
for `role="toolbar"` on the tool rail and skip when the build predates it). Locators therefore use
stable `title` attributes and class names that exist in old and new builds alike.

Every test starts from a clean session (all images closed, ROIs and results cleared through the REST
API) in a fresh browser context, and finishes with `no_console_errors`. Console messages that are
pure GPU/driver chatter (`GL ReadPixels`, `GL Driver Message`, SwiftShader/WebGL fallback notices)
are ignored – see `IGNORED_CONSOLE` in `conftest.py`.

The 3-D view test runs Chrome with SwiftShader (software WebGL2) so it behaves the same on machines
without a GPU; set `LUMEN_E2E_GPU=1` to use the real GPU.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `LUMEN_PYTHON` | interpreter used to start the server (`python -m lumen --no-browser`) |
| `LUMEN_E2E_URL` | test an already running server instead of starting one (its state is reset by the tests!) |
| `LUMEN_E2E_PORT` | port for the spawned server (default: a free port) |
| `LUMEN_E2E_CHANNEL` | Playwright browser channel: `chrome` (default), `chromium`, `msedge` |
| `LUMEN_E2E_GPU` | `1` = do not force SwiftShader |
| `LUMEN_E2E_START_TIMEOUT` | seconds to wait for `/api/meta` (default 90) |

## macOS note (Gatekeeper)

Playwright's `chrome` channel launches the *installed, signed* Google Chrome, so Gatekeeper is
happy. Playwright's own downloaded Chromium (`playwright install chromium`) and its headless shell
are **not** notarised: on macOS with strict Gatekeeper settings they are flagged ("Apple could not
verify … is free of malware") – that is why the project venv deliberately does not ship Playwright
browsers and the default channel is `chrome`. The Node driver bundled in the `playwright` wheel is
signed and runs without prompts.
