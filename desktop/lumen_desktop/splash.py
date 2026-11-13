"""The page shown while the server starts (and the error page when it does not), and the About window."""
from __future__ import annotations

import html
import json

_FONT = '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif'
_MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

SPLASH_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Lumen</title>
<style>
  html,body{height:100%;margin:0}
  body{display:flex;align-items:center;justify-content:center;background:#0d0f14;color:#e8eaf0;
       font:14px/1.5 __FONT__;-webkit-user-select:none;user-select:none}
  .card{text-align:center;max-width:480px;padding:32px}
  .orb{width:64px;height:64px;margin:0 auto 20px;border-radius:50%;
       background:radial-gradient(circle at 35% 35%,#86efac,#10b981 55%,#065f46);box-shadow:0 0 38px rgba(52,211,153,.45)}
  h1{font-size:22px;margin:0 0 4px;font-weight:650;letter-spacing:.2px}
  p{margin:0;color:#9aa3b8}
  .bar{height:3px;background:#222733;border-radius:3px;overflow:hidden;margin:24px auto 0;width:220px}
  .bar i{display:block;height:100%;width:40%;background:#34d399;border-radius:3px;animation:slide 1.2s ease-in-out infinite}
  @keyframes slide{0%{margin-left:-40%}100%{margin-left:100%}}
  @media (prefers-reduced-motion: reduce){.bar i{animation:none;width:100%}}
  .err{color:#fecaca;white-space:pre-wrap;text-align:left;font:12px/1.5 __MONO__;background:#1a1d25;border:1px solid #7f1d1d;
       border-radius:8px;padding:12px;margin-top:16px;max-height:260px;overflow:auto;-webkit-user-select:text;user-select:text;display:none}
  .hint{font-size:12px;color:#6b7386;margin-top:14px;-webkit-user-select:text;user-select:text}
</style></head>
<body><div class="card">
  <div class="orb" aria-hidden="true"></div>
  <h1>Lumen</h1>
  <p id="status" role="status">Starting the image engine…</p>
  <div class="bar" id="bar"><i></i></div>
  <pre class="err" id="err"></pre>
  <div class="hint" id="hint"></div>
</div>
<script>
  window.__setStatus = function (text, error, hint) {
    document.getElementById('status').textContent = text || '';
    var e = document.getElementById('err'); var b = document.getElementById('bar');
    if (error) { e.textContent = error; e.style.display = 'block'; b.style.display = 'none'; } else { e.style.display = 'none'; b.style.display = ''; }
    document.getElementById('hint').textContent = hint || '';
  };
</script></body></html>
""".replace("__FONT__", _FONT).replace("__MONO__", _MONO)


def set_status_js(text: str, error: str | None = None, hint: str | None = None) -> str:
    return f"window.__setStatus && window.__setStatus({json.dumps(text)}, {json.dumps(error)}, {json.dumps(hint)})"


def about_html(info: dict[str, str]) -> str:
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(v)}</td></tr>" for k, v in info.items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>About Lumen</title>
<style>
  body{{margin:0;padding:22px 24px;background:#13161d;color:#e8eaf0;font:13px/1.5 {_FONT}}}
  .head{{display:flex;align-items:center;gap:12px;margin-bottom:14px}}
  .orb{{width:34px;height:34px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#86efac,#10b981 55%,#065f46);box-shadow:0 0 16px rgba(52,211,153,.4)}}
  h1{{font-size:17px;margin:0;font-weight:650}} .sub{{color:#9aa3b8;margin:0}}
  table{{border-collapse:collapse;width:100%}} th{{text-align:left;color:#9aa3b8;font-weight:500;padding:4px 12px 4px 0;white-space:nowrap;vertical-align:top;width:1%}}
  td{{padding:4px 0;word-break:break-all;font-family:{_MONO};font-size:12px;-webkit-user-select:text;user-select:text}}
  .foot{{margin-top:14px;color:#6b7386;font-size:12px}}
</style></head><body>
<div class="head"><div class="orb"></div><div><h1>Lumen</h1><p class="sub">Scientific image analysis – a Fiji re-imagining · desktop app</p></div></div>
<table>{rows}</table>
<p class="foot">ImageJ macros, Fiji menus and ImageJ-identical results in a modern interface.</p>
</body></html>"""
