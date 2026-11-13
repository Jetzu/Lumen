"""Rasterise overlay items (text, arrows, calibration bars, time stamps, scale bars, ROI outlines)
into an RGB array.  Used by Image ▸ Overlay ▸ Burn In and the ImageJ-compat drawString()."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

from . import roi as roimod
from .image import LumenImage
from .luts import get_lut

POSITIONS = ["Upper Right", "Upper Left", "Lower Right", "Lower Left", "Center"]


def parse_color(color: Any, default=(255, 215, 0)) -> tuple[int, int, int]:
    """'#rrggbb', '#rgb', 'red', (r, g, b) → (r, g, b)."""
    if color is None:
        return default
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return tuple(int(v) for v in color[:3])  # type: ignore[return-value]
    s = str(color).strip().lower()
    named = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
             "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255), "orange": (255, 200, 0), "gray": (128, 128, 128),
             "grey": (128, 128, 128), "pink": (255, 175, 175), "lightgray": (192, 192, 192), "darkgray": (64, 64, 64)}
    if s in named:
        return named[s]
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) == 6:
        try:
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            pass
    return default


@lru_cache(maxsize=64)
def _font(size: int):
    from PIL import ImageFont
    size = max(int(size or 12), 4)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def text_size(draw, text: str, font) -> tuple[int, int]:
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return int(r - l), int(b - t)
    except Exception:
        w, h = draw.textsize(text, font=font)  # pragma: no cover
        return int(w), int(h)


def anchor(position: str, W: int, H: int, w: int, h: int, margin: int | None = None) -> tuple[int, int]:
    m = max(int(min(W, H) * 0.03), 6) if margin is None else margin
    pos = position or "Upper Right"
    if pos == "Center":
        return (W - w) // 2, (H - h) // 2
    x = W - m - w if "Right" in pos else m
    y = H - m - h if "Lower" in pos else m
    return x, y


def format_time_stamp(fmt: str, img: LumenImage, t: int, z: int = 0, start_frame: int = 0, start_time: float = 0.0, interval: float = 0.0,
                      label_start: int = 1, label_interval: int = 1) -> str:
    """Text of a time_stamp overlay item on plane (t, z).  The viewport's timeStampText (OverlayItems.tsx) mirrors this exactly:
    time = start_time + (t - start_frame) * (interval or frame_interval or 1); {unit} is the time unit when either interval is
    set, else "frame"; {label} = label_start + n * label_interval with n the frame (several frames) or slice index."""
    cal = img.calibration
    step = float(interval or 0.0) or float(cal.frame_interval or 0.0) or 1.0
    calibrated = bool(interval) or bool(cal.frame_interval)
    n = t if img.nt > 1 else z
    values = {"time": float(start_time or 0.0) + (t - int(start_frame or 0)) * step, "unit": cal.time_unit if calibrated else "frame",
              "frame": t + 1, "slice": z + 1, "t": t + 1, "z": z + 1, "label": int(label_start) + n * int(label_interval)}
    try:
        return str(fmt).format(**values)
    except (KeyError, ValueError, IndexError):
        return str(fmt)


def time_stamp_text(item: dict[str, Any], img: LumenImage, t: int, z: int = 0) -> str:
    """format_time_stamp for a stored time_stamp overlay item (what Burn In / Flatten write)."""
    return format_time_stamp(item.get("format", "t = {time:.1f} {unit}"), img, t, z, item.get("start_frame", 0) or 0, item.get("start_time", 0.0) or 0.0,
                             item.get("interval", 0.0) or 0.0, item.get("label_start", 1) if item.get("label_start") is not None else 1,
                             item.get("label_interval", 1) if item.get("label_interval") is not None else 1)


def _overlay_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    """The session preferences the renderer honours (Image ▸ Overlay ▸ Overlay Options…, see lumen/server/extra/settings.py);
    None = the live session's settings so callers that do not pass them still get the user's options."""
    if settings is None:
        try:
            from .session import SESSION
            settings = SESSION.settings
        except Exception:  # pragma: no cover - the renderer also works without a session
            settings = {}
    return settings if isinstance(settings, dict) else {}


def burn_overlay(rgb: np.ndarray, overlay: list[dict[str, Any]], img: LumenImage | None = None, t: int | None = None, z: int | None = None,
                 settings: dict[str, Any] | None = None) -> np.ndarray:
    """Return a copy of `rgb` (H, W, 3) uint8 with every overlay item drawn into it.

    `settings` are the session preferences (Overlay Options…): ``overlay_stroke_width`` / ``overlay_default_fill`` apply to ROI items
    without their own stroke width / fill, and ``overlay_show_labels`` draws each ROI item's label - its name when ``overlay_use_names``
    and it has one, else its 1-based position in the overlay (the number the viewport's Show labels, Measure Overlay and List
    Elements use: text / arrow / bar items count too, as in ImageJ) - with ``overlay_label_font_size``, ``overlay_label_color`` and
    (``overlay_label_background``) a contrasting box behind the text.  None = the live session's settings; {} = plain defaults."""
    from PIL import Image, ImageDraw
    prefs = _overlay_settings(settings)
    H, W = rgb.shape[:2]
    im = Image.fromarray(np.ascontiguousarray(rgb))
    draw = ImageDraw.Draw(im)
    if img is not None:
        t = img.display.t if t is None else t
        z = img.display.z if z is None else z
    # labels are numbered by the item's 1-based position in the whole overlay (text / arrows / bars count, as in ImageJ and the
    # viewport's Show labels) and over every plane, so an item keeps its number whichever plane is shown
    for index, o in enumerate(overlay, 1):
        typ = o.get("type")
        is_roi = typ in roimod.AREA_TYPES or typ in roimod.LINE_TYPES or typ == "point"
        # per-plane items (ROI 'position') are only drawn on their own plane
        pos = o.get("position") if isinstance(o.get("position"), dict) else None
        if pos and img is not None:
            if pos.get("t") not in (None, t) or pos.get("z") not in (None, z):
                continue
        if typ == "text":
            _draw_text(draw, o, W, H)
        elif typ == "arrow":
            _draw_arrow(draw, o)
        elif typ == "calibration_bar":
            _draw_calibration_bar(im, draw, o, W, H)
        elif typ == "time_stamp":
            if img is None:
                continue
            txt = time_stamp_text(o, img, t or 0, z or 0)
            font = _font(o.get("font_size", 14))
            w, h = text_size(draw, txt, font)
            x, y = anchor(o.get("position", "Upper Left"), W, H, w, h)
            draw.text((x, y), txt, fill=parse_color(o.get("color"), (255, 255, 255)), font=font)
        elif typ == "scalebar":
            _draw_scalebar(draw, o, W, H)
        elif is_roi:
            _draw_roi(draw, o, im, prefs=prefs, index=index)
    return np.asarray(im).copy()


def _draw_label_text(draw, x, y, txt: str, font, col: tuple[int, int, int], bold: bool = False, background: bool = False) -> None:
    """Draw `txt` with its top-left corner at (x, y): an optional contrasting box behind it (black under light text, white under
    dark) and, for bold, a second pass 1 px to the right - PIL's default font has no bold face, so the strokes are thickened instead."""
    if background:
        w, h = text_size(draw, txt, font)
        bg = (0, 0, 0) if sum(col) > 380 else (255, 255, 255)
        draw.rectangle([x - 2, y - 2, x + w + 2 + (1 if bold else 0), y + h + 2], fill=bg)
    draw.text((x, y), txt, fill=col, font=font)
    if bold:
        draw.text((x + 1, y), txt, fill=col, font=font)


def _draw_text(draw, o: dict[str, Any], W: int, H: int) -> None:
    font = _font(o.get("font_size", 14))
    txt = str(o.get("text", ""))
    x, y = int(round(o.get("x", 0))), int(round(o.get("y", 0)))
    col = parse_color(o.get("color"), (255, 255, 0))
    _draw_label_text(draw, x, y, txt, font, col, bold=bool(o.get("bold")), background=bool(o.get("background")))


def _draw_arrow(draw, o: dict[str, Any]) -> None:
    x1, y1, x2, y2 = (float(o.get(k, 0)) for k in ("x1", "y1", "x2", "y2"))
    col = parse_color(o.get("color"), (255, 255, 0))
    sw = max(int(round(o.get("stroke_width", 2))), 1)
    hs = float(o.get("head_size", 10))
    ang = math.atan2(y2 - y1, x2 - x1)
    L = math.hypot(x2 - x1, y2 - y1)
    hs = min(hs, L) if L > 0 else hs
    # shorten the shaft so it does not poke through the head
    bx, by = x2 - hs * 0.8 * math.cos(ang), y2 - hs * 0.8 * math.sin(ang)
    draw.line([(x1, y1), (bx, by)], fill=col, width=sw)
    a = math.radians(25)
    p1 = (x2 - hs * math.cos(ang - a), y2 - hs * math.sin(ang - a))
    p2 = (x2 - hs * math.cos(ang + a), y2 - hs * math.sin(ang + a))
    draw.polygon([(x2, y2), p1, p2], fill=col)
    if o.get("double_headed"):
        p1 = (x1 + hs * math.cos(ang - a), y1 + hs * math.sin(ang - a))
        p2 = (x1 + hs * math.cos(ang + a), y1 + hs * math.sin(ang + a))
        draw.polygon([(x1, y1), p1, p2], fill=col)


def _draw_calibration_bar(im, draw, o: dict[str, Any], W: int, H: int) -> None:
    bw, bh = int(o.get("width", 20)), int(o.get("height", 128))
    font = _font(o.get("font_size", 12))
    col = parse_color(o.get("color"), (255, 255, 255))
    lo, hi = float(o.get("min", 0)), float(o.get("max", 255))
    dec = int(o.get("decimals", 0))
    unit = o.get("unit", "") or ""
    n_ticks = max(int(o.get("ticks", 5)), 2)
    labels = [f"{lo + (hi - lo) * k / (n_ticks - 1):.{dec}f}" for k in range(n_ticks)]
    label_w = max(text_size(draw, s, font)[0] for s in labels) + 6
    unit_h = text_size(draw, unit, font)[1] + 4 if unit else 0
    total_w = bw + label_w + 8
    total_h = bh + unit_h + 8
    x0, y0 = anchor(o.get("position", "Upper Right"), W, H, total_w, total_h)
    if o.get("fill", True):
        draw.rectangle([x0 - 2, y0 - 2, x0 + total_w + 2, y0 + total_h + 2], fill=(0, 0, 0) if sum(col) > 380 else (255, 255, 255))
    try:
        table = get_lut(o.get("lut", "Grays"))
    except KeyError:
        table = get_lut("Grays")
    if o.get("inverted"):  # the bar was added while the channel's LUT was inverted: match the pixels
        table = table[::-1]
    rows = np.linspace(255, 0, bh).astype(int)  # max at the top
    bar = table[rows][:, None, :].repeat(bw, axis=1)
    from PIL import Image
    im.paste(Image.fromarray(bar.astype(np.uint8)), (x0 + 4, y0 + 4 + unit_h))
    draw.rectangle([x0 + 4, y0 + 4 + unit_h, x0 + 4 + bw - 1, y0 + 4 + unit_h + bh - 1], outline=col)
    for k, s in enumerate(labels):
        y = y0 + 4 + unit_h + bh - 1 - int(round((bh - 1) * k / (n_ticks - 1)))
        draw.line([(x0 + 4 + bw, y), (x0 + 4 + bw + 3, y)], fill=col)
        _, th = text_size(draw, s, font)
        draw.text((x0 + 4 + bw + 5, y - th // 2), s, fill=col, font=font)
    if unit:
        draw.text((x0 + 4, y0 + 2), unit, fill=col, font=font)


def _draw_scalebar(draw, o: dict[str, Any], W: int, H: int) -> None:
    px = int(round(o.get("px", 50)))
    th = int(o.get("thickness", 6))
    col = parse_color(o.get("color"), (255, 255, 255))
    font = _font(o.get("font_size", 14))
    label = f"{o.get('length', 0):g} {o.get('unit', '')}".strip() if o.get("show_label", True) else ""
    tw, thh = text_size(draw, label, font) if label else (0, 0)
    w = max(px, tw)
    h = th + (thh + 2 if label else 0)
    x0, y0 = anchor(o.get("position", "Lower Right"), W, H, w, h)
    bx = x0 + (w - px) // 2
    draw.rectangle([bx, y0, bx + px - 1, y0 + th - 1], fill=col)
    if label:
        draw.text((x0 + (w - tw) // 2, y0 + th + 2), label, fill=col, font=font)


def fill_rgba(o: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Interior colour of an area ROI / overlay item as (r, g, b, a): the canonical ``fill`` key (selection.properties)
    or the ImageJ .roi reader's ``fill_color``; '#rrggbbaa' carries an alpha; missing / '' / 'none' = unfilled."""
    v = o.get("fill")
    if v is None or isinstance(v, bool) or (isinstance(v, str) and not v.strip()):
        v = o.get("fill_color")
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (tuple, list)):
        if len(v) < 3:
            return None
        return int(v[0]), int(v[1]), int(v[2]), (int(v[3]) if len(v) > 3 else 255)
    s = str(v).strip()
    if not s or s.lower() in ("none", "no", "off", "-"):
        return None
    h = s[1:] if s.startswith("#") else s
    if len(h) == 8:
        try:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
        except ValueError:
            return None
    r, g, b = parse_color(s, (255, 215, 0))
    return r, g, b, 255


def _paint_area(im, o: dict[str, Any], rgba: tuple[int, int, int, int]) -> None:
    """Blend the fill colour into the interior of an area item (rect / oval / polygon / freehand / mask) of a PIL RGB image."""
    from PIL import Image
    W, H = im.size
    try:
        mask = roimod.to_mask(o, (H, W))
    except (KeyError, TypeError, ValueError):
        return
    if not mask.any():
        return
    r, g, b, a = rgba
    a = max(0, min(255, int(a)))
    im.paste((r, g, b), mask=Image.fromarray(mask.astype(np.uint8) * np.uint8(a)))


def _has_own_fill(o: dict[str, Any]) -> bool:
    """True when the item states its own interior under ``fill`` / ``fill_color`` (None / '' / True = unset; 'none' / False = an
    explicit no-fill, which the Overlay Options default fill must not override)."""
    for k in ("fill", "fill_color"):
        v = o.get(k)
        if v is None or v is True or (isinstance(v, str) and not v.strip()):
            continue
        return True
    return False


def _draw_roi(draw, o: dict[str, Any], im=None, prefs: dict[str, Any] | None = None, index: int = 0) -> None:
    """Stroke (and fill / label) one ROI-shaped overlay item.  `prefs` are the Overlay Options settings (see burn_overlay), `index`
    the item's 1-based position in the whole overlay (0 = unknown, no index label)."""
    prefs = prefs or {}
    col = parse_color(o.get("color"), (255, 215, 0))
    width = o.get("stroke_width")
    if width is None:
        width = prefs.get("overlay_stroke_width")
    try:
        sw = max(int(round(float(1.5 if width is None else width))), 1)
    except (TypeError, ValueError):
        sw = 2
    typ = o.get("type")
    if typ in roimod.AREA_TYPES:
        fill = fill_rgba(o) if _has_own_fill(o) else fill_rgba({"fill": prefs.get("overlay_default_fill") or ""})
        if fill is not None and im is not None:
            _paint_area(im, o, fill)  # interior first; the outline is stroked over it (lines / points / text stay unfilled)
        if typ == "mask":
            from skimage.measure import find_contours
            sub = roimod.rle_decode(o["rle"], (int(o["h"]), int(o["w"])))
            padded = np.zeros((sub.shape[0] + 2, sub.shape[1] + 2), np.float64)
            padded[1:-1, 1:-1] = sub
            for c in find_contours(padded, 0.5):
                if len(c) > 1:
                    pts = np.stack([c[:, 1] - 1 + o["x"] + 0.5, c[:, 0] - 1 + o["y"] + 0.5], axis=1).tolist()
                    draw.line(pts + pts[:1], fill=col, width=sw)
        else:
            pts = np.asarray(roimod.polygon_points(o), dtype=np.float64).reshape(-1, 2).tolist()
            if len(pts) >= 2:
                draw.line(pts + pts[:1], fill=col, width=sw)
    elif typ in roimod.LINE_TYPES:
        pts = np.asarray(roimod.line_points(o), dtype=np.float64).reshape(-1, 2).tolist()
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=sw)
    elif typ == "point":
        r = 2 + sw
        for p in o.get("points", []):
            x, y = float(p[0]), float(p[1])
            draw.line([(x - r, y), (x + r, y)], fill=col, width=1)
            draw.line([(x, y - r), (x, y + r)], fill=col, width=1)
    label = o.get("label") or (o.get("name") if o.get("show_name") else None)
    show_labels = bool(prefs.get("overlay_show_labels"))
    if not label and show_labels:  # Overlay Options… "Show labels": the name (Use names) when the item has one, else its index
        name = o.get("name") if prefs.get("overlay_use_names") else None
        label = str(name) if name else (str(index) if index > 0 else None)
    if label:
        bx, by, bw, bh = roimod.bounding_box(o)
        if show_labels:  # the options' label font / colour / background apply to every label while labels are switched on
            font = _font(o.get("font_size") or prefs.get("overlay_label_font_size") or 12)
            _draw_label_text(draw, bx + 2, by + 2, str(label), font, parse_color(prefs.get("overlay_label_color"), col),
                             background=bool(prefs.get("overlay_label_background")))
        else:
            font = _font(o.get("font_size", 12))
            draw.text((bx + 2, by + 2), str(label), fill=col, font=font)


def draw_text_into_plane(plane: np.ndarray, text: str, x: int, y: int, value: float, font_size: int = 14) -> np.ndarray:
    """Burn `text` into a 2-D plane (any dtype) with the given pixel value (ImageJ drawString)."""
    from PIL import Image, ImageDraw
    H, W = plane.shape
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    font = _font(font_size)
    _, th = text_size(d, text, font)
    d.text((x, y - th), text, fill=255, font=font)  # ImageJ: (x, y) is the text baseline
    mask = np.asarray(m) > 127
    out = plane.copy()
    from .registry import cast_to  # clamp like ImageJ's setValue instead of raising OverflowError on out-of-range values
    out[mask] = cast_to(np.array([value], np.float64), plane.dtype)[0] if plane.dtype != np.float32 else value
    return out
