import json

import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops.paint import stroke_mask
from lumen.core.session import Session


@pytest.fixture
def session():
    s = Session()
    s.add_image(LumenImage(np.zeros((100, 100), np.uint8), name="blank"))
    return s


def test_stroke_mask_round_caps():
    m = stroke_mask([(10, 50), (90, 50)], 6, (100, 100))
    # body: 6 px tall band, x from ~7 to ~93 (round caps)
    assert m[50, 50] and m[47, 50] and m[52, 50]
    assert not m[44, 50] and not m[56, 50]
    assert m[50, 8] and not m[50, 3]  # cap extends ~3 px past the end point
    assert not m[20, 50]
    # pencil (width 1) draws a one-pixel line
    thin = stroke_mask([(0, 0), (99, 99)], 1, (100, 100))
    assert thin.sum() == 100 and thin[10, 10]
    # single click = one disc
    dot = stroke_mask([(50, 50)], 10, (100, 100))
    assert 60 < dot.sum() < 100 and dot[50, 50]


def test_paint_stroke_only_along_path(session):
    img = session.active
    before = img.data.copy()
    session.run_op("edit.paint_stroke", {"points": json.dumps([[10, 50], [90, 50]]), "width": 6})
    p = img.data[0, 0, 0]
    assert p[50, 50] == 255 and p[50, 20] == 255
    assert p[20, 50] == 0 and p[80, 50] == 0 and p[50, 2] == 0
    assert 400 < int((p == 255).sum()) < 600
    # one undo step per stroke
    assert session.histories[img.id].labels()["undo"][-1] == "Paint stroke"
    session.undo(img.id)
    assert np.array_equal(img.data, before)


def test_paint_stroke_erase_and_channels():
    s = Session()
    data = np.full((2, 40, 40), 200, np.uint16)
    s.add_image(LumenImage(data, name="two", axes="CYX"))
    img = s.active
    img.display.c = 1
    s.run_op("edit.paint_stroke", {"points": json.dumps([[5, 20], [35, 20]]), "width": 4, "erase": True})
    assert img.data[0, 0, 1, 20, 20] == 0 and img.data[0, 0, 0, 20, 20] == 200  # only channel 1
    s.run_op("edit.paint_stroke", {"points": json.dumps([[5, 10], [35, 10]]), "width": 4, "all_channels": True})
    assert img.data[0, 0, 0, 10, 20] == 65535 and img.data[0, 0, 1, 10, 20] == 65535  # 16-bit foreground
    assert img.data[0, 0, 0, 30, 20] == 200


def test_paint_stroke_rgb_all_three():
    s = Session()
    s.add_image(LumenImage(np.zeros((30, 30, 3), np.uint8), axes="YXS", rgb=True, name="rgb"))
    img = s.active
    s.run_op("edit.paint_stroke", {"points": json.dumps([[15, 15]]), "width": 6})
    assert tuple(img.data[0, 0, :, 15, 15]) == (255, 255, 255)
    assert img.rgb


def test_flood_fill_region(session):
    img = session.active
    img.data[0, 0, 0, 30:50, 30:50] = 100
    out = session.run_op("edit.flood_fill", {"x": 5, "y": 5})
    p = img.data[0, 0, 0]
    assert p[5, 5] == 255 and p[90, 90] == 255
    assert p[40, 40] == 100  # the square keeps its value
    assert "Filled" in out["message"] and (p == 255).sum() == 100 * 100 - 400
    session.run_op("edit.flood_fill", {"x": 40, "y": 40, "value": 7})
    p = img.data[0, 0, 0]  # replace_data swaps the array
    assert p[40, 40] == 7 and (p == 7).sum() == 400
    session.undo(img.id)
    assert img.data[0, 0, 0, 40, 40] == 100
    with pytest.raises(ValueError):
        session.run_op("edit.flood_fill", {"x": 500, "y": 5})


def test_flood_fill_tolerance(session):
    img = session.active
    plane = img.data[0, 0, 0]
    plane[:, :50] = 10
    plane[:, 50:] = 30
    session.run_op("edit.flood_fill", {"x": 0, "y": 0, "tolerance": 25, "connectivity": "8"})
    assert (img.data[0, 0, 0] == 255).all()


def test_overlay_add_remove_clear(session):
    img = session.active
    item = {"type": "text", "x": 10, "y": 20, "text": "hello", "font_size": 14, "color": "#fff", "background": True}
    session.run_op("overlay.add_item", {"item": json.dumps(item)})
    assert len(img.overlay) == 1 and img.overlay[0]["text"] == "hello" and img.overlay[0]["id"]
    track = {"type": "track", "id": "t1", "name": "Track 1", "points": [[20, 20, 0], [40, 30, 1]], "color": "#f59e0b"}
    session.run_op("overlay.add_item", {"item": json.dumps(track)})
    session.run_op("overlay.add_item", {"item": json.dumps({"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4})})
    assert [o["type"] for o in img.overlay] == ["text", "track", "rect"]
    assert img.overlay[1]["id"] == "t1"
    session.run_op("overlay.remove_item", {"id": "t1"})
    assert [o["type"] for o in img.overlay] == ["text", "rect"]
    with pytest.raises(ValueError):
        session.run_op("overlay.remove_item", {"id": "nope"})
    session.run_op("overlay.clear_type", {"type": "text"})
    assert [o["type"] for o in img.overlay] == ["rect"]
    session.undo(img.id)
    assert [o["type"] for o in img.overlay] == ["text", "rect"]
    with pytest.raises(ValueError):
        session.run_op("overlay.add_item", {"item": "not json"})
    with pytest.raises(ValueError):
        session.run_op("overlay.add_item", {"item": json.dumps({"x": 1})})


def test_rgb_fill_brush_and_bucket_use_the_picked_colour():
    """The Color Picker stores a hex colour; Fill, the brush and the bucket use it per channel on RGB images."""
    import numpy as np
    from lumen.core.image import LumenImage
    from lumen.core.session import Session

    s = Session()
    img = s.add_image(LumenImage(np.zeros((20, 20, 3), np.uint8), axes="YXS", name="rgb.tif", rgb=True))
    s.settings["foreground_color"] = "#ff8000"
    s.run_op("edit.fill", {}, image_id=img.id, roi={"type": "rect", "x": 2, "y": 2, "w": 5, "h": 5})
    assert img.data[0, 0, :, 4, 4].tolist() == [255, 128, 0]
    s.run_op("edit.paint_stroke", {"points": "[[10,10],[12,12]]", "width": 3}, image_id=img.id)
    assert img.data[0, 0, :, 11, 11].tolist() == [255, 128, 0]
    s.run_op("edit.flood_fill", {"x": 0, "y": 19}, image_id=img.id)
    assert img.data[0, 0, :, 19, 0].tolist() == [255, 128, 0]
    s.settings.pop("foreground_color")  # no picked colour: the grey foreground value applies to every channel
    s.run_op("edit.fill", {}, image_id=img.id, roi={"type": "rect", "x": 2, "y": 2, "w": 5, "h": 5})
    assert img.data[0, 0, :, 4, 4].tolist() == [255, 255, 255]
