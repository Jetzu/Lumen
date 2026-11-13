"""Image ▸ Color."""
from __future__ import annotations

import numpy as np

from ..image import ChannelDisplay, LumenImage
from ..luts import LUT_NAMES
from ..registry import Param, OpResult, op

CAT = "Image/Color"


@op("color.split", "Split Channels", CAT, kind="image", scope_param=False, preview=False, description="Open every channel as its own image.")
def split_channels(ctx):
    img = ctx.img
    if img.nc < 2:
        raise ValueError("Image has a single channel")
    out = []
    for c in range(img.nc):
        ch = img.display.channels[c]
        im = img.copy(name=f"C{c + 1}-{img.name}", data=img.data[:, :, c:c + 1].copy())
        im.rgb = False
        im.display.channels = [ChannelDisplay(**ch.to_dict())]
        im.display.c = 0
        im.display.mode = "single"
        out.append(im)
    return OpResult(new_images=out)


@op("color.merge", "Merge Channels", CAT, kind="image", requires_image=False, scope_param=False, preview=False,
    params=[Param("c1", "image", None, label="C1 (red)"), Param("c2", "image", None, label="C2 (green)"), Param("c3", "image", None, label="C3 (blue)"),
            Param("c4", "image", None, label="C4 (gray)"), Param("c5", "image", None, label="C5 (cyan)"), Param("c6", "image", None, label="C6 (magenta)"), Param("c7", "image", None, label="C7 (yellow)"),
            Param("create_composite", "bool", True), Param("keep_source", "bool", False)],
    description="Combine open single-channel images into one multi-channel composite.")
def merge_channels(ctx, c1=None, c2=None, c3=None, c4=None, c5=None, c6=None, c7=None, create_composite=True, keep_source=False):
    s = ctx.session
    luts = ["Red", "Green", "Blue", "Grays", "Cyan", "Magenta", "Yellow"]
    chosen = [(s.get_image(k), lut) for k, lut in zip((c1, c2, c3, c4, c5, c6, c7), luts) if k]
    if len(chosen) < 2:
        raise ValueError("Pick at least two images")
    if len({im.id for im, _ in chosen}) < len(chosen):
        raise ValueError("Pick at least two different images (the same image was chosen twice)")
    ref = chosen[0][0]
    dtype = np.result_type(*(im.data.dtype for im, _ in chosen))  # uint8 < uint16 < float32: never narrower than any source (casting to the first image's dtype wrapped / truncated the others)
    arrays = []
    for im, _ in chosen:
        if (im.nt, im.nz, im.height, im.width) != (ref.nt, ref.nz, ref.height, ref.width):
            raise ValueError(f"{im.name}: size differs from {ref.name}")
        if im.nc > 1:   # ImageJ refuses RGB / multi-channel sources; silently taking channel 0 dropped the other channels
            what = "an RGB image" if im.rgb else f"a {im.nc}-channel image"
            raise ValueError(f"{im.name} is {what}: Merge Channels needs single-channel images (run Image ▸ Color ▸ Split Channels first)")
        arrays.append(im.data.astype(dtype))
    data = np.concatenate(arrays, axis=2)
    res = ref.copy(name="Composite", data=data)
    res.rgb = False
    res.display.channels = [ChannelDisplay(name=im.display.channels[0].name or f"C{i + 1}", lut=lut, min=im.display.channels[0].min, max=im.display.channels[0].max) for i, (im, lut) in enumerate(chosen)]
    res.display.mode = "composite"
    res.display.c = 0
    if not create_composite:
        from .types_ops import to_rgb
        from ..registry import OpContext
        s.add_image(res, activate=True)
        r = to_rgb(OpContext(session=s, image=res))
        res.replace_data(r.data, rgb=True)
        s.close_image(res.id)
        res.name = "RGB"
    if not keep_source:
        for im, _ in chosen:
            s.close_image(im.id)
    return OpResult(new_images=[res])


# "Set LUT" (was "Apply LUT", GUI fix round 2 E5-2 / E7-9): the id stays color.lut, so recorded macros, run("<LUT name>") and
# the ij_compat table keep working; the destructive ImageJ "Apply LUT" (bake the display range into the pixels) is type.apply_lut.
@op("color.lut", "Set LUT", "Image/Lookup Tables", kind="analysis", preview=False, roi_aware=False,
    params=[Param("lut", "choice", "Grays", choices=LUT_NAMES, help="Lookup table (colour map) for the current channel"),
            Param("all_channels", "bool", False, help="Set every channel's LUT, not only the current channel's"),
            Param("channel", "int", 0, min=0, advanced=True, help="Channel to set (1-based); 0 = the current channel. Recorded explicitly on multi-channel images")],
    description="Choose the lookup table (colour map) of the current channel – display only, the pixel values stay unchanged (Image ▸ Lookup Tables ▸ Fire … in Fiji).",
    keywords=["lut", "lookup table", "colormap", "colour map", "color map", "fire", "viridis", "apply lut", "change lut"])
def apply_lut(ctx, lut="Grays", all_channels=False, channel=0):
    img = ctx.img
    target = int(channel) - 1 if int(channel) > 0 else img.display.c  # 1-based, 0 = current
    if target >= img.nc:
        raise ValueError(f"channel {channel} does not exist – the image has {img.nc} channel{'s' if img.nc != 1 else ''}")
    patch = [{"lut": lut} if (all_channels or i == target) else {} for i in range(img.nc)]
    return OpResult(display={"channels": patch})


@op("color.invert_lut", "Invert LUT", "Image/Lookup Tables", kind="analysis", preview=False, roi_aware=False, description="Invert the lookup table of the current channel (pixel values unchanged).")
def invert_lut(ctx):
    img = ctx.img
    cur = img.display.channels[img.display.c]
    patch = [{"inverted": not cur.inverted} if i == img.display.c else {} for i in range(img.nc)]
    return OpResult(display={"channels": patch})


@op("color.bake_lut", "Bake LUT into RGB", "Image/Lookup Tables", kind="image", scope_param=False, preview=False, description="Create an RGB copy with the current LUT applied.")
def bake_lut(ctx):
    from .types_ops import to_rgb
    img = ctx.img
    cp = img.copy(name=f"{img.name} (RGB)")
    from ..registry import OpContext
    r = to_rgb(OpContext(session=ctx.session, image=cp))
    cp.replace_data(r.data, rgb=True)
    return OpResult(new_images=[cp])


@op("color.arrange", "Arrange Channels", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("order", "string", "2,1,3", help="New channel order, 1-based; omit channels to drop them")], description="Re-order or drop channels.")
def arrange(ctx, order="2,1,3"):
    img = ctx.img
    idx = [int(x) - 1 for x in order.replace(" ", "").split(",") if x]
    if not idx or any(i < 0 or i >= img.nc for i in idx):
        raise ValueError(f"Channel indices must be between 1 and {img.nc}")
    data = img.data[:, :, idx].copy()
    img.display.channels = [ChannelDisplay(**img.display.channels[i].to_dict()) for i in idx]
    img.display.c = 0
    return OpResult(data=data, rgb=False)


@op("color.channel_names", "Set Channel Names", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("names", "string", "DAPI, GFP, RFP", help="Comma-separated")], description="Name the channels.")
def channel_names(ctx, names=""):
    img = ctx.img
    parts = [n.strip() for n in names.split(",")]
    patch = [{"name": parts[i]} if i < len(parts) and parts[i] else {} for i in range(img.nc)]
    return OpResult(display={"channels": patch})
