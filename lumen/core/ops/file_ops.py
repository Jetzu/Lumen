"""File menu commands and sample images."""
from __future__ import annotations

import numpy as np

from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op

SAMPLES = ["Blobs", "Cells (3-D, 2 channels)", "Coins", "Astronaut (RGB)", "Cell nuclei (fluorescence)", "Human mitosis", "Moon", "T1 head (MRI stack)",
           "Microaneurysms (retina)", "Synthetic nuclei (3 channels)", "Nuclei timelapse (synthetic)"]

FILLS = ["Black", "White", "Ramp", "Linear gradient", "Radial gradient", "Checkerboard", "Sine grating", "Noise"]
#: the largest width / height File ▸ New accepts; the real limit is the memory guard in new_image() (half of the RAM)
MAX_DIM = 100_000


def new_image_limit_bytes() -> int:
    """Memory a new image may take: half of the physical RAM (the rest is for the undo buffer and the viewer)."""
    from ..lazy import physical_memory_bytes
    return max(int(physical_memory_bytes()) // 2, 256 * 1024 ** 2)


@op("file.new", "New Image", "File", kind="image", requires_image=False, scope_param=False, preview=False,
    params=[Param("title", "string", "Untitled"), Param("type", "choice", "8-bit", choices=["8-bit", "16-bit", "32-bit", "RGB"]),
            Param("width", "int", 512, min=1, max=MAX_DIM, unit="px"), Param("height", "int", 512, min=1, max=MAX_DIM, unit="px"), Param("slices", "int", 1, min=1), Param("channels", "int", 1, min=1, max=16), Param("frames", "int", 1, min=1),
            Param("fill", "choice", "Black", choices=FILLS, help="Ramp = horizontal ramp; Linear gradient = diagonal top-left → bottom-right; "
                  "Radial gradient = bright centre, dark corners; Checkerboard / Sine grating use the period"),
            Param("period", "int", 32, min=1, max=20000, unit="px", help="Square size of the Checkerboard / wavelength of the Sine grating", advanced=True)],
    description="Create a blank image (optionally pre-filled with a ramp, gradient, checkerboard, sine grating or noise).", shortcut="N")
def new_image(ctx, title="Untitled", type="8-bit", width=512, height=512, slices=1, channels=1, frames=1, fill="Black", period=32):
    dtype = {"8-bit": np.uint8, "16-bit": np.uint16, "32-bit": np.float32, "RGB": np.uint8}[type]
    nc = 3 if type == "RGB" else channels
    full = 255 if dtype == np.uint8 else (65535 if dtype == np.uint16 else 1.0)
    # memory guard with a clear message instead of a silently clamped size (E7-7 / E5-18)
    nbytes = int(frames) * int(slices) * int(nc) * int(height) * int(width) * np.dtype(dtype).itemsize
    limit = new_image_limit_bytes()
    if nbytes > limit:
        gb = 1024 ** 3
        dims = f"{width}×{height} {type}" + (f", {slices} slice(s)" if slices > 1 else "") + (f", {nc} channel(s)" if nc > 1 else "") + (f", {frames} frame(s)" if frames > 1 else "")
        raise ValueError(f"A {dims} image needs {nbytes / gb:.1f} GB – more than the {limit / gb:.1f} GB a new image may use on this machine "
                         f"(half of the memory). Reduce the size or the number of planes, or use a smaller bit depth.")
    data = np.zeros((frames, slices, nc, height, width), dtype)
    if fill == "White":
        data[:] = full
    elif fill == "Noise":
        data[:] = (np.random.default_rng().random(data.shape) * full).astype(dtype)
    elif fill != "Black":
        data[:] = fill_pattern(fill, width, height, full, period).astype(dtype)
    return OpResult(new_images=[LumenImage(data, name=title, rgb=(type == "RGB"))])


def fill_pattern(fill: str, width: int, height: int, full: float = 255.0, period: int = 32) -> np.ndarray:
    """(H, W) float32 test pattern in ``0 … full`` (Fiji's New Image fills plus the drawing patterns)."""
    w, h = max(int(width), 1), max(int(height), 1)
    x = np.arange(w, dtype=np.float32) / max(w - 1, 1)
    y = np.arange(h, dtype=np.float32) / max(h - 1, 1)
    if fill == "Ramp":
        pat = np.broadcast_to(x[None, :], (h, w))
    elif fill == "Linear gradient":
        pat = (x[None, :] + y[:, None]) / 2.0
    elif fill == "Radial gradient":
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32) - cx, np.arange(h, dtype=np.float32) - cy)
        r = np.sqrt(xx ** 2 + yy ** 2)
        pat = 1.0 - r / max(float(np.hypot(cx, cy)), 1e-6)
    elif fill == "Checkerboard":
        p = max(int(period), 1)
        pat = (((np.arange(w) // p)[None, :] + (np.arange(h) // p)[:, None]) % 2).astype(np.float32)
    elif fill == "Sine grating":
        p = max(int(period), 1)
        phase = (np.arange(w) % p).astype(np.float32) / p  # exactly periodic, whatever the width
        pat = np.broadcast_to((0.5 * (1.0 + np.sin(2.0 * np.pi * phase)))[None, :], (h, w))
    else:
        raise ValueError(f"Unknown fill {fill!r}")
    out = np.clip(np.asarray(pat, np.float32) * float(full), 0, float(full))
    return np.rint(out).astype(np.float32) if full > 1 else out


@op("file.sample", "Open Sample", "File", kind="image", requires_image=False, scope_param=False, preview=False,
    params=[Param("name", "choice", "Blobs", choices=SAMPLES)], description="Open a bundled sample image (scikit-image data).", keywords=["demo", "example", "test image"])
def open_sample(ctx, name="Blobs"):
    from skimage import data as skd
    if name == "Blobs":
        img = LumenImage(_blobs(), name="blobs.tif")
    elif name.startswith("Cells"):
        vol = skd.cells3d()  # (Z, C, Y, X) uint16
        img = LumenImage(vol, name="cells3d.tif", axes="ZCYX", channel_names=["Membrane", "Nuclei"], calibration=Calibration(0.26, 0.26, 0.29, "µm"))
        img.display.channels[0].lut, img.display.channels[1].lut = "Magenta", "Green"
        img.display.z = img.nz // 2
        img.reset_display(keep_luts=True)
    elif name == "Coins":
        img = LumenImage(skd.coins(), name="coins.png")
    elif name.startswith("Astronaut"):
        img = LumenImage(skd.astronaut(), name="astronaut.png", axes="YXS", rgb=True)
    elif name.startswith("Cell nuclei"):
        img = LumenImage(skd.human_mitosis(), name="human_mitosis.png")
    elif name == "Human mitosis":
        img = LumenImage(skd.human_mitosis(), name="mitosis.png")
    elif name == "Moon":
        img = LumenImage(skd.moon(), name="moon.png")
    elif name.startswith("T1"):
        img = LumenImage(skd.brain(), name="t1-head.tif", axes="ZYX")
        img.display.z = img.nz // 2
        img.reset_display(keep_luts=True)
    elif name.startswith("Micro"):
        img = LumenImage(skd.microaneurysms(), name="microaneurysms.png")
    elif name.startswith("Synthetic nuclei"):
        img = _synthetic_nuclei()
    elif name.startswith("Nuclei timelapse"):
        img = _synthetic_timelapse()
    else:
        raise ValueError(name)
    return OpResult(new_images=[img])


def _blobs(seed: int = 3) -> np.ndarray:
    from skimage.data import binary_blobs
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    b = binary_blobs(length=256, blob_size_fraction=0.08, volume_fraction=0.25, rng=seed).astype(np.float32)
    f = ndi.gaussian_filter(b, 2) * 180 + 40 + rng.normal(0, 8, b.shape)
    return np.clip(f, 0, 255).astype(np.uint8)


def _synthetic_nuclei(n: int = 60, size: int = 512, seed: int = 1) -> LumenImage:
    from skimage.draw import disk
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    nuc = np.zeros((size, size), np.float32)
    cyto = np.zeros((size, size), np.float32)
    spots = np.zeros((size, size), np.float32)
    for _ in range(n):
        cy, cx = rng.integers(20, size - 20, 2)
        r = rng.integers(8, 16)
        rr, cc = disk((cy, cx), r, shape=nuc.shape)
        nuc[rr, cc] = rng.uniform(0.6, 1.0)
        rr, cc = disk((cy, cx), r * 2, shape=nuc.shape)
        cyto[rr, cc] = np.maximum(cyto[rr, cc], rng.uniform(0.2, 0.5))
        for _ in range(rng.integers(0, 6)):
            sy, sx = cy + rng.integers(-r, r), cx + rng.integers(-r, r)
            rr, cc = disk((sy, sx), 2, shape=nuc.shape)
            spots[rr, cc] = 1.0
    nuc = ndi.gaussian_filter(nuc, 1.5)
    cyto = ndi.gaussian_filter(cyto, 3)
    spots = ndi.gaussian_filter(spots, 1)
    bg = ndi.gaussian_filter(rng.random((size, size)), 40) * 0.1
    stack = []
    for ch, gain in ((nuc, 3000), (cyto, 1500), (spots, 2500)):
        im = ch * gain + bg * gain + rng.normal(120, 25, ch.shape)
        stack.append(np.clip(im, 0, 65535))
    data = np.stack(stack).astype(np.uint16)
    img = LumenImage(data, name="synthetic_cells.tif", axes="CYX", channel_names=["DAPI", "Cytoplasm", "Spots"], calibration=Calibration(0.325, 0.325, 1.0, "µm"))
    img.display.channels[0].lut, img.display.channels[1].lut, img.display.channels[2].lut = "Blue", "Green", "Magenta"
    return img


def _synthetic_timelapse(frames: int = 12, size: int = 256, seed: int = 2) -> LumenImage:
    from skimage.draw import disk
    from scipy import ndimage as ndi
    rng = np.random.default_rng(seed)
    n = 15
    pos = rng.uniform(30, size - 30, (n, 2))
    vel = rng.normal(0, 2.0, (n, 2))
    out = []
    for t in range(frames):
        f = np.zeros((size, size), np.float32)
        for k in range(n):
            cy, cx = pos[k]
            rr, cc = disk((cy, cx), 9, shape=f.shape)
            f[rr, cc] = 0.8 + 0.2 * np.sin(t / 2 + k)
        pos += vel
        pos = np.clip(pos, 15, size - 15)
        f = ndi.gaussian_filter(f, 1.2) * 200 + 30 + rng.normal(0, 6, f.shape)
        out.append(np.clip(f, 0, 255))
    data = np.asarray(out).astype(np.uint8)
    img = LumenImage(data, name="timelapse.tif", axes="TYX", calibration=Calibration(0.65, 0.65, 1.0, "µm", frame_interval=5.0, time_unit="min"))
    return img


@op("file.close", "Close", "File", kind="image", scope_param=False, preview=False, description="Close the current image.", shortcut="W")
def close(ctx):
    return OpResult(close_image=True)


@op("file.close_all", "Close All", "File", kind="image", requires_image=False, scope_param=False, preview=False, description="Close every image.")
def close_all(ctx):
    s = ctx.session
    for iid in list(s.images.keys()):
        s.close_image(iid)
    return OpResult()


@op("file.revert", "Revert", "File", kind="image", scope_param=False, preview=False, description="Reload the image from disk, discarding all changes.")
def revert(ctx):
    from pathlib import Path
    from ..io import is_virtual, read_image, read_raw, read_raw_folder, read_sequence
    img = ctx.img
    if not img.path:
        raise ValueError("This image was not opened from a file")
    # Readers that need parameters (image sequences, raw imports, series/dataset selections) record them in
    # metadata["source"]; the record only applies while img.path is still the path it was read from
    # (Save As replaces img.path with the new file, which is then re-read with default parameters).
    src = img.metadata.get("source")
    if not isinstance(src, dict) or src.get("path") != img.path:
        src = {}
    kind = src.get("kind", "file")
    params = {k: v for k, v in src.items() if k not in ("kind", "path")}
    p = Path(img.path)
    if kind == "sequence":
        if not p.is_dir():
            raise ValueError(f"Cannot revert: the image-sequence folder {p} is gone")
        fresh = read_sequence(p, **params)
    elif kind == "raw":
        fresh = read_raw(p, **params)
    elif kind == "raw_folder":
        fresh = read_raw_folder(p, **params)
    else:
        fresh = read_image(p, dataset=params.get("dataset"), series=params.get("series"), level=int(params.get("level", 0) or 0),
                           virtual=bool(params.get("virtual", False)))
    img.calibration = fresh.calibration
    img.display = fresh.display
    img.overlay = []
    if is_virtual(fresh.data):
        # a virtual stack stays virtual (Fiji's ImagePlus.revert never loads a virtual stack): returned as
        # OpResult.data it would go through replace_data() / np.ascontiguousarray and read the whole file into
        # memory – install the lazy array in place instead (run_op sees the rev bump; the frontend refetches on rev)
        img.data = fresh.data
        img.rgb = fresh.rgb
        img.lut_is_binary = False
        img.rev += 1
        img.sync_channels()
        img.clamp_position()
        return OpResult(clear_roi=True)
    return OpResult(data=fresh.data, rgb=fresh.rgb, clear_roi=True)
