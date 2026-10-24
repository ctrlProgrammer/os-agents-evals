"""Prepare a photo for JEV. Needs Pillow: pip install -e ".[images]"."""

import io
import math
from pathlib import Path

# JEV still-image limits: each side 280-1568 px.
MIN_SIDE = 280
MAX_SIDE = 1568
JPEG_QUALITY = 90


def prepare_jpeg(path: Path) -> bytes:
    """Rotate upright and fit JEV's size limits. Same approach as freezer's prepare_jpeg."""
    from PIL import Image, ImageOps

    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    width, height = image.size
    scale = max(MIN_SIDE / width, MIN_SIDE / height, 1.0)
    if scale > 1:
        image = image.resize((math.ceil(width * scale), math.ceil(height * scale)), Image.Resampling.LANCZOS)
        image.thumbnail((MAX_SIDE, MAX_SIDE))
    width, height = image.size
    if width < MIN_SIDE or height < MIN_SIDE:
        canvas = Image.new("RGB", (max(width, MIN_SIDE), max(height, MIN_SIDE)), (0, 0, 0))
        canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
        image = canvas
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return buffer.getvalue()
