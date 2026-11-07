"""Turn a 7x7 attention grid into a coloured heatmap over the photo."""

from __future__ import annotations

import matplotlib
import numpy as np
from PIL import Image, ImageOps


def attention_overlay(image: Image.Image, attn: np.ndarray, alpha: float = 0.5,
                      max_side: int = 320) -> Image.Image:
    """Blend a (grid, grid) attention map (values 0-1) over `image`."""
    img = ImageOps.exif_transpose(image).convert("RGB")
    img.thumbnail((max_side, max_side))
    heat = Image.fromarray(np.uint8(np.clip(attn, 0, 1) * 255))
    heat = heat.resize(img.size, Image.Resampling.BICUBIC)
    rgb = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    return Image.blend(img, Image.fromarray(np.uint8(rgb * 255)), alpha)
