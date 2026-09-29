from __future__ import annotations

from affine import Affine


def scaled_transform(transform: Affine, scale: int) -> Affine:
    """Same origin, pixel size divided by `scale`."""
    return transform * Affine.scale(1.0 / scale)
