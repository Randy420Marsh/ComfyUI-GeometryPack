# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Surface reconstruction module."""

from .reconstruct_surface import NODE_CLASS_MAPPINGS as RECONSTRUCT_MAPPINGS
from .reconstruct_surface import NODE_DISPLAY_NAME_MAPPINGS as RECONSTRUCT_DISPLAY
from .alpha_wrap import NODE_CLASS_MAPPINGS as ALPHA_WRAP_MAPPINGS
from .alpha_wrap import NODE_DISPLAY_NAME_MAPPINGS as ALPHA_WRAP_DISPLAY

NODE_CLASS_MAPPINGS = {
    **RECONSTRUCT_MAPPINGS,
    **ALPHA_WRAP_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **RECONSTRUCT_DISPLAY,
    **ALPHA_WRAP_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
