# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Combine/split operations module."""

from .combine_meshes import NODE_CLASS_MAPPINGS as COMBINE_MAPPINGS
from .combine_meshes import NODE_DISPLAY_NAME_MAPPINGS as COMBINE_DISPLAY

# Also import from parent combine.py for other nodes (SplitComponents, FilterComponents)

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **COMBINE_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **COMBINE_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
