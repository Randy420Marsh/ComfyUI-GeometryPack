# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Analysis module - mesh information and quality metrics."""

from .mesh_info import NODE_CLASS_MAPPINGS as MESH_INFO_MAPPINGS
from .mesh_info import NODE_DISPLAY_NAME_MAPPINGS as MESH_INFO_DISPLAY
from .mesh_quality import NODE_CLASS_MAPPINGS as MESH_QUALITY_MAPPINGS
from .mesh_quality import NODE_DISPLAY_NAME_MAPPINGS as MESH_QUALITY_DISPLAY

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **MESH_INFO_MAPPINGS,
    **MESH_QUALITY_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_INFO_DISPLAY,
    **MESH_QUALITY_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
