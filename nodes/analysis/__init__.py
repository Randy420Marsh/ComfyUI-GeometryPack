# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Analysis module - mesh information and quality metrics."""

from .mesh_info import NODE_CLASS_MAPPINGS as MESH_INFO_MAPPINGS
from .mesh_info import NODE_DISPLAY_NAME_MAPPINGS as MESH_INFO_DISPLAY
from .mesh_quality import NODE_CLASS_MAPPINGS as MESH_QUALITY_MAPPINGS
from .mesh_quality import NODE_DISPLAY_NAME_MAPPINGS as MESH_QUALITY_DISPLAY
from .connected_components import NODE_CLASS_MAPPINGS as CONNECTED_COMPONENTS_MAPPINGS
from .connected_components import NODE_DISPLAY_NAME_MAPPINGS as CONNECTED_COMPONENTS_DISPLAY

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **MESH_INFO_MAPPINGS,
    **MESH_QUALITY_MAPPINGS,
    **CONNECTED_COMPONENTS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_INFO_DISPLAY,
    **MESH_QUALITY_DISPLAY,
    **CONNECTED_COMPONENTS_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
