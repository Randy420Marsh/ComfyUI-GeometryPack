# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Texture remeshing module - remesh with texture preservation."""

from .remesh_uv import NODE_CLASS_MAPPINGS as REMESH_UV_MAPPINGS
from .remesh_uv import NODE_DISPLAY_NAME_MAPPINGS as REMESH_UV_DISPLAY
from .texture_to_geometry import NODE_CLASS_MAPPINGS as TEX_TO_GEOM_MAPPINGS
from .texture_to_geometry import NODE_DISPLAY_NAME_MAPPINGS as TEX_TO_GEOM_DISPLAY
from .depth_normals_to_mesh import NODE_CLASS_MAPPINGS as DEPTH_NORMALS_MAPPINGS
from .depth_normals_to_mesh import NODE_DISPLAY_NAME_MAPPINGS as DEPTH_NORMALS_DISPLAY

# Also import from parent texture_remeshing.py for other nodes (BlenderRemeshWithTexture, XAtlasRemeshWithTexture)

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **REMESH_UV_MAPPINGS,
    **TEX_TO_GEOM_MAPPINGS,
    **DEPTH_NORMALS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **REMESH_UV_DISPLAY,
    **TEX_TO_GEOM_DISPLAY,
    **DEPTH_NORMALS_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
