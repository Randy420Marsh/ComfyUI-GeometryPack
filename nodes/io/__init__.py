# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
I/O Nodes - Load and save mesh files
"""

from .load_mesh import NODE_CLASS_MAPPINGS as LoadMesh_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMesh_display
from .load_mesh_batch import NODE_CLASS_MAPPINGS as LoadMeshBatch_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshBatch_display
from .load_mesh_fbx import NODE_CLASS_MAPPINGS as LoadMeshFBX_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshFBX_display
from .load_mesh_blend import NODE_CLASS_MAPPINGS as LoadMeshBlend_mappings, NODE_DISPLAY_NAME_MAPPINGS as LoadMeshBlend_display
from .save_mesh import NODE_CLASS_MAPPINGS as SaveMesh_mappings, NODE_DISPLAY_NAME_MAPPINGS as SaveMesh_display

# Aggregate all node mappings
NODE_CLASS_MAPPINGS = {
    **LoadMesh_mappings,
    **LoadMeshBatch_mappings,
    **LoadMeshFBX_mappings,
    **LoadMeshBlend_mappings,
    **SaveMesh_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LoadMesh_display,
    **LoadMeshBatch_display,
    **LoadMeshFBX_display,
    **LoadMeshBlend_display,
    **SaveMesh_display,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
