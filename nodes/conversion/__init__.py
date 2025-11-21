"""Conversion module - mesh to point cloud."""

from .mesh_to_pointcloud import NODE_CLASS_MAPPINGS as MESH_TO_PC_MAPPINGS
from .mesh_to_pointcloud import NODE_DISPLAY_NAME_MAPPINGS as MESH_TO_PC_DISPLAY

# Combine all mappings
NODE_CLASS_MAPPINGS = {
    **MESH_TO_PC_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **MESH_TO_PC_DISPLAY,
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
