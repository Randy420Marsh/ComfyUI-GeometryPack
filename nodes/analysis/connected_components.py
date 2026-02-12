# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Connected Components Node - Label disconnected mesh parts with part_id field.

Uses trimesh's graph.connected_components() to identify disconnected regions
and assigns a unique part_id to each face based on which component it belongs to.
"""

import numpy as np


class ConnectedComponentsNode:
    """
    Label disconnected mesh components with a part_id face attribute.

    Each disconnected region of the mesh gets a unique integer ID (0, 1, 2, ...).
    The part_id is stored as a face attribute that can be visualized with the
    field-based mesh preview nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "INT")
    RETURN_NAMES = ("trimesh", "num_components")
    FUNCTION = "label_components"
    CATEGORY = "geompack/analysis"

    def label_components(self, trimesh):
        """
        Label each face with its connected component ID.

        Args:
            trimesh: Input trimesh object

        Returns:
            tuple: (trimesh with part_id face attribute, number of components)
        """
        import trimesh as trimesh_module

        # Get connected components using face adjacency
        # Returns list of arrays, each containing face indices for one component
        components = trimesh_module.graph.connected_components(
            trimesh.face_adjacency,
            nodes=np.arange(len(trimesh.faces))
        )

        num_components = len(components)
        print(f"[ConnectedComponents] Found {num_components} disconnected component(s)")

        # Create part_id array for all faces
        part_ids = np.zeros(len(trimesh.faces), dtype=np.int32)

        for component_id, face_indices in enumerate(components):
            part_ids[face_indices] = component_id
            print(f"[ConnectedComponents] Component {component_id}: {len(face_indices)} faces")

        # Store as face attribute
        # Make a copy to avoid modifying the original
        result_mesh = trimesh.copy()
        result_mesh.face_attributes['part_id'] = part_ids

        # Also store in visual facets metadata for compatibility
        if not hasattr(result_mesh, 'metadata'):
            result_mesh.metadata = {}
        result_mesh.metadata['part_ids'] = part_ids
        result_mesh.metadata['num_components'] = num_components

        return (result_mesh, num_components)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackConnectedComponents": ConnectedComponentsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackConnectedComponents": "Connected Components",
}
