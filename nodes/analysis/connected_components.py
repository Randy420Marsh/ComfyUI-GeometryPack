# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Connected Components Node - Label disconnected mesh parts with part_id field.

Uses trimesh's graph.connected_components() to identify disconnected regions
and assigns a unique part_id to each face based on which component it belongs to.

Supports batch processing: input a list of meshes, get a list of results.
"""

import os
import numpy as np


class ConnectedComponentsNode:
    """
    Label disconnected mesh components with a part_id face attribute.

    Each disconnected region of the mesh gets a unique integer ID (0, 1, 2, ...).
    The part_id is stored as a face attribute that can be visualized with the
    field-based mesh preview nodes.

    Supports batch processing: input a list of meshes, get a list of results.
    """

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "component_summary")
    OUTPUT_IS_LIST = (True, False)  # TRIMESH is list, STRING is single summary
    FUNCTION = "label_components"
    CATEGORY = "geompack/analysis"

    def label_components(self, trimesh):
        """
        Label each face with its connected component ID.

        Args:
            trimesh: Input trimesh object(s)

        Returns:
            tuple: (list of trimesh with part_id face attribute, summary string)
        """
        import trimesh as trimesh_module

        # Handle batch input
        meshes = trimesh if isinstance(trimesh, list) else [trimesh]

        result_meshes = []
        summary_lines = []

        for mesh in meshes:
            # Get connected components using face adjacency
            # Returns list of arrays, each containing face indices for one component
            components = trimesh_module.graph.connected_components(
                mesh.face_adjacency,
                nodes=np.arange(len(mesh.faces))
            )

            num_components = len(components)

            # Create part_id array for all faces
            part_ids = np.zeros(len(mesh.faces), dtype=np.int32)

            component_sizes = []
            for component_id, face_indices in enumerate(components):
                part_ids[face_indices] = component_id
                component_sizes.append(len(face_indices))

            # Get mesh name for summary
            mesh_name = mesh.metadata.get('file_name', 'mesh') if hasattr(mesh, 'metadata') else 'mesh'
            # Remove extension for cleaner display
            mesh_name_short = os.path.splitext(mesh_name)[0]

            # Add to summary
            summary_lines.append(f"{mesh_name_short}: {num_components}")

            # Print to console
            if num_components <= 5:
                sizes_str = ", ".join(str(s) for s in component_sizes)
                print(f"[ConnectedComponents] {mesh_name}: {num_components} component(s): [{sizes_str}] faces each")
            else:
                largest = max(component_sizes)
                smallest = min(component_sizes)
                print(f"[ConnectedComponents] {mesh_name}: {num_components} component(s) (largest: {largest} faces, smallest: {smallest} faces)")

            # Store as face attribute
            # Make a copy to avoid modifying the original
            result_mesh = mesh.copy()
            result_mesh.face_attributes['part_id'] = part_ids

            # Also store in visual facets metadata for compatibility
            if not hasattr(result_mesh, 'metadata'):
                result_mesh.metadata = {}
            result_mesh.metadata['part_ids'] = part_ids
            result_mesh.metadata['num_components'] = num_components

            result_meshes.append(result_mesh)

        # Create summary string
        summary = "\n".join(summary_lines)

        print(f"[ConnectedComponents] Processed {len(meshes)} mesh(es)")
        return (result_meshes, summary)


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackConnectedComponents": ConnectedComponentsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackConnectedComponents": "Connected Components",
}
