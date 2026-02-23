# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Combine Meshes Node - Concatenate multiple meshes into one
"""

import logging

import numpy as np
import trimesh as trimesh_module

log = logging.getLogger("geometrypack")


class CombineMeshesNode:
    """
    Combine Meshes - Concatenate multiple meshes into one.

    Simply concatenates vertices and faces without performing boolean operations.
    The result contains all geometry from input meshes as separate components.
    Useful for grouping objects or preparing batch operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_a": ("TRIMESH",),
            },
            "optional": {
                "mesh_b": ("TRIMESH",),
                "mesh_c": ("TRIMESH",),
                "mesh_d": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("combined_mesh", "info")
    FUNCTION = "combine"
    CATEGORY = "geompack/combine"
    OUTPUT_NODE = True

    def combine(self, mesh_a, mesh_b=None, mesh_c=None, mesh_d=None):
        """
        Combine multiple meshes into one.

        Args:
            mesh_a: First mesh (required)
            mesh_b, mesh_c, mesh_d: Optional additional meshes

        Returns:
            tuple: (combined_mesh, info_string)
        """
        meshes = [mesh_a]
        if mesh_b is not None:
            meshes.append(mesh_b)
        if mesh_c is not None:
            meshes.append(mesh_c)
        if mesh_d is not None:
            meshes.append(mesh_d)

        log.info("Combining %d meshes", len(meshes))

        # Track input stats
        input_stats = []
        total_vertices = 0
        total_faces = 0

        for i, mesh in enumerate(meshes):
            input_stats.append({
                'index': i + 1,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces)
            })
            total_vertices += len(mesh.vertices)
            total_faces += len(mesh.faces)
            log.info("Mesh %d: %d vertices, %d faces", i+1, len(mesh.vertices), len(mesh.faces))

        # Concatenate meshes
        if len(meshes) == 1:
            result = mesh_a.copy()
        else:
            result = trimesh_module.util.concatenate(meshes)

        # Preserve metadata from first mesh
        result.metadata = mesh_a.metadata.copy()
        result.metadata['combined'] = {
            'num_meshes': len(meshes),
            'input_stats': input_stats,
            'total_vertices': len(result.vertices),
            'total_faces': len(result.faces)
        }

        # Build info string
        mesh_lines = []
        for stat in input_stats:
            mesh_lines.append(f"  Mesh {stat['index']}: {stat['vertices']:,} vertices, {stat['faces']:,} faces")

        info = f"""Combine Meshes Results:

Number of Meshes Combined: {len(meshes)}

Input Meshes:
{chr(10).join(mesh_lines)}

Combined Result:
  Total Vertices: {len(result.vertices):,}
  Total Faces: {len(result.faces):,}
  Connected Components: {len(trimesh_module.graph.connected_components(result.face_adjacency)[1])}

Note: Meshes are concatenated without boolean operations.
Components remain separate within the combined mesh.
"""

        log.info("Result: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return {"ui": {"text": [info]}, "result": (result, info)}


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCombineMeshes": CombineMeshesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCombineMeshes": "Combine Meshes",
}
