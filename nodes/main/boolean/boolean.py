# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Boolean CGAL Node - CSG operations using libigl+CGAL
Requires igl.copyleft.cgal.
"""

import logging

import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class BooleanCGALNode(io.ComfyNode):
    """
    Boolean CGAL - Union, Difference, and Intersection of meshes using CGAL.

    Performs Constructive Solid Geometry (CSG) operations:
    - union: Combine two meshes into one
    - difference: Subtract mesh_b from mesh_a
    - intersection: Keep only overlapping parts

    Uses libigl with CGAL backend for robust boolean operations.
    For Blender-based booleans, use "Boolean Blender" node.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackBooleanCGAL",
            display_name="Boolean CGAL",
            category="geompack/boolean",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("mesh_a"),
                io.Custom("TRIMESH").Input("mesh_b"),
                io.Combo.Input("operation", options=["union", "difference", "intersection"]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="result_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, mesh_a, mesh_b, operation):
        """
        Perform boolean operation on two meshes using libigl+CGAL.

        Args:
            mesh_a: First mesh (base mesh for difference)
            mesh_b: Second mesh (subtracted mesh for difference)
            operation: Boolean operation type

        Returns:
            tuple: (result_mesh, info_string)
        """
        log.info("Mesh A: %d vertices, %d faces", len(mesh_a.vertices), len(mesh_a.faces))
        log.info("Mesh B: %d vertices, %d faces", len(mesh_b.vertices), len(mesh_b.faces))
        log.info("Operation: %s", operation)

        try:
            import igl.copyleft.cgal as cgal
            log.info("Using libigl+CGAL backend...")

            # Convert trimesh to numpy arrays
            VA = np.asarray(mesh_a.vertices, dtype=np.float64)
            FA = np.asarray(mesh_a.faces, dtype=np.int64)
            VB = np.asarray(mesh_b.vertices, dtype=np.float64)
            FB = np.asarray(mesh_b.faces, dtype=np.int64)

            # Map operation to igl type_str
            op_map = {
                "union": "union",
                "difference": "difference",
                "intersection": "intersection"
            }

            if operation not in op_map:
                raise ValueError(f"Unknown operation: {operation}")

            # Perform boolean operation using CGAL
            VC, FC, J = cgal.mesh_boolean(VA, FA, VB, FB, op_map[operation])

            # Create result trimesh
            result = trimesh_module.Trimesh(vertices=VC, faces=FC, process=False)

            # Preserve metadata from mesh_a
            result.metadata = mesh_a.metadata.copy()
            result.metadata['boolean'] = {
                'operation': operation,
                'engine': 'libigl_cgal',
                'mesh_a_vertices': len(mesh_a.vertices),
                'mesh_a_faces': len(mesh_a.faces),
                'mesh_b_vertices': len(mesh_b.vertices),
                'mesh_b_faces': len(mesh_b.faces),
                'result_vertices': len(result.vertices),
                'result_faces': len(result.faces)
            }

            info = f"""Boolean Operation Results:

Operation: {operation.upper()}
Engine: libigl + CGAL

Mesh A:
  Vertices: {len(mesh_a.vertices):,}
  Faces: {len(mesh_a.faces):,}

Mesh B:
  Vertices: {len(mesh_b.vertices):,}
  Faces: {len(mesh_b.faces):,}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""

            log.info("Success: %d vertices, %d faces", len(result.vertices), len(result.faces))
            return io.NodeOutput(result, info, ui={"text": [info]})

        except Exception as e:
            raise RuntimeError(f"Boolean CGAL operation failed: {e}")


NODE_CLASS_MAPPINGS = {
    "GeomPackBooleanCGAL": BooleanCGALNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBooleanCGAL": "Boolean CGAL",
}
