# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh Node - Universal topology-changing remeshing operations
"""

import numpy as np
import trimesh as trimesh_module

from .._utils import mesh_ops
from .._utils import blender_bridge


class RemeshNode:
    """
    Universal Remesh - Unified topology-changing remeshing operations.

    Consolidates multiple remeshing backends into a single node:
    - pymeshlab_isotropic: PyMeshLab isotropic remeshing
    - cgal_isotropic: CGAL high-quality isotropic remeshing
    - blender_voxel: Blender voxel-based remeshing (watertight output)
    - blender_quadriflow: Blender Quadriflow quad remeshing
    - instant_meshes: Field-aligned quad remeshing

    Parameters are backend-specific; unused parameters are ignored.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "backend": ([
                    "pymeshlab_isotropic",
                    "cgal_isotropic",
                    "blender_voxel",
                    "blender_quadriflow",
                    "instant_meshes"
                ], {"default": "pymeshlab_isotropic"}),
            },
            "optional": {
                # Isotropic params (pymeshlab, cgal)
                "target_edge_length": ("FLOAT", {
                    "default": 1.00,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                # CGAL-specific
                "protect_boundaries": (["true", "false"], {"default": "true"}),
                # Blender voxel
                "voxel_size": ("FLOAT", {
                    "default": 1,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                # Quadriflow/Instant Meshes
                "target_face_count": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100
                }),
                "target_vertex_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100
                }),
                # Instant Meshes specific
                "deterministic": (["true", "false"], {"default": "true"}),
                "crease_angle": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "info")
    FUNCTION = "remesh"
    CATEGORY = "geompack/remeshing"

    def remesh(self, trimesh, backend, target_edge_length=0.05, iterations=3,
               protect_boundaries="true", voxel_size=0.02, target_face_count=10000,
               target_vertex_count=5000, deterministic="true", crease_angle=0.0):
        """
        Apply remeshing based on selected backend.

        Args:
            trimesh: Input trimesh.Trimesh object
            backend: Remeshing backend to use
            [other params]: Backend-specific parameters

        Returns:
            tuple: (remeshed_mesh, info_string)
        """
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"[Remesh] Input: {initial_vertices} vertices, {initial_faces} faces")
        print(f"[Remesh] Backend: {backend}")

        if backend == "pymeshlab_isotropic":
            remeshed_mesh, info = self._pymeshlab_isotropic(
                trimesh, target_edge_length, iterations
            )
        elif backend == "cgal_isotropic":
            remeshed_mesh, info = self._cgal_isotropic(
                trimesh, target_edge_length, iterations, protect_boundaries
            )
        elif backend == "blender_voxel":
            remeshed_mesh, info = self._blender_voxel(trimesh, voxel_size)
        elif backend == "blender_quadriflow":
            remeshed_mesh, info = self._blender_quadriflow(trimesh, target_face_count)
        elif backend == "instant_meshes":
            remeshed_mesh, info = self._instant_meshes(
                trimesh, target_vertex_count, deterministic, crease_angle
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        print(f"[Remesh] Output: {len(remeshed_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(remeshed_mesh.faces)} faces ({face_change:+d})")

        return (remeshed_mesh, info)

    def _pymeshlab_isotropic(self, trimesh, target_edge_length, iterations):
        """PyMeshLab isotropic remeshing."""
        remeshed_mesh, error = mesh_ops.pymeshlab_isotropic_remesh(
            trimesh, target_edge_length, iterations
        )
        if remeshed_mesh is None:
            raise ValueError(f"PyMeshLab remeshing failed: {error}")

        info = f"""Remesh Results (PyMeshLab Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _cgal_isotropic(self, trimesh, target_edge_length, iterations, protect_boundaries):
        """CGAL isotropic remeshing."""
        protect = (protect_boundaries == "true")
        remeshed_mesh, error = mesh_ops.cgal_isotropic_remesh(
            trimesh, target_edge_length, iterations, protect
        )
        if remeshed_mesh is None:
            raise ValueError(f"CGAL remeshing failed: {error}")

        info = f"""Remesh Results (CGAL Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}
Protect Boundaries: {protect_boundaries}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _blender_voxel(self, trimesh, voxel_size):
        """Blender voxel remeshing."""
        script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{{input_path}}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Apply voxel remesh
obj.data.remesh_voxel_size = {voxel_size}
bpy.ops.object.voxel_remesh()

# Export remeshed object
bpy.ops.wm.obj_export(
    filepath='{{output_path}}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

        print(f"[Remesh] Running Blender voxel remesh (voxel_size={voxel_size})...")
        remeshed_mesh = blender_bridge.run_blender_mesh_operation(
            trimesh, script,
            metadata_key='remeshing',
            metadata_values={
                'algorithm': 'blender_voxel',
                'voxel_size': voxel_size,
                'original_vertices': len(trimesh.vertices),
                'original_faces': len(trimesh.faces)
            }
        )

        info = f"""Remesh Results (Blender Voxel):

Voxel Size: {voxel_size}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    def _blender_quadriflow(self, trimesh, target_face_count):
        """Blender Quadriflow remeshing."""
        script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{{input_path}}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Apply Quadriflow remesh
bpy.ops.object.quadriflow_remesh(
    use_mesh_symmetry=False,
    use_preserve_sharp=False,
    use_preserve_boundary=False,
    smooth_normals=False,
    mode='FACES',
    target_faces={target_face_count},
    seed=0
)

# Export remeshed object
bpy.ops.wm.obj_export(
    filepath='{{output_path}}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

        print(f"[Remesh] Running Blender Quadriflow (target_faces={target_face_count})...")
        remeshed_mesh = blender_bridge.run_blender_mesh_operation(
            trimesh, script,
            metadata_key='remeshing',
            metadata_values={
                'algorithm': 'blender_quadriflow',
                'target_face_count': target_face_count,
                'original_vertices': len(trimesh.vertices),
                'original_faces': len(trimesh.faces)
            }
        )

        info = f"""Remesh Results (Blender Quadriflow):

Target Face Count: {target_face_count:,}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Quadriflow creates quad-dominant meshes with good topology.
"""
        return remeshed_mesh, info

    def _instant_meshes(self, trimesh, target_vertex_count, deterministic, crease_angle):
        """Instant Meshes field-aligned remeshing."""
        try:
            import pynanoinstantmeshes as pynano
        except ImportError:
            raise ImportError(
                "PyNanoInstantMeshes not installed. Install with: pip install PyNanoInstantMeshes"
            )

        V = trimesh.vertices.astype(np.float32)
        F = trimesh.faces.astype(np.uint32)

        V_out, F_out = pynano.remesh(
            V, F,
            vertex_count=target_vertex_count,
            deterministic=(deterministic == "true"),
            creaseAngle=crease_angle
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out,
            faces=F_out,
            process=False
        )

        # Preserve metadata
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'instant_meshes',
            'target_vertex_count': target_vertex_count,
            'deterministic': deterministic == "true",
            'crease_angle': crease_angle,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (Instant Meshes):

Target Vertex Count: {target_vertex_count:,}
Deterministic: {deterministic}
Crease Angle: {crease_angle}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Instant Meshes creates flow-aligned quad meshes.
"""
        return remeshed_mesh, info
