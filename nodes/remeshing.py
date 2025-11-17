"""
Remeshing Nodes - Mesh remeshing and optimization algorithms
"""

import numpy as np
import trimesh as trimesh_module
import os
import subprocess
import tempfile
import shutil
from pathlib import Path

from . import mesh_utils
from . import blender_utils


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
                    "default": 0.05,
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
                    "default": 0.02,
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
        remeshed_mesh, error = mesh_utils.pymeshlab_isotropic_remesh(
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
        remeshed_mesh, error = mesh_utils.cgal_isotropic_remesh(
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
        remeshed_mesh = blender_utils.run_blender_mesh_operation(
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
        remeshed_mesh = blender_utils.run_blender_mesh_operation(
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
            import PyNanoInstantMeshes as pynano
        except ImportError:
            raise ImportError(
                "PyNanoInstantMeshes not installed. Install with: pip install PyNanoInstantMeshes"
            )

        V = trimesh.vertices.astype(np.float64)
        F = trimesh.faces.astype(np.int32)

        V_out, F_out = pynano.instant_meshes(
            V, F,
            target_vertex_count=target_vertex_count,
            deterministic=(deterministic == "true"),
            crease_angle=crease_angle
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


class RefineMeshNode:
    """
    Refine Mesh - Unified non-destructive mesh refinement operations.

    Consolidates mesh refinement operations:
    - decimation: Reduce face count (quadric error metrics)
    - subdivision_loop: Smooth subdivision (Loop algorithm)
    - subdivision_midpoint: Simple midpoint subdivision
    - laplacian_smoothing: Iterative Laplacian smoothing

    Parameters are operation-specific; unused parameters are ignored.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "operation": ([
                    "decimation",
                    "subdivision_loop",
                    "subdivision_midpoint",
                    "laplacian_smoothing"
                ], {"default": "decimation"}),
            },
            "optional": {
                # Decimation
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 4,
                    "max": 10000000,
                    "step": 100
                }),
                "decimation_method": (["trimesh", "pymeshlab"], {"default": "trimesh"}),
                # Subdivision
                "subdivision_iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                # Smoothing
                "smoothing_iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "lambda_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("refined_mesh", "info")
    FUNCTION = "refine"
    CATEGORY = "geompack/remeshing"

    def refine(self, trimesh, operation, target_face_count=5000, decimation_method="trimesh",
               subdivision_iterations=1, smoothing_iterations=5, lambda_factor=0.5):
        """
        Apply mesh refinement based on selected operation.

        Args:
            trimesh: Input trimesh.Trimesh object
            operation: Refinement operation to apply
            [other params]: Operation-specific parameters

        Returns:
            tuple: (refined_mesh, info_string)
        """
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        print(f"[RefineMesh] Input: {initial_vertices} vertices, {initial_faces} faces")
        print(f"[RefineMesh] Operation: {operation}")

        if operation == "decimation":
            refined_mesh, info = self._decimate(trimesh, target_face_count, decimation_method)
        elif operation == "subdivision_loop":
            refined_mesh, info = self._subdivide(trimesh, subdivision_iterations, "loop")
        elif operation == "subdivision_midpoint":
            refined_mesh, info = self._subdivide(trimesh, subdivision_iterations, "midpoint")
        elif operation == "laplacian_smoothing":
            refined_mesh, info = self._smooth(trimesh, smoothing_iterations, lambda_factor)
        else:
            raise ValueError(f"Unknown operation: {operation}")

        vertex_change = len(refined_mesh.vertices) - initial_vertices
        face_change = len(refined_mesh.faces) - initial_faces

        print(f"[RefineMesh] Output: {len(refined_mesh.vertices)} vertices ({vertex_change:+d}), "
              f"{len(refined_mesh.faces)} faces ({face_change:+d})")

        return (refined_mesh, info)

    def _decimate(self, trimesh, target_face_count, method):
        """Decimate mesh to target face count."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        if method == "trimesh":
            decimated = trimesh.simplify_quadric_decimation(face_count=target_face_count)
        elif method == "pymeshlab":
            try:
                import pymeshlab
            except ImportError:
                raise ImportError("pymeshlab not installed. Install with: pip install pymeshlab")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=trimesh.vertices,
                face_matrix=trimesh.faces
            )
            ms.add_mesh(pml_mesh)

            ms.simplification_quadric_edge_collapse_decimation(
                targetfacenum=target_face_count,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=False
            )

            decimated_pml = ms.current_mesh()
            decimated = trimesh_module.Trimesh(
                vertices=decimated_pml.vertex_matrix(),
                faces=decimated_pml.face_matrix()
            )
        else:
            raise ValueError(f"Unknown decimation method: {method}")

        # Preserve metadata
        decimated.metadata = trimesh.metadata.copy()
        decimated.metadata['decimation'] = {
            'method': method,
            'target_face_count': target_face_count,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'reduction_ratio': len(decimated.faces) / initial_faces if initial_faces > 0 else 0
        }

        reduction_pct = 100.0 * (initial_faces - len(decimated.faces)) / initial_faces if initial_faces > 0 else 0

        info = f"""Refine Mesh Results (Decimation):

Method: {method}
Target Faces: {target_face_count:,}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}

Reduction: {reduction_pct:.1f}%
"""
        return decimated, info

    def _subdivide(self, trimesh, iterations, method):
        """Subdivide mesh to increase resolution."""
        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        subdivided = trimesh.copy()

        for i in range(iterations):
            if method == "loop":
                subdivided = subdivided.subdivide_loop(iterations=1)
            elif method == "midpoint":
                subdivided = subdivided.subdivide()
            else:
                raise ValueError(f"Unknown subdivision method: {method}")

            print(f"[RefineMesh] Subdivision iteration {i+1}/{iterations}: "
                  f"{len(subdivided.vertices)} vertices, {len(subdivided.faces)} faces")

        # Preserve metadata
        subdivided.metadata = trimesh.metadata.copy()
        subdivided.metadata['subdivision'] = {
            'method': method,
            'iterations': iterations,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'multiplier': len(subdivided.faces) / initial_faces if initial_faces > 0 else 0
        }

        info = f"""Refine Mesh Results (Subdivision):

Method: {method}
Iterations: {iterations}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(subdivided.vertices):,}
  Faces: {len(subdivided.faces):,}

Multiplier: {len(subdivided.faces) / initial_faces:.2f}x
"""
        return subdivided, info

    def _smooth(self, trimesh, iterations, lambda_factor):
        """Apply Laplacian smoothing."""
        smoothed = trimesh.copy()
        smoothed = trimesh_module.smoothing.filter_laplacian(
            smoothed,
            lamb=lambda_factor,
            iterations=iterations
        )

        # Preserve metadata
        smoothed.metadata = trimesh.metadata.copy()
        smoothed.metadata['smoothing'] = {
            'algorithm': 'laplacian',
            'iterations': iterations,
            'lambda': lambda_factor
        }

        info = f"""Refine Mesh Results (Laplacian Smoothing):

Iterations: {iterations}
Lambda Factor: {lambda_factor}

Vertices: {len(smoothed.vertices):,}
Faces: {len(smoothed.faces):,}

Smoothing reduces surface roughness and noise.
"""
        return smoothed, info


# Legacy node classes for backwards compatibility
class PyMeshLabRemeshNode:
    """
    PyMeshLab Isotropic Remeshing - Create uniform triangle meshes.

    Uses PyMeshLab's implementation of isotropic remeshing.
    This remeshing technique creates triangles with target edge length,
    resulting in more uniform mesh quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_edge_length": ("FLOAT", {
                    "default": 0.1,
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
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "remesh"
    CATEGORY = "geompack/pymeshlab"

    def remesh(self, trimesh, target_edge_length, iterations):
        """
        Apply PyMeshLab isotropic remeshing.

        Args:
            trimesh: Input trimesh.Trimesh object
            target_edge_length: Target edge length for remeshed triangles
            iterations: Number of remeshing iterations

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[PyMeshLabRemesh] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[PyMeshLabRemesh] Target edge length: {target_edge_length}, Iterations: {iterations}")

        remeshed_mesh, error = mesh_utils.pymeshlab_isotropic_remesh(
            trimesh,
            target_edge_length,
            iterations
        )

        if remeshed_mesh is None:
            raise ValueError(f"Remeshing failed: {error}")

        print(f"[PyMeshLabRemesh] Output: {len(remeshed_mesh.vertices)} vertices, {len(remeshed_mesh.faces)} faces")

        return (remeshed_mesh,)


class CGALIsotropicRemeshNode:
    """
    CGAL Isotropic Remeshing - Create uniform triangle meshes using CGAL.

    Uses CGAL's high-quality isotropic remeshing algorithm to create meshes
    with uniform triangle sizes and improved quality. Preserves volume and
    surface features while creating more regular triangulations.

    Requires the CGAL Python package: pip install cgal
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_edge_length": ("FLOAT", {
                    "default": 0.1,
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
                "protect_boundaries": (["true", "false"], {
                    "default": "true"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "remesh"
    CATEGORY = "geompack/cgal"

    def remesh(self, trimesh, target_edge_length, iterations, protect_boundaries="true"):
        """
        Apply CGAL isotropic remeshing.

        Args:
            trimesh: Input trimesh.Trimesh object
            target_edge_length: Target edge length for remeshed triangles
            iterations: Number of remeshing iterations (1-20)
            protect_boundaries: Whether to preserve boundary edges ("true" or "false")

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[CGALRemesh] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[CGALRemesh] Target edge length: {target_edge_length}, Iterations: {iterations}")
        print(f"[CGALRemesh] Protect boundaries: {protect_boundaries}")

        protect = (protect_boundaries == "true")

        remeshed_mesh, error = mesh_utils.cgal_isotropic_remesh(
            trimesh,
            target_edge_length,
            iterations,
            protect
        )

        if remeshed_mesh is None:
            raise ValueError(f"CGAL remeshing failed: {error}")

        print(f"[CGALRemesh] Output: {len(remeshed_mesh.vertices)} vertices, {len(remeshed_mesh.faces)} faces")

        return (remeshed_mesh,)


class BlenderVoxelRemeshNode:
    """
    Voxel-based remeshing using Blender.

    Creates a new mesh by voxelizing the input mesh and reconstructing
    the surface. Good for creating uniform, watertight meshes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "voxel_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "voxel_remesh"
    CATEGORY = "geompack/blender"

    def voxel_remesh(self, trimesh,voxel_size):
        """
        Apply voxel remeshing using Blender.

        Args:
            trimesh: Input trimesh.Trimesh object
            voxel_size: Voxel size for remeshing (smaller = higher resolution)

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[BlenderVoxelRemesh] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[BlenderVoxelRemesh] Voxel size: {voxel_size}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            trimesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for voxel remeshing
            script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Apply voxel remesh
obj.data.remesh_voxel_size = {voxel_size}
bpy.ops.object.voxel_remesh()

# Export remeshed object
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

            print(f"[BlenderVoxelRemesh] Running Blender in background mode...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the remeshed mesh
            print(f"[BlenderVoxelRemesh] Loading remeshed trimesh...")
            remeshed = trimesh_module.load(output_path, process=False)

            # If it's a scene, dump to single mesh
            if isinstance(remeshed, trimesh_module.Scene):
                remeshed = remeshed.dump(concatenate=True)

            # Preserve metadata
            remeshed.metadata = trimesh.metadata.copy()
            remeshed.metadata['remeshing'] = {
                'algorithm': 'blender_voxel',
                'voxel_size': voxel_size,
                'original_vertices': len(trimesh.vertices),
                'original_faces': len(trimesh.faces),
                'remeshed_vertices': len(remeshed.vertices),
                'remeshed_faces': len(remeshed.faces)
            }

            vertex_change = len(remeshed.vertices) - len(trimesh.vertices)
            face_change = len(remeshed.faces) - len(trimesh.faces)

            print(f"[BlenderVoxelRemesh] ✓ Complete:")
            print(f"[BlenderVoxelRemesh]   Vertices: {len(trimesh.vertices)} -> {len(remeshed.vertices)} ({vertex_change:+d})")
            print(f"[BlenderVoxelRemesh]   Faces:    {len(trimesh.faces)} -> {len(remeshed.faces)} ({face_change:+d})")

            return (remeshed,)

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class BlenderQuadriflowRemeshNode:
    """
    Quadriflow remeshing using Blender.

    Creates a quad-dominant mesh with good topology. Better for animation
    and subdivision surfaces than triangle-based remeshing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 100000,
                    "step": 100
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("remeshed_mesh",)
    FUNCTION = "quadriflow_remesh"
    CATEGORY = "geompack/blender"

    def quadriflow_remesh(self, trimesh,target_face_count):
        """
        Apply Quadriflow remeshing using Blender.

        Args:
            trimesh: Input trimesh.Trimesh object
            target_face_count: Target number of faces in output mesh

        Returns:
            tuple: (remeshed_trimesh.Trimesh,)
        """
        print(f"[BlenderQuadriflow] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[BlenderQuadriflow] Target face count: {target_face_count}")

        # Find Blender
        blender_path = _find_blender()

        # Create temp files
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_in:
            input_path = f_in.name
            trimesh.export(input_path)

        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
            output_path = f_out.name

        try:
            # Blender script for Quadriflow remeshing
            script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh
bpy.ops.wm.obj_import(filepath='{input_path}')

# Get imported object
obj = bpy.context.selected_objects[0]
bpy.context.view_layer.objects.active = obj

# Apply Quadriflow remesh
# Note: Different Blender versions have different parameters
# Using minimal set for maximum compatibility
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
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

            print(f"[BlenderQuadriflow] Running Blender in background mode...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                raise RuntimeError(f"Blender failed: {result.stderr}")

            # Load the remeshed mesh
            print(f"[BlenderQuadriflow] Loading remeshed trimesh...")
            remeshed = trimesh_module.load(output_path, process=False)

            # If it's a scene, dump to single mesh
            if isinstance(remeshed, trimesh_module.Scene):
                remeshed = remeshed.dump(concatenate=True)

            # Quadriflow produces quads, but trimesh will triangulate them
            # Preserve metadata
            remeshed.metadata = trimesh.metadata.copy()
            remeshed.metadata['remeshing'] = {
                'algorithm': 'blender_quadriflow',
                'target_face_count': target_face_count,
                'original_vertices': len(trimesh.vertices),
                'original_faces': len(trimesh.faces),
                'remeshed_vertices': len(remeshed.vertices),
                'remeshed_faces': len(remeshed.faces)
            }

            vertex_change = len(remeshed.vertices) - len(trimesh.vertices)
            face_change = len(remeshed.faces) - len(trimesh.faces)

            print(f"[BlenderQuadriflow] ✓ Complete:")
            print(f"[BlenderQuadriflow]   Vertices: {len(trimesh.vertices)} -> {len(remeshed.vertices)} ({vertex_change:+d})")
            print(f"[BlenderQuadriflow]   Faces:    {len(trimesh.faces)} -> {len(remeshed.faces)} ({face_change:+d})")

            return (remeshed,)

        finally:
            # Cleanup temp files
            if os.path.exists(input_path):
                os.unlink(input_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


class MeshDecimationNode:
    """
    Reduce mesh triangle count while preserving shape.

    Uses quadric error metrics to intelligently remove vertices and faces
    while minimizing geometric error. Essential for LOD generation, game
    assets, and performance optimization.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_face_count": ("INT", {
                    "default": 1000,
                    "min": 4,
                    "max": 10000000,
                    "step": 100
                }),
            },
            "optional": {
                "method": (["trimesh", "pymeshlab"], {
                    "default": "trimesh"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("decimated_mesh", "info")
    FUNCTION = "decimate"
    CATEGORY = "geompack/remeshing"

    def decimate(self, trimesh,target_face_count, method="trimesh"):
        """
        Decimate mesh to target face count.

        Args:
            trimesh: Input trimesh.Trimesh object
            target_face_count: Target number of faces
            method: Algorithm to use ("trimesh" or "pymeshlab")

        Returns:
            tuple: (decimated_mesh, info_string)
        """
        print(f"[MeshDecimation] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[MeshDecimation] Target: {target_face_count} faces using {method}")

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        if method == "trimesh":
            # Use trimesh's built-in decimation (wrapper around fast_simplification)
            decimated = trimesh.simplify_quadric_decimation(face_count=target_face_count)

        elif method == "pymeshlab":
            # Use PyMeshLab's decimation
            try:
                import pymeshlab
            except ImportError:
                raise ImportError("pymeshlab not installed. Install with: pip install pymeshlab")

            ms = pymeshlab.MeshSet()
            pml_mesh = pymeshlab.Mesh(
                vertex_matrix=trimesh.vertices,
                face_matrix=trimesh.faces
            )
            ms.add_mesh(pml_mesh)

            # Apply decimation
            ms.simplification_quadric_edge_collapse_decimation(
                targetfacenum=target_face_count,
                preserveboundary=True,
                preservenormal=True,
                preservetopology=False
            )

            # Convert back to trimesh
            decimated_pml = ms.current_mesh()
            decimated = trimesh_module.Trimesh(
                vertices=decimated_pml.vertex_matrix(),
                faces=decimated_pml.face_matrix()
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # Preserve metadata
        decimated.metadata = trimesh.metadata.copy()
        decimated.metadata['decimation'] = {
            'method': method,
            'target_face_count': target_face_count,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'decimated_vertices': len(decimated.vertices),
            'decimated_faces': len(decimated.faces),
            'reduction_ratio': len(decimated.faces) / initial_faces if initial_faces > 0 else 0
        }

        vertex_reduction = initial_vertices - len(decimated.vertices)
        face_reduction = initial_faces - len(decimated.faces)
        reduction_pct = 100.0 * face_reduction / initial_faces if initial_faces > 0 else 0

        info = f"""Mesh Decimation Results:

Method: {method}
Target Faces: {target_face_count:,}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,} (-{vertex_reduction:,}, -{100.0*vertex_reduction/initial_vertices:.1f}%)
  Faces: {len(decimated.faces):,} (-{face_reduction:,}, -{reduction_pct:.1f}%)

Reduction Ratio: {len(decimated.faces) / initial_faces:.2%}
"""

        print(f"[MeshDecimation] ✓ Complete: {initial_faces} -> {len(decimated.faces)} faces ({reduction_pct:.1f}% reduction)")

        return (decimated, info)


class MeshSubdivisionNode:
    """
    Increase mesh resolution through subdivision.

    Subdivides each triangle into smaller triangles, creating a smoother,
    higher-resolution trimesh. Uses Loop subdivision for smooth surfaces.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
            },
            "optional": {
                "method": (["loop", "midpoint"], {
                    "default": "loop"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("subdivided_mesh", "info")
    FUNCTION = "subdivide"
    CATEGORY = "geompack/remeshing"

    def subdivide(self, trimesh,iterations, method="loop"):
        """
        Subdivide mesh to increase resolution.

        Args:
            trimesh: Input trimesh.Trimesh object
            iterations: Number of subdivision iterations
            method: Subdivision method ("loop" or "midpoint")

        Returns:
            tuple: (subdivided_mesh, info_string)
        """
        print(f"[MeshSubdivision] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[MeshSubdivision] Method: {method}, Iterations: {iterations}")

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        # Create copy
        subdivided = trimesh.copy()

        for i in range(iterations):
            if method == "loop":
                # Loop subdivision - creates smooth surfaces
                subdivided = subdivided.subdivide_loop(iterations=1)
            elif method == "midpoint":
                # Simple midpoint subdivision - less smooth
                subdivided = subdivided.subdivide()
            else:
                raise ValueError(f"Unknown method: {method}")

            print(f"[MeshSubdivision] Iteration {i+1}/{iterations}: {len(subdivided.vertices)} vertices, {len(subdivided.faces)} faces")

        # Preserve metadata
        subdivided.metadata = trimesh.metadata.copy()
        subdivided.metadata['subdivision'] = {
            'method': method,
            'iterations': iterations,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'subdivided_vertices': len(subdivided.vertices),
            'subdivided_faces': len(subdivided.faces),
            'multiplier': len(subdivided.faces) / initial_faces if initial_faces > 0 else 0
        }

        vertex_increase = len(subdivided.vertices) - initial_vertices
        face_increase = len(subdivided.faces) - initial_faces

        info = f"""Mesh Subdivision Results:

Method: {method}
Iterations: {iterations}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(subdivided.vertices):,} (+{vertex_increase:,}, {len(subdivided.vertices)/initial_vertices:.2f}x)
  Faces: {len(subdivided.faces):,} (+{face_increase:,}, {len(subdivided.faces)/initial_faces:.2f}x)

Each iteration approximately 4x the face count for Loop subdivision.
"""

        print(f"[MeshSubdivision] ✓ Complete: {initial_faces} -> {len(subdivided.faces)} faces ({len(subdivided.faces)/initial_faces:.2f}x)")

        return (subdivided, info)


class InstantMeshesRemeshNode:
    """
    Instant Meshes - Field-aligned quad remeshing.

    Uses Instant Meshes algorithm to create high-quality quad-dominant meshes
    with flow-aligned edges. Excellent for animation, simulation, and
    subdivision surfaces. Better topology than simple remeshing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_vertex_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100
                }),
            },
            "optional": {
                "deterministic": (["true", "false"], {
                    "default": "true"
                }),
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

    def remesh(self, trimesh,target_vertex_count, deterministic="true", crease_angle=0.0):
        """
        Apply Instant Meshes remeshing.

        Args:
            trimesh: Input trimesh.Trimesh object
            target_vertex_count: Target number of vertices
            deterministic: Use deterministic mode for reproducible results
            crease_angle: Angle threshold for feature detection (degrees)

        Returns:
            tuple: (remeshed_mesh, info_string)
        """
        try:
            import PyNanoInstantMeshes as pynano
        except ImportError:
            raise ImportError(
                "PyNanoInstantMeshes not installed. Install with: pip install PyNanoInstantMeshes"
            )

        print(f"[InstantMeshes] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[InstantMeshes] Target vertex count: {target_vertex_count}")

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        # Instant Meshes expects vertices and faces as numpy arrays
        V = trimesh.vertices.astype(np.float64)
        F = trimesh.faces.astype(np.int32)

        # Run Instant Meshes
        V_out, F_out = pynano.instant_meshes(
            V, F,
            target_vertex_count=target_vertex_count,
            deterministic=(deterministic == "true"),
            crease_angle=crease_angle
        )

        # Create remeshed mesh (Instant Meshes outputs quads, trimesh will triangulate)
        remeshed = trimesh_module.Trimesh(
            vertices=V_out,
            faces=F_out,
            process=False
        )

        # Preserve metadata
        remeshed.metadata = trimesh.metadata.copy()
        remeshed.metadata['remeshing'] = {
            'algorithm': 'instant_meshes',
            'target_vertex_count': target_vertex_count,
            'deterministic': deterministic == "true",
            'crease_angle': crease_angle,
            'original_vertices': initial_vertices,
            'original_faces': initial_faces,
            'remeshed_vertices': len(remeshed.vertices),
            'remeshed_faces': len(remeshed.faces),
            'field_aligned': True
        }

        vertex_change = len(remeshed.vertices) - initial_vertices
        face_change = len(remeshed.faces) - initial_faces

        info = f"""Instant Meshes Remeshing Results:

Algorithm: Field-aligned quad remeshing
Target Vertices: {target_vertex_count:,}
Deterministic: {deterministic}
Crease Angle: {crease_angle}°

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(remeshed.vertices):,} ({vertex_change:+d})
  Faces: {len(remeshed.faces):,} ({face_change:+d})

Instant Meshes creates flow-aligned quad meshes with
better topology for animation and subdivision.
"""

        print(f"[InstantMeshes] ✓ Complete: {initial_vertices} -> {len(remeshed.vertices)} vertices")

        return (remeshed, info)


class LaplacianSmoothingNode:
    """
    Smooth mesh using Laplacian smoothing.

    Applies iterative Laplacian smoothing to reduce surface roughness and
    noise. Moves each vertex toward the average of its neighbors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "iterations": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "lambda_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("smoothed_mesh",)
    FUNCTION = "smooth"
    CATEGORY = "geompack/remeshing"

    def smooth(self, trimesh,iterations, lambda_factor):
        """
        Apply Laplacian smoothing to trimesh.

        Args:
            trimesh: Input trimesh.Trimesh object
            iterations: Number of smoothing iterations
            lambda_factor: Smoothing strength (0-1)

        Returns:
            tuple: (smoothed_mesh,)
        """
        print(f"[LaplacianSmoothing] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[LaplacianSmoothing] Iterations: {iterations}, Lambda: {lambda_factor}")

        # Create copy
        smoothed = trimesh.copy()

        # Apply Laplacian smoothing
        smoothed = trimesh_module.smoothing.filter_laplacian(
            smoothed,
            lamb=lambda_factor,
            iterations=iterations
        )

        # Preserve metadata
        smoothed.metadata = trimesh.metadata.copy()
        smoothed.metadata['smoothing'] = {
            'algorithm': 'laplacian',
            'iterations': iterations,
            'lambda': lambda_factor
        }

        print(f"[LaplacianSmoothing] ✓ Complete")

        return (smoothed,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh": RemeshNode,
    "GeomPackRefineMesh": RefineMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh": "Remesh",
    "GeomPackRefineMesh": "Refine Mesh",
}
