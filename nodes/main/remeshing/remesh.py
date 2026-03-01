# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Remesh Node - Main backends (pymeshlab, instant_meshes, quadriflow, mmg,
geogram, pmp, quadwild)
"""

import logging
import numpy as np
import trimesh as trimesh_module
from typing import Tuple, Optional
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _pymeshlab_isotropic_remesh(
    mesh: trimesh_module.Trimesh,
    target_edge_length: float,
    iterations: int = 3,
    adaptive: bool = False,
    feature_angle: float = 30.0
) -> Tuple[Optional[trimesh_module.Trimesh], str]:
    """Apply isotropic remeshing using PyMeshLab."""
    log.info("Starting Isotropic Remeshing")
    log.info("Input mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
    log.info("Parameters: target_edge_length=%s, iterations=%s, adaptive=%s, feature_angle=%s",
             target_edge_length, iterations, adaptive, feature_angle)

    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    if not isinstance(mesh, trimesh_module.Trimesh):
        return None, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, "Mesh is empty"

    if target_edge_length <= 0:
        return None, f"Target edge length must be positive, got {target_edge_length}"

    if iterations < 1:
        return None, f"Iterations must be at least 1, got {iterations}"

    try:
        log.info("Converting to PyMeshLab format...")
        ms = pymeshlab.MeshSet()

        pml_mesh = pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces
        )
        ms.add_mesh(pml_mesh)

        log.info("Applying isotropic remeshing...")
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        target_pct = (target_edge_length / bbox_diag) * 100.0

        try:
            ms.meshing_isotropic_explicit_remeshing(
                targetlen=pymeshlab.PercentageValue(target_pct),
                iterations=iterations,
                adaptive=adaptive,
                featuredeg=feature_angle
            )
        except AttributeError:
            try:
                ms.remeshing_isotropic_explicit_remeshing(
                    targetlen=pymeshlab.PercentageValue(target_pct),
                    iterations=iterations,
                    adaptive=adaptive,
                    featuredeg=feature_angle
                )
            except AttributeError:
                return None, (
                    "PyMeshLab meshing filter not available. "
                    "This usually means the libfilter_meshing.so plugin failed to load. "
                    "On Linux, install OpenGL libraries: sudo apt-get install libgl1-mesa-glx libglu1-mesa"
                )

        log.info("Converting back to trimesh...")
        remeshed_pml = ms.current_mesh()
        remeshed_mesh = trimesh_module.Trimesh(
            vertices=remeshed_pml.vertex_matrix(),
            faces=remeshed_pml.face_matrix()
        )

        remeshed_mesh.metadata = mesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pymeshlab_isotropic',
            'target_edge_length': target_edge_length,
            'target_percentage': target_pct,
            'iterations': iterations,
            'adaptive': adaptive,
            'feature_angle': feature_angle,
            'original_vertices': len(mesh.vertices),
            'original_faces': len(mesh.faces),
            'remeshed_vertices': len(remeshed_mesh.vertices),
            'remeshed_faces': len(remeshed_mesh.faces)
        }

        vertex_change = len(remeshed_mesh.vertices) - len(mesh.vertices)
        face_change = len(remeshed_mesh.faces) - len(mesh.faces)
        vertex_pct = (vertex_change / len(mesh.vertices)) * 100 if len(mesh.vertices) > 0 else 0
        face_pct = (face_change / len(mesh.faces)) * 100 if len(mesh.faces) > 0 else 0

        log.info("Remeshing Complete")
        log.info("Vertices: %d -> %d (%+d, %+.1f%%)", len(mesh.vertices), len(remeshed_mesh.vertices), vertex_change, vertex_pct)
        log.info("Faces: %d -> %d (%+d, %+.1f%%)", len(mesh.faces), len(remeshed_mesh.faces), face_change, face_pct)

        return remeshed_mesh, ""

    except Exception as e:
        log.error("Error during remeshing", exc_info=True)
        return None, f"Error during remeshing: {str(e)}"


def _mmg_adaptive_remesh(
    mesh: trimesh_module.Trimesh,
    hausd: float = 0.01,
    hmin: float = 0.0,
    hmax: float = 0.0,
    hgrad: float = 1.3,
) -> Tuple[Optional[trimesh_module.Trimesh], str]:
    """Apply adaptive surface remeshing using mmgpy (MMG mmgs)."""
    log.info("Starting MMG Adaptive Surface Remeshing")
    log.info("Input mesh: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces))
    log.info("Parameters: hausd=%s, hmin=%s, hmax=%s, hgrad=%s",
             hausd, hmin, hmax, hgrad)

    try:
        import mmgpy
    except (ImportError, OSError):
        return None, "mmgpy is not installed. Install with: pip install mmgpy"

    if not isinstance(mesh, trimesh_module.Trimesh):
        return None, "Input must be a trimesh.Trimesh object"

    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, "Mesh is empty"

    if hausd <= 0:
        return None, f"Hausdorff distance must be positive, got {hausd}"

    try:
        log.info("Converting to mmgpy format...")
        vertices = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        mmg_mesh = mmgpy.Mesh(vertices, faces)

        opts_kwargs = {
            "hausd": hausd,
            "hgrad": hgrad,
            "verbose": -1,
        }
        if hmin > 0:
            opts_kwargs["hmin"] = hmin
        if hmax > 0:
            opts_kwargs["hmax"] = hmax

        opts = mmgpy.MmgSOptions(**opts_kwargs)

        log.info("Running mmgs surface remeshing...")
        result = mmg_mesh.remesh(opts)

        if not result.success:
            return None, f"MMG remeshing failed (return code {result.return_code})"

        log.info("Converting back to trimesh...")
        out_vertices = mmg_mesh.get_vertices()
        out_faces = mmg_mesh.get_triangles()

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=out_vertices,
            faces=out_faces,
            process=False
        )

        remeshed_mesh.metadata = mesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'mmg_adaptive',
            'hausd': hausd,
            'hmin': hmin,
            'hmax': hmax,
            'hgrad': hgrad,
            'quality_mean_before': result.quality_mean_before,
            'quality_mean_after': result.quality_mean_after,
            'original_vertices': len(mesh.vertices),
            'original_faces': len(mesh.faces),
            'remeshed_vertices': len(remeshed_mesh.vertices),
            'remeshed_faces': len(remeshed_mesh.faces)
        }

        vertex_change = len(remeshed_mesh.vertices) - len(mesh.vertices)
        face_change = len(remeshed_mesh.faces) - len(mesh.faces)
        vertex_pct = (vertex_change / len(mesh.vertices)) * 100 if len(mesh.vertices) > 0 else 0
        face_pct = (face_change / len(mesh.faces)) * 100 if len(mesh.faces) > 0 else 0

        log.info("Remeshing Complete")
        log.info("Vertices: %d -> %d (%+d, %+.1f%%)", len(mesh.vertices), len(remeshed_mesh.vertices), vertex_change, vertex_pct)
        log.info("Faces: %d -> %d (%+d, %+.1f%%)", len(mesh.faces), len(remeshed_mesh.faces), face_change, face_pct)
        log.info("Quality: %.3f -> %.3f", result.quality_mean_before, result.quality_mean_after)

        return remeshed_mesh, ""

    except Exception as e:
        log.error("Error during mmg remeshing", exc_info=True)
        return None, f"Error during mmg remeshing: {str(e)}"


class RemeshNode(io.ComfyNode):
    """
    Remesh - Topology-changing remeshing operations (main backends).

    Available backends:
    - pymeshlab_isotropic: PyMeshLab isotropic remeshing (fast)
    - instant_meshes: Field-aligned quad remeshing
    - quadriflow: QuadriFlow quad remeshing (good topology)
    - mmg_adaptive: MMG curvature-adaptive surface remeshing
    - geogram_smooth: Geogram CVT isotropic remeshing (high quality)
    - geogram_anisotropic: Geogram curvature-adapted CVT remeshing
    - pmp_uniform: PMP uniform isotropic remeshing (edge split/collapse/flip)
    - pmp_adaptive: PMP curvature-driven adaptive remeshing
    - quadwild: QuadWild BiMDF tri-to-quad remeshing

    For CGAL isotropic remeshing, use "Remesh CGAL" node.
    For Blender voxel/modifier remeshing, use "Remesh Blender" node.
    For GPU-accelerated remeshing, use "Remesh GPU" node.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackRemesh",
            display_name="Remesh",
            category="geompack/remeshing",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip="Remeshing algorithm. pymeshlab=fast isotropic, instant_meshes=field-aligned quads, quadriflow=quad remesh with good topology, mmg_adaptive=curvature-adaptive surface remeshing", options=[
                    io.DynamicCombo.Option("pymeshlab_isotropic", [
                        io.Float.Input("target_edge_length", default=1.00, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Target edge length for output triangles. Value is relative to mesh scale."),
                        io.Int.Input("iterations", default=3, min=1, max=20, step=1, tooltip="Number of remeshing passes. More iterations = smoother result, slower processing."),
                        io.Float.Input("feature_angle", default=30.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold (degrees) for feature edge detection. Edges with dihedral angle greater than this are preserved as sharp creases."),
                        io.Combo.Input("adaptive", options=["true", "false"], default="false", tooltip="Use curvature-adaptive edge lengths. Creates smaller triangles in high-curvature areas, larger triangles in flat areas."),
                    ]),
                    io.DynamicCombo.Option("instant_meshes", [
                        io.Int.Input("target_vertex_count", default=5000, min=100, max=1000000, step=100, tooltip="Target vertex count for Instant Meshes output. Creates field-aligned quad-dominant mesh."),
                        io.Combo.Input("deterministic", options=["true", "false"], default="true", tooltip="Use deterministic algorithm for reproducible results. Disable for potentially better quality but non-reproducible output."),
                        io.Float.Input("crease_angle", default=0.0, min=0.0, max=180.0, step=1.0, tooltip="Angle threshold (degrees) for preserving sharp/crease edges in Instant Meshes. 0 = no crease preservation."),
                    ]),
                    io.DynamicCombo.Option("quadriflow", [
                        io.Int.Input("target_face_count", default=5000, min=100, max=5000000, step=100, tooltip="Target number of output faces for QuadriFlow. Creates quad-dominant mesh with good topology."),
                        io.Combo.Input("preserve_sharp", options=["true", "false"], default="false", tooltip="Preserve sharp edges during QuadriFlow remeshing."),
                        io.Combo.Input("preserve_boundary", options=["true", "false"], default="true", tooltip="Preserve mesh boundary edges during QuadriFlow remeshing."),
                    ]),
                    io.DynamicCombo.Option("mmg_adaptive", [
                        io.Float.Input("hausd", default=0.01, min=0.0001, max=10.0, step=0.001, display_mode="number", tooltip="Hausdorff distance: maximum geometric deviation from original surface. Smaller values preserve detail better but produce more triangles."),
                        io.Float.Input("hmin", default=0.0, min=0.0, max=10.0, step=0.001, display_mode="number", tooltip="Minimum edge length. 0 = auto (MMG computes from mesh geometry). Setting this prevents overly small triangles."),
                        io.Float.Input("hmax", default=0.0, min=0.0, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length. 0 = auto (MMG computes from mesh geometry). Setting this prevents overly large triangles in flat areas."),
                        io.Float.Input("hgrad", default=1.3, min=1.0, max=5.0, step=0.1, display_mode="number", tooltip="Gradation: controls how fast element sizes change across the mesh. 1.3 = smooth transitions (default). Lower = more uniform, higher = faster size changes."),
                    ]),
                    io.DynamicCombo.Option("geogram_smooth", [
                        io.Int.Input("nb_points", default=5000, min=0, max=1000000, step=100, tooltip="Target number of output vertices. 0 = same count as input."),
                        io.Int.Input("nb_lloyd_iter", default=5, min=1, max=50, step=1, tooltip="Lloyd relaxation iterations (initial uniform distribution)."),
                        io.Int.Input("nb_newton_iter", default=30, min=1, max=100, step=1, tooltip="Newton optimization iterations (refines point placement)."),
                        io.Int.Input("newton_m", default=7, min=1, max=20, step=1, tooltip="Number of L-BFGS Hessian evaluations."),
                    ]),
                    io.DynamicCombo.Option("geogram_anisotropic", [
                        io.Int.Input("nb_points_aniso", default=5000, min=0, max=1000000, step=100, tooltip="Target number of output vertices."),
                        io.Float.Input("anisotropy", default=0.04, min=0.005, max=0.5, step=0.005, display_mode="number", tooltip="Anisotropy factor. Controls how much triangle shape adapts to curvature. Lower = more anisotropic. Typical range: 0.02-0.1."),
                    ]),
                    io.DynamicCombo.Option("pmp_uniform", [
                        io.Float.Input("pmp_edge_length", default=1.0, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Target edge length for uniform remeshing."),
                        io.Int.Input("pmp_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations. More iterations produce better quality."),
                        io.Combo.Input("pmp_use_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto the input surface after each iteration."),
                    ]),
                    io.DynamicCombo.Option("pmp_adaptive", [
                        io.Float.Input("pmp_min_edge", default=0.1, min=0.001, max=100.0, step=0.01, display_mode="number", tooltip="Minimum edge length (used in high-curvature areas)."),
                        io.Float.Input("pmp_max_edge", default=2.0, min=0.01, max=100.0, step=0.01, display_mode="number", tooltip="Maximum edge length (used in flat areas)."),
                        io.Float.Input("pmp_approx_error", default=0.1, min=0.001, max=10.0, step=0.01, display_mode="number", tooltip="Maximum geometric approximation error."),
                        io.Int.Input("pmp_adaptive_iterations", default=10, min=1, max=100, step=1, tooltip="Number of remeshing iterations."),
                        io.Combo.Input("pmp_adaptive_projection", options=["true", "false"], default="true", tooltip="Project vertices back onto the input surface after each iteration."),
                    ]),
                    io.DynamicCombo.Option("quadwild", [
                        io.Float.Input("qw_sharp_angle", default=35.0, min=0.0, max=180.0, step=1.0, tooltip="Dihedral angle threshold (degrees) for sharp feature detection. Edges with angle above this are preserved as creases."),
                        io.Float.Input("qw_alpha", default=0.02, min=0.005, max=0.1, step=0.005, display_mode="number", tooltip="Balance between regularity and isometry of the quad layout. Lower = more regular quads. Typical range: 0.005-0.1."),
                        io.Float.Input("qw_scale_factor", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Quad size multiplier. Larger values produce bigger quads (fewer output faces)."),
                        io.Combo.Input("qw_remesh", options=["true", "false"], default="true", tooltip="Pre-remesh input for better triangle quality before quad conversion."),
                        io.Combo.Input("qw_smooth", options=["true", "false"], default="true", tooltip="Smooth output mesh topology after quadrangulation."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="remeshed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        """Apply remeshing based on selected backend."""
        selected = backend["backend"]

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        log.info("Backend: %s", selected)
        log.info("Input: %s vertices, %s faces", f"{initial_vertices:,}", f"{initial_faces:,}")

        if selected == "pymeshlab_isotropic":
            target_edge_length = backend.get("target_edge_length", 1.0)
            iterations = backend.get("iterations", 3)
            feature_angle = backend.get("feature_angle", 30.0)
            adaptive = backend.get("adaptive", "false")
            log.info("Parameters: target_edge_length=%s, iterations=%s, feature_angle=%s, adaptive=%s",
                     target_edge_length, iterations, feature_angle, adaptive)
            remeshed_mesh, info = cls._pymeshlab_isotropic(
                trimesh, target_edge_length, iterations, feature_angle, adaptive
            )
        elif selected == "instant_meshes":
            target_vertex_count = backend.get("target_vertex_count", 5000)
            deterministic = backend.get("deterministic", "true")
            crease_angle = backend.get("crease_angle", 0.0)
            log.info("Parameters: target_vertex_count=%s, deterministic=%s, crease_angle=%s",
                     f"{target_vertex_count:,}", deterministic, crease_angle)
            remeshed_mesh, info = cls._instant_meshes(
                trimesh, target_vertex_count, deterministic, crease_angle
            )
        elif selected == "quadriflow":
            target_face_count = backend.get("target_face_count", 5000)
            preserve_sharp = backend.get("preserve_sharp", "false")
            preserve_boundary = backend.get("preserve_boundary", "true")
            log.info("Parameters: target_face_count=%s, preserve_sharp=%s, preserve_boundary=%s",
                     f"{target_face_count:,}", preserve_sharp, preserve_boundary)
            remeshed_mesh, info = cls._quadriflow(
                trimesh, target_face_count, preserve_sharp, preserve_boundary
            )
        elif selected == "mmg_adaptive":
            hausd = backend.get("hausd", 0.01)
            hmin = backend.get("hmin", 0.0)
            hmax = backend.get("hmax", 0.0)
            hgrad = backend.get("hgrad", 1.3)
            log.info("Parameters: hausd=%s, hmin=%s, hmax=%s, hgrad=%s",
                     hausd, hmin, hmax, hgrad)
            remeshed_mesh, info = cls._mmg_adaptive(
                trimesh, hausd, hmin, hmax, hgrad
            )
        elif selected == "geogram_smooth":
            nb_points = backend.get("nb_points", 5000)
            nb_lloyd_iter = backend.get("nb_lloyd_iter", 5)
            nb_newton_iter = backend.get("nb_newton_iter", 30)
            newton_m = backend.get("newton_m", 7)
            log.info("Parameters: nb_points=%s, nb_lloyd_iter=%s, nb_newton_iter=%s, newton_m=%s",
                     f"{nb_points:,}", nb_lloyd_iter, nb_newton_iter, newton_m)
            remeshed_mesh, info = cls._geogram_smooth(
                trimesh, nb_points, nb_lloyd_iter, nb_newton_iter, newton_m
            )
        elif selected == "geogram_anisotropic":
            nb_points = backend.get("nb_points_aniso", 5000)
            anisotropy = backend.get("anisotropy", 0.04)
            log.info("Parameters: nb_points=%s, anisotropy=%s",
                     f"{nb_points:,}", anisotropy)
            remeshed_mesh, info = cls._geogram_anisotropic(
                trimesh, nb_points, anisotropy
            )
        elif selected == "pmp_uniform":
            edge_length = backend.get("pmp_edge_length", 1.0)
            iterations = backend.get("pmp_iterations", 10)
            use_projection = backend.get("pmp_use_projection", "true")
            log.info("Parameters: edge_length=%s, iterations=%s, use_projection=%s",
                     edge_length, iterations, use_projection)
            remeshed_mesh, info = cls._pmp_uniform(
                trimesh, edge_length, iterations, use_projection
            )
        elif selected == "pmp_adaptive":
            min_edge = backend.get("pmp_min_edge", 0.1)
            max_edge = backend.get("pmp_max_edge", 2.0)
            approx_error = backend.get("pmp_approx_error", 0.1)
            iterations = backend.get("pmp_adaptive_iterations", 10)
            use_projection = backend.get("pmp_adaptive_projection", "true")
            log.info("Parameters: min_edge=%s, max_edge=%s, approx_error=%s, iterations=%s, use_projection=%s",
                     min_edge, max_edge, approx_error, iterations, use_projection)
            remeshed_mesh, info = cls._pmp_adaptive(
                trimesh, min_edge, max_edge, approx_error, iterations, use_projection
            )
        elif selected == "quadwild":
            sharp_angle = backend.get("qw_sharp_angle", 35.0)
            alpha = backend.get("qw_alpha", 0.02)
            scale_factor = backend.get("qw_scale_factor", 1.0)
            remesh_input = backend.get("qw_remesh", "true")
            smooth = backend.get("qw_smooth", "true")
            log.info("Parameters: sharp_angle=%s, alpha=%s, scale_factor=%s, remesh=%s, smooth=%s",
                     sharp_angle, alpha, scale_factor, remesh_input, smooth)
            remeshed_mesh, info = cls._quadwild(
                trimesh, sharp_angle, alpha, scale_factor, remesh_input, smooth
            )
        else:
            raise ValueError(f"Unknown backend: {selected}")

        vertex_change = len(remeshed_mesh.vertices) - initial_vertices
        face_change = len(remeshed_mesh.faces) - initial_faces

        log.info("Output: %d vertices (%+d), %d faces (%+d)",
                 len(remeshed_mesh.vertices), vertex_change, len(remeshed_mesh.faces), face_change)

        return io.NodeOutput(remeshed_mesh, info, ui={"text": [info]})

    @staticmethod
    def _pymeshlab_isotropic(trimesh, target_edge_length, iterations, feature_angle, adaptive):
        """PyMeshLab isotropic remeshing."""
        adaptive_bool = (adaptive == "true")
        remeshed_mesh, error = _pymeshlab_isotropic_remesh(
            trimesh, target_edge_length, iterations,
            adaptive=adaptive_bool, feature_angle=feature_angle
        )
        if remeshed_mesh is None:
            raise ValueError(f"PyMeshLab remeshing failed: {error}")

        info = f"""Remesh Results (PyMeshLab Isotropic):

Target Edge Length: {target_edge_length}
Iterations: {iterations}
Feature Angle: {feature_angle}\u00b0
Adaptive: {adaptive}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}
"""
        return remeshed_mesh, info

    @staticmethod
    def _instant_meshes(trimesh, target_vertex_count, deterministic, crease_angle):
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

    @staticmethod
    def _quadriflow(trimesh, target_face_count, preserve_sharp, preserve_boundary):
        """QuadriFlow quad remeshing using pyQuadriFlow."""
        try:
            from pyQuadriFlow.pyQuadriFlow import pyquadriflow
        except ImportError:
            raise ImportError(
                "pyQuadriFlow not installed. Install with: pip install pyQuadriFlow"
            )

        V = np.asarray(trimesh.vertices, dtype=np.float64).tolist()
        F = np.asarray(trimesh.faces, dtype=np.int32).tolist()

        log.info("Running QuadriFlow (target_faces=%d)...", target_face_count)
        result = pyquadriflow(
            target_face_count,
            0,  # seed
            V,
            F,
            preserve_sharp == "true",
            preserve_boundary == "true",
            False,  # adaptive_scale
            False,  # aggressive_sat
            False,  # minimum_cost_flow
        )
        V_out = result['vertices']
        F_out = result['faces']

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=np.array(V_out, dtype=np.float32),
            faces=np.array(F_out, dtype=np.int32),
            process=False
        )

        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'quadriflow',
            'target_face_count': target_face_count,
            'preserve_sharp': preserve_sharp == "true",
            'preserve_boundary': preserve_boundary == "true",
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces)
        }

        info = f"""Remesh Results (QuadriFlow):

Target Face Count: {target_face_count:,}
Preserve Sharp: {preserve_sharp}
Preserve Boundary: {preserve_boundary}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

QuadriFlow creates quad-dominant meshes with good topology.
"""
        return remeshed_mesh, info

    @staticmethod
    def _mmg_adaptive(trimesh, hausd, hmin, hmax, hgrad):
        """MMG adaptive surface remeshing."""
        remeshed_mesh, error = _mmg_adaptive_remesh(
            trimesh, hausd=hausd, hmin=hmin, hmax=hmax, hgrad=hgrad
        )
        if remeshed_mesh is None:
            raise ValueError(f"MMG remeshing failed: {error}")

        hmin_str = f"{hmin}" if hmin > 0 else "auto"
        hmax_str = f"{hmax}" if hmax > 0 else "auto"

        info = f"""Remesh Results (MMG Adaptive):

Hausdorff Distance: {hausd}
Min Edge Length: {hmin_str}
Max Edge Length: {hmax_str}
Gradation: {hgrad}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

MMG creates curvature-adaptive surface meshes.
"""
        return remeshed_mesh, info

    @staticmethod
    def _geogram_smooth(trimesh, nb_points, nb_lloyd_iter, nb_newton_iter, newton_m):
        """Geogram CVT isotropic remeshing."""
        try:
            import pygeogram
        except ImportError:
            raise ImportError("pygeogram-ap not installed. Install with: pip install pygeogram-ap")

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        log.info("Running Geogram CVT smooth remeshing (nb_points=%d)...", nb_points)
        V_out, F_out = pygeogram.remesh_smooth(
            V, F, nb_points,
            nb_lloyd_iter=nb_lloyd_iter,
            nb_newton_iter=nb_newton_iter,
            newton_m=newton_m,
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out, faces=F_out, process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'geogram_smooth',
            'nb_points': nb_points,
            'nb_lloyd_iter': nb_lloyd_iter,
            'nb_newton_iter': nb_newton_iter,
            'newton_m': newton_m,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces),
        }

        info = f"""Remesh Results (Geogram CVT Smooth):

Target Points: {nb_points:,}
Lloyd Iterations: {nb_lloyd_iter}
Newton Iterations: {nb_newton_iter}
Newton M: {newton_m}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Geogram CVT produces high-quality isotropic triangle meshes.
"""
        return remeshed_mesh, info

    @staticmethod
    def _geogram_anisotropic(trimesh, nb_points, anisotropy):
        """Geogram curvature-adapted CVT remeshing."""
        try:
            import pygeogram
        except ImportError:
            raise ImportError("pygeogram-ap not installed. Install with: pip install pygeogram-ap")

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        log.info("Running Geogram CVT anisotropic remeshing (nb_points=%d, anisotropy=%s)...",
                 nb_points, anisotropy)
        V_out, F_out = pygeogram.remesh_anisotropic(
            V, F, nb_points, anisotropy=anisotropy,
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out, faces=F_out, process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'geogram_anisotropic',
            'nb_points': nb_points,
            'anisotropy': anisotropy,
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces),
        }

        info = f"""Remesh Results (Geogram CVT Anisotropic):

Target Points: {nb_points:,}
Anisotropy: {anisotropy}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

Geogram anisotropic CVT adapts triangle shape to surface curvature.
"""
        return remeshed_mesh, info

    @staticmethod
    def _pmp_uniform(trimesh, edge_length, iterations, use_projection):
        """PMP uniform isotropic remeshing."""
        try:
            import pypmp
        except ImportError:
            raise ImportError("pypmp not installed. Install with: pip install pypmp")

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        log.info("Running PMP uniform remeshing (edge_length=%s, iterations=%d)...",
                 edge_length, iterations)
        V_out, F_out = pypmp.remesh_uniform(
            V, F, edge_length,
            iterations=iterations,
            use_projection=(use_projection == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out, faces=F_out, process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pmp_uniform',
            'edge_length': edge_length,
            'iterations': iterations,
            'use_projection': use_projection == "true",
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces),
        }

        info = f"""Remesh Results (PMP Uniform):

Target Edge Length: {edge_length}
Iterations: {iterations}
Use Projection: {use_projection}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

PMP uniform remeshing uses edge split/collapse/flip with tangential relaxation.
"""
        return remeshed_mesh, info

    @staticmethod
    def _pmp_adaptive(trimesh, min_edge, max_edge, approx_error, iterations, use_projection):
        """PMP curvature-driven adaptive remeshing."""
        try:
            import pypmp
        except ImportError:
            raise ImportError("pypmp not installed. Install with: pip install pypmp")

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        log.info("Running PMP adaptive remeshing (min=%s, max=%s, error=%s, iterations=%d)...",
                 min_edge, max_edge, approx_error, iterations)
        V_out, F_out = pypmp.remesh_adaptive(
            V, F, min_edge, max_edge, approx_error,
            iterations=iterations,
            use_projection=(use_projection == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out, faces=F_out, process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'pmp_adaptive',
            'min_edge_length': min_edge,
            'max_edge_length': max_edge,
            'approx_error': approx_error,
            'iterations': iterations,
            'use_projection': use_projection == "true",
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces),
        }

        info = f"""Remesh Results (PMP Adaptive):

Min Edge Length: {min_edge}
Max Edge Length: {max_edge}
Approx Error: {approx_error}
Iterations: {iterations}
Use Projection: {use_projection}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces: {len(remeshed_mesh.faces):,}

PMP adaptive remeshing uses smaller triangles in high-curvature areas.
"""
        return remeshed_mesh, info

    @staticmethod
    def _quadwild(trimesh, sharp_angle, alpha, scale_factor, remesh_input, smooth):
        """QuadWild BiMDF tri-to-quad remeshing."""
        try:
            import pyquadwild
        except ImportError:
            raise ImportError("pyquadwild not installed. Install with: pip install pyquadwild")

        V = np.ascontiguousarray(trimesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(trimesh.faces, dtype=np.int32)

        log.info("Running QuadWild (sharp_angle=%s, alpha=%s, scale=%s)...",
                 sharp_angle, alpha, scale_factor)
        V_out, F_out = pyquadwild.quadwild_remesh(
            V, F,
            remesh=(remesh_input == "true"),
            sharp_angle=sharp_angle,
            alpha=alpha,
            scale_factor=scale_factor,
            smooth=(smooth == "true"),
        )

        remeshed_mesh = trimesh_module.Trimesh(
            vertices=V_out, faces=F_out, process=False
        )
        remeshed_mesh.metadata = trimesh.metadata.copy()
        remeshed_mesh.metadata['remeshing'] = {
            'algorithm': 'quadwild',
            'sharp_angle': sharp_angle,
            'alpha': alpha,
            'scale_factor': scale_factor,
            'remesh': remesh_input == "true",
            'smooth': smooth == "true",
            'original_vertices': len(trimesh.vertices),
            'original_faces': len(trimesh.faces),
        }

        info = f"""Remesh Results (QuadWild BiMDF):

Sharp Angle: {sharp_angle}
Alpha: {alpha}
Scale Factor: {scale_factor}
Pre-Remesh: {remesh_input}
Smooth: {smooth}

Before:
  Vertices: {len(trimesh.vertices):,}
  Faces (tri): {len(trimesh.faces):,}

After:
  Vertices: {len(remeshed_mesh.vertices):,}
  Faces (quad): {len(remeshed_mesh.faces):,}

QuadWild produces quad-dominant meshes via field-line tracing and ILP solving.
"""
        return remeshed_mesh, info


NODE_CLASS_MAPPINGS = {
    "GeomPackRemesh": RemeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackRemesh": "Remesh",
}
