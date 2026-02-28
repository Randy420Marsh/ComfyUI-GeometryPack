# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Smooth Mesh Node - Multiple smoothing backends via pymeshlab and trimesh.

Available backends:
- laplacian: Classic Laplacian smoothing (fast, may shrink)
- taubin: Two-step Laplacian that prevents shrinkage
- hc_laplacian: Humphrey Classes correction (low shrinkage, good detail)
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

def _pymeshlab_laplacian_smooth(mesh, iterations, cotangent_weight, selected_only):
    """Laplacian smoothing via PyMeshLab."""
    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.apply_coord_laplacian_smoothing(
            stepsmoothnum=iterations,
            cotangentweight=cotangent_weight,
            selected=selected_only,
        )
    except AttributeError:
        try:
            ms.laplacian_smooth(
                stepsmoothnum=iterations,
                cotangentweight=cotangent_weight,
                selected=selected_only,
            )
        except AttributeError:
            return None, (
                "PyMeshLab Laplacian smoothing filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""

def _pymeshlab_taubin_smooth(mesh, iterations, lambda_val, mu_val, selected_only):
    """Taubin smoothing via PyMeshLab (shrinkage-free)."""
    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.apply_coord_taubin_smoothing(
            lambda_=lambda_val,
            mu=mu_val,
            stepsmoothnum=iterations,
            selected=selected_only,
        )
    except AttributeError:
        try:
            ms.taubin_smooth(
                lambda_=lambda_val,
                mu=mu_val,
                stepsmoothnum=iterations,
                selected=selected_only,
            )
        except AttributeError:
            return None, (
                "PyMeshLab Taubin smoothing filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""

def _pymeshlab_hc_laplacian_smooth(mesh, selected_only):
    """HC Laplacian smoothing via PyMeshLab (low shrinkage)."""
    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed. Install with: pip install pymeshlab"

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.apply_coord_hc_laplacian_smoothing(selected=selected_only)
    except AttributeError:
        try:
            ms.hc_laplacian_smooth(selected=selected_only)
        except AttributeError:
            return None, (
                "PyMeshLab HC Laplacian smoothing filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""

def _trimesh_laplacian_smooth(mesh, iterations, lamb):
    """Laplacian smoothing via trimesh (uniform weights)."""
    result = mesh.copy()

    try:
        from trimesh.smoothing import filter_laplacian
        filter_laplacian(result, lamb=lamb, iterations=iterations)
    except ImportError:
        return None, "trimesh smoothing module not available."

    return result, ""

def _trimesh_taubin_smooth(mesh, iterations, lamb, mu):
    """Taubin smoothing via trimesh."""
    result = mesh.copy()

    try:
        from trimesh.smoothing import filter_taubin
        filter_taubin(result, lamb=lamb, mu=mu, iterations=iterations)
    except ImportError:
        return None, "trimesh Taubin smoothing not available."

    return result, ""

class SmoothMeshNode(io.ComfyNode):
    """
    Smooth Mesh - Various mesh smoothing algorithms.

    Available backends:
    - laplacian: Classic Laplacian smoothing. Fast but causes mesh shrinkage.
      Cotangent weights give geometrically better results than uniform weights.
    - taubin: Two-step (lambda/mu) Laplacian that prevents shrinkage.
      Best general-purpose smoothing for preserving volume.
    - hc_laplacian: Humphrey Classes Laplacian correction.
      Minimal shrinkage, good detail preservation.
    - trimesh_laplacian: Uniform Laplacian via trimesh (fast, no pymeshlab needed).
    - trimesh_taubin: Taubin smoothing via trimesh (no pymeshlab needed).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSmoothMesh",
            display_name="Smooth Mesh",
            category="geompack/smoothing",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Smoothing algorithm. "
                        "taubin=shrinkage-free (recommended), "
                        "laplacian=fast but shrinks, "
                        "hc_laplacian=low shrinkage, "
                        "trimesh_*=lightweight alternatives"
                    ), options=[
                    io.DynamicCombo.Option("taubin", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip="Inflation factor (negative). Counteracts shrinkage from lambda. Must satisfy |mu| > lambda for stability. Typical: -0.53 for lambda=0.5."),
                    ]),
                    io.DynamicCombo.Option("laplacian", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Combo.Input("cotangent_weight", options=["true", "false"], default="true", tooltip="Use cotangent weights instead of uniform weights. Cotangent weights respect mesh geometry better but may be unstable on degenerate meshes."),
                    ]),
                    io.DynamicCombo.Option("hc_laplacian", []),
                    io.DynamicCombo.Option("trimesh_laplacian", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                    ]),
                    io.DynamicCombo.Option("trimesh_taubin", [
                        io.Int.Input("iterations", default=5, min=1, max=200, step=1, tooltip="Number of smoothing passes. More = smoother but slower."),
                        io.Float.Input("lambda_", default=0.5, min=0.01, max=1.0, step=0.01, tooltip="Smoothing strength per step. Higher = more aggressive smoothing per iteration."),
                        io.Float.Input("mu", default=-0.53, min=-1.0, max=-0.01, step=0.01, tooltip="Inflation factor (negative). Counteracts shrinkage from lambda. Must satisfy |mu| > lambda for stability. Typical: -0.53 for lambda=0.5."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="smoothed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        """Apply mesh smoothing based on selected backend."""
        selected = backend["backend"]
        iterations = backend.get("iterations", 5)
        lambda_ = backend.get("lambda_", 0.5)
        mu = backend.get("mu", -0.53)
        cotangent_weight = backend.get("cotangent_weight", "true")

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        log.info("Smooth backend: %s", selected)
        log.info("Input: %s vertices, %s faces",
                 f"{initial_vertices:,}", f"{initial_faces:,}")

        if selected == "laplacian":
            cot = (cotangent_weight == "true")
            log.info("Parameters: iterations=%d, cotangent_weight=%s", iterations, cot)
            smoothed, error = _pymeshlab_laplacian_smooth(
                trimesh, iterations, cot, False
            )
        elif selected == "taubin":
            log.info("Parameters: iterations=%d, lambda=%.3f, mu=%.3f",
                     iterations, lambda_, mu)
            smoothed, error = _pymeshlab_taubin_smooth(
                trimesh, iterations, lambda_, mu, False
            )
        elif selected == "hc_laplacian":
            log.info("Parameters: (none)")
            smoothed, error = _pymeshlab_hc_laplacian_smooth(trimesh, False)
        elif selected == "trimesh_laplacian":
            log.info("Parameters: iterations=%d, lambda=%.3f", iterations, lambda_)
            smoothed, error = _trimesh_laplacian_smooth(
                trimesh, iterations, lambda_
            )
        elif selected == "trimesh_taubin":
            log.info("Parameters: iterations=%d, lambda=%.3f, mu=%.3f",
                     iterations, lambda_, mu)
            smoothed, error = _trimesh_taubin_smooth(
                trimesh, iterations, lambda_, mu
            )
        else:
            raise ValueError(f"Unknown backend: {selected}")

        if smoothed is None:
            raise ValueError(f"Smoothing failed ({selected}): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            smoothed.metadata = trimesh.metadata.copy()
        smoothed.metadata["smoothing"] = {
            "algorithm": selected,
            "iterations": iterations,
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
        }

        # Compute displacement stats
        disp = np.linalg.norm(
            np.asarray(smoothed.vertices) - np.asarray(trimesh.vertices), axis=1
        )
        avg_disp = float(np.mean(disp))
        max_disp = float(np.max(disp))

        log.info("Output: %d vertices, %d faces",
                 len(smoothed.vertices), len(smoothed.faces))
        log.info("Avg vertex displacement: %.6f, max: %.6f", avg_disp, max_disp)

        # Build backend-specific param block
        if selected == "laplacian":
            param_text = (
                f"Iterations: {iterations}\n"
                f"Cotangent Weight: {cotangent_weight}"
            )
        elif selected in ("taubin", "trimesh_taubin"):
            param_text = (
                f"Iterations: {iterations}\n"
                f"Lambda: {lambda_}\n"
                f"Mu: {mu}"
            )
        elif selected == "hc_laplacian":
            param_text = "(single-pass HC correction)"
        elif selected == "trimesh_laplacian":
            param_text = (
                f"Iterations: {iterations}\n"
                f"Lambda: {lambda_}"
            )
        else:
            param_text = ""

        info = f"""Smooth Mesh Results ({selected}):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(smoothed, info, ui={"text": [info]})

NODE_CLASS_MAPPINGS = {
    "GeomPackSmoothMesh": SmoothMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSmoothMesh": "Smooth Mesh",
}
