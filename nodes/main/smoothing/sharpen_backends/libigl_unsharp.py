# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Cotangent-weighted geometric unsharp mask backend node (libigl)."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def _libigl_unsharp_sharpen(mesh, lambda_, iterations):
    """Cotangent-weighted geometric unsharp mask via libigl.

    V_sharp = V + lambda * M^{-1} * L * V
    where L is the cotangent Laplacian and M is the Voronoi mass matrix.
    """
    try:
        import igl
    except ImportError:
        return None, "libigl is not installed. Install with: pip install libigl"

    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    L = igl.cotmatrix(V, F)
    M = igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI)

    # M is diagonal -- invert via element-wise reciprocal
    M_diag = M.diagonal().copy()
    M_diag[M_diag == 0] = 1e-12
    M_inv = 1.0 / M_diag

    for _ in range(iterations):
        # L @ V gives curvature-weighted displacement (n, 3)
        LV = L @ V
        delta = M_inv[:, None] * LV
        V = V + lambda_ * delta

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenLibiglUnsharpNode(io.ComfyNode):
    """Cotangent-weighted geometric unsharp mask backend (libigl)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_LibiglUnsharp",
            display_name="Sharpen Libigl Unsharp (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Float.Input("lambda_", default=0.5, min=0.01, max=5.0, step=0.01, tooltip=(
                    "Unsharp mask strength. Controls how much the cotangent "
                    "Laplacian displacement is amplified. Higher values produce "
                    "stronger sharpening but may cause overshooting. "
                    "Start with 0.3-0.5."
                )),
                io.Int.Input("iterations", default=3, min=1, max=50, step=1, tooltip=(
                    "Number of unsharp mask passes. Multiple light passes "
                    "are more stable than a single heavy pass."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, lambda_=0.5, iterations=3):
        log.info("Backend: libigl_unsharp")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: lambda=%.3f, iterations=%d", lambda_, iterations)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _libigl_unsharp_sharpen(
            trimesh, lambda_, iterations,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (libigl_unsharp): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "libigl_unsharp",
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
        }

        # Compute displacement stats
        disp = np.linalg.norm(
            np.asarray(sharpened.vertices) - np.asarray(trimesh.vertices), axis=1
        )
        avg_disp = float(np.mean(disp))
        max_disp = float(np.max(disp))

        log.info("Output: %d vertices, %d faces",
                 len(sharpened.vertices), len(sharpened.faces))
        log.info("Avg vertex displacement: %.6f, max: %.6f", avg_disp, max_disp)

        param_text = (
            f"Lambda: {lambda_}\n"
            f"Iterations: {iterations}"
        )

        info = f"""Sharpen Mesh Results (libigl_unsharp):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_LibiglUnsharp": SharpenLibiglUnsharpNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_LibiglUnsharp": "Sharpen Libigl Unsharp (backend)"}
