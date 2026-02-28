# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Sharpen Mesh Node - Edge-recovering sharpening via pymeshlab.

Available backends:
- two_step: Two-phase bilateral normal filtering. Smooths face normals
  (respecting dihedral angle thresholds), then repositions vertices to
  match. Sharpens creases while keeping faces flat. Best for CAD-like
  geometry from marching cubes, scanning, or neural SDF extraction.
- unsharp_mask: Geometric unsharp masking. Subtracts a smoothed version
  from the original to amplify ridges and valleys.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

def _pymeshlab_two_step_sharpen(mesh, smooth_steps, normal_threshold,
                                normal_iterations, fit_iterations, selected_only):
    """Two-step bilateral normal sharpening via PyMeshLab."""
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
        ms.apply_coord_two_steps_smoothing(
            stepsmoothnum=smooth_steps,
            normalthr=normal_threshold,
            stepnormalnum=normal_iterations,
            stepfitnum=fit_iterations,
            selected=selected_only,
        )
    except AttributeError:
        try:
            ms.two_steps_smooth(
                stepsmoothnum=smooth_steps,
                normalthr=normal_threshold,
                stepnormalnum=normal_iterations,
                stepfitnum=fit_iterations,
                selected=selected_only,
            )
        except AttributeError:
            return None, (
                "PyMeshLab two-step smoothing filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""

def _pymeshlab_unsharp_mask_sharpen(mesh, weight, weight_original, iterations):
    """Geometric unsharp mask sharpening via PyMeshLab."""
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
        ms.apply_coord_unsharp_mask(
            weight=weight,
            weightorig=weight_original,
            iterations=iterations,
        )
    except AttributeError:
        try:
            ms.coord_unsharp_mask(
                weight=weight,
                weightorig=weight_original,
                iterations=iterations,
            )
        except AttributeError:
            return None, (
                "PyMeshLab coordinate unsharp mask filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""

class SharpenMeshNode(io.ComfyNode):
    """
    Sharpen Mesh - Edge-recovering sharpening algorithms.

    Available backends:
    - two_step: Bilateral normal filtering then vertex repositioning.
      Sharpens creases while flattening faces. Best for CAD-like geometry.
      The normal_threshold is the key parameter: edges with dihedral angle
      sharper than this are preserved as features.
    - unsharp_mask: Geometric unsharp masking. Subtracts smoothed from
      original to amplify ridges and valleys. Good for general enhancement.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpenMesh",
            display_name="Sharpen Mesh",
            category="geompack/smoothing",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.DynamicCombo.Input("backend", tooltip=(
                        "Sharpening algorithm. "
                        "two_step=bilateral normal filtering (recommended for CAD-like edges), "
                        "unsharp_mask=geometric unsharp masking (general enhancement)"
                    ), options=[
                    io.DynamicCombo.Option("two_step", [
                        io.Int.Input("smooth_steps", default=3, min=1, max=50, step=1, tooltip=(
                            "Number of two-step smoothing passes. "
                            "More steps = stronger sharpening effect."
                        )),
                        io.Float.Input("normal_threshold", default=60.0, min=0.0, max=180.0, step=0.5, tooltip=(
                            "Dihedral angle threshold in degrees. "
                            "Edges sharper than this angle are preserved as features. "
                            "Lower = more aggressive (more edges treated as creases). "
                            "60 is a good default for most CAD models."
                        )),
                    ]),
                    io.DynamicCombo.Option("unsharp_mask", [
                        io.Float.Input("weight", default=0.3, min=0.0, max=3.0, step=0.01, tooltip=(
                            "Unsharp mask weight controlling sharpening strength. "
                            "Higher = more pronounced sharpening."
                        )),
                        io.Int.Input("iterations", default=5, min=1, max=50, step=1, tooltip=(
                            "Smoothing iterations for the reference smooth mesh. "
                            "More iterations = larger-scale sharpening."
                        )),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, backend):
        """Apply mesh sharpening based on selected backend."""
        selected = backend["backend"]

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        log.info("Sharpen backend: %s", selected)
        log.info("Input: %s vertices, %s faces",
                 f"{initial_vertices:,}", f"{initial_faces:,}")

        if selected == "two_step":
            smooth_steps = backend.get("smooth_steps", 3)
            normal_threshold = backend.get("normal_threshold", 60.0)
            log.info("Parameters: smooth_steps=%d, normal_threshold=%.1f",
                     smooth_steps, normal_threshold)
            sharpened, error = _pymeshlab_two_step_sharpen(
                trimesh, smooth_steps, normal_threshold,
                normal_iterations=20, fit_iterations=20,
                selected_only=False,
            )
        elif selected == "unsharp_mask":
            weight = backend.get("weight", 0.3)
            iterations = backend.get("iterations", 5)
            log.info("Parameters: weight=%.3f, iterations=%d", weight, iterations)
            sharpened, error = _pymeshlab_unsharp_mask_sharpen(
                trimesh, weight, weight_original=1.0, iterations=iterations,
            )
        else:
            raise ValueError(f"Unknown backend: {selected}")

        if sharpened is None:
            raise ValueError(f"Sharpening failed ({backend}): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": selected,
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

        # Build backend-specific param block
        if selected == "two_step":
            param_text = (
                f"Smooth Steps: {smooth_steps}\n"
                f"Normal Threshold: {normal_threshold}\u00b0"
            )
        elif selected == "unsharp_mask":
            param_text = (
                f"Weight: {weight}\n"
                f"Iterations: {iterations}"
            )
        else:
            param_text = ""

        info = f"""Sharpen Mesh Results ({selected}):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})

NODE_CLASS_MAPPINGS = {
    "GeomPackSharpenMesh": SharpenMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackSharpenMesh": "Sharpen Mesh",
}
