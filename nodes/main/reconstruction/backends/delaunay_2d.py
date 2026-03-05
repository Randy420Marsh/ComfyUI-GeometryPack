# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""2D Delaunay triangulation surface reconstruction backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructDelaunay2DNode(io.ComfyNode):
    """2D Delaunay triangulation (for height fields)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstruct_Delaunay2D",
            display_name="Reconstruct Delaunay 2D (backend)",
            category="geompack/reconstruction",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points):
        log.info("Backend: delaunay_2d")
        vertices = np.asarray(points.vertices, dtype=np.float64)

        log.info("Computing 2D Delaunay triangulation...")

        try:
            from scipy.spatial import Delaunay

            points_2d = vertices[:, :2]
            tri = Delaunay(points_2d)

            result = trimesh_module.Trimesh(
                vertices=vertices,
                faces=tri.simplices,
                process=False
            )

            info = f"""Reconstruct Surface Results (2D Delaunay):

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

2D Delaunay projects points to XY plane for triangulation.
Best for height fields and terrain data.
"""
            # Preserve metadata
            if hasattr(points, 'metadata') and points.metadata:
                result.metadata = points.metadata.copy()
            else:
                result.metadata = {}
            result.metadata['reconstruction'] = {
                'method': 'delaunay_2d',
                'input_points': len(vertices),
                'output_vertices': len(result.vertices),
                'output_faces': len(result.faces),
            }

            return io.NodeOutput(result, info, ui={"text": [info]})

        except ImportError:
            raise ImportError("2D Delaunay requires scipy. Install with: pip install scipy")


NODE_CLASS_MAPPINGS = {"GeomPackReconstruct_Delaunay2D": ReconstructDelaunay2DNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackReconstruct_Delaunay2D": "Reconstruct Delaunay 2D (backend)"}
