# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Unified Reconstruct Surface Node - Single frontend with method selector.

Uses ComfyUI's node expansion (GraphBuilder) to dispatch to hidden
backend-specific nodes, each running in its own isolation env.
"""

import logging
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


class ReconstructSurfaceNode(io.ComfyNode):
    """
    Reconstruct Surface - Convert point cloud to mesh.

    Dispatches to hidden backend nodes via node expansion.
    """

    BACKEND_MAP = {
        "poisson": "GeomPackReconstruct_Poisson",
        "ball_pivoting": "GeomPackReconstruct_BallPivoting",
        "alpha_shape": "GeomPackReconstruct_AlphaShape",
        "convex_hull": "GeomPackReconstruct_ConvexHull",
        "delaunay_2d": "GeomPackReconstruct_Delaunay2D",
        "alpha_wrap": "GeomPackAlphaWrap",
    }

    # Remap frontend param names to backend param names where they differ
    PARAM_REMAP = {
        "alpha_wrap": {"points": "input_mesh"},
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstructSurface",
            display_name="Reconstruct Surface",
            category="geompack/reconstruction",
            enable_expand=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.DynamicCombo.Input("backend", tooltip=(
                    "Reconstruction algorithm. "
                    "poisson=smooth watertight, "
                    "ball_pivoting=preserves detail, "
                    "alpha_shape=controllable detail, "
                    "convex_hull=fast rough, "
                    "delaunay_2d=height fields, "
                    "alpha_wrap=shrink wrap (CGAL)"
                ), options=[
                    io.DynamicCombo.Option("poisson", [
                        io.Int.Input("poisson_depth", default=8, min=1, max=12, step=1, tooltip="Octree depth for Poisson reconstruction. Higher = more detail, slower."),
                        io.Float.Input("poisson_scale", default=1.1, min=1.0, max=2.0, step=0.1, tooltip="Scale factor for reconstruction grid."),
                        io.Combo.Input("estimate_normals", options=["true", "false"], default="true", tooltip="Estimate normals from point cloud."),
                        io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, tooltip="Search radius for normal estimation."),
                    ]),
                    io.DynamicCombo.Option("ball_pivoting", [
                        io.Float.Input("ball_radius", default=0.0, min=0.0, max=100.0, step=0.01, tooltip="Ball radius for pivoting. 0 = auto."),
                        io.Combo.Input("estimate_normals", options=["true", "false"], default="true", tooltip="Estimate normals from point cloud."),
                        io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, tooltip="Search radius for normal estimation."),
                    ]),
                    io.DynamicCombo.Option("alpha_shape", [
                        io.Float.Input("alpha", default=0.0, min=0.0, max=100.0, step=0.01, tooltip="Alpha value. 0 = auto (10%% of bbox diagonal)."),
                    ]),
                    io.DynamicCombo.Option("convex_hull", []),
                    io.DynamicCombo.Option("delaunay_2d", []),
                    io.DynamicCombo.Option("alpha_wrap", [
                        io.Float.Input("alpha_percent", default=2.0, min=0.001, max=50.0, step=0.1, tooltip="Wrap tightness as %% of bounding box diagonal. Smaller = tighter wrap with more detail, but MUCH slower."),
                        io.Float.Input("offset_percent", default=2.0, min=0.01, max=10.0, step=0.1, tooltip="Surface offset as %% of bounding box diagonal. Smaller = closer to original surface."),
                    ]),
                ]),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, backend):
        from comfy_execution.graph_utils import GraphBuilder

        if cls.SCHEMA is None:
            cls.GET_SCHEMA()

        selected = backend["backend"]
        node_id = cls.BACKEND_MAP[selected]

        log.info("Reconstruct dispatch: %s -> %s", selected, node_id)

        # Build kwargs, applying param remaps for cross-env backends
        remap = cls.PARAM_REMAP.get(selected, {})
        kwargs = {remap.get("points", "points"): points}
        for k, v in backend.items():
            if k == "backend":
                continue
            kwargs[remap.get(k, k)] = v

        graph = GraphBuilder()
        backend_node = graph.node(node_id, **kwargs)

        return {
            "result": (backend_node.out(0), backend_node.out(1)),
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "GeomPackReconstructSurface": ReconstructSurfaceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackReconstructSurface": "Reconstruct Surface",
}
