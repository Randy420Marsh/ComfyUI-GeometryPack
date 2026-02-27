# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Extract Skeleton Node - Extract skeleton from 3D mesh
"""

import logging

import numpy as np
import trimesh
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


def normalize_skeleton(vertices: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton vertices to [-1, 1] range.

    Args:
        vertices: Array of shape [N, 3]

    Returns:
        Normalized vertices in [-1, 1] range
    """
    # Find bounding box
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Center at origin
    center = (min_coords + max_coords) / 2
    vertices = vertices - center

    # Scale to [-1, 1]
    scale = (max_coords - min_coords).max() / 2
    if scale > 0:
        vertices = vertices / scale

    return vertices


class ExtractSkeleton(io.ComfyNode):
    """
    Extract skeleton from 3D mesh using Skeletor library.

    Outputs skeleton data (vertices + edges) with optional normalization to [-1, 1] range.
    By default, preserves the original mesh scale.
    """


    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackExtractSkeleton",
            display_name="Extract Skeleton",
            category="geompack/skeleton",
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Combo.Input("algorithm", options=["wavefront", "vertex_clusters", "edge_collapse", "teasar"], default="wavefront"),
                io.Boolean.Input("fix_mesh", default=True, tooltip="Fix mesh issues before skeletonization"),
                io.Boolean.Input("normalize", default=False, tooltip="Normalize skeleton to [-1, 1] range (False preserves original mesh scale)"),
                io.Int.Input("waves", default=1, min=1, max=20, tooltip="Wavefront: number of waves", optional=True),
                io.Float.Input("step_size", default=1.0, min=0.1, max=20.0, tooltip="Wavefront: step size (higher = coarser)", optional=True),
                io.Float.Input("sampling_dist", default=1.0, min=0.1, max=50.0, tooltip="Vertex clusters: max distance for clustering", optional=True),
                io.Combo.Input("cluster_pos", options=["median", "center"], default="median", tooltip="Vertex clusters: cluster position method", optional=True),
                io.Float.Input("shape_weight", default=1.0, min=0.0, max=10.0, tooltip="Edge collapse: shape preservation weight", optional=True),
                io.Float.Input("sample_weight", default=0.1, min=0.0, max=10.0, tooltip="Edge collapse: sampling quality weight", optional=True),
                io.Float.Input("inv_dist", default=10.0, min=1.0, max=100.0, tooltip="TEASAR: invalidation distance (lower = more detail)", optional=True),
                io.Float.Input("min_length", default=0.0, min=0.0, max=100.0, tooltip="TEASAR: minimum branch length to keep", optional=True),
            ],
            outputs=[
                io.Custom("SKELETON").Output(display_name="skeleton"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, algorithm, fix_mesh, normalize,
                waves=1, step_size=1.0,
                sampling_dist=1.0, cluster_pos="median",
                shape_weight=1.0, sample_weight=0.1,
                inv_dist=10.0, min_length=0.0):
        """Extract skeleton from mesh.

        Args:
            trimesh: Input mesh
            algorithm: Skeletonization algorithm to use
            fix_mesh: Whether to fix mesh issues before extraction
            normalize: If True, normalize skeleton to [-1, 1] range. If False, preserve original scale.
            ... (algorithm-specific parameters)
        """
        try:
            import skeletor as sk
        except ImportError:
            raise ImportError(
                "Skeletor library not found. Please install: pip install skeletor"
            )

        log.info("Extracting skeleton using %s algorithm...", algorithm)

        # Print input mesh bounding box
        mesh_min = trimesh.bounds[0]
        mesh_max = trimesh.bounds[1]
        mesh_size = mesh_max - mesh_min
        mesh_center = (mesh_min + mesh_max) / 2
        log.info("Input mesh bounding box:")
        log.info("  Min: [%.3f, %.3f, %.3f]", mesh_min[0], mesh_min[1], mesh_min[2])
        log.info("  Max: [%.3f, %.3f, %.3f]", mesh_max[0], mesh_max[1], mesh_max[2])
        log.info("  Size: [%.3f, %.3f, %.3f]", mesh_size[0], mesh_size[1], mesh_size[2])
        log.info("  Center: [%.3f, %.3f, %.3f]", mesh_center[0], mesh_center[1], mesh_center[2])

        # Fix mesh if requested
        if fix_mesh:
            log.info("Fixing mesh...")
            mesh = sk.pre.fix_mesh(trimesh, remove_disconnected=5, inplace=False)
        else:
            mesh = trimesh

        # Extract skeleton based on algorithm
        try:
            if algorithm == "wavefront":
                log.info("  Parameters: waves=%d, step_size=%s", waves, step_size)
                skel = sk.skeletonize.by_wavefront(mesh, waves=waves, step_size=step_size)

            elif algorithm == "vertex_clusters":
                log.info("  Parameters: sampling_dist=%s, cluster_pos=%s", sampling_dist, cluster_pos)
                skel = sk.skeletonize.by_vertex_clusters(
                    mesh,
                    sampling_dist=sampling_dist,
                    cluster_pos=cluster_pos
                )

            elif algorithm == "edge_collapse":
                log.info("  Parameters: shape_weight=%s, sample_weight=%s", shape_weight, sample_weight)
                skel = sk.skeletonize.by_edge_collapse(
                    mesh,
                    shape_weight=shape_weight,
                    sample_weight=sample_weight
                )

            elif algorithm == "teasar":
                log.info("  Parameters: inv_dist=%s, min_length=%s", inv_dist, min_length)
                skel = sk.skeletonize.by_teasar(
                    mesh,
                    inv_dist=inv_dist,
                    min_length=min_length if min_length > 0 else None
                )

            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

        except Exception as e:
            log.error("Error during skeletonization: %s", e)
            raise RuntimeError(f"Skeletonization failed: {e}")

        # Get vertices and edges
        vertices = np.array(skel.vertices)
        edges = np.array(skel.edges)

        log.info("Extracted %d joints, %d bones", len(vertices), len(edges))

        # Print skeleton bounding box before any normalization
        skel_min = vertices.min(axis=0)
        skel_max = vertices.max(axis=0)
        skel_size = skel_max - skel_min
        skel_center = (skel_min + skel_max) / 2
        log.info("Skeleton bounding box (original):")
        log.info("  Min: [%.3f, %.3f, %.3f]", skel_min[0], skel_min[1], skel_min[2])
        log.info("  Max: [%.3f, %.3f, %.3f]", skel_max[0], skel_max[1], skel_max[2])
        log.info("  Size: [%.3f, %.3f, %.3f]", skel_size[0], skel_size[1], skel_size[2])
        log.info("  Center: [%.3f, %.3f, %.3f]", skel_center[0], skel_center[1], skel_center[2])

        # Store original scale and center for metadata
        original_scale = float((skel_max - skel_min).max() / 2)
        original_center = skel_center.copy()

        # Conditionally normalize
        if normalize:
            vertices = normalize_skeleton(vertices)

            # Print skeleton bounding box after normalization
            norm_min = vertices.min(axis=0)
            norm_max = vertices.max(axis=0)
            norm_size = norm_max - norm_min
            log.info("Skeleton bounding box AFTER normalization:")
            log.info("  Min: [%.3f, %.3f, %.3f]", norm_min[0], norm_min[1], norm_min[2])
            log.info("  Max: [%.3f, %.3f, %.3f]", norm_max[0], norm_max[1], norm_max[2])
            log.info("  Size: [%.3f, %.3f, %.3f]", norm_size[0], norm_size[1], norm_size[2])
            log.info("  Overall range: [%.3f, %.3f]", vertices.min(), vertices.max())
        else:
            log.info("Normalization skipped - preserving original scale")

        # Package as skeleton data
        skeleton = {
            "vertices": vertices,  # [N, 3] joint positions
            "edges": edges,        # [M, 2] bone connections (vertex indices)
            "scale": original_scale,  # Original scale factor (for denormalization if needed)
            "center": original_center.tolist(),  # Original center point
            "normalized": normalize,  # Whether this skeleton was normalized
        }

        return io.NodeOutput(skeleton)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackExtractSkeleton": ExtractSkeleton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackExtractSkeleton": "Extract Skeleton",
}
