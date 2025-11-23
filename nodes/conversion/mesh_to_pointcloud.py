# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh to Point Cloud Node - Sample points from mesh surface
"""

import numpy as np
import trimesh


class MeshToPointCloudNode:
    """
    Convert mesh to point cloud by sampling surface points.

    Samples points from the mesh surface using various sampling methods.
    Can optionally include normals and colors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "sample_count": ("INT", {
                    "default": 10000,
                    "min": 100,
                    "max": 10000000,
                    "step": 100
                }),
                "sampling_method": (["uniform", "even", "face_weighted"], {
                    "default": "uniform"
                }),
            },
            "optional": {
                "include_normals": (["true", "false"], {
                    "default": "true"
                }),
            }
        }

    RETURN_TYPES = ("TRIMESH",)  # Changed from POINT_CLOUD to TRIMESH for compatibility
    RETURN_NAMES = ("point_cloud",)
    FUNCTION = "mesh_to_pointcloud"
    CATEGORY = "geompack/conversion"

    def mesh_to_pointcloud(self, trimesh, sample_count, sampling_method, include_normals="true"):
        """
        Sample points from mesh surface.

        Args:
            trimesh: Input trimesh.Trimesh object
            sample_count: Number of points to sample
            sampling_method: Sampling strategy
            include_normals: Whether to compute surface normals

        Returns:
            tuple: (point_cloud_as_trimesh,) - TRIMESH with vertices only (no faces)
        """
        print(f"[MeshToPointCloud] Sampling {sample_count:,} points using {sampling_method} method")

        if sampling_method == "uniform":
            # Uniform random sampling
            points, face_indices = trimesh.sample(sample_count, return_index=True)

        elif sampling_method == "even":
            # Approximately even spacing (rejection sampling)
            # Calculate radius based on surface area and desired point count
            radius = np.sqrt(trimesh.area / sample_count) * 2.0
            points, face_indices = trimesh.sample.sample_surface_even(
                trimesh, sample_count, radius=radius
            )
            print(f"[MeshToPointCloud] Even sampling produced {len(points):,} points (target: {sample_count:,})")

        elif sampling_method == "face_weighted":
            # Weight by face area (default behavior)
            points, face_indices = trimesh.sample(
                sample_count,
                return_index=True,
                face_weight=trimesh.area_faces
            )

        # Optional: compute normals at sample points
        normals = None
        if include_normals == "true":
            # Get face normals for each sampled point
            normals = trimesh.face_normals[face_indices]

        # Create point cloud as TRIMESH object (vertices only, no faces)
        # This ensures compatibility with all TRIMESH-expecting nodes
        import trimesh as trimesh_module
        point_cloud = trimesh_module.PointCloud(vertices=points)

        # Add normals as vertex_normals if computed
        if normals is not None:
            point_cloud.vertex_normals = normals

        # Store point cloud metadata
        point_cloud.metadata = {
            'is_point_cloud': True,
            'face_indices': face_indices,
            'source_mesh_vertices': len(trimesh.vertices),
            'source_mesh_faces': len(trimesh.faces),
            'sample_count': len(points),
            'sampling_method': sampling_method,
            'has_normals': normals is not None
        }

        print(f"[MeshToPointCloud] Generated point cloud as TRIMESH with {len(points):,} points")

        return (point_cloud,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackMeshToPointCloud": MeshToPointCloudNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackMeshToPointCloud": "Mesh to Point Cloud",
}
