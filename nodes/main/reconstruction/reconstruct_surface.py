# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Surface Reconstruction Node - Point cloud to mesh conversion
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")

class ReconstructSurfaceNode(io.ComfyNode):
    """
    Reconstruct Surface - Convert point cloud to mesh.

    Accepts TRIMESH objects (including point clouds from MeshToPointCloud node).
    Point clouds are represented as TRIMESH with vertices only (0 faces).

    Multiple reconstruction algorithms:
    - poisson: Screened Poisson surface reconstruction (smooth, watertight)
    - ball_pivoting: Ball pivoting algorithm (preserves detail)
    - alpha_shape: Alpha shape reconstruction
    - convex_hull: Simple convex hull (fast, rough)
    - delaunay_2d: 2D Delaunay triangulation (for height fields)

    Requires Open3D or PyMeshLab for advanced methods.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackReconstructSurface",
            display_name="Reconstruct Surface",
            category="geompack/reconstruction",
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("points"),
                io.Combo.Input("method", options=[
                    "poisson",
                    "ball_pivoting",
                    "alpha_shape",
                    "convex_hull",
                    "delaunay_2d"
                ], default="poisson"),
                io.Int.Input("poisson_depth", default=8, min=1, max=12, step=1, optional=True),
                io.Float.Input("poisson_scale", default=1.1, min=1.0, max=2.0, step=0.1, optional=True),
                io.Float.Input("ball_radius", default=0.0, min=0.0, max=100.0, step=0.01, optional=True),
                io.Float.Input("alpha", default=0.0, min=0.0, max=100.0, step=0.01, optional=True),
                io.Combo.Input("estimate_normals", options=["true", "false"], default="true", optional=True),
                io.Float.Input("normal_radius", default=0.1, min=0.001, max=10.0, step=0.01, optional=True),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="reconstructed_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, points, method,
                    poisson_depth=8, poisson_scale=1.1,
                    ball_radius=0.0, alpha=0.0,
                    estimate_normals="true", normal_radius=0.1):
        """
        Reconstruct surface from point cloud.

        Args:
            points: Point cloud as TRIMESH (may have 0 faces for pure point clouds)
            method: Reconstruction algorithm
            [other params]: Method-specific parameters

        Returns:
            tuple: (reconstructed_mesh, info_string)
        """
        # Extract vertices from TRIMESH
        if not hasattr(points, 'vertices'):
            raise ValueError("Input must be a TRIMESH object with vertices")

        vertices = points.vertices
        normals = None

        # Extract normals if available
        if hasattr(points, 'vertex_normals') and len(points.vertex_normals) > 0:
            normals = points.vertex_normals
            log.info("Using normals from input")

        # Check if this is a point cloud
        is_point_cloud = False
        face_count = len(points.faces) if hasattr(points, 'faces') and points.faces is not None else 0

        if face_count == 0 or (hasattr(points, 'metadata') and points.metadata.get('is_point_cloud', False)):
            is_point_cloud = True
            log.info("Input type: Point cloud (%d points)", len(vertices))
        else:
            log.info("Input type: TRIMESH (%d vertices, %d faces)", len(vertices), face_count)

        log.info("Method: %s", method)

        if method == "poisson":
            result, info = cls._poisson(vertices, normals, poisson_depth, poisson_scale,
                                         estimate_normals == "true", normal_radius)
        elif method == "ball_pivoting":
            result, info = cls._ball_pivoting(vertices, normals, ball_radius,
                                               estimate_normals == "true", normal_radius)
        elif method == "alpha_shape":
            result, info = cls._alpha_shape(vertices, alpha)
        elif method == "convex_hull":
            result, info = cls._convex_hull(vertices)
        elif method == "delaunay_2d":
            result, info = cls._delaunay_2d(vertices)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Preserve metadata
        if hasattr(points, 'metadata'):
            result.metadata = points.metadata.copy()
        else:
            result.metadata = {}

        # Add reconstruction info
        result.metadata['reconstruction'] = {
            'method': method,
            'was_point_cloud': is_point_cloud,
            'input_points': len(vertices),
            'output_vertices': len(result.vertices),
            'output_faces': len(result.faces)
        }

        log.info("Output: %d vertices, %d faces", len(result.vertices), len(result.faces))
        return io.NodeOutput(result, info, ui={"text": [info]})

    @staticmethod
    def _poisson(vertices, normals, depth, scale, estimate_normals, normal_radius):
        """Poisson surface reconstruction using Open3D or PyMeshLab."""
        # Try Open3D first
        try:
            import open3d as o3d

            log.info("Using Open3D Poisson reconstruction...")
            log.info("Step 1/5: Creating point cloud...")

            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)

            # Estimate normals if needed
            log.info("Step 2/5: Estimating normals...")
            if normals is None or estimate_normals:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=normal_radius, max_nn=30
                    )
                )
                log.info("Step 3/5: Orienting normals...")
                pcd.orient_normals_consistent_tangent_plane(k=10)
            else:
                pcd.normals = o3d.utility.Vector3dVector(normals)

            # Poisson reconstruction
            log.info("Step 4/5: Running Poisson reconstruction (depth=%d)... This may take a while.", depth)
            mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, scale=scale, linear_fit=False
            )

            # Remove low density vertices (noise)
            log.info("Step 5/5: Cleaning up mesh...")
            densities = np.asarray(densities)
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh_o3d.remove_vertices_by_mask(vertices_to_remove)

            # Convert to trimesh
            result = trimesh_module.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                process=False
            )

            log.info("Done! Output: %s vertices, %s faces", f"{len(result.vertices):,}", f"{len(result.faces):,}")

            info = f"""Reconstruct Surface Results (Poisson):

Engine: Open3D
Depth: {depth}
Scale: {scale}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}

Poisson reconstruction creates smooth, watertight surfaces.
"""
            return result, info

        except ImportError:
            pass

        # Fallback to PyMeshLab
        try:
            import pymeshlab

            log.info("Using PyMeshLab Poisson reconstruction...")

            ms = pymeshlab.MeshSet()

            # Create point cloud mesh
            if normals is not None and not estimate_normals:
                pml_mesh = pymeshlab.Mesh(
                    vertex_matrix=vertices,
                    v_normals_matrix=normals
                )
            else:
                pml_mesh = pymeshlab.Mesh(vertex_matrix=vertices)

            ms.add_mesh(pml_mesh)

            # Estimate normals if needed
            if normals is None or estimate_normals:
                ms.compute_normal_for_point_clouds(k=10)

            # Poisson reconstruction
            ms.generate_surface_reconstruction_screened_poisson(
                depth=depth,
                scale=scale
            )

            # Get result
            result_mesh = ms.current_mesh()
            result = trimesh_module.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            info = f"""Reconstruct Surface Results (Poisson):

Engine: PyMeshLab
Depth: {depth}
Scale: {scale}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""
            return result, info

        except ImportError:
            raise ImportError(
                "Poisson reconstruction requires Open3D or PyMeshLab.\n"
                "Install with: pip install open3d  or  pip install pymeshlab"
            )

    @staticmethod
    def _ball_pivoting(vertices, normals, ball_radius, estimate_normals, normal_radius):
        """Ball pivoting algorithm using PyMeshLab."""
        try:
            import pymeshlab

            log.info("Using PyMeshLab Ball Pivoting...")

            ms = pymeshlab.MeshSet()

            # Create point cloud mesh
            if normals is not None and not estimate_normals:
                pml_mesh = pymeshlab.Mesh(
                    vertex_matrix=vertices,
                    v_normals_matrix=normals
                )
            else:
                pml_mesh = pymeshlab.Mesh(vertex_matrix=vertices)

            ms.add_mesh(pml_mesh)

            # Estimate normals if needed
            if normals is None or estimate_normals:
                ms.compute_normal_for_point_clouds(k=10)

            # Ball pivoting reconstruction
            if ball_radius == 0.0:
                # Auto radius based on point cloud density
                ms.generate_surface_reconstruction_ball_pivoting()
            else:
                ms.generate_surface_reconstruction_ball_pivoting(
                    ballradius=pymeshlab.PercentageValue(ball_radius * 100)
                )

            result_mesh = ms.current_mesh()
            result = trimesh_module.Trimesh(
                vertices=result_mesh.vertex_matrix(),
                faces=result_mesh.face_matrix(),
                process=False
            )

            info = f"""Reconstruct Surface Results (Ball Pivoting):

Engine: PyMeshLab
Ball Radius: {'auto' if ball_radius == 0.0 else f'{ball_radius:.3f}'}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Ball pivoting preserves fine details but may have holes.
"""
            return result, info

        except ImportError:
            raise ImportError(
                "Ball pivoting requires PyMeshLab.\n"
                "Install with: pip install pymeshlab"
            )

    @staticmethod
    def _alpha_shape(vertices, alpha_value):
        """Alpha shape reconstruction."""
        log.info("Computing alpha shape...")

        if alpha_value == 0.0:
            # Auto alpha: use 10% of bounding box diagonal
            bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
            alpha_value = bbox_diag * 0.1
            log.info("Auto alpha: %.4f", alpha_value)

        # Use trimesh's alpha shape
        try:
            from scipy.spatial import Delaunay

            # Compute Delaunay triangulation
            tri = Delaunay(vertices)

            # Filter simplices by alpha criterion
            # Alpha shape: keep tetrahedra with circumradius < alpha
            valid_faces = []
            for simplex in tri.simplices:
                # Get tetrahedron vertices
                tet_verts = vertices[simplex]

                # Compute circumradius (approximation using edge lengths)
                edges = []
                for i in range(4):
                    for j in range(i + 1, 4):
                        edges.append(np.linalg.norm(tet_verts[i] - tet_verts[j]))
                max_edge = max(edges)

                if max_edge < alpha_value * 2:
                    # Add faces of this tetrahedron
                    for i in range(4):
                        face = tuple(sorted([simplex[j] for j in range(4) if j != i]))
                        valid_faces.append(face)

            # Remove duplicate faces (interior faces appear twice)
            from collections import Counter
            face_counts = Counter(valid_faces)
            boundary_faces = [list(f) for f, count in face_counts.items() if count == 1]

            if len(boundary_faces) == 0:
                raise ValueError("Alpha value too small, no faces generated")

            result = trimesh_module.Trimesh(
                vertices=vertices,
                faces=boundary_faces,
                process=True
            )

            info = f"""Reconstruct Surface Results (Alpha Shape):

Alpha Value: {alpha_value:.4f}

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Alpha shapes capture the overall shape with controllable detail level.
"""
            return result, info

        except ImportError:
            raise ImportError("Alpha shape requires scipy. Install with: pip install scipy")

    @staticmethod
    def _convex_hull(vertices):
        """Simple convex hull reconstruction."""
        log.info("Computing convex hull...")

        # Use trimesh's convex hull
        result = trimesh_module.Trimesh(vertices=vertices).convex_hull

        info = f"""Reconstruct Surface Results (Convex Hull):

Input Points: {len(vertices):,}
Output Vertices: {len(result.vertices):,}
Output Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
Volume: {result.volume:.6f}

Convex hull is fast but loses all concave features.
"""
        return result, info

    @staticmethod
    def _delaunay_2d(vertices):
        """2D Delaunay triangulation (for height fields)."""
        log.info("Computing 2D Delaunay triangulation...")

        try:
            from scipy.spatial import Delaunay

            # Project to XY plane for triangulation
            points_2d = vertices[:, :2]
            tri = Delaunay(points_2d)

            # Create mesh with original 3D coordinates
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
            return result, info

        except ImportError:
            raise ImportError("2D Delaunay requires scipy. Install with: pip install scipy")

# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackReconstructSurface": ReconstructSurfaceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackReconstructSurface": "Reconstruct Surface",
}
