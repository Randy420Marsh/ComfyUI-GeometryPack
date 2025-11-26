# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Backdraft detection visualization node.

Renders a mesh from above (Z-axis parallel projection) and colors pixels based on
ray intersection count:
- BLACK: Ray misses mesh (background)
- GREEN: Ray hits exactly 1 face (clean geometry, no undercut)
- RED: Ray hits 2+ faces (backdraft/undercut - material would be trapped)

This is useful for manufacturing analysis to detect undercuts that would
prevent clean mold release.

Supports two ray tracing backends:
- trimesh: Uses embree for fast batch ray casting (recommended)
- pyvista: Uses VTK's multi_ray_trace
"""

import numpy as np
import torch


class BackdraftViewNode:
    """
    Render mesh from Z-axis and detect backdraft regions.

    Backdraft = when a ray from above hits MORE than 1 face.
    This indicates an undercut that would trap material in manufacturing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "resolution": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 4096,
                    "step": 64,
                    "tooltip": "Output image resolution. Higher = more detail but slower."
                }),
                "backend": (["trimesh", "pyvista"], {
                    "default": "trimesh",
                    "tooltip": "Ray tracing backend. trimesh (embree) is faster, pyvista uses VTK."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("backdraft_image",)
    FUNCTION = "render_backdraft"
    CATEGORY = "geompack/visualization"

    def render_backdraft(self, trimesh, resolution=1024, backend="trimesh"):
        """
        Render mesh with backdraft detection using batch ray casting.

        Args:
            trimesh: Input trimesh object
            resolution: Output image size
            backend: Ray tracing backend ("trimesh" or "pyvista")

        Returns:
            tuple: (IMAGE tensor in BHWC format)
        """
        print(f"[BackdraftView] Processing mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[BackdraftView] Resolution: {resolution}x{resolution}, Backend: {backend}")

        # Get mesh bounds
        bounds = trimesh.bounds
        xmin, ymin, zmin = bounds[0]
        xmax, ymax, zmax = bounds[1]
        print(f"[BackdraftView] Mesh bounds: X[{xmin:.2f}, {xmax:.2f}] Y[{ymin:.2f}, {ymax:.2f}] Z[{zmin:.2f}, {zmax:.2f}]")

        # Add padding around bounds
        x_range = xmax - xmin
        y_range = ymax - ymin
        padding = max(x_range, y_range) * 0.05
        xmin -= padding
        xmax += padding
        ymin -= padding
        ymax += padding

        # Calculate grid dimensions maintaining aspect ratio
        x_range = xmax - xmin
        y_range = ymax - ymin
        aspect = x_range / y_range

        if aspect > 1:
            nx = resolution
            ny = max(1, int(resolution / aspect))
        else:
            ny = resolution
            nx = max(1, int(resolution * aspect))

        total_rays = nx * ny
        print(f"[BackdraftView] Ray grid: {nx} x {ny} = {total_rays} rays")

        # Create ray grid
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        z_start = zmax + 1.0
        z_end = zmin - 1.0

        # Dispatch to appropriate backend
        if backend == "trimesh":
            hit_counts = self._raytrace_trimesh(trimesh, xs, ys, z_start, nx, ny, total_rays)
        else:
            hit_counts = self._raytrace_pyvista(trimesh, xs, ys, z_start, z_end, nx, ny)

        # Flip Y axis for image coordinates (image origin is top-left)
        hit_counts = hit_counts[::-1, :]

        # Vectorized color assignment
        # BLACK (0,0,0) for no hits, GREEN (0,255,0) for 1 hit, RED (255,0,0) for 2+ hits
        image = np.zeros((ny, nx, 3), dtype=np.uint8)
        image[hit_counts == 1] = [0, 255, 0]   # GREEN - clean geometry
        image[hit_counts >= 2] = [255, 0, 0]   # RED - backdraft
        # hit_counts == 0 stays BLACK (background)

        # Calculate stats
        hit_count = np.sum(hit_counts >= 1)
        backdraft_count = np.sum(hit_counts >= 2)
        backdraft_pct = backdraft_count / max(1, hit_count) * 100

        print(f"[BackdraftView] Complete: {hit_count} pixels with geometry, {backdraft_count} backdraft pixels ({backdraft_pct:.1f}%)")

        # Convert to ComfyUI IMAGE format: (B, H, W, C) float32 0-1
        img_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        return (img_tensor,)

    def _raytrace_trimesh(self, mesh, xs, ys, z_start, nx, ny, total_rays):
        """Batch ray casting using trimesh (embree backend) with progress display."""
        print(f"[BackdraftView] Using trimesh backend (embree)...")

        # Create batch ray origins and directions
        xx, yy = np.meshgrid(xs, ys)
        origins = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(total_rays, z_start)
        ]).astype(np.float64)

        # All rays point straight down (-Z)
        directions = np.tile([0.0, 0.0, -1.0], (total_rays, 1))

        # Process in chunks for progress display (each chunk = ~100k rays)
        chunk_size = 100000
        hit_counts = np.zeros(total_rays, dtype=np.int32)
        total_intersections = 0

        if total_rays <= chunk_size:
            # Small enough to do in one batch
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=True
            )
            np.add.at(hit_counts, index_ray, 1)
            total_intersections = len(locations)
        else:
            # Process in chunks with progress
            num_chunks = (total_rays + chunk_size - 1) // chunk_size
            print(f"[BackdraftView] Processing {num_chunks} chunks of ~{chunk_size} rays...")

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_rays)

                chunk_origins = origins[start_idx:end_idx]
                chunk_directions = directions[start_idx:end_idx]

                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=chunk_origins,
                    ray_directions=chunk_directions,
                    multiple_hits=True
                )

                # Offset indices back to global array
                np.add.at(hit_counts, index_ray + start_idx, 1)
                total_intersections += len(locations)

                progress = (i + 1) / num_chunks * 100
                print(f"[BackdraftView] Progress: {progress:.0f}% ({end_idx}/{total_rays} rays)")

        print(f"[BackdraftView] Ray casting complete, {total_intersections} total intersections")

        hit_counts = hit_counts.reshape(ny, nx)
        return hit_counts

    def _raytrace_pyvista(self, mesh, xs, ys, z_start, z_end, nx, ny):
        """Batch ray casting using PyVista multi_ray_trace."""
        import pyvista as pv

        print(f"[BackdraftView] Using pyvista backend (multi_ray_trace)...")

        # Convert trimesh to PyVista PolyData
        faces_pv = np.hstack([
            np.full((len(mesh.faces), 1), 3),
            mesh.faces
        ]).astype(np.int64).flatten()
        pv_mesh = pv.PolyData(np.array(mesh.vertices), faces_pv)

        # Create batch ray origins and directions
        total_rays = nx * ny
        xx, yy = np.meshgrid(xs, ys)
        origins = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(total_rays, z_start)
        ])
        # Directions: pointing down (-Z)
        directions = np.zeros((total_rays, 3))
        directions[:, 2] = -1.0

        print(f"[BackdraftView] Casting {total_rays} rays...")

        # Use multi_ray_trace with first_point=False for ALL intersections
        points, ind_ray, ind_tri = pv_mesh.multi_ray_trace(
            origins=origins,
            directions=directions,
            first_point=False
        )

        print(f"[BackdraftView] Ray casting complete, {len(points)} total intersections")

        # Count hits per ray
        hit_counts = np.zeros(total_rays, dtype=np.int32)
        if len(ind_ray) > 0:
            np.add.at(hit_counts, ind_ray, 1)
        hit_counts = hit_counts.reshape(ny, nx)

        return hit_counts


# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "GeomPackBackdraftView": BackdraftViewNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBackdraftView": "Backdraft View",
}
