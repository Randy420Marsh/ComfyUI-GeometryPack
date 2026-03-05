# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""Variational Shape Approximation face clustering + vertex snapping backend node."""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io
from ._helpers import _compute_face_geometry

log = logging.getLogger("geometrypack")


def _vsa_snap_sharpen(mesh, num_proxies, lloyd_iterations, snap_strength):
    """Variational Shape Approximation clustering with vertex snapping.

    Clusters mesh faces into proxy plane groups via Lloyd iteration (using
    area-weighted normal distance as the L2,1 metric), then snaps vertices:
    - Interior vertices (one proxy): project onto proxy plane
    - Edge vertices (two proxies): snap to plane intersection line
    - Corner vertices (3+ proxies): snap to plane intersection point
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)
    k = min(num_proxies, m)

    normals, centroids, areas = _compute_face_geometry(V, F)

    # --- Step 1: farthest-point seeding on area-weighted normal space ---
    seeds = [0]
    min_dists = np.full(m, np.inf)

    for _ in range(k - 1):
        last = seeds[-1]
        d = areas * np.linalg.norm(normals - normals[last], axis=1)
        min_dists = np.minimum(min_dists, d)
        seeds.append(int(np.argmax(min_dists)))

    proxy_normals = normals[seeds].copy()
    proxy_centroids = centroids[seeds].copy()

    # --- Step 2: Lloyd iteration ---
    labels = np.zeros(m, dtype=np.int32)

    for _ in range(lloyd_iterations):
        # Assign each face to nearest proxy (L2,1 metric)
        # (m, k) distance matrix
        normal_diffs = np.linalg.norm(
            normals[:, None, :] - proxy_normals[None, :, :], axis=2
        )
        weighted_diffs = areas[:, None] * normal_diffs
        labels = np.argmin(weighted_diffs, axis=1).astype(np.int32)

        # Refit proxies
        for p in range(k):
            mask = labels == p
            if not np.any(mask):
                # Empty cluster -- re-seed from farthest face
                assigned_dists = weighted_diffs[np.arange(m), labels]
                worst = int(np.argmax(assigned_dists))
                proxy_normals[p] = normals[worst]
                proxy_centroids[p] = centroids[worst]
                continue
            a = areas[mask]
            total_a = np.sum(a) + 1e-12
            avg_n = np.sum(normals[mask] * a[:, None], axis=0) / total_a
            norm_len = np.linalg.norm(avg_n)
            if norm_len > 1e-12:
                avg_n /= norm_len
            proxy_normals[p] = avg_n
            proxy_centroids[p] = np.sum(centroids[mask] * a[:, None], axis=0) / total_a

    # --- Step 3: snap vertices ---
    # Build vertex-to-face adjacency
    vertex_faces = [[] for _ in range(len(V))]
    for fi in range(m):
        for vi in F[fi]:
            vertex_faces[vi].append(fi)

    V_snapped = V.copy()

    for vi in range(len(V)):
        adj_fi = vertex_faces[vi]
        if not adj_fi:
            continue

        adj_labels = list(set(labels[fi] for fi in adj_fi))

        if len(adj_labels) == 1:
            # Interior: project onto single proxy plane
            p = adj_labels[0]
            n = proxy_normals[p]
            c = proxy_centroids[p]
            d = np.dot(V[vi] - c, n)
            V_snapped[vi] = V[vi] - snap_strength * d * n

        elif len(adj_labels) == 2:
            # Edge: snap to intersection of two planes
            p1, p2 = adj_labels
            n1, c1 = proxy_normals[p1], proxy_centroids[p1]
            n2, c2 = proxy_normals[p2], proxy_centroids[p2]

            line_dir = np.cross(n1, n2)
            line_dir_norm = np.linalg.norm(line_dir)

            if line_dir_norm < 1e-8:
                # Nearly parallel planes -- project onto average
                avg_n = (n1 + n2)
                avg_n_norm = np.linalg.norm(avg_n)
                if avg_n_norm > 1e-12:
                    avg_n /= avg_n_norm
                avg_c = (c1 + c2) / 2.0
                d = np.dot(V[vi] - avg_c, avg_n)
                V_snapped[vi] = V[vi] - snap_strength * d * avg_n
            else:
                line_dir /= line_dir_norm
                # Find a point on the intersection line
                A = np.array([n1, n2])
                b = np.array([np.dot(n1, c1), np.dot(n2, c2)])
                try:
                    AAT = A @ A.T
                    p0 = A.T @ np.linalg.solve(AAT, b)
                except np.linalg.LinAlgError:
                    continue
                # Project vertex onto the line
                v_rel = V[vi] - p0
                t = np.dot(v_rel, line_dir)
                closest = p0 + t * line_dir
                V_snapped[vi] = V[vi] + snap_strength * (closest - V[vi])

        else:
            # Corner: snap to intersection of 3+ planes
            A = np.array([proxy_normals[p] for p in adj_labels])
            b = np.array([np.dot(proxy_normals[p], proxy_centroids[p])
                          for p in adj_labels])
            try:
                if len(adj_labels) == 3:
                    corner = np.linalg.solve(A, b)
                else:
                    corner, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V_snapped[vi] = V[vi] + snap_strength * (corner - V[vi])
            except np.linalg.LinAlgError:
                continue

    result = trimesh_module.Trimesh(
        vertices=V_snapped,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


class SharpenVSASnapNode(io.ComfyNode):
    """Variational Shape Approximation face clustering + vertex snapping backend."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GeomPackSharpen_VSASnap",
            display_name="Sharpen VSA Snap (backend)",
            category="geompack/smoothing",
            is_dev_only=True,
            is_output_node=True,
            inputs=[
                io.Custom("TRIMESH").Input("trimesh"),
                io.Int.Input("num_proxies", default=20, min=2, max=500, step=1, tooltip=(
                    "Number of proxy planes (face clusters). More proxies "
                    "preserve more detail. Fewer proxies create a more "
                    "abstract/simplified appearance. 10-50 for simple shapes, "
                    "50-200 for complex shapes."
                )),
                io.Int.Input("lloyd_iterations", default=10, min=1, max=100, step=1, tooltip=(
                    "Number of Lloyd clustering iterations. 10 is usually "
                    "sufficient for convergence."
                )),
                io.Float.Input("snap_strength", default=1.0, min=0.0, max=1.0, step=0.01, tooltip=(
                    "How aggressively to snap vertices to proxy planes. "
                    "0.0 = no change, 1.0 = full snap. "
                    "Values like 0.5-0.8 give a subtle sharpening effect."
                )),
            ],
            outputs=[
                io.Custom("TRIMESH").Output(display_name="sharpened_mesh"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, trimesh, num_proxies=20, lloyd_iterations=10, snap_strength=1.0):
        log.info("Backend: vsa_snap")
        log.info("Input: %d vertices, %d faces", len(trimesh.vertices), len(trimesh.faces))
        log.info("Parameters: num_proxies=%d, lloyd_iter=%d, snap_strength=%.2f",
                 num_proxies, lloyd_iterations, snap_strength)

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        sharpened, error = _vsa_snap_sharpen(
            trimesh, num_proxies, lloyd_iterations, snap_strength,
        )

        if sharpened is None:
            raise ValueError(f"Sharpening failed (vsa_snap): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            sharpened.metadata = trimesh.metadata.copy()
        sharpened.metadata["sharpening"] = {
            "algorithm": "vsa_snap",
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
            f"Num Proxies: {num_proxies}\n"
            f"Lloyd Iterations: {lloyd_iterations}\n"
            f"Snap Strength: {snap_strength}"
        )

        info = f"""Sharpen Mesh Results (vsa_snap):

{param_text}

Vertices: {initial_vertices:,} (unchanged)
Faces: {initial_faces:,} (unchanged)

Displacement:
  Average: {avg_disp:.6f}
  Maximum: {max_disp:.6f}
"""
        return io.NodeOutput(sharpened, info, ui={"text": [info]})


NODE_CLASS_MAPPINGS = {"GeomPackSharpen_VSASnap": SharpenVSASnapNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeomPackSharpen_VSASnap": "Sharpen VSA Snap (backend)"}
