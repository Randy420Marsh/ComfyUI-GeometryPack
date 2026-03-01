# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Sharpen Mesh Node - Edge-recovering sharpening for CAD reconstruction prep.

Available backends:
- two_step: Two-phase bilateral normal filtering via pymeshlab. Smooths face
  normals (respecting dihedral angle thresholds), then repositions vertices to
  match. Sharpens creases while keeping faces flat. Best for CAD-like geometry
  from marching cubes, scanning, or neural SDF extraction.
- unsharp_mask: Geometric unsharp masking via pymeshlab. Subtracts a smoothed
  version from the original to amplify ridges and valleys.
- libigl_unsharp: Cotangent-weighted geometric unsharp mask via libigl.
  Geometrically superior to uniform-weight unsharp masking because cotangent
  Laplacian respects mesh geometry (triangle shape/area).
- l0_minimize: L0 normal minimization (He & Schaefer 2013). Minimizes the
  number of distinct face normal orientations, forcing the mesh into
  piecewise-flat regions with sharp edges at boundaries. Best for aggressive
  CAD-like sharpening.
- guided_normal: Guided mesh normal filtering (Zhang et al. 2015). Uses a
  min-range-metric guidance signal to drive bilateral normal filtering while
  preserving sharp edges. Interleaves vertex updates within normal iterations.
- vsa_snap: Variational Shape Approximation face clustering with vertex
  snapping. Clusters faces into proxy plane groups via Lloyd iteration, then
  snaps vertices to proxy planes (interior), plane intersection lines (edges),
  or plane intersection points (corners).
- fast_effective: Fast and Effective Feature-Preserving Mesh Denoising
  (Sun et al. TVCG 2007). Uses thresholded cosine-similarity weights for
  normal filtering: w = max(0, dot(ni,nj) - T)^2. Simple and fast.
- non_iterative: Non-Iterative Feature-Preserving Mesh Smoothing
  (Jones et al. SIGGRAPH 2003). Mollifies normals on a smoothed mesh copy,
  then does a single-pass bilateral vertex update using spatial and influence
  Gaussian weights with BFS face neighbor search.
"""

import logging
import numpy as np
import trimesh as trimesh_module
from comfy_api.latest import io

log = logging.getLogger("geometrypack")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _compute_face_geometry(V, F):
    """Compute face normals, centroids, and areas from vertex/face arrays.

    Returns:
        normals: (m, 3) unit face normals
        centroids: (m, 3) face centroids
        areas: (m,) face areas
    """
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    cross = np.cross(e1, e2)
    area_2x = np.linalg.norm(cross, axis=1, keepdims=True)
    normals = cross / (area_2x + 1e-12)
    areas = area_2x.ravel() * 0.5
    centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    return normals, centroids, areas


def _update_vertices_from_normals(V, F, target_normals, vertex_iterations,
                                  fixed_boundary=False):
    """Update vertex positions to match target face normals via iterative
    projection. For each iteration, projects each vertex onto the planes
    defined by the target normals of its adjacent faces, then averages.

    Matches MeshDenoisingBase::updateVertexPosition from the C++ reference:
    p += (1/N) * sum_j n_j * dot(n_j, c_j - p)

    Args:
        fixed_boundary: If True, boundary vertices are kept in place.

    Returns:
        V_new: (n, 3) updated vertex positions
    """
    V = V.copy()
    n_verts = len(V)
    m_faces = len(F)

    # Detect boundary vertices if needed
    if fixed_boundary:
        edge_face_count = {}
        for fi in range(m_faces):
            for i in range(3):
                e = (min(F[fi][i], F[fi][(i + 1) % 3]),
                     max(F[fi][i], F[fi][(i + 1) % 3]))
                edge_face_count[e] = edge_face_count.get(e, 0) + 1
        is_boundary = np.zeros(n_verts, dtype=bool)
        for (v0, v1), cnt in edge_face_count.items():
            if cnt == 1:
                is_boundary[v0] = True
                is_boundary[v1] = True
    else:
        is_boundary = None

    for _ in range(vertex_iterations):
        new_V = np.zeros_like(V)
        counts = np.zeros(n_verts)

        # Compute current centroids
        centroids = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0

        for fi in range(m_faces):
            c = centroids[fi]
            n = target_normals[fi]
            for vi in F[fi]:
                if is_boundary is not None and is_boundary[vi]:
                    continue
                d = np.dot(V[vi] - c, n)
                new_V[vi] += V[vi] - d * n
                counts[vi] += 1

        mask = counts > 0
        new_V[mask] /= counts[mask, None]
        new_V[~mask] = V[~mask]
        V = new_V

    return V


def _build_vertex_to_faces(n_verts, F):
    """Build vertex-to-face adjacency list. Returns list of lists."""
    vtf = [[] for _ in range(n_verts)]
    for fi in range(len(F)):
        for vi in F[fi]:
            vtf[vi].append(fi)
    return vtf


def _build_vertex_based_face_neighbors(F, vert_to_faces, include_central=True):
    """Build vertex-based face neighbor lists.

    Two faces are neighbors if they share at least one vertex.
    Returns list of sorted index lists, one per face.
    """
    m = len(F)
    neighbors = []
    for fi in range(m):
        nbrs = set()
        for vi in F[fi]:
            for fj in vert_to_faces[vi]:
                nbrs.add(fj)
        if not include_central:
            nbrs.discard(fi)
        neighbors.append(sorted(nbrs))
    return neighbors


# ---------------------------------------------------------------------------
# Backend: pymeshlab two-step bilateral normal sharpening
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backend: pymeshlab geometric unsharp mask
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Backend: libigl cotangent-weighted unsharp mask
# ---------------------------------------------------------------------------

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

    # M is diagonal — invert via element-wise reciprocal
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


# ---------------------------------------------------------------------------
# Backend: L0 normal minimization (He & Schaefer 2013)
# ---------------------------------------------------------------------------

def _l0_minimize_sharpen(mesh, alpha, beta, iterations):
    """L0 normal minimization for piecewise-flat sharpening.

    Iteratively thresholds face normal differences across edges: if the
    difference is below the current alpha threshold, adjacent normals are
    snapped to their area-weighted average. Alpha grows by factor beta each
    iteration, progressively eliminating small variations and forcing the
    mesh into piecewise-constant normal regions with sharp edges at
    boundaries.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    adj_pairs = np.asarray(mesh.face_adjacency)
    if len(adj_pairs) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate)."

    normals, centroids, areas = _compute_face_geometry(V, F)
    current_alpha = alpha

    for it in range(iterations):
        # Recompute normals from current vertex positions
        normals, centroids, areas = _compute_face_geometry(V, F)
        target_normals = normals.copy()

        # Threshold: snap adjacent normals whose difference^2 < current_alpha
        for ei in range(len(adj_pairs)):
            fi, fj = adj_pairs[ei]
            ni, nj = target_normals[fi], target_normals[fj]
            diff_sq = np.sum((ni - nj) ** 2)
            if diff_sq < current_alpha:
                ai, aj = areas[fi], areas[fj]
                avg = (ai * ni + aj * nj) / (ai + aj + 1e-12)
                norm_len = np.linalg.norm(avg)
                if norm_len > 1e-12:
                    avg /= norm_len
                target_normals[fi] = avg
                target_normals[fj] = avg

        # Update vertex positions to match target normals
        V = _update_vertices_from_normals(V, F, target_normals, vertex_iterations=1)

        current_alpha *= beta
        log.debug("L0 iteration %d: alpha=%.6f", it, current_alpha)

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


# ---------------------------------------------------------------------------
# Backend: guided bilateral normal filtering + vertex update
# ---------------------------------------------------------------------------

def _guided_normal_sharpen(mesh, normal_iterations, vertex_iterations,
                           sigma_s, sigma_r):
    """Guided mesh normal filtering with interleaved vertex update.

    Matches GuidedMeshNormalFiltering::updateFilteredNormalsLocalScheme from
    the C++ reference (Zhang et al. 2015).

    Per iteration:
    1. Recompute geometry from current vertices
    2. Compute guidance normals via min-range-metric: for each face's
       vertex-based 1-ring, compute a sharpness metric (maxdiff * max_tv /
       sum_tv). Pick the neighbor with minimum metric and use its area-weighted
       average normal as guidance.
    3. Bilateral filter normals using guidance for range weight
    4. Immediately update vertex positions (interleaved)
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)
    adj_pairs = np.asarray(mesh.face_adjacency)
    if len(adj_pairs) == 0:
        return None, "Mesh has no face adjacency (disconnected or degenerate)."

    # Build vertex-based face neighbor structures
    vert_to_faces = _build_vertex_to_faces(len(V), F)
    # Guided neighborhood: vertex-based including central face
    guided_neighbors = _build_vertex_based_face_neighbors(F, vert_to_faces, include_central=True)
    # Filtering neighborhood: same vertex-based including central
    filter_neighbors = guided_neighbors

    # Precompute face-to-adjacency mapping for inner edge computation
    face_to_adj = [[] for _ in range(m)]
    for ai in range(len(adj_pairs)):
        fi, fj = adj_pairs[ai]
        face_to_adj[fi].append(ai)
        face_to_adj[fj].append(ai)

    for normal_iter in range(normal_iterations):
        # Recompute geometry from current vertex positions
        normals, centroids, areas = _compute_face_geometry(V, F)

        # Compute sigma_s in absolute units (avg centroid-to-centroid distance * multiple)
        edge_lens = np.concatenate([
            np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
            np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
            np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1),
        ])
        avg_edge_len = float(np.mean(edge_lens))
        sigma_s_abs = sigma_s * avg_edge_len
        sigma_s_sq2 = 2.0 * sigma_s_abs * sigma_s_abs
        sigma_r_sq2 = 2.0 * sigma_r * sigma_r

        # --- Step 1: Compute guidance normals via min-range-metric ---
        # For each face compute (metric, area_weighted_avg_normal) over its
        # guided neighborhood, matching getRangeAndMeanNormal in the C++ code.
        metrics = np.zeros(m)
        avg_normals = np.zeros((m, 3))

        for fi in range(m):
            patch = guided_neighbors[fi]
            n_patch = len(patch)

            # Area-weighted average normal
            patch_areas = areas[patch]
            patch_normals = normals[patch]
            avg_n = np.sum(patch_normals * patch_areas[:, None], axis=0)
            norm_len = np.linalg.norm(avg_n)
            if norm_len > 1e-12:
                avg_n /= norm_len
            avg_normals[fi] = avg_n

            # Max pairwise normal difference in patch
            if n_patch > 1:
                pn = normals[patch]
                diffs = np.linalg.norm(pn[:, None, :] - pn[None, :, :], axis=2)
                maxdiff = float(np.max(diffs))
            else:
                maxdiff = 0.0

            # Inner edges: adjacency pairs where both faces are in the patch
            patch_set = set(patch)
            max_tv = 0.0
            sum_tv = 0.0
            seen_edges = set()
            for pf in patch:
                for ai in face_to_adj[pf]:
                    if ai not in seen_edges:
                        seen_edges.add(ai)
                        fa, fb = adj_pairs[ai]
                        if fa in patch_set and fb in patch_set:
                            tv = float(np.linalg.norm(normals[fa] - normals[fb]))
                            if tv > max_tv:
                                max_tv = tv
                            sum_tv += tv

            metrics[fi] = maxdiff * max_tv / (sum_tv + 1e-9)

        # For each face, find the guided neighbor with minimum metric and use
        # that neighbor's area-weighted average normal as guidance.
        guided_normals = np.zeros((m, 3))
        for fi in range(m):
            patch = guided_neighbors[fi]
            min_metric = 1e8
            min_idx = patch[0]
            for pf in patch:
                if metrics[pf] < min_metric:
                    min_metric = metrics[pf]
                    min_idx = pf
            guided_normals[fi] = avg_normals[min_idx]

        # --- Step 2: Bilateral filter normals using guidance ---
        filtered = np.zeros((m, 3))
        for fi in range(m):
            patch = filter_neighbors[fi]
            if not patch:
                filtered[fi] = normals[fi]
                continue

            w_total = 0.0
            n_acc = np.zeros(3)
            for fj in patch:
                dist_sq = float(np.sum((centroids[fi] - centroids[fj]) ** 2))
                ws = np.exp(-dist_sq / (sigma_s_sq2 + 1e-12))

                # Range weight uses guidance normal distance (matching C++ reference)
                gdiff_sq = float(np.sum((guided_normals[fi] - guided_normals[fj]) ** 2))
                wr = np.exp(-gdiff_sq / (sigma_r_sq2 + 1e-12))

                w = areas[fj] * ws * wr
                n_acc += normals[fj] * w
                w_total += w

            if w_total > 1e-12:
                filtered[fi] = n_acc / w_total
            else:
                filtered[fi] = normals[fi]

        f_norms = np.linalg.norm(filtered, axis=1, keepdims=True)
        filtered_normals = filtered / (f_norms + 1e-12)

        # --- Step 3: Interleaved vertex update (matching C++ reference) ---
        V = _update_vertices_from_normals(V, F, filtered_normals, vertex_iterations)

        log.debug("Guided normal iteration %d/%d complete", normal_iter + 1, normal_iterations)

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


# ---------------------------------------------------------------------------
# Backend: Variational Shape Approximation face clustering + vertex snapping
# ---------------------------------------------------------------------------

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
                # Empty cluster — re-seed from farthest face
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
                # Nearly parallel planes — project onto average
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


# ---------------------------------------------------------------------------
# Backend: Fast and Effective Feature-Preserving Mesh Denoising
# (Sun et al. TVCG 2007)
# ---------------------------------------------------------------------------

def _fast_effective_sharpen(mesh, threshold_T, normal_iterations, vertex_iterations):
    """Fast and Effective Feature-Preserving Mesh Denoising.

    Ported from FastAndEffectiveFeaturePreservingMeshDenoising.cpp.

    Normal filtering weight: w = max(0, dot(ni, nj) - T)^2
    where T is a threshold controlling which normals contribute.
    Faces with normals more similar than T contribute with quadratic weight;
    faces with normals less similar than T contribute nothing. This produces
    sharp edges at dihedral angles corresponding to the threshold.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)

    # Build vertex-based face neighbors (including central face, matching C++)
    vert_to_faces = _build_vertex_to_faces(len(V), F)
    all_face_neighbor = _build_vertex_based_face_neighbors(
        F, vert_to_faces, include_central=True)

    normals, _, _ = _compute_face_geometry(V, F)
    previous_normals = normals.copy()

    # Iterative normal filtering
    for it in range(normal_iterations):
        filtered = np.zeros((m, 3))
        for fi in range(m):
            ni = previous_normals[fi]
            temp_normal = np.zeros(3)
            for fj in all_face_neighbor[fi]:
                nj = previous_normals[fj]
                value = float(np.dot(ni, nj)) - threshold_T
                weight = value * value if value > 0.0 else 0.0
                temp_normal += nj * weight
            norm_len = np.linalg.norm(temp_normal)
            if norm_len > 1e-12:
                temp_normal /= norm_len
            filtered[fi] = temp_normal
        previous_normals = filtered
        log.debug("Fast effective normal iter %d/%d", it + 1, normal_iterations)

    # Update vertex positions (with fixed boundary matching C++ reference)
    V = _update_vertices_from_normals(V, F, previous_normals, vertex_iterations,
                                      fixed_boundary=True)

    result = trimesh_module.Trimesh(
        vertices=V,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


# ---------------------------------------------------------------------------
# Backend: Non-Iterative, Feature-Preserving Mesh Smoothing
# (Jones et al. SIGGRAPH 2003)
# ---------------------------------------------------------------------------

def _non_iterative_sharpen(mesh, sigma_f_ratio, sigma_g_ratio):
    """Non-Iterative Feature-Preserving Mesh Smoothing.

    Ported from NonIterativeFeaturePreservingMeshFiltering.cpp.

    1. Mollify normals: smooth vertex positions with Gaussian (sigma_m = sigma_f/2),
       then recompute face normals on the smoothed mesh.
    2. For each vertex, BFS-expand to find faces within radius 2*sigma_f.
    3. Project vertex onto each neighbor face plane (using mollified normal +
       original centroid). Weight by area * spatial_Gaussian(sigma_f) *
       influence_Gaussian(sigma_g). Average projections.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    if len(V) == 0 or len(F) == 0:
        return None, "Empty mesh (no vertices or faces)."

    m = len(F)
    n = len(V)

    normals, centroids, areas = _compute_face_geometry(V, F)

    # Compute average edge length
    edge_lens = np.concatenate([
        np.linalg.norm(V[F[:, 1]] - V[F[:, 0]], axis=1),
        np.linalg.norm(V[F[:, 2]] - V[F[:, 1]], axis=1),
        np.linalg.norm(V[F[:, 0]] - V[F[:, 2]], axis=1),
    ])
    avg_edge_len = float(np.mean(edge_lens))

    sigma_f = sigma_f_ratio * avg_edge_len
    sigma_g = sigma_g_ratio * avg_edge_len
    sigma_m = sigma_f / 2.0
    radius = 2.0 * sigma_f

    # Build vertex-to-face adjacency
    vert_to_faces = _build_vertex_to_faces(n, F)

    # Build vertex-based face adjacency for BFS expansion (matching C++ kVertexBased)
    vbased_face_adj = _build_vertex_based_face_neighbors(
        F, vert_to_faces, include_central=False)

    def _bfs_faces_in_radius(vertex_pos, seed_faces, rad):
        """BFS from seed faces, expanding via vertex-based neighbors within radius."""
        visited = set()
        queue = []
        for fi in seed_faces:
            dist = float(np.linalg.norm(vertex_pos - centroids[fi]))
            if dist <= rad:
                queue.append(fi)
            visited.add(fi)
        result_faces = []
        head = 0
        while head < len(queue):
            fi = queue[head]
            result_faces.append(fi)
            head += 1
            for fj in vbased_face_adj[fi]:
                if fj not in visited:
                    visited.add(fj)
                    dist = float(np.linalg.norm(vertex_pos - centroids[fj]))
                    if dist <= rad:
                        queue.append(fj)
        return result_faces

    # --- Step 1: Mollified normals ---
    # Smooth vertex positions with area-weighted Gaussian (sigma_m)
    V_smooth = V.copy()
    for vi in range(n):
        pt = V[vi]
        seed_faces = vert_to_faces[vi]
        if not seed_faces:
            continue

        faces_in_range = _bfs_faces_in_radius(pt, seed_faces, radius)

        new_pt = np.zeros(3)
        w_sum = 0.0
        for fi in faces_in_range:
            c = centroids[fi]
            dist = float(np.linalg.norm(c - pt))
            w = np.exp(-0.5 * dist * dist / (sigma_m * sigma_m + 1e-12))
            a = areas[fi]
            new_pt += c * a * w
            w_sum += a * w

        if w_sum > 1e-12:
            V_smooth[vi] = new_pt / w_sum

    # Recompute normals on the smoothed mesh
    mollified_normals, _, _ = _compute_face_geometry(V_smooth, F)

    # --- Step 2: Single-pass bilateral vertex update ---
    V_new = V.copy()
    sigma_f_sq = sigma_f * sigma_f
    sigma_g_sq = sigma_g * sigma_g

    for vi in range(n):
        pt = V[vi]
        seed_faces = vert_to_faces[vi]
        if not seed_faces:
            continue

        faces_in_range = _bfs_faces_in_radius(pt, seed_faces, radius)
        if not faces_in_range:
            continue

        temp_pt = np.zeros(3)
        w_sum = 0.0
        for fi in faces_in_range:
            c = centroids[fi]
            mn = mollified_normals[fi]

            # Project vertex onto face plane (mollified normal + original centroid)
            proj = pt - mn * float(np.dot(pt - c, mn))

            dist_spatial = float(np.linalg.norm(c - pt))
            w_spatial = np.exp(-0.5 * dist_spatial * dist_spatial / (sigma_f_sq + 1e-12))

            dist_influence = float(np.linalg.norm(proj - pt))
            w_influence = np.exp(-0.5 * dist_influence * dist_influence / (sigma_g_sq + 1e-12))

            a = areas[fi]
            temp_pt += proj * a * w_spatial * w_influence
            w_sum += a * w_spatial * w_influence

        if w_sum > 1e-12:
            V_new[vi] = temp_pt / w_sum

    result = trimesh_module.Trimesh(
        vertices=V_new,
        faces=np.asarray(mesh.faces, dtype=np.int32),
        process=False,
    )
    return result, ""


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class SharpenMeshNode(io.ComfyNode):
    """
    Sharpen Mesh - Edge-recovering sharpening algorithms for CAD reconstruction prep.

    Available backends:
    - two_step: Bilateral normal filtering then vertex repositioning (pymeshlab).
    - unsharp_mask: Geometric unsharp masking (pymeshlab).
    - libigl_unsharp: Cotangent-weighted geometric unsharp mask (libigl).
    - l0_minimize: L0 normal minimization (He & Schaefer 2013).
    - guided_normal: Guided mesh normal filtering (Zhang et al. 2015).
    - vsa_snap: Variational Shape Approximation face clustering + vertex snapping.
    - fast_effective: Thresholded cosine-weight normal filtering (Sun et al. 2007).
    - non_iterative: Mollified-normal single-pass bilateral filtering (Jones et al. 2003).
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
                        "unsharp_mask=geometric unsharp masking (pymeshlab), "
                        "libigl_unsharp=cotangent-weighted unsharp (geometry-aware), "
                        "l0_minimize=piecewise-flat L0 optimization (aggressive CAD prep), "
                        "guided_normal=guided normal filtering with min-range-metric (controllable), "
                        "vsa_snap=face clustering + vertex snapping (explicit patch control), "
                        "fast_effective=thresholded cosine weight normal filtering (fast), "
                        "non_iterative=mollified normal single-pass bilateral (non-iterative)"
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
                    io.DynamicCombo.Option("libigl_unsharp", [
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
                    ]),
                    io.DynamicCombo.Option("l0_minimize", [
                        io.Float.Input("alpha", default=0.001, min=0.0001, max=0.1, step=0.0001, tooltip=(
                            "Initial regularization weight for L0 minimization. "
                            "Controls the threshold below which normal differences "
                            "are snapped to zero. Smaller = gentler start, "
                            "larger = more aggressive initial flattening."
                        )),
                        io.Float.Input("beta", default=2.0, min=1.1, max=10.0, step=0.1, tooltip=(
                            "Growth rate for alpha each iteration. Alpha is multiplied "
                            "by beta each step. 2.0 doubles per iteration. "
                            "Higher = faster convergence to piecewise-flat."
                        )),
                        io.Int.Input("iterations", default=10, min=1, max=50, step=1, tooltip=(
                            "Number of L0 optimization iterations. The algorithm "
                            "gradually increases the threshold, snapping more normals "
                            "flat each step."
                        )),
                    ]),
                    io.DynamicCombo.Option("guided_normal", [
                        io.Int.Input("normal_iterations", default=5, min=1, max=50, step=1, tooltip=(
                            "Iterations for guided bilateral normal filtering. "
                            "More iterations produce smoother/flatter regions while "
                            "preserving sharp edges."
                        )),
                        io.Int.Input("vertex_iterations", default=10, min=1, max=100, step=1, tooltip=(
                            "Iterations for updating vertex positions to match filtered "
                            "normals. More iterations give better convergence."
                        )),
                        io.Float.Input("sigma_s", default=1.0, min=0.1, max=10.0, step=0.1, tooltip=(
                            "Spatial weight sigma as a multiple of average edge length. "
                            "Controls the neighborhood size for normal filtering. "
                            "Larger = smoother but may blur sharp features."
                        )),
                        io.Float.Input("sigma_r", default=0.35, min=0.01, max=1.0, step=0.01, tooltip=(
                            "Normal similarity threshold. Controls which normals are "
                            "averaged together. Smaller = more aggressive edge "
                            "preservation. 0.35 corresponds to roughly 40 degree "
                            "dihedral angle threshold."
                        )),
                    ]),
                    io.DynamicCombo.Option("vsa_snap", [
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
                    ]),
                    io.DynamicCombo.Option("fast_effective", [
                        io.Float.Input("threshold_T", default=0.5, min=1e-10, max=1.0, step=0.01, tooltip=(
                            "Cosine similarity threshold (Sun et al. TVCG 2007). "
                            "Normals with dot(ni,nj) > T contribute with weight "
                            "(dot-T)^2; below T they contribute nothing. "
                            "Lower = more normals averaged (smoother), "
                            "higher = only very similar normals averaged (sharper). "
                            "0.5 is a good default."
                        )),
                        io.Int.Input("normal_iterations", default=20, min=1, max=500, step=1, tooltip=(
                            "Iterations for normal filtering. More iterations "
                            "produce stronger flattening of near-flat regions."
                        )),
                        io.Int.Input("vertex_iterations", default=50, min=1, max=500, step=1, tooltip=(
                            "Iterations for vertex position update from filtered "
                            "normals. Boundary vertices are kept fixed."
                        )),
                    ]),
                    io.DynamicCombo.Option("non_iterative", [
                        io.Float.Input("sigma_f", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                            "Spatial sigma as multiple of average edge length "
                            "(Jones et al. SIGGRAPH 2003). Controls spatial extent "
                            "of the bilateral filter. Face neighbors are searched "
                            "within radius 2*sigma_f. Larger = smoother."
                        )),
                        io.Float.Input("sigma_g", default=1.0, min=0.001, max=10.0, step=0.1, tooltip=(
                            "Influence sigma as multiple of average edge length. "
                            "Controls sensitivity to projection distance (how far "
                            "the vertex moves toward each face plane). Smaller = "
                            "more feature-preserving."
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
        elif selected == "libigl_unsharp":
            lambda_ = backend.get("lambda_", 0.5)
            iterations = backend.get("iterations", 3)
            log.info("Parameters: lambda=%.3f, iterations=%d", lambda_, iterations)
            sharpened, error = _libigl_unsharp_sharpen(
                trimesh, lambda_, iterations,
            )
        elif selected == "l0_minimize":
            alpha = backend.get("alpha", 0.001)
            beta_val = backend.get("beta", 2.0)
            iterations = backend.get("iterations", 10)
            log.info("Parameters: alpha=%.4f, beta=%.1f, iterations=%d",
                     alpha, beta_val, iterations)
            sharpened, error = _l0_minimize_sharpen(
                trimesh, alpha, beta_val, iterations,
            )
        elif selected == "guided_normal":
            normal_iterations = backend.get("normal_iterations", 5)
            vertex_iterations = backend.get("vertex_iterations", 10)
            sigma_s = backend.get("sigma_s", 1.0)
            sigma_r = backend.get("sigma_r", 0.35)
            log.info("Parameters: normal_iter=%d, vertex_iter=%d, sigma_s=%.2f, sigma_r=%.3f",
                     normal_iterations, vertex_iterations, sigma_s, sigma_r)
            sharpened, error = _guided_normal_sharpen(
                trimesh, normal_iterations, vertex_iterations, sigma_s, sigma_r,
            )
        elif selected == "vsa_snap":
            num_proxies = backend.get("num_proxies", 20)
            lloyd_iterations = backend.get("lloyd_iterations", 10)
            snap_strength = backend.get("snap_strength", 1.0)
            log.info("Parameters: num_proxies=%d, lloyd_iter=%d, snap_strength=%.2f",
                     num_proxies, lloyd_iterations, snap_strength)
            sharpened, error = _vsa_snap_sharpen(
                trimesh, num_proxies, lloyd_iterations, snap_strength,
            )
        elif selected == "fast_effective":
            threshold_T = backend.get("threshold_T", 0.5)
            normal_iterations = backend.get("normal_iterations", 20)
            vertex_iterations = backend.get("vertex_iterations", 50)
            log.info("Parameters: threshold_T=%.3f, normal_iter=%d, vertex_iter=%d",
                     threshold_T, normal_iterations, vertex_iterations)
            sharpened, error = _fast_effective_sharpen(
                trimesh, threshold_T, normal_iterations, vertex_iterations,
            )
        elif selected == "non_iterative":
            sigma_f = backend.get("sigma_f", 1.0)
            sigma_g = backend.get("sigma_g", 1.0)
            log.info("Parameters: sigma_f=%.3f, sigma_g=%.3f", sigma_f, sigma_g)
            sharpened, error = _non_iterative_sharpen(
                trimesh, sigma_f, sigma_g,
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
        elif selected == "libigl_unsharp":
            param_text = (
                f"Lambda: {lambda_}\n"
                f"Iterations: {iterations}"
            )
        elif selected == "l0_minimize":
            param_text = (
                f"Alpha: {alpha}\n"
                f"Beta: {beta_val}\n"
                f"Iterations: {iterations}"
            )
        elif selected == "guided_normal":
            param_text = (
                f"Normal Iterations: {normal_iterations}\n"
                f"Vertex Iterations: {vertex_iterations}\n"
                f"Sigma S: {sigma_s}\n"
                f"Sigma R: {sigma_r}"
            )
        elif selected == "vsa_snap":
            param_text = (
                f"Num Proxies: {num_proxies}\n"
                f"Lloyd Iterations: {lloyd_iterations}\n"
                f"Snap Strength: {snap_strength}"
            )
        elif selected == "fast_effective":
            param_text = (
                f"Threshold T: {threshold_T}\n"
                f"Normal Iterations: {normal_iterations}\n"
                f"Vertex Iterations: {vertex_iterations}"
            )
        elif selected == "non_iterative":
            param_text = (
                f"Sigma F: {sigma_f}\n"
                f"Sigma G: {sigma_g}"
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
