# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Decimate Node - Multiple mesh decimation/simplification backends.

Available backends:
- quadric_edge_collapse: PyMeshLab Garland-Heckbert quadric error (best quality)
- fast_simplification: Fast-Quadric-Mesh-Simplification (~10x faster)
- vertex_clustering: PyMeshLab vertex clustering (fastest, aggressive)
- cgal_edge_collapse: CGAL Lindstrom-Turk edge collapse (high geometric fidelity)
- decimate_pro: PyVista/VTK DecimatePro (topology-preserving)
"""

import logging
import numpy as np
import trimesh as trimesh_module

log = logging.getLogger("geometrypack")


def _pymeshlab_quadric_edge_collapse(mesh, target_face_count, quality_threshold,
                                     preserve_boundary, preserve_normal,
                                     preserve_topology, planar_quadric):
    """Quadric edge collapse decimation via PyMeshLab."""
    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed."

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    kwargs = {
        "targetfacenum": target_face_count,
        "qualitythr": quality_threshold,
        "preserveboundary": preserve_boundary,
        "preservenormal": preserve_normal,
        "preservetopology": preserve_topology,
        "planarquadric": planar_quadric,
        "autoclean": True,
    }

    try:
        ms.meshing_decimation_quadric_edge_collapse(**kwargs)
    except AttributeError:
        try:
            ms.simplification_quadric_edge_collapse_decimation(**kwargs)
        except AttributeError:
            return None, (
                "PyMeshLab quadric edge collapse filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""


def _fast_simplification_decimate(mesh, target_reduction, agg):
    """Fast quadric mesh simplification."""
    try:
        import fast_simplification
    except (ImportError, OSError):
        return None, "fast-simplification is not installed."

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)

    v_out, f_out = fast_simplification.simplify(
        v, f,
        target_reduction=target_reduction,
        agg=agg,
    )

    result = trimesh_module.Trimesh(
        vertices=v_out,
        faces=f_out,
        process=False,
    )
    return result, ""


def _pymeshlab_vertex_clustering(mesh, threshold_percentage):
    """Vertex clustering decimation via PyMeshLab."""
    try:
        import pymeshlab
    except (ImportError, OSError):
        return None, "pymeshlab is not installed."

    ms = pymeshlab.MeshSet()
    pml_mesh = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pml_mesh)

    try:
        ms.meshing_decimation_clustering(
            threshold=pymeshlab.PercentageValue(threshold_percentage),
        )
    except AttributeError:
        try:
            ms.simplification_clustering_decimation(
                threshold=pymeshlab.PercentageValue(threshold_percentage),
            )
        except AttributeError:
            return None, (
                "PyMeshLab vertex clustering filter not available. "
                "Ensure pymeshlab >= 2023.12 is installed."
            )

    out = ms.current_mesh()
    result = trimesh_module.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )
    return result, ""


def _cgal_edge_collapse(mesh, target_edge_count, cost_strategy):
    """CGAL surface mesh simplification via edge collapse."""
    try:
        from CGAL import CGAL_Surface_mesh_simplification
        from CGAL.CGAL_Kernel import Point_3
        from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
        from CGAL import CGAL_Polygon_mesh_processing
    except (ImportError, OSError):
        return None, "CGAL Python bindings not available."

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if len(vertices) == 0 or len(faces) == 0:
        return None, "Mesh is empty"

    # Build CGAL Polyhedron_3
    points = CGAL_Polygon_mesh_processing.Point_3_Vector()
    points.reserve(len(vertices))
    for v in vertices:
        points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))

    polygons = [[int(idx) for idx in face] for face in faces]

    P = Polyhedron_3()
    CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, P)

    # Choose cost strategy
    if cost_strategy == "lindstrom_turk":
        cost = CGAL_Surface_mesh_simplification.LindstromTurk_cost()
        placement = CGAL_Surface_mesh_simplification.LindstromTurk_placement()
    else:
        # edge_length cost
        cost = CGAL_Surface_mesh_simplification.Edge_length_cost()
        placement = CGAL_Surface_mesh_simplification.Midpoint_placement()

    stop = CGAL_Surface_mesh_simplification.Count_stop_predicate(target_edge_count)

    CGAL_Surface_mesh_simplification.edge_collapse(P, stop, cost, placement)

    # Extract result
    new_vertices = []
    vertex_map = {}
    for i, vertex in enumerate(P.vertices()):
        point = vertex.point()
        new_vertices.append([float(point.x()), float(point.y()), float(point.z())])
        vertex_map[vertex] = i

    new_faces = []
    for facet in P.facets():
        halfedge = facet.halfedge()
        face_vertices = []
        start = halfedge
        current = start
        while True:
            face_vertices.append(vertex_map[current.vertex()])
            current = current.next()
            if current == start:
                break
        if len(face_vertices) == 3:
            new_faces.append(face_vertices)

    result = trimesh_module.Trimesh(
        vertices=np.array(new_vertices, dtype=np.float64),
        faces=np.array(new_faces, dtype=np.int32),
        process=False,
    )
    return result, ""


def _pyvista_decimate_pro(mesh, target_reduction, preserve_topology, feature_angle):
    """VTK DecimatePro via PyVista."""
    try:
        import pyvista as pv
    except (ImportError, OSError):
        return None, "pyvista is not installed."

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces_raw = np.asarray(mesh.faces, dtype=np.int32)

    # PyVista face format: [3, v0, v1, v2, 3, v0, v1, v2, ...]
    n_faces = len(faces_raw)
    pv_faces = np.empty((n_faces, 4), dtype=np.int32)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces_raw
    pv_faces = pv_faces.ravel()

    pv_mesh = pv.PolyData(vertices, pv_faces)

    decimated = pv_mesh.decimate_pro(
        target_reduction,
        preserve_topology=preserve_topology,
        feature_angle=feature_angle,
    )

    # Extract back
    out_verts = np.array(decimated.points, dtype=np.float64)
    out_faces_pv = decimated.faces.reshape(-1, 4)[:, 1:]

    result = trimesh_module.Trimesh(
        vertices=out_verts,
        faces=out_faces_pv,
        process=False,
    )
    return result, ""


class DecimateMeshNode:
    """
    Decimate Mesh - Reduce mesh face/vertex count using multiple algorithms.

    Available backends:
    - quadric_edge_collapse: Garland-Heckbert quadric error metric (PyMeshLab).
      Best quality, most options. Industry standard.
    - fast_simplification: Fast Quadric Mesh Simplification.
      ~10x faster than PyMeshLab, slightly lower quality.
    - vertex_clustering: Group nearby vertices (PyMeshLab).
      Very fast, aggressive reduction. Good for massive meshes.
    - cgal_edge_collapse: CGAL Lindstrom-Turk edge collapse.
      Highest geometric fidelity, optimizes volume + boundary.
    - decimate_pro: VTK DecimatePro (PyVista).
      Topology-preserving vertex deletion with local re-triangulation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "backend": ([
                    "quadric_edge_collapse",
                    "fast_simplification",
                    "vertex_clustering",
                    "cgal_edge_collapse",
                    "decimate_pro",
                ], {
                    "default": "quadric_edge_collapse",
                    "tooltip": (
                        "Decimation algorithm. "
                        "quadric_edge_collapse=best quality, "
                        "fast_simplification=fastest, "
                        "vertex_clustering=aggressive, "
                        "cgal_edge_collapse=highest fidelity, "
                        "decimate_pro=topology-preserving"
                    ),
                }),
            },
            "optional": {
                # --- Shared: target face count (quadric, cgal) ---
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 4,
                    "max": 10000000,
                    "step": 100,
                    "tooltip": "Target number of output faces.",
                    "visible_when": {"backend": [
                        "quadric_edge_collapse", "cgal_edge_collapse",
                    ]},
                }),
                # --- Shared: target reduction ratio (fast_simplification, decimate_pro) ---
                "target_reduction": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.01,
                    "max": 0.99,
                    "step": 0.01,
                    "tooltip": (
                        "Fraction of faces to REMOVE. "
                        "0.5 = reduce to ~50%% of original faces, "
                        "0.9 = reduce to ~10%% of original."
                    ),
                    "visible_when": {"backend": [
                        "fast_simplification", "decimate_pro",
                    ]},
                }),
                # --- quadric_edge_collapse specific ---
                "quality_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": (
                        "Quality threshold for edge collapse. "
                        "Higher = more conservative, better triangle quality."
                    ),
                    "visible_when": {"backend": ["quadric_edge_collapse"]},
                }),
                "preserve_boundary": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Preserve mesh boundary edges during decimation.",
                    "visible_when": {"backend": ["quadric_edge_collapse"]},
                }),
                "preserve_normal": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Prevent face normal flips during decimation.",
                    "visible_when": {"backend": ["quadric_edge_collapse"]},
                }),
                "preserve_topology": (["true", "false"], {
                    "default": "true",
                    "tooltip": "Preserve mesh topology (genus) during decimation.",
                    "visible_when": {"backend": [
                        "quadric_edge_collapse", "decimate_pro",
                    ]},
                }),
                "planar_quadric": (["true", "false"], {
                    "default": "false",
                    "tooltip": (
                        "Add penalty for non-planar faces. "
                        "Helps preserve flat regions."
                    ),
                    "visible_when": {"backend": ["quadric_edge_collapse"]},
                }),
                # --- fast_simplification specific ---
                "aggressiveness": ("INT", {
                    "default": 7,
                    "min": 1,
                    "max": 15,
                    "step": 1,
                    "tooltip": (
                        "How aggressively to simplify. "
                        "Higher = faster but lower quality. Default 7."
                    ),
                    "visible_when": {"backend": ["fast_simplification"]},
                }),
                # --- vertex_clustering specific ---
                "cluster_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": (
                        "Clustering cell size as percentage of bounding box diagonal. "
                        "Larger = more aggressive reduction."
                    ),
                    "visible_when": {"backend": ["vertex_clustering"]},
                }),
                # --- cgal_edge_collapse specific ---
                "cost_strategy": ([
                    "lindstrom_turk",
                    "edge_length",
                ], {
                    "default": "lindstrom_turk",
                    "tooltip": (
                        "CGAL cost strategy. "
                        "lindstrom_turk=optimizes geometry+volume+boundary (best), "
                        "edge_length=collapses shortest edges first (fast)."
                    ),
                    "visible_when": {"backend": ["cgal_edge_collapse"]},
                }),
                # --- decimate_pro specific ---
                "feature_angle": ("FLOAT", {
                    "default": 15.0,
                    "min": 0.0,
                    "max": 180.0,
                    "step": 1.0,
                    "tooltip": (
                        "Feature angle threshold (degrees). "
                        "Edges with dihedral angle above this are preserved."
                    ),
                    "visible_when": {"backend": ["decimate_pro"]},
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("decimated_mesh", "info")
    FUNCTION = "decimate"
    CATEGORY = "geompack/decimation"
    OUTPUT_NODE = True

    def decimate(
        self,
        trimesh,
        backend,
        target_face_count=5000,
        target_reduction=0.5,
        quality_threshold=0.3,
        preserve_boundary="true",
        preserve_normal="true",
        preserve_topology="true",
        planar_quadric="false",
        aggressiveness=7,
        cluster_threshold=1.0,
        cost_strategy="lindstrom_turk",
        feature_angle=15.0,
    ):
        """Apply mesh decimation based on selected backend."""
        # Sanitize hidden widget values
        target_face_count = int(target_face_count) if target_face_count not in (None, "") else 5000
        target_reduction = float(target_reduction) if target_reduction not in (None, "") else 0.5
        quality_threshold = float(quality_threshold) if quality_threshold not in (None, "") else 0.3
        aggressiveness = int(aggressiveness) if aggressiveness not in (None, "") else 7
        cluster_threshold = float(cluster_threshold) if cluster_threshold not in (None, "") else 1.0
        feature_angle = float(feature_angle) if feature_angle not in (None, "") else 15.0

        initial_vertices = len(trimesh.vertices)
        initial_faces = len(trimesh.faces)

        log.info("Decimate backend: %s", backend)
        log.info("Input: %s vertices, %s faces",
                 f"{initial_vertices:,}", f"{initial_faces:,}")

        if backend == "quadric_edge_collapse":
            log.info("Parameters: target_face_count=%d, quality_thr=%.2f, "
                     "preserve_boundary=%s, preserve_normal=%s, preserve_topology=%s",
                     target_face_count, quality_threshold,
                     preserve_boundary, preserve_normal, preserve_topology)
            decimated, error = _pymeshlab_quadric_edge_collapse(
                trimesh, target_face_count, quality_threshold,
                preserve_boundary == "true",
                preserve_normal == "true",
                preserve_topology == "true",
                planar_quadric == "true",
            )

        elif backend == "fast_simplification":
            log.info("Parameters: target_reduction=%.2f, aggressiveness=%d",
                     target_reduction, aggressiveness)
            decimated, error = _fast_simplification_decimate(
                trimesh, target_reduction, aggressiveness,
            )

        elif backend == "vertex_clustering":
            log.info("Parameters: cluster_threshold=%.1f%%", cluster_threshold)
            decimated, error = _pymeshlab_vertex_clustering(
                trimesh, cluster_threshold,
            )

        elif backend == "cgal_edge_collapse":
            # CGAL uses edge count as stop predicate; approximate from face count
            # For a closed triangle mesh: E ~ 1.5 * F
            target_edges = int(target_face_count * 1.5)
            log.info("Parameters: target_edges=%d (from target_faces=%d), cost=%s",
                     target_edges, target_face_count, cost_strategy)
            decimated, error = _cgal_edge_collapse(
                trimesh, target_edges, cost_strategy,
            )

        elif backend == "decimate_pro":
            log.info("Parameters: target_reduction=%.2f, preserve_topology=%s, feature_angle=%.1f",
                     target_reduction, preserve_topology, feature_angle)
            decimated, error = _pyvista_decimate_pro(
                trimesh, target_reduction,
                preserve_topology == "true",
                feature_angle,
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")

        if decimated is None:
            raise ValueError(f"Decimation failed ({backend}): {error}")

        # Copy metadata
        if hasattr(trimesh, "metadata") and trimesh.metadata:
            decimated.metadata = trimesh.metadata.copy()
        decimated.metadata["decimation"] = {
            "algorithm": backend,
            "original_vertices": initial_vertices,
            "original_faces": initial_faces,
            "result_vertices": len(decimated.vertices),
            "result_faces": len(decimated.faces),
        }

        vertex_change = len(decimated.vertices) - initial_vertices
        face_change = len(decimated.faces) - initial_faces
        face_pct = (face_change / initial_faces) * 100 if initial_faces > 0 else 0

        log.info("Output: %d vertices (%+d), %d faces (%+d, %.1f%%)",
                 len(decimated.vertices), vertex_change,
                 len(decimated.faces), face_change, face_pct)

        # Build backend-specific param block
        if backend == "quadric_edge_collapse":
            param_text = (
                f"Target Face Count: {target_face_count:,}\n"
                f"Quality Threshold: {quality_threshold}\n"
                f"Preserve Boundary: {preserve_boundary}\n"
                f"Preserve Normal: {preserve_normal}\n"
                f"Preserve Topology: {preserve_topology}\n"
                f"Planar Quadric: {planar_quadric}"
            )
        elif backend == "fast_simplification":
            param_text = (
                f"Target Reduction: {target_reduction:.0%}\n"
                f"Aggressiveness: {aggressiveness}"
            )
        elif backend == "vertex_clustering":
            param_text = f"Cluster Threshold: {cluster_threshold}%"
        elif backend == "cgal_edge_collapse":
            param_text = (
                f"Target Face Count: {target_face_count:,}\n"
                f"Cost Strategy: {cost_strategy}"
            )
        elif backend == "decimate_pro":
            param_text = (
                f"Target Reduction: {target_reduction:.0%}\n"
                f"Preserve Topology: {preserve_topology}\n"
                f"Feature Angle: {feature_angle}\u00b0"
            )
        else:
            param_text = ""

        info = f"""Decimate Results ({backend}):

{param_text}

Before:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}

After:
  Vertices: {len(decimated.vertices):,}
  Faces: {len(decimated.faces):,}
  Reduction: {abs(face_pct):.1f}%
"""
        return {"ui": {"text": [info]}, "result": (decimated, info)}


NODE_CLASS_MAPPINGS = {
    "GeomPackDecimateMesh": DecimateMeshNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackDecimateMesh": "Decimate Mesh",
}
