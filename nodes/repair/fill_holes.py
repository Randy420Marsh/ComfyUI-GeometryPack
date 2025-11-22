"""
Fill holes in mesh by adding new faces.
"""

import trimesh
import numpy as np

try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False


class FillHolesNode:
    """
    Fill holes in mesh by adding new faces.

    Identifies boundary loops (holes) and fills them with new triangular faces.
    Useful for mesh repair, creating watertight meshes for 3D printing, or
    closing gaps in scanned geometry.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("TRIMESH",),
                "method": (["trimesh", "pymeshlab", "igl_fan"], {"default": "trimesh"}),
                "maxholesize": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 10000.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("filled_mesh", "info")
    FUNCTION = "fill_holes"
    CATEGORY = "geompack/repair"

    def fill_holes(self, mesh, method="trimesh", maxholesize=100.0):
        """
        Fill holes in the mesh.

        Args:
            mesh: Input trimesh.Trimesh object
            method: Hole filling method ("trimesh", "pymeshlab", or "igl_fan")
            maxholesize: Maximum hole size to fill (used by trimesh method)

        Returns:
            tuple: (filled_trimesh, info_string)
        """
        print(f"[FillHoles] Input: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Check initial state
        was_watertight = mesh.is_watertight
        initial_vertices = len(mesh.vertices)
        initial_faces = len(mesh.faces)

        # Create a copy
        filled_mesh = mesh.copy()

        # Track method actually used (for fallback cases)
        method_used = method
        num_holes_filled = None

        # Fill holes using selected method
        if method == "pymeshlab" and HAS_PYMESHLAB:
            # Use PyMeshLab's hole closing
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=filled_mesh.vertices,
                face_matrix=filled_mesh.faces
            ))

            # Close holes
            ms.meshing_close_holes(maxholesize=int(maxholesize))

            # Extract result
            m = ms.current_mesh()
            filled_mesh = trimesh.Trimesh(
                vertices=m.vertex_matrix(),
                faces=m.face_matrix(),
                process=False
            )

            print(f"[FillHoles] PyMeshLab method completed")

        elif method == "pymeshlab" and not HAS_PYMESHLAB:
            # Fallback to trimesh if pymeshlab not available
            print(f"[FillHoles] PyMeshLab not available, falling back to trimesh method")
            filled_mesh.fill_holes()
            method_used = "trimesh (fallback)"

        elif method == "igl_fan" and HAS_IGL:
            # Use libigl to find boundary and fill with fan triangulation
            V = np.asarray(filled_mesh.vertices, dtype=np.float64)
            F = np.asarray(filled_mesh.faces, dtype=np.int32)

            # Get boundary loop (returns single loop as 1D array)
            try:
                loop = igl.boundary_loop(F)

                # Check if we got a valid loop
                if isinstance(loop, np.ndarray) and loop.size > 0 and loop.ndim == 1:
                    if len(loop) >= 3:
                        # Create fan triangulation from first vertex
                        new_faces = []
                        center_idx = loop[0]
                        for i in range(1, len(loop) - 1):
                            new_faces.append([center_idx, loop[i], loop[i + 1]])

                        # Add new faces to mesh
                        if new_faces:
                            all_faces = np.vstack([F, np.array(new_faces, dtype=np.int32)])
                            filled_mesh = trimesh.Trimesh(
                                vertices=V,
                                faces=all_faces,
                                process=False
                            )
                            num_holes_filled = 1
                            print(f"[FillHoles] igl_fan filled boundary loop with {len(new_faces)} faces")
                    else:
                        print(f"[FillHoles] Boundary loop too small ({len(loop)} vertices)")
                else:
                    print(f"[FillHoles] No boundary loop found or invalid format")
            except Exception as e:
                print(f"[FillHoles] igl boundary_loop failed: {e}, using trimesh fallback")
                filled_mesh.fill_holes()
                method_used = "trimesh (igl error fallback)"

        elif method == "igl_fan" and not HAS_IGL:
            # Fallback to trimesh if igl not available
            print(f"[FillHoles] libigl not available, falling back to trimesh method")
            filled_mesh.fill_holes()
            method_used = "trimesh (fallback)"

        else:
            # Use trimesh's built-in method
            filled_mesh.fill_holes()
            print(f"[FillHoles] Trimesh method completed")

        # Check result
        is_watertight = filled_mesh.is_watertight
        final_vertices = len(filled_mesh.vertices)
        final_faces = len(filled_mesh.faces)

        added_vertices = final_vertices - initial_vertices
        added_faces = final_faces - initial_faces

        # Build holes info
        holes_info = ""
        if num_holes_filled is not None:
            holes_info = f"\n  Holes Filled: {num_holes_filled}"

        info = f"""Hole Filling Results:

Method: {method_used}
Max Hole Size: {maxholesize}

Initial State:
  Vertices: {initial_vertices:,}
  Faces: {initial_faces:,}
  Watertight: {'Yes' if was_watertight else 'No'}

After Filling:
  Vertices: {final_vertices:,} (+{added_vertices})
  Faces: {final_faces:,} (+{added_faces})
  Watertight: {'✓ Yes' if is_watertight else '⚠ No'}{holes_info}

{'✓ All holes successfully filled!' if is_watertight and added_faces > 0 else ''}
{'ℹ No holes detected - mesh was already watertight.' if was_watertight else ''}
{'⚠ Some holes may remain (check mesh topology).' if not is_watertight and added_faces > 0 else ''}
"""

        print(f"[FillHoles] Added {added_faces} faces, Watertight: {was_watertight} -> {is_watertight}")

        return (filled_mesh, info)


NODE_CLASS_MAPPINGS = {
    "GeomPackFillHoles": FillHolesNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFillHoles": "Fill Holes",
}
