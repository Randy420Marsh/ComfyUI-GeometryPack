"""
Fix inconsistent normal orientations.
"""

import trimesh
import numpy as np

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False


class FixNormalsNode:
    """
    Fix inconsistent normal orientations.

    Ensures all face normals point consistently (all outward or all inward).
    Uses graph traversal to propagate consistent orientation across the trimesh.
    Essential for proper rendering and boolean operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "method": (["trimesh", "igl_bfs"], {"default": "trimesh"}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("fixed_mesh", "info")
    FUNCTION = "fix_normals"
    CATEGORY = "geompack/repair"

    def fix_normals(self, trimesh, method="trimesh"):
        """
        Fix inconsistent face normal orientations.

        Args:
            trimesh: Input trimesh.Trimesh object
            method: Orientation method ("trimesh" or "igl_bfs")

        Returns:
            tuple: (fixed_trimesh, info_string)
        """
        print(f"[FixNormals] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Create a copy to avoid modifying the original
        fixed_mesh = trimesh.copy()

        # Check initial winding consistency
        was_consistent = fixed_mesh.is_winding_consistent

        # Fix normals using selected method
        method_used = method
        num_components = None

        if method == "igl_bfs" and HAS_IGL:
            # Use libigl's BFS-based orientation
            V = np.asarray(fixed_mesh.vertices, dtype=np.float64)
            F = np.asarray(fixed_mesh.faces, dtype=np.int64)

            # Orient faces using BFS
            FF, C = igl.bfs_orient(F)

            # Update mesh faces with oriented version
            fixed_mesh.faces = FF

            # Track number of orientation components
            num_components = len(np.unique(C))
            print(f"[FixNormals] igl.bfs_orient: {num_components} orientation components")

        elif method == "igl_bfs" and not HAS_IGL:
            # Fallback to trimesh if igl not available
            print(f"[FixNormals] igl not available, falling back to trimesh method")
            fixed_mesh.fix_normals()
            method_used = "trimesh (fallback)"

        else:
            # Use trimesh's built-in method
            fixed_mesh.fix_normals()

        # Check if it's now consistent
        is_consistent = fixed_mesh.is_winding_consistent

        # Build info string
        components_info = f"\nOrientation Components: {num_components}" if num_components is not None else ""

        info = f"""Normal Orientation Fix:

Method: {method_used}
Before: {'Consistent' if was_consistent else 'Inconsistent'}
After:  {'Consistent' if is_consistent else 'Inconsistent'}{components_info}

Vertices: {len(fixed_mesh.vertices):,}
Faces: {len(fixed_mesh.faces):,}

{'✓ Normals are now consistently oriented!' if is_consistent else '⚠ Some inconsistencies may remain (check mesh topology)'}
"""

        print(f"[FixNormals] {'✓' if is_consistent else '⚠'} Normal orientation: {was_consistent} -> {is_consistent}")

        return (fixed_mesh, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackFixNormals": FixNormalsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackFixNormals": "Fix Normals",
}
