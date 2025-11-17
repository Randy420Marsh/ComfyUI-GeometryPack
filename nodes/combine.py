"""
Combine/Split Nodes - Merge and separate mesh components
"""

import numpy as np
import trimesh as trimesh_module


class CombineMeshesNode:
    """
    Combine Meshes - Concatenate multiple meshes into one.

    Simply concatenates vertices and faces without performing boolean operations.
    The result contains all geometry from input meshes as separate components.
    Useful for grouping objects or preparing batch operations.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_a": ("TRIMESH",),
            },
            "optional": {
                "mesh_b": ("TRIMESH",),
                "mesh_c": ("TRIMESH",),
                "mesh_d": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("combined_mesh", "info")
    FUNCTION = "combine"
    CATEGORY = "geompack/combine"

    def combine(self, mesh_a, mesh_b=None, mesh_c=None, mesh_d=None):
        """
        Combine multiple meshes into one.

        Args:
            mesh_a: First mesh (required)
            mesh_b, mesh_c, mesh_d: Optional additional meshes

        Returns:
            tuple: (combined_mesh, info_string)
        """
        meshes = [mesh_a]
        if mesh_b is not None:
            meshes.append(mesh_b)
        if mesh_c is not None:
            meshes.append(mesh_c)
        if mesh_d is not None:
            meshes.append(mesh_d)

        print(f"[CombineMeshes] Combining {len(meshes)} meshes")

        # Track input stats
        input_stats = []
        total_vertices = 0
        total_faces = 0

        for i, mesh in enumerate(meshes):
            input_stats.append({
                'index': i + 1,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces)
            })
            total_vertices += len(mesh.vertices)
            total_faces += len(mesh.faces)
            print(f"[CombineMeshes] Mesh {i+1}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Concatenate meshes
        if len(meshes) == 1:
            result = mesh_a.copy()
        else:
            result = trimesh_module.util.concatenate(meshes)

        # Preserve metadata from first mesh
        result.metadata = mesh_a.metadata.copy()
        result.metadata['combined'] = {
            'num_meshes': len(meshes),
            'input_stats': input_stats,
            'total_vertices': len(result.vertices),
            'total_faces': len(result.faces)
        }

        # Build info string
        mesh_lines = []
        for stat in input_stats:
            mesh_lines.append(f"  Mesh {stat['index']}: {stat['vertices']:,} vertices, {stat['faces']:,} faces")

        info = f"""Combine Meshes Results:

Number of Meshes Combined: {len(meshes)}

Input Meshes:
{chr(10).join(mesh_lines)}

Combined Result:
  Total Vertices: {len(result.vertices):,}
  Total Faces: {len(result.faces):,}
  Connected Components: {len(trimesh_module.graph.connected_components(result.face_adjacency)[1])}

Note: Meshes are concatenated without boolean operations.
Components remain separate within the combined mesh.
"""

        print(f"[CombineMeshes] Result: {len(result.vertices)} vertices, {len(result.faces)} faces")
        return (result, info)


class SplitComponentsNode:
    """
    Split Components - Separate mesh into disconnected components.

    Identifies and extracts individual connected components from a mesh.
    Returns up to 3 largest components and the total count.
    Useful for cleaning up meshes or processing individual parts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "min_faces": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 1000000,
                    "step": 1
                }),
                "sort_by": (["face_count", "vertex_count", "volume"], {"default": "face_count"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "TRIMESH", "TRIMESH", "INT", "STRING")
    RETURN_NAMES = ("largest", "second", "third", "component_count", "info")
    FUNCTION = "split"
    CATEGORY = "geompack/combine"

    def split(self, trimesh, min_faces=0, sort_by="face_count"):
        """
        Split mesh into connected components.

        Args:
            trimesh: Input mesh
            min_faces: Minimum faces to keep a component
            sort_by: Metric for sorting components

        Returns:
            tuple: (largest, second, third, count, info)
        """
        print(f"[SplitComponents] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Split into components
        components = trimesh.split(only_watertight=False)

        # Convert to list if single mesh
        if isinstance(components, trimesh_module.Trimesh):
            components = [components]
        else:
            components = list(components)

        print(f"[SplitComponents] Found {len(components)} components")

        # Filter by minimum face count
        if min_faces > 0:
            components = [c for c in components if len(c.faces) >= min_faces]
            print(f"[SplitComponents] After filtering (min_faces={min_faces}): {len(components)} components")

        # Sort components
        if sort_by == "face_count":
            components.sort(key=lambda c: len(c.faces), reverse=True)
        elif sort_by == "vertex_count":
            components.sort(key=lambda c: len(c.vertices), reverse=True)
        elif sort_by == "volume":
            components.sort(key=lambda c: abs(c.volume) if c.is_watertight else 0, reverse=True)

        # Extract up to 3 largest
        largest = components[0] if len(components) > 0 else trimesh.copy()
        second = components[1] if len(components) > 1 else trimesh_module.Trimesh()
        third = components[2] if len(components) > 2 else trimesh_module.Trimesh()

        # Preserve metadata
        for comp in [largest, second, third]:
            if len(comp.vertices) > 0:
                comp.metadata = trimesh.metadata.copy()

        # Build component info
        component_lines = []
        for i, comp in enumerate(components[:10]):  # Show up to 10
            watertight = "Yes" if comp.is_watertight else "No"
            component_lines.append(
                f"  {i+1}. Vertices: {len(comp.vertices):,}, "
                f"Faces: {len(comp.faces):,}, Watertight: {watertight}"
            )

        if len(components) > 10:
            component_lines.append(f"  ... and {len(components) - 10} more components")

        info = f"""Split Components Results:

Input Mesh:
  Vertices: {len(trimesh.vertices):,}
  Faces: {len(trimesh.faces):,}

Components Found: {len(components)}
Sorted By: {sort_by}
Min Faces Filter: {min_faces}

Component Details:
{chr(10).join(component_lines)}

Output:
  Largest: {len(largest.vertices):,} vertices, {len(largest.faces):,} faces
  Second: {len(second.vertices):,} vertices, {len(second.faces):,} faces
  Third: {len(third.vertices):,} vertices, {len(third.faces):,} faces
"""

        print(f"[SplitComponents] Returning {len(components)} components")
        return (largest, second, third, len(components), info)


class FilterComponentsNode:
    """
    Filter Components - Keep or remove components based on size criteria.

    Filters mesh components by face count, vertex count, or volume.
    Useful for removing small debris or keeping only significant parts.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filter_mode": (["keep_largest", "remove_small", "keep_by_count"],),
            },
            "optional": {
                "min_faces": ("INT", {"default": 100, "min": 1}),
                "min_vertices": ("INT", {"default": 50, "min": 1}),
                "keep_count": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("filtered_mesh", "info")
    FUNCTION = "filter_components"
    CATEGORY = "geompack/combine"

    def filter_components(self, trimesh, filter_mode,
                          min_faces=100, min_vertices=50, keep_count=1):
        """
        Filter mesh components based on criteria.

        Args:
            trimesh: Input mesh
            filter_mode: Filtering strategy
            min_faces: Minimum faces for remove_small mode
            min_vertices: Minimum vertices for remove_small mode
            keep_count: Number of components to keep for keep_by_count mode

        Returns:
            tuple: (filtered_mesh, info_string)
        """
        print(f"[FilterComponents] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[FilterComponents] Mode: {filter_mode}")

        # Split into components
        components = trimesh.split(only_watertight=False)
        if isinstance(components, trimesh_module.Trimesh):
            components = [components]
        else:
            components = list(components)

        original_count = len(components)
        print(f"[FilterComponents] Found {original_count} components")

        # Sort by face count (descending)
        components.sort(key=lambda c: len(c.faces), reverse=True)

        # Apply filter
        if filter_mode == "keep_largest":
            filtered = [components[0]] if components else []
        elif filter_mode == "remove_small":
            filtered = [c for c in components
                        if len(c.faces) >= min_faces and len(c.vertices) >= min_vertices]
        elif filter_mode == "keep_by_count":
            filtered = components[:keep_count]
        else:
            raise ValueError(f"Unknown filter mode: {filter_mode}")

        removed_count = original_count - len(filtered)

        # Recombine filtered components
        if len(filtered) == 0:
            result = trimesh_module.Trimesh()
        elif len(filtered) == 1:
            result = filtered[0]
        else:
            result = trimesh_module.util.concatenate(filtered)

        # Preserve metadata
        result.metadata = trimesh.metadata.copy()
        result.metadata['filtered'] = {
            'filter_mode': filter_mode,
            'original_components': original_count,
            'kept_components': len(filtered),
            'removed_components': removed_count
        }

        info = f"""Filter Components Results:

Filter Mode: {filter_mode}
Original Components: {original_count}
Kept Components: {len(filtered)}
Removed Components: {removed_count}

Parameters:
  Min Faces: {min_faces}
  Min Vertices: {min_vertices}
  Keep Count: {keep_count}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}
"""

        print(f"[FilterComponents] Result: {len(result.vertices)} vertices, {len(result.faces)} faces")
        return (result, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackCombineMeshes": CombineMeshesNode,
    "GeomPackSplitComponents": SplitComponentsNode,
    "GeomPackFilterComponents": FilterComponentsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackCombineMeshes": "Combine Meshes",
    "GeomPackSplitComponents": "Split Components",
    "GeomPackFilterComponents": "Filter Components",
}
