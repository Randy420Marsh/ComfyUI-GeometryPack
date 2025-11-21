"""
Unified dual mesh preview with VTK.js - supports both side-by-side and overlay layouts.

Combines and enhances PreviewMeshVTKDual and PreviewMeshVTKSideBySide with full
field visualization support. Displays two meshes either:
- Side-by-side: Synchronized cameras in separate viewports
- Overlaid: Combined in single viewport with color coding

Supports scalar field visualization with shared colormap when meshes have fields.
"""

import trimesh as trimesh_module
import numpy as np
import os
import tempfile
import uuid
import sys

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from _utils.mesh_ops import is_point_cloud, get_face_count, get_geometry_type

from ._vtp_export import export_mesh_with_scalars_vtp

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    COMFYUI_OUTPUT_FOLDER = None


def extract_field_names(mesh):
    """Extract all vertex and face attribute field names from a mesh."""
    field_names = []
    if hasattr(mesh, 'vertex_attributes') and mesh.vertex_attributes:
        field_names.extend(list(mesh.vertex_attributes.keys()))
    if hasattr(mesh, 'face_attributes') and mesh.face_attributes:
        field_names.extend([f"face.{k}" for k in mesh.face_attributes.keys()])
    return field_names


def has_fields(mesh):
    """Check if mesh has any vertex or face attributes."""
    has_vertex_attrs = hasattr(mesh, 'vertex_attributes') and len(mesh.vertex_attributes) > 0
    has_face_attrs = hasattr(mesh, 'face_attributes') and len(mesh.face_attributes) > 0
    return has_vertex_attrs or has_face_attrs


class PreviewMeshDualNode:
    """
    Unified dual mesh preview with VTK.js - supports both side-by-side and overlay layouts.

    Combines two meshes for comparison with full field visualization support.
    Choose between synchronized side-by-side viewports or single overlaid viewport.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_1": ("TRIMESH",),
                "mesh_2": ("TRIMESH",),
            },
            "optional": {
                "layout": (["side_by_side", "overlay"], {"default": "side_by_side"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_dual"
    CATEGORY = "geompack/visualization"

    def preview_dual(self, mesh_1, mesh_2, layout="side_by_side", opacity=1.0):
        """
        Preview two meshes with chosen layout and field visualization.

        Args:
            mesh_1: First trimesh object
            mesh_2: Second trimesh object
            layout: "side_by_side" or "overlay"
            opacity: Opacity for both meshes (0.0-1.0)

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshDual] Layout: {layout}")
        print(f"[PreviewMeshDual] Mesh 1: {get_geometry_type(mesh_1)} - {len(mesh_1.vertices)} vertices, {get_face_count(mesh_1)} faces")
        print(f"[PreviewMeshDual] Mesh 2: {get_geometry_type(mesh_2)} - {len(mesh_2.vertices)} vertices, {get_face_count(mesh_2)} faces")

        # Check for field data
        mesh_1_has_fields = has_fields(mesh_1)
        mesh_2_has_fields = has_fields(mesh_2)
        field_names_1 = extract_field_names(mesh_1)
        field_names_2 = extract_field_names(mesh_2)
        common_fields = list(set(field_names_1) & set(field_names_2))

        print(f"[PreviewMeshDual] Mesh 1 fields: {field_names_1}")
        print(f"[PreviewMeshDual] Mesh 2 fields: {field_names_2}")
        print(f"[PreviewMeshDual] Common fields: {common_fields}")

        # Check if meshes are point clouds (need VTP, STL doesn't support point clouds)
        mesh_1_is_pc = is_point_cloud(mesh_1)
        mesh_2_is_pc = is_point_cloud(mesh_2)

        # Generate unique ID for this preview
        preview_id = uuid.uuid4().hex[:8]

        if layout == "side_by_side":
            # Export meshes separately - use VTP for fields OR point clouds
            filename_1, filepath_1 = self._export_mesh(mesh_1, f"preview_dual_1_{preview_id}", mesh_1_has_fields or mesh_1_is_pc)
            filename_2, filepath_2 = self._export_mesh(mesh_2, f"preview_dual_2_{preview_id}", mesh_2_has_fields or mesh_2_is_pc)

            # Build UI data for side-by-side mode
            ui_data = {
                "layout": [layout],
                "mesh_1_file": [filename_1],
                "mesh_2_file": [filename_2],
                "vertex_count_1": [len(mesh_1.vertices)],
                "vertex_count_2": [len(mesh_2.vertices)],
                "face_count_1": [get_face_count(mesh_1)],
                "face_count_2": [get_face_count(mesh_2)],
                "bounds_min_1": [mesh_1.bounds[0].tolist()],
                "bounds_max_1": [mesh_1.bounds[1].tolist()],
                "bounds_min_2": [mesh_2.bounds[0].tolist()],
                "bounds_max_2": [mesh_2.bounds[1].tolist()],
                "extents_1": [mesh_1.extents.tolist()],
                "extents_2": [mesh_2.extents.tolist()],
                "is_watertight_1": [bool(mesh_1.is_watertight) if not is_point_cloud(mesh_1) else False],
                "is_watertight_2": [bool(mesh_2.is_watertight) if not is_point_cloud(mesh_2) else False],
                "field_names_1": [field_names_1],
                "field_names_2": [field_names_2],
                "common_fields": [common_fields],
            }

        else:  # overlay
            # Combine meshes with color coding
            filename, filepath = self._export_combined_mesh(
                mesh_1, mesh_2, preview_id, opacity,
                mesh_1_has_fields, mesh_2_has_fields
            )

            # Calculate combined bounds
            combined_bounds_min = np.minimum(mesh_1.bounds[0], mesh_2.bounds[0])
            combined_bounds_max = np.maximum(mesh_1.bounds[1], mesh_2.bounds[1])
            combined_extents = combined_bounds_max - combined_bounds_min

            # Build UI data for overlay mode
            ui_data = {
                "layout": [layout],
                "mesh_file": [filename],
                "vertex_count_1": [len(mesh_1.vertices)],
                "vertex_count_2": [len(mesh_2.vertices)],
                "face_count_1": [get_face_count(mesh_1)],
                "face_count_2": [get_face_count(mesh_2)],
                "bounds_min": [combined_bounds_min.tolist()],
                "bounds_max": [combined_bounds_max.tolist()],
                "extents": [combined_extents.tolist()],
                "opacity": [float(opacity)],
                "is_watertight_1": [bool(mesh_1.is_watertight) if not is_point_cloud(mesh_1) else False],
                "is_watertight_2": [bool(mesh_2.is_watertight) if not is_point_cloud(mesh_2) else False],
                "field_names_1": [field_names_1],
                "field_names_2": [field_names_2],
                "common_fields": [common_fields],
            }

        print(f"[PreviewMeshDual] Preview ready")
        return {"ui": ui_data}

    def _export_mesh(self, mesh, base_filename, use_vtp):
        """Export a single mesh to appropriate format."""
        if use_vtp:
            filename = f"{base_filename}.vtp"
        else:
            filename = f"{base_filename}.stl"

        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        try:
            if use_vtp:
                export_mesh_with_scalars_vtp(mesh, filepath)
                print(f"[PreviewMeshDual] Exported VTP with fields: {filepath}")
            else:
                mesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshDual] Exported STL: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshDual] Export failed: {e}, trying fallback")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            mesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshDual] Exported OBJ fallback: {filepath}")

        return filename, filepath

    def _export_combined_mesh(self, mesh_1, mesh_2, preview_id, opacity,
                              mesh_1_has_fields, mesh_2_has_fields):
        """Export combined mesh for overlay mode as VTP."""

        # Combine meshes (with or without fields)
        try:
            combined = trimesh_module.util.concatenate([mesh_1, mesh_2])
            filename = f"preview_dual_overlay_{preview_id}.vtp"

            if COMFYUI_OUTPUT_FOLDER:
                filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
            else:
                filepath = os.path.join(tempfile.gettempdir(), filename)

            export_mesh_with_scalars_vtp(combined, filepath)
            print(f"[PreviewMeshDual] Exported combined VTP: {filepath}")
            print(f"[PreviewMeshDual] Combined {get_geometry_type(combined)}: {len(combined.vertices)} vertices, {get_face_count(combined)} faces")
            return filename, filepath
        except Exception as e:
            print(f"[PreviewMeshDual] Failed to export combined mesh: {e}")
            raise


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshDual": PreviewMeshDualNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshDual": "Preview Mesh Dual",
}
