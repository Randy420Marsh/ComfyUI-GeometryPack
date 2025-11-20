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
                "mesh_1_color": (["default", "red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"], {"default": "default"}),
                "mesh_2_color": (["default", "red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple"], {"default": "default"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_dual"
    CATEGORY = "geompack/visualization"

    def preview_dual(self, mesh_1, mesh_2, layout="side_by_side",
                     mesh_1_color="default", mesh_2_color="default", opacity=1.0):
        """
        Preview two meshes with chosen layout and field visualization.

        Args:
            mesh_1: First trimesh object
            mesh_2: Second trimesh object
            layout: "side_by_side" or "overlay"
            mesh_1_color: Color for mesh_1 (overlay mode only, ignored if fields present)
            mesh_2_color: Color for mesh_2 (overlay mode only, ignored if fields present)
            opacity: Opacity for both meshes (0.0-1.0)

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshDual] Layout: {layout}")
        print(f"[PreviewMeshDual] Mesh 1: {len(mesh_1.vertices)} vertices, {len(mesh_1.faces)} faces")
        print(f"[PreviewMeshDual] Mesh 2: {len(mesh_2.vertices)} vertices, {len(mesh_2.faces)} faces")

        # Check for field data
        mesh_1_has_fields = has_fields(mesh_1)
        mesh_2_has_fields = has_fields(mesh_2)
        field_names_1 = extract_field_names(mesh_1)
        field_names_2 = extract_field_names(mesh_2)
        common_fields = list(set(field_names_1) & set(field_names_2))

        print(f"[PreviewMeshDual] Mesh 1 fields: {field_names_1}")
        print(f"[PreviewMeshDual] Mesh 2 fields: {field_names_2}")
        print(f"[PreviewMeshDual] Common fields: {common_fields}")

        # Generate unique ID for this preview
        preview_id = uuid.uuid4().hex[:8]

        if layout == "side_by_side":
            # Export meshes separately
            filename_1, filepath_1 = self._export_mesh(mesh_1, f"preview_dual_1_{preview_id}", mesh_1_has_fields)
            filename_2, filepath_2 = self._export_mesh(mesh_2, f"preview_dual_2_{preview_id}", mesh_2_has_fields)

            # Build UI data for side-by-side mode
            ui_data = {
                "layout": [layout],
                "mesh_1_file": [filename_1],
                "mesh_2_file": [filename_2],
                "vertex_count_1": [len(mesh_1.vertices)],
                "vertex_count_2": [len(mesh_2.vertices)],
                "face_count_1": [len(mesh_1.faces)],
                "face_count_2": [len(mesh_2.faces)],
                "bounds_min_1": [mesh_1.bounds[0].tolist()],
                "bounds_max_1": [mesh_1.bounds[1].tolist()],
                "bounds_min_2": [mesh_2.bounds[0].tolist()],
                "bounds_max_2": [mesh_2.bounds[1].tolist()],
                "extents_1": [mesh_1.extents.tolist()],
                "extents_2": [mesh_2.extents.tolist()],
                "is_watertight_1": [bool(mesh_1.is_watertight)],
                "is_watertight_2": [bool(mesh_2.is_watertight)],
                "field_names_1": [field_names_1],
                "field_names_2": [field_names_2],
                "common_fields": [common_fields],
            }

        else:  # overlay
            # Combine meshes with color coding
            filename, filepath = self._export_combined_mesh(
                mesh_1, mesh_2, preview_id,
                mesh_1_color, mesh_2_color, opacity,
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
                "face_count_1": [len(mesh_1.faces)],
                "face_count_2": [len(mesh_2.faces)],
                "bounds_min": [combined_bounds_min.tolist()],
                "bounds_max": [combined_bounds_max.tolist()],
                "extents": [combined_extents.tolist()],
                "mesh_1_color": [mesh_1_color],
                "mesh_2_color": [mesh_2_color],
                "opacity": [float(opacity)],
                "is_watertight_1": [bool(mesh_1.is_watertight)],
                "is_watertight_2": [bool(mesh_2.is_watertight)],
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

    def _export_combined_mesh(self, mesh_1, mesh_2, preview_id,
                              color_1, color_2, opacity,
                              mesh_1_has_fields, mesh_2_has_fields):
        """Export combined mesh for overlay mode."""

        # If both meshes have fields, export as VTP
        if mesh_1_has_fields and mesh_2_has_fields:
            # Combine meshes preserving their fields
            try:
                combined = trimesh_module.util.concatenate([mesh_1, mesh_2])
                filename = f"preview_dual_overlay_{preview_id}.vtp"

                if COMFYUI_OUTPUT_FOLDER:
                    filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
                else:
                    filepath = os.path.join(tempfile.gettempdir(), filename)

                export_mesh_with_scalars_vtp(combined, filepath)
                print(f"[PreviewMeshDual] Exported combined VTP with fields: {filepath}")
                return filename, filepath
            except Exception as e:
                print(f"[PreviewMeshDual] Failed to combine meshes with fields: {e}")
                # Fall through to color-based export

        # Otherwise, use vertex coloring for distinction
        color_map = {
            "red": [255, 100, 100],
            "blue": [100, 150, 255],
            "green": [100, 255, 100],
            "yellow": [255, 255, 100],
            "cyan": [100, 255, 255],
            "magenta": [255, 100, 255],
            "orange": [255, 180, 100],
            "purple": [200, 100, 255],
            "default": [255, 255, 255],  # White
        }

        alpha = int(opacity * 255)

        # Create colored copies
        mesh_1_colored = mesh_1.copy()
        mesh_2_colored = mesh_2.copy()

        # Assign vertex colors
        color_1_rgba = color_map.get(color_1, [255, 100, 100]) + [alpha]
        mesh_1_colored.visual.vertex_colors = np.tile(
            color_1_rgba, (len(mesh_1.vertices), 1)
        ).astype(np.uint8)

        color_2_rgba = color_map.get(color_2, [100, 150, 255]) + [alpha]
        mesh_2_colored.visual.vertex_colors = np.tile(
            color_2_rgba, (len(mesh_2.vertices), 1)
        ).astype(np.uint8)

        # Combine meshes
        try:
            combined = trimesh_module.util.concatenate([mesh_1_colored, mesh_2_colored])
            print(f"[PreviewMeshDual] Combined mesh: {len(combined.vertices)} vertices, {len(combined.faces)} faces")
        except Exception as e:
            print(f"[PreviewMeshDual] Failed to combine meshes: {e}")
            combined = mesh_1_colored

        # Export to GLB (preserves vertex colors)
        filename = f"preview_dual_overlay_{preview_id}.glb"

        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        try:
            combined.export(filepath, file_type='glb', include_normals=True)
            print(f"[PreviewMeshDual] Exported combined GLB: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshDual] GLB export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.glb', '.obj')
            filepath = filepath.replace('.glb', '.obj')
            combined.export(filepath, file_type='obj')
            print(f"[PreviewMeshDual] Exported OBJ fallback: {filepath}")

        return filename, filepath


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshDual": PreviewMeshDualNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshDual": "Preview Mesh Dual",
}
