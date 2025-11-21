"""
Preview mesh with VTK.js scientific visualization viewer.

Displays mesh in an interactive VTK.js viewer with trackball controls.
Better for scientific visualization, mesh analysis, and large datasets.

Supports scalar field visualization: automatically detects vertex and face
attributes and exports to VTP format to preserve field data for visualization.
"""

import trimesh as trimesh_module
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


class PreviewMeshVTKNode:
    """
    Preview mesh with VTK.js scientific visualization viewer.

    Displays mesh in an interactive VTK.js viewer with trackball controls.
    Better for scientific visualization, mesh analysis, and large datasets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_vtk"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk(self, trimesh):
        """
        Export mesh and prepare for VTK.js preview.

        Automatically detects scalar fields (vertex/face attributes) and exports
        to VTP format if present, otherwise exports to STL.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTK] Preparing preview: {get_geometry_type(trimesh)} - {len(trimesh.vertices)} vertices, {get_face_count(trimesh)} faces")

        # Check for scalar fields (vertex/face attributes)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0
        has_face_attrs = hasattr(trimesh, 'face_attributes') and len(trimesh.face_attributes) > 0
        has_fields = has_vertex_attrs or has_face_attrs

        print(f"[PreviewMeshVTK] DEBUG - hasattr vertex_attributes: {hasattr(trimesh, 'vertex_attributes')}")
        print(f"[PreviewMeshVTK] DEBUG - hasattr face_attributes: {hasattr(trimesh, 'face_attributes')}")
        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTK] DEBUG - vertex_attributes: {trimesh.vertex_attributes}")
            print(f"[PreviewMeshVTK] DEBUG - len(vertex_attributes): {len(trimesh.vertex_attributes)}")
        if hasattr(trimesh, 'face_attributes'):
            print(f"[PreviewMeshVTK] DEBUG - face_attributes: {trimesh.face_attributes}")
            print(f"[PreviewMeshVTK] DEBUG - len(face_attributes): {len(trimesh.face_attributes)}")
        print(f"[PreviewMeshVTK] DEBUG - has_vertex_attrs: {has_vertex_attrs}")
        print(f"[PreviewMeshVTK] DEBUG - has_face_attrs: {has_face_attrs}")
        print(f"[PreviewMeshVTK] DEBUG - has_fields: {has_fields}")

        # Check if this is a point cloud
        is_pc = is_point_cloud(trimesh)

        # Choose export format based on whether fields exist or if it's a point cloud
        if has_fields or is_pc:
            # Export to VTP for: scalar fields OR point clouds (STL doesn't support point clouds)
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.vtp"
            print(f"[PreviewMeshVTK] Using VTP format (fields={has_fields}, point_cloud={is_pc})")
        else:
            # Export to STL (compact format for simple surface meshes)
            filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.stl"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh
        try:
            if has_fields or is_pc:
                # Use VTP exporter for fields or point clouds
                export_mesh_with_scalars_vtp(trimesh, filepath)
                print(f"[PreviewMeshVTK] Exported VTP to: {filepath}")
            else:
                # Use STL for simple surface meshes
                trimesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshVTK] Exported STL to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTK] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTK] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight (only for actual meshes, not point clouds)
        is_watertight = False if is_point_cloud(trimesh) else trimesh.is_watertight

        # Calculate volume and area (only for meshes with faces, not point clouds)
        volume = None
        area = None
        if not is_point_cloud(trimesh):
            try:
                if is_watertight:
                    volume = float(trimesh.volume)
                area = float(trimesh.area)
            except Exception as e:
                print(f"[PreviewMeshVTK] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays) - for field visualization UI
        field_names = []
        if has_vertex_attrs:
            field_names.extend(list(trimesh.vertex_attributes.keys()))
            print(f"[PreviewMeshVTK] Vertex attributes: {list(trimesh.vertex_attributes.keys())}")
        if has_face_attrs:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])
            print(f"[PreviewMeshVTK] Face attributes: {list(trimesh.face_attributes.keys())}")

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [get_face_count(trimesh)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
            "field_names": [field_names],  # Always include (empty array if no fields)
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]

        if field_names:
            print(f"[PreviewMeshVTK] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, fields={field_names}")
        else:
            print(f"[PreviewMeshVTK] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, no fields")

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshVTK": PreviewMeshVTKNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshVTK": "Preview Mesh (VTK)",
}
