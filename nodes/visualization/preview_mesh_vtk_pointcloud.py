"""
Preview mesh with VTK.js viewer optimized for scalar field visualization.

Specifically designed for visualizing vertex scalar fields (like boundary edges,
curvature, distance fields, etc.) and face scalar fields (like segmentation labels).
Always exports as VTP format to preserve scalar data.
"""

import trimesh as trimesh_module
import os
import tempfile
import uuid
from ._vtp_export import export_mesh_with_scalars_vtp

try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    COMFYUI_OUTPUT_FOLDER = None


class PreviewMeshVTKPointCloudNode:
    """
    Preview mesh with VTK.js viewer optimized for scalar field visualization.

    Specifically designed for visualizing vertex scalar fields (like boundary edges,
    curvature, distance fields, etc.) and face scalar fields (like segmentation labels).
    Always exports as VTP format to preserve scalar data.
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
    FUNCTION = "preview_mesh_vtk_pointcloud"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_pointcloud(self, trimesh):
        """
        Export mesh with scalar fields to VTP and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKPointCloud] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check if mesh has vertex or face attributes (scalar fields)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0
        has_face_attrs = hasattr(trimesh, 'face_attributes') and len(trimesh.face_attributes) > 0

        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTKPointCloud] trimesh.vertex_attributes: {trimesh.vertex_attributes.keys()}")
            print(f"[PreviewMeshVTKPointCloud] Number of vertex attributes: {len(trimesh.vertex_attributes)}")

        if hasattr(trimesh, 'face_attributes'):
            print(f"[PreviewMeshVTKPointCloud] trimesh.face_attributes: {trimesh.face_attributes.keys()}")
            print(f"[PreviewMeshVTKPointCloud] Number of face attributes: {len(trimesh.face_attributes)}")

        if not has_vertex_attrs and not has_face_attrs:
            print(f"[PreviewMeshVTKPointCloud] WARNING: No vertex or face attributes found. This node is for scalar field visualization.")

        # Always use VTP format for this node (designed for scalar fields)
        filename = f"preview_vtk_pointcloud_{uuid.uuid4().hex[:8]}.vtp"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh with vertex and face attributes as VTP
        try:
            export_mesh_with_scalars_vtp(trimesh, filepath)
            num_vertex_fields = len(trimesh.vertex_attributes) if has_vertex_attrs else 0
            num_face_fields = len(trimesh.face_attributes) if has_face_attrs else 0
            print(f"[PreviewMeshVTKPointCloud] Exported VTP with {num_vertex_fields} vertex fields and {num_face_fields} face fields to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKPointCloud] Export failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to export mesh with scalar fields: {e}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend(list(trimesh.vertex_attributes.keys()))
        if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
            field_names.extend([f"face.{k}" for k in trimesh.face_attributes.keys()])

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "is_watertight": [bool(is_watertight)],
            "field_names": [field_names],
        }

        print(f"[PreviewMeshVTKPointCloud] Scalar fields: {field_names}")

        return {"ui": ui_data}


NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMeshVTKFields": PreviewMeshVTKPointCloudNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMeshVTKFields": "Preview Mesh (VTK with Fields)",
}
