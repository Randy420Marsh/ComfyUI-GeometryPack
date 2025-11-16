"""
Visualization Nodes - 3D mesh preview and visualization
"""

import trimesh as trimesh_module
import os
import tempfile
import uuid
import sys
import json
import numpy as np
from pathlib import Path

# Import create_bbox_visualization from Hunyuan3D-Part
# Add parent directory to path to allow cross-custom-node imports
custom_nodes_dir = Path(__file__).parent.parent.parent
hunyuan_utils_path = custom_nodes_dir / "ComfyUI-Hunyuan3D-Part" / "node_utils"
if str(hunyuan_utils_path) not in sys.path:
    sys.path.insert(0, str(hunyuan_utils_path))

try:
    from mesh_utils import create_bbox_visualization, export_scene_to_vtp
    BBOX_VIZ_AVAILABLE = True
except ImportError:
    print("[GeomPack] WARNING: Could not import bbox functions from Hunyuan3D-Part")
    print("[GeomPack] The Preview Bounding Boxes node will not be available")
    BBOX_VIZ_AVAILABLE = False
    export_scene_to_vtp = None

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_OUTPUT_FOLDER = folder_paths.get_output_directory()
except:
    COMFYUI_OUTPUT_FOLDER = None


def _export_mesh_with_scalars_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
    """
    Export trimesh to VTK PolyData XML format (.vtp) with vertex scalar attributes.

    VTP format preserves vertex attributes (scalar fields) which can be visualized
    in VTK.js with color mapping.

    Args:
        trimesh: Trimesh object with optional vertex_attributes
        filepath: Output .vtp file path
    """
    import xml.etree.ElementTree as ET

    print(f"[_export_mesh_with_scalars_vtp] Exporting to VTP: {filepath}")

    # Create VTK PolyData XML structure
    vtk_file = ET.Element('VTKFile', type='PolyData', version='1.0', byte_order='LittleEndian')
    poly_data = ET.SubElement(vtk_file, 'PolyData')

    num_verts = len(trimesh.vertices)
    num_faces = len(trimesh.faces)

    piece = ET.SubElement(poly_data, 'Piece', NumberOfPoints=str(num_verts), NumberOfPolys=str(num_faces))

    # Points section
    points = ET.SubElement(piece, 'Points')
    points_data_array = ET.SubElement(points, 'DataArray',
                                       type='Float32',
                                       NumberOfComponents='3',
                                       format='ascii')
    # Flatten vertices to space-separated string
    verts_flat = trimesh.vertices.flatten()
    points_data_array.text = ' '.join(map(str, verts_flat))

    # PointData section (scalar fields)
    point_data = ET.SubElement(piece, 'PointData')

    # Add vertex attributes as scalar arrays
    if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
        for attr_name, attr_values in trimesh.vertex_attributes.items():
            print(f"[_export_mesh_with_scalars_vtp]   Adding scalar field: {attr_name}")
            scalar_array = ET.SubElement(point_data, 'DataArray',
                                          type='Float32',
                                          Name=attr_name,
                                          format='ascii')
            scalar_array.text = ' '.join(map(str, attr_values.flatten()))

    # Polys section (faces/triangles)
    polys = ET.SubElement(piece, 'Polys')

    # Connectivity: vertex indices for each face
    connectivity = ET.SubElement(polys, 'DataArray',
                                   type='Int32',
                                   Name='connectivity',
                                   format='ascii')
    faces_flat = trimesh.faces.flatten()
    connectivity.text = ' '.join(map(str, faces_flat))

    # Offsets: cumulative count of indices (each triangle has 3 vertices)
    offsets = ET.SubElement(polys, 'DataArray',
                             type='Int32',
                             Name='offsets',
                             format='ascii')
    offset_values = [(i + 1) * 3 for i in range(num_faces)]
    offsets.text = ' '.join(map(str, offset_values))

    # Write to file with pretty formatting
    tree = ET.ElementTree(vtk_file)
    ET.indent(tree, space='  ')
    tree.write(filepath, encoding='utf-8', xml_declaration=True)

    print(f"[_export_mesh_with_scalars_vtp] Export complete: {num_verts} vertices, {num_faces} faces")


class PreviewMeshNode:
    """
    Preview mesh with interactive 3D viewer.

    Displays mesh in an interactive Three.js viewer with orbit controls.
    Allows rotating, panning, and zooming to inspect the mesh geometry.
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
    FUNCTION = "preview_mesh"
    CATEGORY = "geompack/visualization"

    def preview_mesh(self, trimesh):
        """
        Export mesh to GLB and prepare for 3D preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMesh] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to GLB (best format for Three.js)
        try:
            trimesh.export(filepath, file_type='glb')
            print(f"[PreviewMesh] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMesh] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.glb', '.obj')
            filepath = filepath.replace('.glb', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMesh] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        return {
            "ui": {
                "mesh_file": [filename],
                "vertex_count": [len(trimesh.vertices)],
                "face_count": [len(trimesh.faces)],
                "bounds_min": [bounds[0].tolist()],
                "bounds_max": [bounds[1].tolist()],
                "extents": [extents.tolist()],
                "max_extent": [float(max_extent)],
            }
        }


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
        Export mesh to STL and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTK] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_vtk_{uuid.uuid4().hex[:8]}.stl"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to STL (native format for VTK.js)
        try:
            trimesh.export(filepath, file_type='stl')
            print(f"[PreviewMeshVTK] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTK] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.stl', '.obj')
            filepath = filepath.replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTK] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Calculate volume and area (only if watertight)
        volume = None
        area = None
        try:
            if is_watertight:
                volume = float(trimesh.volume)
            area = float(trimesh.area)
        except Exception as e:
            print(f"[PreviewMeshVTK] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend([f"vertex.{k}" for k in trimesh.vertex_attributes.keys()])
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
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]
        if field_names:
            ui_data["field_names"] = [field_names]

        print(f"[PreviewMeshVTK] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, fields={len(field_names)}")

        return {"ui": ui_data}


class PreviewMeshVTKHidableMenuNode:
    """
    Preview mesh with VTK.js scientific visualization viewer (with hidable menu).

    Displays mesh in an interactive VTK.js viewer with collapsible controls menu.
    Provides a cleaner interface with toggle button to show/hide the control options.
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
    FUNCTION = "preview_mesh_vtk_hidable"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_hidable(self, trimesh):
        """
        Export mesh to STL and prepare for VTK.js preview with hidable menu.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKHidable] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Generate unique filename
        filename = f"preview_vtk_hidable_{uuid.uuid4().hex[:8]}.stl"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to STL (native format for VTK.js)
        try:
            trimesh.export(filepath, file_type='stl')
            print(f"[PreviewMeshVTKHidable] Exported to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKHidable] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.stl', '.obj')
            filepath = filepath.replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTKHidable] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Calculate volume and area (only if watertight)
        volume = None
        area = None
        try:
            if is_watertight:
                volume = float(trimesh.volume)
            area = float(trimesh.area)
        except Exception as e:
            print(f"[PreviewMeshVTKHidable] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend([f"vertex.{k}" for k in trimesh.vertex_attributes.keys()])
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
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]
        if field_names:
            ui_data["field_names"] = [field_names]

        print(f"[PreviewMeshVTKHidable] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, fields={len(field_names)}")

        return {"ui": ui_data}


class PreviewMeshVTKFiltersNode:
    """
    Preview mesh with VTK.js scientific visualization viewer including filters.

    Displays mesh in an interactive VTK.js viewer with trackball controls
    and filter controls (smoothing, outline, cutting plane).
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
    FUNCTION = "preview_mesh_vtk_filters"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_filters(self, trimesh):
        """
        Export mesh to STL and prepare for VTK.js preview with filters.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKFilters] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check if mesh has vertex attributes (scalar fields)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0

        print(f"[PreviewMeshVTKFilters] hasattr(trimesh, 'vertex_attributes'): {hasattr(trimesh, 'vertex_attributes')}")
        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTKFilters] trimesh.vertex_attributes: {trimesh.vertex_attributes.keys()}")
            print(f"[PreviewMeshVTKFilters] Number of vertex attributes: {len(trimesh.vertex_attributes)}")

        # Generate unique filename
        # Use VTP format if we have vertex attributes, otherwise STL
        if has_vertex_attrs:
            filename = f"preview_vtk_filters_{uuid.uuid4().hex[:8]}.vtp"
            file_format = 'vtp'
            print(f"[PreviewMeshVTKFilters] Mesh has vertex attributes, using VTP format")
        else:
            filename = f"preview_vtk_filters_{uuid.uuid4().hex[:8]}.stl"
            file_format = 'stl'
            print(f"[PreviewMeshVTKFilters] No vertex attributes found, using STL format")

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh with vertex attributes
        try:
            if file_format == 'vtp':
                # Export to VTK PolyData format (XML) which preserves vertex attributes
                _export_mesh_with_scalars_vtp(trimesh, filepath)
                print(f"[PreviewMeshVTKFilters] Exported VTP with {len(trimesh.vertex_attributes)} scalar fields to: {filepath}")
            else:
                # Export to STL (no scalar data)
                trimesh.export(filepath, file_type='stl')
                print(f"[PreviewMeshVTKFilters] Exported to STL: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKFilters] Export failed: {e}")
            # Fallback to OBJ
            filename = filename.replace('.vtp', '.obj').replace('.stl', '.obj')
            filepath = filepath.replace('.vtp', '.obj').replace('.stl', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTKFilters] Exported to OBJ: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Calculate volume and area (only if watertight)
        volume = None
        area = None
        try:
            if is_watertight:
                volume = float(trimesh.volume)
            area = float(trimesh.area)
        except Exception as e:
            print(f"[PreviewMeshVTKFilters] Could not calculate volume/area: {e}")

        # Get field names (vertex/face data arrays)
        field_names = []
        if hasattr(trimesh, 'vertex_attributes') and trimesh.vertex_attributes:
            field_names.extend([f"vertex.{k}" for k in trimesh.vertex_attributes.keys()])
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
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]
        if field_names:
            ui_data["field_names"] = [field_names]

        print(f"[PreviewMeshVTKFilters] Mesh info: watertight={is_watertight}, volume={volume}, area={area}, fields={len(field_names)}")

        return {"ui": ui_data}


class PreviewMeshVTKFieldsNode:
    """
    Preview mesh with VTK.js viewer optimized for scalar field visualization.

    Specifically designed for visualizing vertex scalar fields (like boundary edges,
    curvature, distance fields, etc.) with color mapping. Always exports as VTP
    format to preserve scalar data.
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
    FUNCTION = "preview_mesh_vtk_fields"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_fields(self, trimesh):
        """
        Export mesh with scalar fields to VTP and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKFields] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check if mesh has vertex attributes (scalar fields)
        has_vertex_attrs = hasattr(trimesh, 'vertex_attributes') and len(trimesh.vertex_attributes) > 0

        if hasattr(trimesh, 'vertex_attributes'):
            print(f"[PreviewMeshVTKFields] trimesh.vertex_attributes: {trimesh.vertex_attributes.keys()}")
            print(f"[PreviewMeshVTKFields] Number of vertex attributes: {len(trimesh.vertex_attributes)}")

        if not has_vertex_attrs:
            print(f"[PreviewMeshVTKFields] WARNING: No vertex attributes found. This node is for scalar field visualization.")

        # Always use VTP format for this node (designed for scalar fields)
        filename = f"preview_vtk_fields_{uuid.uuid4().hex[:8]}.vtp"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export mesh with vertex attributes as VTP
        try:
            _export_mesh_with_scalars_vtp(trimesh, filepath)
            num_fields = len(trimesh.vertex_attributes) if has_vertex_attrs else 0
            print(f"[PreviewMeshVTKFields] Exported VTP with {num_fields} scalar fields to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKFields] Export failed: {e}")
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

        print(f"[PreviewMeshVTKFields] Scalar fields: {field_names}")

        return {"ui": ui_data}


class PreviewMeshVTKTexturedNode:
    """
    Preview mesh with VTK.js viewer optimized for textured/PBR meshes.

    Exports mesh to GLB format to preserve textures, materials, UV coordinates,
    and vertex colors. Uses VTK.js GLTFImporter for loading. Best choice for
    meshes with textures or PBR materials.
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
    FUNCTION = "preview_mesh_vtk_textured"
    CATEGORY = "geompack/visualization"

    def preview_mesh_vtk_textured(self, trimesh):
        """
        Export mesh to GLB and prepare for VTK.js preview with texture support.

        Args:
            trimesh: Input trimesh_module.Trimesh object

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshVTKTextured] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check for texture/material information
        has_visual = hasattr(trimesh, 'visual') and trimesh.visual is not None
        visual_kind = None
        has_texture = False
        has_vertex_colors = False
        has_material = False

        if has_visual:
            visual_kind = trimesh.visual.kind
            if visual_kind == 'texture':
                has_texture = hasattr(trimesh.visual, 'material') and trimesh.visual.material is not None
                has_material = has_texture
                print(f"[PreviewMeshVTKTextured] Mesh has texture visual with material: {has_material}")
            elif visual_kind == 'vertex':
                has_vertex_colors = True
                print(f"[PreviewMeshVTKTextured] Mesh has vertex colors")
            elif visual_kind == 'face':
                print(f"[PreviewMeshVTKTextured] Mesh has face colors")

        # Generate unique filename - always use GLB for texture/material support
        filename = f"preview_vtk_textured_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export to GLB (preserves textures, materials, UVs, colors)
        try:
            trimesh.export(filepath, file_type='glb', include_normals=True)
            print(f"[PreviewMeshVTKTextured] Exported GLB to: {filepath}")
        except Exception as e:
            print(f"[PreviewMeshVTKTextured] GLB export failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to OBJ (loses textures but keeps geometry)
            filename = filename.replace('.glb', '.obj')
            filepath = filepath.replace('.glb', '.obj')
            trimesh.export(filepath, file_type='obj')
            print(f"[PreviewMeshVTKTextured] Exported to OBJ fallback: {filepath}")

        # Calculate bounding box info for camera setup
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Check if mesh is watertight
        is_watertight = trimesh.is_watertight

        # Calculate volume and area
        volume = None
        area = None
        try:
            if is_watertight:
                volume = float(trimesh.volume)
            area = float(trimesh.area)
        except Exception as e:
            print(f"[PreviewMeshVTKTextured] Could not calculate volume/area: {e}")

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
            "has_texture": [has_texture],
            "has_vertex_colors": [has_vertex_colors],
            "has_material": [has_material],
            "visual_kind": [visual_kind if visual_kind else "none"],
        }

        # Add optional fields if available
        if volume is not None:
            ui_data["volume"] = [volume]
        if area is not None:
            ui_data["area"] = [area]

        print(f"[PreviewMeshVTKTextured] Mesh info: watertight={is_watertight}, texture={has_texture}, vertex_colors={has_vertex_colors}")

        return {"ui": ui_data}


class PreviewBoundingBoxesVTKNode:
    """
    Preview mesh with bounding boxes using VTK.js viewer.

    Displays the original mesh with wireframe bounding boxes overlaid,
    showing the segmentation results from P3-SAM or other bbox-generating nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "bboxes": ("BBOXES_3D",),
                "line_width": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "slider",
                    "tooltip": "Thickness of bounding box wireframe lines in pixels. Rendered by VTK.js."
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_bboxes_vtk"
    CATEGORY = "geompack/visualization"

    def preview_bboxes_vtk(self, trimesh, bboxes, line_width):
        """
        Export mesh with bounding boxes to VTP and prepare for VTK.js preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object
            bboxes: BBOXES_3D dict with 'bboxes' (array [N, 2, 3]) and 'num_parts'
            line_width: Thickness of wireframe lines in pixels (for VTK.js rendering)

        Returns:
            dict: UI data for frontend widget
        """
        if not BBOX_VIZ_AVAILABLE:
            raise RuntimeError(
                "create_bbox_visualization not available. "
                "Make sure ComfyUI-Hunyuan3D-Part is installed."
            )

        # Extract bboxes array from BBOXES_3D dictionary
        if isinstance(bboxes, dict):
            bboxes_array = bboxes['bboxes']
            num_parts = bboxes.get('num_parts', len(bboxes_array))
        else:
            # Fallback for raw array
            bboxes_array = bboxes
            num_parts = len(bboxes_array)

        print(f"[PreviewBoundingBoxesVTK] Preparing preview with {num_parts} bounding boxes")
        print(f"[PreviewBoundingBoxesVTK] Mesh: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[PreviewBoundingBoxesVTK] Line width: {line_width}px")

        # Create scene with mesh and bounding boxes
        # Use Path3D wireframes (rendered as lines in VTK.js)
        scene = create_bbox_visualization(trimesh, bboxes_array, use_tubes=False)

        # Generate unique filename - use VTP which supports wireframes
        filename = f"preview_bboxes_vtk_{uuid.uuid4().hex[:8]}.vtp"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, filename)
        else:
            filepath = os.path.join(tempfile.gettempdir(), filename)

        # Export scene to VTP (preserves wireframes)
        try:
            # Use custom VTP exporter that preserves both mesh and wireframe lines
            export_scene_to_vtp(scene, filepath)
            print(f"[PreviewBoundingBoxesVTK] Exported scene with {num_parts} bounding boxes to: {filepath}")
        except Exception as e:
            print(f"[PreviewBoundingBoxesVTK] VTP export failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to STL with tubes
            print(f"[PreviewBoundingBoxesVTK] Falling back to STL with tube meshes")
            scene = create_bbox_visualization(trimesh, bboxes_array, use_tubes=True)
            filename = filename.replace('.vtp', '.stl')
            filepath = filepath.replace('.vtp', '.stl')
            scene.export(filepath, file_type='stl')
            print(f"[PreviewBoundingBoxesVTK] Exported to: {filepath}")

        # Calculate bounding box info for camera setup
        # Use the original mesh bounds for camera positioning
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "num_bboxes": [num_parts],
            "line_width": [float(line_width)],
        }

        print(f"[PreviewBoundingBoxesVTK] Created visualization with {num_parts} bounding boxes")

        return {"ui": ui_data}


class PreviewMeshUVNode:
    """
    Preview mesh with synchronized 3D and UV layout views.

    Displays mesh in a split-pane interactive viewer:
    - Left: 3D mesh view with wireframe overlay (rotatable)
    - Right: 2D UV layout in 0-1 space
    - Click on either view highlights corresponding point on the other

    Useful for inspecting UV unwrapping quality, seam placement, and texture mapping.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "show_checker": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Apply checker pattern to visualize UV distortion"
                }),
                "show_wireframe": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show mesh wireframe on 3D view"
                }),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_mesh_uv"
    CATEGORY = "geompack/visualization"

    def preview_mesh_uv(self, trimesh, show_checker=False, show_wireframe=True):
        """
        Export mesh and UV data for synchronized 3D + UV layout preview.

        Args:
            trimesh: Input trimesh_module.Trimesh object
            show_checker: Apply checker pattern texture
            show_wireframe: Show wireframe overlay on 3D mesh

        Returns:
            dict: UI data for frontend widget
        """
        print(f"[PreviewMeshUV] Preparing preview: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # Check for UV data
        has_uvs = False
        uv_data = None

        if hasattr(trimesh, 'visual') and trimesh.visual is not None:
            if hasattr(trimesh.visual, 'uv') and trimesh.visual.uv is not None:
                uvs = trimesh.visual.uv
                if len(uvs) > 0:
                    has_uvs = True
                    # Convert UV data to serializable format
                    uv_data = {
                        "uvs": uvs.tolist(),
                        "faces": trimesh.faces.tolist(),
                    }
                    print(f"[PreviewMeshUV] Found UV data: {len(uvs)} UV coordinates")

                    # Calculate UV statistics
                    uv_min = uvs.min(axis=0)
                    uv_max = uvs.max(axis=0)
                    uv_range = uv_max - uv_min

                    # Check if UVs are in 0-1 range
                    in_unit_square = bool(uv_min[0] >= 0 and uv_min[1] >= 0 and
                                         uv_max[0] <= 1 and uv_max[1] <= 1)

                    # Estimate UV coverage (area of UV triangles / unit square area)
                    try:
                        uv_area = 0.0
                        for face in trimesh.faces:
                            # Get UV coordinates for this face
                            uv0 = uvs[face[0]]
                            uv1 = uvs[face[1]]
                            uv2 = uvs[face[2]]
                            # Triangle area using cross product
                            v1 = uv1 - uv0
                            v2 = uv2 - uv0
                            uv_area += abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0
                        uv_coverage = float(uv_area)
                    except Exception as e:
                        print(f"[PreviewMeshUV] Could not calculate UV coverage: {e}")
                        uv_coverage = 0.0

                    print(f"[PreviewMeshUV] UV stats: range={uv_range}, in_unit_square={in_unit_square}, coverage={uv_coverage:.4f}")

        if not has_uvs:
            print(f"[PreviewMeshUV] WARNING: No UV data found. UV layout view will be empty.")

        # Generate unique filename for mesh
        mesh_filename = f"preview_uv_{uuid.uuid4().hex[:8]}.glb"

        # Use ComfyUI's output directory
        if COMFYUI_OUTPUT_FOLDER:
            mesh_filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, mesh_filename)
        else:
            mesh_filepath = os.path.join(tempfile.gettempdir(), mesh_filename)

        # Export mesh to GLB (preserves UVs)
        try:
            trimesh.export(mesh_filepath, file_type='glb')
            print(f"[PreviewMeshUV] Exported mesh to: {mesh_filepath}")
        except Exception as e:
            print(f"[PreviewMeshUV] GLB export failed: {e}")
            # Fallback to OBJ
            mesh_filename = mesh_filename.replace('.glb', '.obj')
            mesh_filepath = mesh_filepath.replace('.glb', '.obj')
            trimesh.export(mesh_filepath, file_type='obj')
            print(f"[PreviewMeshUV] Exported to OBJ: {mesh_filepath}")

        # Save UV data as JSON for the frontend
        uv_json_filename = None
        if uv_data:
            uv_json_filename = f"preview_uv_{uuid.uuid4().hex[:8]}_uvdata.json"
            if COMFYUI_OUTPUT_FOLDER:
                uv_json_filepath = os.path.join(COMFYUI_OUTPUT_FOLDER, uv_json_filename)
            else:
                uv_json_filepath = os.path.join(tempfile.gettempdir(), uv_json_filename)

            with open(uv_json_filepath, 'w') as f:
                json.dump(uv_data, f)
            print(f"[PreviewMeshUV] Exported UV data to: {uv_json_filepath}")

        # Calculate bounding box info
        bounds = trimesh.bounds
        extents = trimesh.extents
        max_extent = max(extents)

        # Return metadata for frontend widget
        ui_data = {
            "mesh_file": [mesh_filename],
            "vertex_count": [len(trimesh.vertices)],
            "face_count": [len(trimesh.faces)],
            "bounds_min": [bounds[0].tolist()],
            "bounds_max": [bounds[1].tolist()],
            "extents": [extents.tolist()],
            "max_extent": [float(max_extent)],
            "has_uvs": [has_uvs],
            "show_checker": [show_checker],
            "show_wireframe": [show_wireframe],
        }

        # Add UV-specific data if available
        if uv_json_filename:
            ui_data["uv_data_file"] = [uv_json_filename]
        if has_uvs:
            ui_data["uv_coverage"] = [uv_coverage]
            ui_data["uv_in_unit_square"] = [in_unit_square]
            ui_data["uv_min"] = [uv_min.tolist()]
            ui_data["uv_max"] = [uv_max.tolist()]

        print(f"[PreviewMeshUV] Preview ready: has_uvs={has_uvs}, checker={show_checker}, wireframe={show_wireframe}")

        return {"ui": ui_data}


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackPreviewMesh": PreviewMeshNode,
    "GeomPackPreviewMeshVTK": PreviewMeshVTKNode,
    "GeomPackPreviewMeshVTKHidableMenu": PreviewMeshVTKHidableMenuNode,
    "GeomPackPreviewMeshVTKFilters": PreviewMeshVTKFiltersNode,
    "GeomPackPreviewMeshVTKFields": PreviewMeshVTKFieldsNode,
    "GeomPackPreviewMeshVTKTextured": PreviewMeshVTKTexturedNode,
    "GeomPackPreviewBoundingBoxesVTK": PreviewBoundingBoxesVTKNode,
    "GeomPackPreviewMeshUV": PreviewMeshUVNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackPreviewMesh": "Preview Mesh (3D)",
    "GeomPackPreviewMeshVTK": "Preview Mesh (VTK)",
    "GeomPackPreviewMeshVTKHidableMenu": "Preview Mesh VTK (Hidable Menu)",
    "GeomPackPreviewMeshVTKFilters": "Preview Mesh (VTK with Filters)",
    "GeomPackPreviewMeshVTKFields": "Preview Mesh (VTK with Fields)",
    "GeomPackPreviewMeshVTKTextured": "Preview Mesh (VTK with Textures)",
    "GeomPackPreviewBoundingBoxesVTK": "Preview Bounding Boxes (VTK)",
    "GeomPackPreviewMeshUV": "Preview Mesh (UV Layout)",
}
