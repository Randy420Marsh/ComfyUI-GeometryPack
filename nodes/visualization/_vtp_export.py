"""
Helper function for exporting mesh to VTK PolyData XML format (.vtp) with scalar attributes.

VTP format preserves vertex attributes (PointData) and face attributes (CellData)
which can be visualized in VTK.js with color mapping.
"""

import trimesh as trimesh_module
import xml.etree.ElementTree as ET


def export_mesh_with_scalars_vtp(trimesh: trimesh_module.Trimesh, filepath: str):
    """
    Export trimesh to VTK PolyData XML format (.vtp) with scalar attributes.

    VTP format preserves vertex attributes (PointData) and face attributes (CellData)
    which can be visualized in VTK.js with color mapping.

    Args:
        trimesh: Trimesh object with optional vertex_attributes and face_attributes
        filepath: Output .vtp file path
    """
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

    # CellData section (face attributes)
    cell_data = ET.SubElement(piece, 'CellData')

    # Add face attributes as scalar arrays
    if hasattr(trimesh, 'face_attributes') and trimesh.face_attributes:
        for attr_name, attr_values in trimesh.face_attributes.items():
            print(f"[_export_mesh_with_scalars_vtp]   Adding face field: {attr_name}")
            scalar_array = ET.SubElement(cell_data, 'DataArray',
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
