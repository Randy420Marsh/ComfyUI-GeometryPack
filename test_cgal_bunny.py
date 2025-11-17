#!/usr/bin/env python3
"""
Test script for CGAL isotropic remeshing with Stanford Bunny
Uses the same approach as mesh_utils.py
"""
import sys
import numpy as np

# Test CGAL import
print("=" * 60)
print("CGAL Isotropic Remeshing Test - Stanford Bunny")
print("=" * 60)

try:
    from CGAL.CGAL_Kernel import Point_3
    from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
    from CGAL import CGAL_Polygon_mesh_processing
    print("✓ CGAL imports successful")

    # Show available methods
    methods = [m for m in dir(CGAL_Polygon_mesh_processing) if not m.startswith('_')]
    print(f"  Available methods: {methods[:15]}...")
except ImportError as e:
    print(f"✗ CGAL import failed: {e}")
    sys.exit(1)

try:
    import trimesh
    print("✓ Trimesh import successful")
except ImportError as e:
    print(f"✗ Trimesh import failed: {e}")
    sys.exit(1)

# Load Stanford Bunny
print("\nLoading Stanford Bunny...")
mesh = trimesh.load('/home/shadeform/miniconda3/envs/geom/lib/python3.10/site-packages/pymeshlab/tests/sample_meshes/bunny.obj')
print(f"  Vertices: {len(mesh.vertices):,}")
print(f"  Faces: {len(mesh.faces):,}")
print(f"  Bounds: {mesh.bounds.tolist()}")

# Parameters
target_edge_length = 0.01  # Larger edge length for faster test
iterations = 2

print(f"\nRemeshing parameters:")
print(f"  Target edge length: {target_edge_length}")
print(f"  Iterations: {iterations}")

# Build CGAL Polyhedron using polygon_soup_to_polygon_mesh
print("\nBuilding CGAL Polyhedron_3...")

# Create points using CGAL_Polygon_mesh_processing.Point_3_Vector
points = CGAL_Polygon_mesh_processing.Point_3_Vector()
points.reserve(len(mesh.vertices))
for v in mesh.vertices:
    points.append(Point_3(float(v[0]), float(v[1]), float(v[2])))
print(f"  Created {len(points)} Point_3 objects")

# Create polygons using CGAL_Polygon_mesh_processing.Polygon_Vector
print("\nCreating polygon list...")
polygons = CGAL_Polygon_mesh_processing.Polygon_Vector()

# Debug first face
first_face_raw = mesh.faces[0]
print(f"  First face (raw): {first_face_raw}")
print(f"  First face types: {[type(x).__name__ for x in first_face_raw]}")

# Convert with explicit Python int - THE CRITICAL FIX
first_face_converted = [int(idx) for idx in first_face_raw]
print(f"  First face (converted): {first_face_converted}")
print(f"  Converted types: {[type(x).__name__ for x in first_face_converted]}")

# Add all faces
for i, face in enumerate(mesh.faces):
    polygons.append([int(idx) for idx in face])
    if i == 0:
        print(f"  ✓ First face added successfully")

print(f"  Created polygon vector with {len(polygons)} faces")

# Build polyhedron using polygon_soup_to_polygon_mesh
print("\nBuilding polyhedron from polygon soup...")
poly = Polyhedron_3()
CGAL_Polygon_mesh_processing.polygon_soup_to_polygon_mesh(points, polygons, poly)
print(f"  ✓ Polyhedron built successfully")
print(f"  Number of vertices: {poly.size_of_vertices()}")
print(f"  Number of facets: {poly.size_of_facets()}")

# Collect facets for remeshing
print(f"\nCollecting facets for remeshing...")
flist = []
for fh in poly.facets():
    flist.append(fh)
print(f"  Collected {len(flist)} facets")

# Run remeshing
print(f"\nRunning isotropic_remeshing...")
CGAL_Polygon_mesh_processing.isotropic_remeshing(
    flist,
    target_edge_length,
    poly,
    iterations
)
print(f"  ✓ Remeshing complete")
print(f"  New vertices: {poly.size_of_vertices()}")
print(f"  New facets: {poly.size_of_facets()}")

# Extract vertices
print("\nExtracting result mesh...")
new_vertices = []
vertex_map = {}
for i, v in enumerate(poly.vertices()):
    pt = v.point()
    new_vertices.append([pt.x(), pt.y(), pt.z()])
    vertex_map[v] = i

# Extract faces
new_faces = []
for f in poly.facets():
    he = f.halfedge()
    face_indices = []
    start_he = he
    while True:
        face_indices.append(vertex_map[he.vertex()])
        he = he.next()
        if he == start_he:
            break
    if len(face_indices) >= 3:
        new_faces.append(face_indices[:3])

new_vertices = np.array(new_vertices)
new_faces = np.array(new_faces)

print(f"  ✓ Extracted {len(new_vertices)} vertices, {len(new_faces)} faces")

# Create trimesh
result_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
print(f"\nFinal mesh:")
print(f"  Vertices: {len(result_mesh.vertices):,}")
print(f"  Faces: {len(result_mesh.faces):,}")
print(f"  Is watertight: {result_mesh.is_watertight}")

# Save result
output_path = '/tmp/bunny_cgal_remeshed.obj'
result_mesh.export(output_path)
print(f"\n✓ Saved remeshed mesh to: {output_path}")

print("\n" + "=" * 60)
print("TEST PASSED - CGAL Isotropic Remeshing Works!")
print("=" * 60)
