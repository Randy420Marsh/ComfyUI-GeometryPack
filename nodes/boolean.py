"""
Boolean Operations Nodes - CSG operations on meshes
"""

import numpy as np
import trimesh as trimesh_module


class BooleanOpNode:
    """
    Boolean Operations - Union, Difference, and Intersection of meshes.

    Performs Constructive Solid Geometry (CSG) operations:
    - union: Combine two meshes into one
    - difference: Subtract mesh_b from mesh_a
    - intersection: Keep only overlapping parts

    Uses trimesh's boolean operations (requires manifold3d backend).
    Fallback to Blender if manifold3d is not available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_a": ("TRIMESH",),
                "mesh_b": ("TRIMESH",),
                "operation": (["union", "difference", "intersection"],),
            },
            "optional": {
                "engine": (["auto", "manifold", "blender"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("result_mesh", "info")
    FUNCTION = "boolean_op"
    CATEGORY = "geompack/boolean"

    def boolean_op(self, mesh_a, mesh_b, operation, engine="auto"):
        """
        Perform boolean operation on two meshes.

        Args:
            mesh_a: First mesh (base mesh for difference)
            mesh_b: Second mesh (subtracted mesh for difference)
            operation: Boolean operation type
            engine: Backend to use

        Returns:
            tuple: (result_mesh, info_string)
        """
        print(f"[Boolean] Mesh A: {len(mesh_a.vertices)} vertices, {len(mesh_a.faces)} faces")
        print(f"[Boolean] Mesh B: {len(mesh_b.vertices)} vertices, {len(mesh_b.faces)} faces")
        print(f"[Boolean] Operation: {operation}, Engine: {engine}")

        # Try manifold backend first
        if engine in ["auto", "manifold"]:
            result, info = self._try_manifold(mesh_a, mesh_b, operation)
            if result is not None:
                return (result, info)
            if engine == "manifold":
                raise RuntimeError("Manifold backend failed and was explicitly requested")

        # Fallback to Blender
        if engine in ["auto", "blender"]:
            result, info = self._try_blender(mesh_a, mesh_b, operation)
            if result is not None:
                return (result, info)

        raise RuntimeError(f"Boolean operation failed with all available backends")

    def _try_manifold(self, mesh_a, mesh_b, operation):
        """Try boolean operation using manifold3d (via trimesh)."""
        try:
            print(f"[Boolean] Attempting manifold backend...")

            # trimesh uses manifold3d for boolean operations
            if operation == "union":
                result = mesh_a.union(mesh_b, engine="manifold")
            elif operation == "difference":
                result = mesh_a.difference(mesh_b, engine="manifold")
            elif operation == "intersection":
                result = mesh_a.intersection(mesh_b, engine="manifold")
            else:
                raise ValueError(f"Unknown operation: {operation}")

            # Preserve metadata from mesh_a
            result.metadata = mesh_a.metadata.copy()
            result.metadata['boolean'] = {
                'operation': operation,
                'engine': 'manifold',
                'mesh_a_vertices': len(mesh_a.vertices),
                'mesh_a_faces': len(mesh_a.faces),
                'mesh_b_vertices': len(mesh_b.vertices),
                'mesh_b_faces': len(mesh_b.faces),
                'result_vertices': len(result.vertices),
                'result_faces': len(result.faces)
            }

            info = f"""Boolean Operation Results:

Operation: {operation.upper()}
Engine: manifold

Mesh A:
  Vertices: {len(mesh_a.vertices):,}
  Faces: {len(mesh_a.faces):,}

Mesh B:
  Vertices: {len(mesh_b.vertices):,}
  Faces: {len(mesh_b.faces):,}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""

            print(f"[Boolean] Manifold success: {len(result.vertices)} vertices, {len(result.faces)} faces")
            return result, info

        except Exception as e:
            print(f"[Boolean] Manifold backend failed: {e}")
            return None, str(e)

    def _try_blender(self, mesh_a, mesh_b, operation):
        """Try boolean operation using Blender."""
        try:
            from . import blender_utils
            import tempfile
            import os

            print(f"[Boolean] Attempting Blender backend...")

            # Create temp files for both meshes
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_a:
                input_a_path = f_a.name
                mesh_a.export(input_a_path)

            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_b:
                input_b_path = f_b.name
                mesh_b.export(input_b_path)

            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f_out:
                output_path = f_out.name

            try:
                # Map operation to Blender modifier type
                blender_op = {
                    "union": "UNION",
                    "difference": "DIFFERENCE",
                    "intersection": "INTERSECT"
                }[operation]

                script = f"""
import bpy

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import mesh A
bpy.ops.wm.obj_import(filepath='{input_a_path}')
obj_a = bpy.context.selected_objects[0]
obj_a.name = "MeshA"

# Import mesh B
bpy.ops.wm.obj_import(filepath='{input_b_path}')
obj_b = bpy.context.selected_objects[0]
obj_b.name = "MeshB"

# Select mesh A as active
bpy.ops.object.select_all(action='DESELECT')
obj_a.select_set(True)
bpy.context.view_layer.objects.active = obj_a

# Add boolean modifier
bool_mod = obj_a.modifiers.new(name="Boolean", type='BOOLEAN')
bool_mod.operation = '{blender_op}'
bool_mod.object = obj_b
bool_mod.solver = 'EXACT'

# Apply modifier
bpy.ops.object.modifier_apply(modifier="Boolean")

# Delete mesh B
bpy.data.objects.remove(obj_b, do_unlink=True)

# Export result
bpy.ops.wm.obj_export(
    filepath='{output_path}',
    export_selected_objects=True,
    export_uv=False,
    export_materials=False
)
"""

                blender_utils.run_blender_script(script, timeout=300)

                # Load result
                result = trimesh_module.load(output_path, process=False)
                if isinstance(result, trimesh_module.Scene):
                    result = result.dump(concatenate=True)

                # Preserve metadata
                result.metadata = mesh_a.metadata.copy()
                result.metadata['boolean'] = {
                    'operation': operation,
                    'engine': 'blender',
                    'mesh_a_vertices': len(mesh_a.vertices),
                    'mesh_a_faces': len(mesh_a.faces),
                    'mesh_b_vertices': len(mesh_b.vertices),
                    'mesh_b_faces': len(mesh_b.faces),
                    'result_vertices': len(result.vertices),
                    'result_faces': len(result.faces)
                }

                info = f"""Boolean Operation Results:

Operation: {operation.upper()}
Engine: blender (EXACT solver)

Mesh A:
  Vertices: {len(mesh_a.vertices):,}
  Faces: {len(mesh_a.faces):,}

Mesh B:
  Vertices: {len(mesh_b.vertices):,}
  Faces: {len(mesh_b.faces):,}

Result:
  Vertices: {len(result.vertices):,}
  Faces: {len(result.faces):,}

Watertight: {result.is_watertight}
"""

                print(f"[Boolean] Blender success: {len(result.vertices)} vertices, {len(result.faces)} faces")
                return result, info

            finally:
                # Cleanup temp files
                for path in [input_a_path, input_b_path, output_path]:
                    if os.path.exists(path):
                        os.unlink(path)

        except Exception as e:
            print(f"[Boolean] Blender backend failed: {e}")
            return None, str(e)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackBooleanOp": BooleanOpNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBooleanOp": "Boolean",
}
