"""
Texture Remeshing Nodes - Remesh with texture preservation
"""

import numpy as np
import trimesh as trimesh_module
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Optional

from . import mesh_utils
from . import blender_utils

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[texture_remeshing] Warning: PIL not available")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[texture_remeshing] Warning: torch not available")

try:
    import xatlas
    XATLAS_AVAILABLE = True
except ImportError:
    XATLAS_AVAILABLE = False
    print("[texture_remeshing] Warning: xatlas not available")

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[texture_remeshing] Warning: scipy not available")


def _extract_texture(mesh: trimesh_module.Trimesh) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """
    Extract texture from trimesh visual to temp file.

    Args:
        mesh: trimesh.Trimesh object with texture

    Returns:
        tuple: (texture_path, uv_coordinates) or (None, None) if no texture
    """
    if not PIL_AVAILABLE:
        print("[extract_texture] PIL not available, cannot extract texture")
        return None, None

    # Check if mesh has texture
    if not hasattr(mesh, 'visual'):
        print("[extract_texture] No visual attribute")
        return None, None

    # Get UV coordinates
    uvs = None
    if hasattr(mesh.visual, 'uv'):
        uvs = mesh.visual.uv
        print(f"[extract_texture] Found UVs: {uvs.shape if uvs is not None else 'None'}")
    else:
        print("[extract_texture] No UV coordinates")

    # Try to get texture image
    texture_image = None

    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        print(f"[extract_texture] Material type: {type(material).__name__}")

        # Try SimpleMaterial.image
        if hasattr(material, 'image'):
            img = material.image
            if isinstance(img, Image.Image):
                texture_image = img
                print(f"[extract_texture] Found SimpleMaterial.image (PIL): {img.size}")
            elif isinstance(img, str) and os.path.exists(img):
                texture_image = Image.open(img)
                print(f"[extract_texture] Found SimpleMaterial.image (path): {img}")

        # Try PBRMaterial.baseColorTexture
        if texture_image is None and hasattr(material, 'baseColorTexture'):
            base_color_tex = material.baseColorTexture
            if base_color_tex is not None:
                if isinstance(base_color_tex, Image.Image):
                    texture_image = base_color_tex
                    print(f"[extract_texture] Found PBRMaterial.baseColorTexture (PIL): {texture_image.size}")
                elif isinstance(base_color_tex, str) and os.path.exists(base_color_tex):
                    texture_image = Image.open(base_color_tex)
                    print(f"[extract_texture] Found PBRMaterial.baseColorTexture (path): {base_color_tex}")

        # Try other PBRMaterial textures as fallback
        if texture_image is None:
            for tex_attr in ['emissiveTexture', 'metallicRoughnessTexture', 'normalTexture', 'occlusionTexture']:
                if hasattr(material, tex_attr):
                    tex = getattr(material, tex_attr)
                    if tex is not None:
                        if isinstance(tex, Image.Image):
                            texture_image = tex
                            print(f"[extract_texture] Found PBRMaterial.{tex_attr} (PIL): {texture_image.size}")
                            break
                        elif isinstance(tex, str) and os.path.exists(tex):
                            texture_image = Image.open(tex)
                            print(f"[extract_texture] Found PBRMaterial.{tex_attr} (path): {tex}")
                            break

    if texture_image is None:
        print("[extract_texture] No texture image found")
        # Generate a dummy texture if we have UVs but no texture
        if uvs is not None:
            print("[extract_texture] Creating checkerboard placeholder texture")
            # Create a simple checkerboard pattern
            texture_image = Image.new('RGB', (512, 512), color=(200, 200, 200))
            import numpy as np
            arr = np.array(texture_image)
            # Add checkerboard pattern
            for i in range(0, 512, 64):
                for j in range(0, 512, 64):
                    if (i // 64 + j // 64) % 2 == 0:
                        arr[i:i+64, j:j+64] = [100, 100, 100]
            texture_image = Image.fromarray(arr)
        else:
            return None, uvs

    # Save texture to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    texture_image.save(temp_file.name)
    temp_file.close()

    print(f"[extract_texture] Extracted texture: {texture_image.size}, saved to {temp_file.name}")
    return temp_file.name, uvs


def _attach_texture(mesh: trimesh_module.Trimesh, texture_path: str, uvs: np.ndarray) -> trimesh_module.Trimesh:
    """
    Attach texture to mesh.

    Args:
        mesh: trimesh.Trimesh object
        texture_path: Path to texture image
        uvs: UV coordinates (N x 2)

    Returns:
        trimesh.Trimesh with texture attached
    """
    if not PIL_AVAILABLE:
        print("[attach_texture] PIL not available, cannot attach texture")
        return mesh

    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import SimpleMaterial

    img = Image.open(texture_path)
    material = SimpleMaterial(image=img)
    mesh.visual = TextureVisuals(uv=uvs, material=material)

    print(f"[attach_texture] Attached texture: {img.size}, UVs shape: {uvs.shape}")
    return mesh


def _load_as_comfy_image(texture_path: str):
    """
    Convert texture to ComfyUI IMAGE format (torch tensor).

    Args:
        texture_path: Path to texture image

    Returns:
        torch.Tensor in ComfyUI IMAGE format (1, H, W, 3) with values in [0, 1]
    """
    if not PIL_AVAILABLE or not TORCH_AVAILABLE:
        print("[load_as_comfy_image] PIL or torch not available")
        return None

    img = Image.open(texture_path).convert("RGB")
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


class BlenderRemeshWithTexture:
    """
    Remesh with texture preservation using Blender baking.

    This node remeshes a textured mesh while preserving its appearance through
    texture baking. It combines remeshing, UV unwrapping, and texture baking
    into one operation using Blender's Cycles renderer.

    Workflow:
    1. Remeshes the input geometry using selected algorithm
    2. Generates new UV coordinates for the remeshed surface
    3. Bakes the original texture onto the new UVs using Blender
    4. Returns remeshed mesh with baked texture

    Best for: Retopology of scanned/sculpted textured models
    Requires: Mesh with texture data (material.image) and Blender
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "remesh_method": (["voxel", "quadriflow"], {"default": "quadriflow"}),
                "voxel_size": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.001,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "target_face_count": ("INT", {
                    "default": 5000,
                    "min": 100,
                    "max": 1000000,
                    "step": 100
                }),
                "texture_size": ([512, 1024, 2048, 4096], {"default": 2048}),
                "bake_margin": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 64,
                    "step": 1
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "baked_texture", "info")
    FUNCTION = "remesh_with_texture"
    CATEGORY = "geompack/blender"

    def remesh_with_texture(self, trimesh, remesh_method, voxel_size, target_face_count,
                           texture_size, bake_margin):
        """
        Remesh a textured mesh while preserving texture through Blender baking.
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required for texture operations. Install with: pip install Pillow")

        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required for ComfyUI IMAGE output. Install with: pip install torch")

        print(f"[BlenderRemeshWithTexture] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")
        print(f"[BlenderRemeshWithTexture] Method: {remesh_method}")

        # 1. Extract texture from source mesh
        texture_path, original_uvs = _extract_texture(trimesh)
        if texture_path is None:
            raise ValueError("Input mesh must have texture data. Use a mesh loaded from OBJ/GLB with textures.")

        # 2. Find Blender
        blender_path = blender_utils.find_blender()

        # 3. Create temp files
        source_obj = tempfile.NamedTemporaryFile(suffix='_source.obj', delete=False)
        output_obj = tempfile.NamedTemporaryFile(suffix='_output.obj', delete=False)
        output_mtl = tempfile.NamedTemporaryFile(suffix='_output.mtl', delete=False)
        baked_texture = tempfile.NamedTemporaryFile(suffix='_baked.png', delete=False)

        try:
            # Export source mesh with UVs
            print(f"[BlenderRemeshWithTexture] Exporting source mesh to OBJ...")
            print(f"[BlenderRemeshWithTexture]   Source mesh has UVs: {trimesh.visual.uv is not None}")
            if trimesh.visual.uv is not None:
                print(f"[BlenderRemeshWithTexture]   UV shape: {trimesh.visual.uv.shape}")
            print(f"[BlenderRemeshWithTexture]   Visual kind: {trimesh.visual.kind}")
            trimesh.export(source_obj.name, include_texture=True)
            source_obj.close()
            print(f"[BlenderRemeshWithTexture]   Exported to: {source_obj.name}")

            # 4. Build Blender script for remeshing + UV unwrapping + texture baking
            if remesh_method == "voxel":
                remesh_code = f"obj.data.remesh_voxel_size = {voxel_size}\nbpy.ops.object.voxel_remesh()"
            else:  # quadriflow
                remesh_code = f"bpy.ops.object.quadriflow_remesh(target_faces={target_face_count})"

            script = f"""
import bpy
import os

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import source mesh (with original UVs and texture)
bpy.ops.wm.obj_import(filepath='{source_obj.name}')
source_obj = bpy.context.selected_objects[0]
source_obj.name = 'Source'

# DEBUG: Verify UV import
print(f"[Blender] Source mesh vertices: {{len(source_obj.data.vertices)}}")
print(f"[Blender] Source mesh UV layers: {{len(source_obj.data.uv_layers)}}")
if len(source_obj.data.uv_layers) > 0:
    for i, uv_layer in enumerate(source_obj.data.uv_layers):
        print(f"[Blender]   UV Layer {{i}}: {{uv_layer.name}}")
else:
    print("[Blender] WARNING: No UV layers found after import!")

# Duplicate for remeshing
bpy.ops.object.duplicate()
target_obj = bpy.context.selected_objects[0]
target_obj.name = 'Target'

# Apply remeshing
bpy.context.view_layer.objects.active = target_obj
{remesh_code}

# UV unwrap the remeshed mesh using Smart UV Project
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
bpy.ops.object.mode_set(mode='OBJECT')

# Setup source material with texture
if len(source_obj.data.materials) > 0:
    source_mat = source_obj.data.materials[0]
else:
    source_mat = bpy.data.materials.new('SourceMat')
    source_obj.data.materials.append(source_mat)

source_mat.use_nodes = True
source_nodes = source_mat.node_tree.nodes
source_nodes.clear()

# Create texture node with source texture
tex_coord = source_nodes.new('ShaderNodeTexCoord')
tex_image = source_nodes.new('ShaderNodeTexImage')
loaded_image = bpy.data.images.load('{texture_path}')
tex_image.image = loaded_image
print(f"[Blender] Loaded texture: {{loaded_image.name}}, size: {{loaded_image.size[0]}}x{{loaded_image.size[1]}}")
bsdf = source_nodes.new('ShaderNodeBsdfDiffuse')
output_node = source_nodes.new('ShaderNodeOutputMaterial')
source_mat.node_tree.links.new(tex_coord.outputs['UV'], tex_image.inputs['Vector'])
source_mat.node_tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Color'])
source_mat.node_tree.links.new(bsdf.outputs['BSDF'], output_node.inputs['Surface'])
print(f"[Blender] Source material shader nodes connected")

# Setup target material with bake image
if len(target_obj.data.materials) > 0:
    target_mat = target_obj.data.materials[0]
else:
    target_mat = bpy.data.materials.new('TargetMat')
    target_obj.data.materials.append(target_mat)

target_mat.use_nodes = True
target_nodes = target_mat.node_tree.nodes
target_nodes.clear()

# Create bake target image
bake_image = bpy.data.images.new('BakedTexture', {texture_size}, {texture_size})
bake_node = target_nodes.new('ShaderNodeTexImage')
bake_node.image = bake_image
bake_node.select = True
target_nodes.active = bake_node

# Select both objects for baking (source first, then target)
bpy.ops.object.select_all(action='DESELECT')
source_obj.select_set(True)
target_obj.select_set(True)
bpy.context.view_layer.objects.active = target_obj

# Configure bake settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'CPU'
bpy.context.scene.cycles.samples = 32
bpy.context.scene.render.bake.use_selected_to_active = True
bpy.context.scene.render.bake.margin = {bake_margin}
bpy.context.scene.render.bake.cage_extrusion = 0.1
bpy.context.scene.render.bake.max_ray_distance = 1.0

# CRITICAL: Configure DIFFUSE pass to only capture color, not lighting
bpy.context.scene.render.bake.use_pass_direct = False
bpy.context.scene.render.bake.use_pass_indirect = False
bpy.context.scene.render.bake.use_pass_color = True

print("[Blender] Starting texture bake...")
try:
    bpy.ops.object.bake(type='DIFFUSE')
    print("[Blender] Bake completed successfully")
except Exception as e:
    print(f"[Blender] Bake error: {{e}}")
    raise

# Save baked texture
bake_image.filepath_raw = '{baked_texture.name}'
bake_image.file_format = 'PNG'
bake_image.save()
print(f"[Blender] Saved baked texture to: {baked_texture.name}")

# Export target mesh with new UVs
bpy.ops.object.select_all(action='DESELECT')
target_obj.select_set(True)
bpy.ops.wm.obj_export(
    filepath='{output_obj.name}',
    export_selected_objects=True,
    export_uv=True,
    export_materials=True
)
print(f"[Blender] Exported remeshed mesh to: {output_obj.name}")
"""

            print(f"[BlenderRemeshWithTexture] Running Blender for remeshing + baking...")
            result = subprocess.run(
                [blender_path, '--background', '--python-expr', script],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode != 0:
                print(f"[BlenderRemeshWithTexture] Blender stderr: {result.stderr}")
                raise RuntimeError(f"Blender baking failed: {result.stderr}")

            # Print Blender output for debugging
            if result.stdout:
                print(f"[BlenderRemeshWithTexture] Blender output:\n{result.stdout}")

            # Load the remeshed mesh
            print(f"[BlenderRemeshWithTexture] Loading remeshed mesh from {output_obj.name}")
            remeshed = trimesh_module.load(output_obj.name, process=False)

            # If it's a scene, dump to single mesh
            if isinstance(remeshed, trimesh_module.Scene):
                remeshed = remeshed.dump(concatenate=True)

            # Attach baked texture
            if hasattr(remeshed.visual, 'uv'):
                remeshed = _attach_texture(remeshed, baked_texture.name, remeshed.visual.uv)

            # Load texture as ComfyUI IMAGE
            comfy_image = _load_as_comfy_image(baked_texture.name)

            # Generate info string
            info = f"Remesh with Texture (Blender)\n"
            info += f"Method: {remesh_method}\n"
            info += f"Vertices: {len(trimesh.vertices)} -> {len(remeshed.vertices)}\n"
            info += f"Faces: {len(trimesh.faces)} -> {len(remeshed.faces)}\n"
            info += f"Texture size: {texture_size}x{texture_size}\n"
            info += f"Bake margin: {bake_margin} pixels"

            print(f"[BlenderRemeshWithTexture] ✓ Complete:")
            print(f"[BlenderRemeshWithTexture]   {info.replace(chr(10), chr(10) + '  ')}")

            return (remeshed, comfy_image, info)

        finally:
            # Cleanup temp files
            for path in [source_obj.name, output_obj.name, output_mtl.name, texture_path, baked_texture.name]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass


class XAtlasRemeshWithTexture:
    """
    Remesh with texture preservation using xatlas + simple transfer.

    This node remeshes a textured mesh and transfers the texture using closest-point
    sampling. Faster than Blender baking but lower quality.

    Workflow:
    1. Remeshes the input geometry using PyMeshLab
    2. UV unwraps with xatlas
    3. Samples texture from original mesh using nearest-neighbor lookup
    4. Returns remeshed mesh with transferred texture

    Best for: Quick prototyping where perfect quality isn't required
    Requires: Mesh with texture data, xatlas, scipy
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_edge_length": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 10.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 20,
                    "step": 1
                }),
                "texture_size": ([512, 1024, 2048, 4096], {"default": 1024}),
            },
        }

    RETURN_TYPES = ("TRIMESH", "IMAGE", "STRING")
    RETURN_NAMES = ("remeshed_mesh", "transferred_texture", "info")
    FUNCTION = "remesh_with_texture"
    CATEGORY = "geompack/remeshing"

    def remesh_with_texture(self, trimesh, target_edge_length, iterations, texture_size):
        """
        Remesh a textured mesh and transfer texture using simple sampling.
        """
        if not PIL_AVAILABLE:
            raise RuntimeError("PIL is required. Install with: pip install Pillow")

        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required. Install with: pip install torch")

        if not XATLAS_AVAILABLE:
            raise RuntimeError("xatlas is required. Install with: pip install xatlas")

        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required. Install with: pip install scipy")

        print(f"[XAtlasRemeshWithTexture] Input: {len(trimesh.vertices)} vertices, {len(trimesh.faces)} faces")

        # 1. Extract texture from source mesh
        texture_path, original_uvs = _extract_texture(trimesh)
        if texture_path is None or original_uvs is None:
            raise ValueError("Input mesh must have texture data and UVs")

        # Load texture image
        texture_img = Image.open(texture_path)
        texture_array = np.array(texture_img)
        # Ensure texture is RGB (not RGBA) for consistent handling
        if len(texture_array.shape) == 3 and texture_array.shape[2] == 4:
            texture_array = texture_array[:, :, :3]  # Drop alpha channel
        tex_height, tex_width = texture_array.shape[:2]

        # 2. Remesh using PyMeshLab
        print(f"[XAtlasRemeshWithTexture] Remeshing...")
        remeshed, error = mesh_utils.pymeshlab_isotropic_remesh(
            trimesh, target_edge_length, iterations
        )

        if remeshed is None:
            raise ValueError(f"Remeshing failed: {error}")

        # 3. UV unwrap with xatlas
        print(f"[XAtlasRemeshWithTexture] UV unwrapping with xatlas...")
        vmapping, indices, uvs = xatlas.parametrize(
            remeshed.vertices.astype(np.float32),
            remeshed.faces.astype(np.uint32)
        )

        # Apply UV mapping
        new_vertices = remeshed.vertices[vmapping]
        remeshed = trimesh_module.Trimesh(vertices=new_vertices, faces=indices, process=False)

        # 4. Transfer texture using nearest-neighbor sampling
        print(f"[XAtlasRemeshWithTexture] Transferring texture...")

        # Build KD-tree for original mesh vertices
        tree = cKDTree(trimesh.vertices)

        # For each new vertex, find closest original vertex
        distances, closest_indices = tree.query(remeshed.vertices)

        # Get UV coordinates of closest vertices
        closest_uvs = original_uvs[closest_indices]

        # Create new texture by sampling original texture
        new_texture = Image.new('RGB', (texture_size, texture_size))
        new_texture_array = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

        # Rasterize: for each face, sample texture
        for face_idx, face in enumerate(remeshed.faces):
            # Get UV coordinates for this face
            face_uvs = uvs[face]

            # Get closest original UVs
            orig_uvs = closest_uvs[face]

            # Sample original texture at mean UV position
            mean_uv = orig_uvs.mean(axis=0)
            u, v = mean_uv
            u = np.clip(u, 0, 1)
            v = np.clip(1 - v, 0, 1)  # Flip V

            tex_x = int(u * (tex_width - 1))
            tex_y = int(v * (tex_height - 1))
            color = texture_array[tex_y, tex_x]

            # Fill new texture at new UV positions
            for new_uv in face_uvs:
                u_new, v_new = new_uv
                u_new = np.clip(u_new, 0, 1)
                v_new = np.clip(1 - v_new, 0, 1)

                new_x = int(u_new * (texture_size - 1))
                new_y = int(v_new * (texture_size - 1))
                new_texture_array[new_y, new_x] = color

        # Apply simple dilation to fill gaps
        from scipy.ndimage import grey_dilation
        for channel in range(3):
            new_texture_array[:,:,channel] = grey_dilation(new_texture_array[:,:,channel], size=3)

        new_texture = Image.fromarray(new_texture_array)

        # Save new texture
        new_texture_path = tempfile.NamedTemporaryFile(suffix='_transferred.png', delete=False).name
        new_texture.save(new_texture_path)

        # 5. Attach texture to remeshed mesh
        remeshed = _attach_texture(remeshed, new_texture_path, uvs)

        # Load as ComfyUI IMAGE
        comfy_image = _load_as_comfy_image(new_texture_path)

        # Generate info
        info = f"Remesh with Texture (xatlas)\n"
        info += f"Vertices: {len(trimesh.vertices)} -> {len(remeshed.vertices)}\n"
        info += f"Faces: {len(trimesh.faces)} -> {len(remeshed.faces)}\n"
        info += f"Texture size: {texture_size}x{texture_size}\n"
        info += f"Method: Nearest-neighbor sampling"

        print(f"[XAtlasRemeshWithTexture] ✓ Complete")

        # Cleanup
        try:
            os.unlink(texture_path)
            os.unlink(new_texture_path)
        except:
            pass

        return (remeshed, comfy_image, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackBlenderRemeshWithTexture": BlenderRemeshWithTexture,
    "GeomPackXAtlasRemeshWithTexture": XAtlasRemeshWithTexture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackBlenderRemeshWithTexture": "Blender Remesh with Texture",
    "GeomPackXAtlasRemeshWithTexture": "xAtlas Remesh with Texture",
}
