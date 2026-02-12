"""
Texture to Geometry Node - Convert texture heightmap to 3D geometry
"""

import numpy as np
import trimesh


class TextureToGeometryNode:
    """
    Texture to Geometry - Convert a heightmap texture to 3D mesh geometry.

    Takes an IMAGE (heightmap) and converts it to a displacement-mapped mesh.
    The brightness of each pixel determines the height/displacement of the
    corresponding vertex. Useful for creating terrain, embossed surfaces, or
    converting 2D textures to 3D relief.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "height_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "base_resolution": ("INT", {
                    "default": 128,
                    "min": 16,
                    "max": 512,
                    "step": 16,
                    "tooltip": "Resolution of the base mesh (width/height in vertices)"
                }),
            },
            "optional": {
                "invert_height": (["false", "true"], {"default": "false"}),
                "smooth_normals": (["true", "false"], {"default": "true"}),
            }
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "texture_to_geometry"
    CATEGORY = "geompack/texture_remeshing"

    def texture_to_geometry(self, mask, height_scale, base_resolution,
                           invert_height="false", smooth_normals="true"):
        """
        Convert binary mask to 3D mesh with height displacement.

        Args:
            mask: Input MASK tensor (B, H, W) from ComfyUI (binary 0/1 values)
            height_scale: Scale factor for height displacement
            base_resolution: Resolution of base mesh grid
            invert_height: Invert the mask (0=high, 1=low)
            smooth_normals: Compute smooth vertex normals

        Returns:
            tuple: (mesh, info_string)
        """
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise RuntimeError("torch and PIL required. Install with: pip install torch Pillow")

        print(f"[TextureToGeometry] Converting mask to geometry")

        # Extract mask from ComfyUI tensor format (B, H, W)
        if isinstance(mask, torch.Tensor):
            # Get first mask in batch
            heightmap = mask[0].cpu().numpy()
        else:
            heightmap = np.array(mask)

        # Ensure 2D array (masks are single-channel)
        if len(heightmap.shape) > 2:
            heightmap = heightmap[:, :, 0] if heightmap.shape[2] == 1 else heightmap

        print(f"[TextureToGeometry] Mask size: {heightmap.shape}, range: [{heightmap.min():.3f}, {heightmap.max():.3f}]")

        # Resize mask to base resolution (already normalized 0-1)
        heightmap_pil = Image.fromarray((heightmap * 255).astype(np.uint8))
        heightmap_pil = heightmap_pil.resize((base_resolution, base_resolution), Image.Resampling.LANCZOS)
        heightmap = np.array(heightmap_pil).astype(np.float32) / 255.0

        # Invert if requested
        if invert_height == "true":
            heightmap = 1.0 - heightmap

        # Create grid mesh
        width, height = base_resolution, base_resolution

        # Generate vertices
        vertices = []
        for y in range(height):
            for x in range(width):
                # Normalize x, y to [-1, 1]
                nx = (x / (width - 1)) * 2.0 - 1.0
                ny = (y / (height - 1)) * 2.0 - 1.0

                # Get height from heightmap
                h = heightmap[y, x] * height_scale

                vertices.append([nx, ny, h])

        vertices = np.array(vertices, dtype=np.float32)

        # Generate faces (triangles)
        faces = []
        for y in range(height - 1):
            for x in range(width - 1):
                # Current vertex index
                i = y * width + x

                # Create two triangles per quad
                # Triangle 1: (i, i+1, i+width)
                faces.append([i, i + 1, i + width])
                # Triangle 2: (i+1, i+width+1, i+width)
                faces.append([i + 1, i + width + 1, i + width])

        faces = np.array(faces, dtype=np.int32)

        # Create trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Compute normals
        if smooth_normals == "true":
            mesh.fix_normals()

        print(f"[TextureToGeometry] Created mesh: {len(vertices)} vertices, {len(faces)} faces")

        # Compute statistics
        height_min = vertices[:, 2].min()
        height_max = vertices[:, 2].max()
        height_range = height_max - height_min

        info = f"""Depth Map to Mesh Results:

Input:
  Mask Size: {mask.shape[1] if isinstance(mask, torch.Tensor) else heightmap.shape[0]}x{mask.shape[2] if isinstance(mask, torch.Tensor) else heightmap.shape[1]}
  Base Resolution: {base_resolution}x{base_resolution}
  Height Scale: {height_scale}
  Inverted: {invert_height}

Output Mesh:
  Vertices: {len(vertices):,}
  Faces: {len(faces):,}
  Height Range: [{height_min:.3f}, {height_max:.3f}] (span: {height_range:.3f})
  Bounds: {mesh.bounds.tolist()}

Note: Binary mask values (0/1) control vertex displacement in Z-axis.
"""

        return (mesh, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackTextureToGeometry": TextureToGeometryNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackTextureToGeometry": "Depth Map to Mesh",
}
