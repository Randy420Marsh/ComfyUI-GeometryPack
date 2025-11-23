# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh FBX Node - Load FBX files with automatic conversion to GLB
"""

import os
import subprocess

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None

from .._utils import mesh_ops, blender_bridge


def _convert_fbx_to_glb(fbx_path, cache_dir=None):
    """
    Convert FBX file to GLB using Blender.

    Args:
        fbx_path: Path to FBX file
        cache_dir: Directory for caching converted files (optional)

    Returns:
        str: Path to converted GLB file

    Raises:
        RuntimeError: If conversion fails
    """
    print(f"[FBX→GLB] Converting: {fbx_path}")

    # Setup cache
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(fbx_path), ".fbx_cache")

    os.makedirs(cache_dir, exist_ok=True)

    # Generate cache filename
    fbx_basename = os.path.basename(fbx_path)
    fbx_name_no_ext = os.path.splitext(fbx_basename)[0]
    glb_cache_path = os.path.join(cache_dir, f"{fbx_name_no_ext}.glb")

    # Check if cached GLB exists and is newer than FBX
    if os.path.exists(glb_cache_path):
        fbx_mtime = os.path.getmtime(fbx_path)
        glb_mtime = os.path.getmtime(glb_cache_path)
        if glb_mtime > fbx_mtime:
            print(f"[FBX→GLB] Using cached GLB: {glb_cache_path}")
            return glb_cache_path

    # Find Blender
    try:
        blender_path = blender_bridge.find_blender()
    except RuntimeError as e:
        raise RuntimeError(f"FBX conversion requires Blender: {e}")

    # Convert FBX to GLB using Blender
    script = f"""
import bpy
import sys

try:
    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    print("[Blender] Importing FBX...")
    bpy.ops.import_scene.fbx(filepath='{fbx_path}')

    # Export as GLB
    print("[Blender] Exporting GLB...")
    bpy.ops.export_scene.gltf(
        filepath='{glb_cache_path}',
        export_format='GLB',
        export_image_format='AUTO',
        export_materials='EXPORT'
    )

    print("[Blender] Conversion complete!")
    sys.exit(0)

except Exception as e:
    print(f"[Blender] Error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""

    print(f"[FBX→GLB] Running Blender conversion...")
    result = subprocess.run(
        [blender_path, '--background', '--python-expr', script],
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        error_msg = f"Blender conversion failed:\n{result.stderr}"
        print(f"[FBX→GLB] {error_msg}")
        raise RuntimeError(error_msg)

    if not os.path.exists(glb_cache_path):
        raise RuntimeError(f"GLB file was not created: {glb_cache_path}")

    print(f"[FBX→GLB] ✓ Converted successfully: {glb_cache_path}")
    return glb_cache_path


class LoadMeshFBX:
    """
    Load FBX files with automatic conversion to GLB.

    FBX files are not natively supported by trimesh, so this node automatically
    converts them to GLB using Blender, then loads the result. Converted files
    are cached to avoid repeated conversions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of FBX files only
        fbx_files = cls.get_fbx_files()

        if not fbx_files:
            fbx_files = ["No FBX files found in input/3d or input folders"]

        return {
            "required": {
                "file_path": (fbx_files, ),
            },
        }

    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("mesh", "info")
    FUNCTION = "load_fbx"
    CATEGORY = "geompack/io"

    @classmethod
    def get_fbx_files(cls):
        """Get list of available FBX files in input/3d and input folders."""
        fbx_files = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Scan input/3d first
            input_3d = os.path.join(COMFYUI_INPUT_FOLDER, "3d")
            if os.path.exists(input_3d):
                for file in os.listdir(input_3d):
                    if file.lower().endswith('.fbx'):
                        fbx_files.append(f"3d/{file}")

            # Then scan input root
            for file in os.listdir(COMFYUI_INPUT_FOLDER):
                file_path = os.path.join(COMFYUI_INPUT_FOLDER, file)
                if os.path.isfile(file_path) and file.lower().endswith('.fbx'):
                    fbx_files.append(file)

        return sorted(fbx_files)

    @classmethod
    def IS_CHANGED(cls, file_path):
        """Force re-execution when file changes."""
        if COMFYUI_INPUT_FOLDER is not None:
            # Check file modification time
            full_path = None
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)

            if os.path.exists(input_3d_path):
                full_path = input_3d_path
            elif os.path.exists(input_path):
                full_path = input_path

            if full_path and os.path.exists(full_path):
                return os.path.getmtime(full_path)

        return file_path

    def load_fbx(self, file_path):
        """
        Load FBX file by converting to GLB first.

        Args:
            file_path: Path to FBX file (relative to input folder or absolute)

        Returns:
            tuple: (trimesh.Trimesh, info_string)
        """
        if not file_path or file_path.strip() == "":
            raise ValueError("File path cannot be empty")

        # Try to find the FBX file
        full_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # First, try in ComfyUI input/3d folder
            input_3d_path = os.path.join(COMFYUI_INPUT_FOLDER, "3d", file_path)
            searched_paths.append(input_3d_path)
            if os.path.exists(input_3d_path):
                full_path = input_3d_path
                print(f"[LoadMeshFBX] Found FBX in input/3d folder: {file_path}")

            # Second, try in ComfyUI input folder
            if full_path is None:
                input_path = os.path.join(COMFYUI_INPUT_FOLDER, file_path)
                searched_paths.append(input_path)
                if os.path.exists(input_path):
                    full_path = input_path
                    print(f"[LoadMeshFBX] Found FBX in input folder: {file_path}")

        # If not found in input folders, try as absolute path
        if full_path is None:
            searched_paths.append(file_path)
            if os.path.exists(file_path):
                full_path = file_path
                print(f"[LoadMeshFBX] Loading from absolute path: {file_path}")
            else:
                # Generate error message with all searched paths
                error_msg = f"File not found: '{file_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Convert FBX to GLB
        try:
            glb_path = _convert_fbx_to_glb(full_path)
        except RuntimeError as e:
            raise ValueError(f"Failed to convert FBX to GLB: {e}")

        # Load the GLB mesh
        loaded_mesh, error = mesh_ops.load_mesh_file(glb_path)

        if loaded_mesh is None:
            raise ValueError(f"Failed to load converted GLB: {error}")

        # Generate info string
        info = f"FBX Loaded (auto-converted to GLB)\n"
        info += f"Original: {os.path.basename(full_path)}\n"
        info += f"Converted: {os.path.basename(glb_path)}\n"
        info += f"Vertices: {len(loaded_mesh.vertices):,}\n"
        info += f"Faces: {len(loaded_mesh.faces):,}"

        print(f"[LoadMeshFBX] ✓ Loaded: {len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")

        return (loaded_mesh, info)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshFBX": LoadMeshFBX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshFBX": "Load Mesh (FBX)",
}
