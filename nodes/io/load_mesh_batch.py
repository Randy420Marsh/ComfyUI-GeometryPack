# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Load Mesh Batch Node - Load multiple meshes from a folder (batch loading)
"""

import os

# ComfyUI folder paths
try:
    import folder_paths
    COMFYUI_INPUT_FOLDER = folder_paths.get_input_directory()
except (ImportError, AttributeError):
    # Fallback if folder_paths not available (e.g., during testing)
    COMFYUI_INPUT_FOLDER = None

from .._utils import mesh_ops


class LoadMeshBatch:
    """
    Load multiple meshes from a folder (batch loading).
    Similar to ComfyUI's image batch loading, with start_index and max_meshes controls.
    """

    # Supported mesh file extensions
    SUPPORTED_EXTENSIONS = ['.obj', '.ply', '.stl', '.off', '.gltf', '.glb', '.fbx', '.dae', '.3ds', '.vtp']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "3d",
                    "multiline": False
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000
                }),
                "max_meshes": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100000
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("meshes",)
    FUNCTION = "load_mesh_batch"
    CATEGORY = "geompack/io"
    OUTPUT_IS_LIST = (True,)

    def load_mesh_batch(self, folder_path, start_index, max_meshes):
        """
        Load multiple meshes from a folder.

        Args:
            folder_path: Path to folder containing mesh files (relative to input folder or absolute)
            start_index: Skip first N meshes (0 = start from beginning)
            max_meshes: Load up to N meshes (-1 = unlimited)

        Returns:
            tuple: (list of trimesh.Trimesh objects,)
        """
        if not folder_path or folder_path.strip() == "":
            raise ValueError("Folder path cannot be empty")

        # Resolve folder path
        full_folder_path = None
        searched_paths = []

        if COMFYUI_INPUT_FOLDER is not None:
            # Try in ComfyUI input folder
            input_path = os.path.join(COMFYUI_INPUT_FOLDER, folder_path)
            searched_paths.append(input_path)
            if os.path.exists(input_path) and os.path.isdir(input_path):
                full_folder_path = input_path
                print(f"[LoadMeshBatch] Found folder in input: {folder_path}")

        # If not found, try as absolute path
        if full_folder_path is None:
            searched_paths.append(folder_path)
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                full_folder_path = folder_path
                print(f"[LoadMeshBatch] Using absolute path: {folder_path}")
            else:
                error_msg = f"Folder not found: '{folder_path}'\nSearched in:"
                for path in searched_paths:
                    error_msg += f"\n  - {path}"
                raise ValueError(error_msg)

        # Scan folder for mesh files
        mesh_files = []
        for filename in os.listdir(full_folder_path):
            file_lower = filename.lower()
            if any(file_lower.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                mesh_files.append(filename)

        # Sort files alphabetically for consistent ordering
        mesh_files.sort()

        if len(mesh_files) == 0:
            raise ValueError(f"No mesh files found in folder: {full_folder_path}\n"
                           f"Supported extensions: {', '.join(self.SUPPORTED_EXTENSIONS)}")

        print(f"[LoadMeshBatch] Found {len(mesh_files)} mesh files")

        # Apply start_index and max_meshes
        if start_index > 0:
            if start_index >= len(mesh_files):
                raise ValueError(f"start_index ({start_index}) is >= number of mesh files ({len(mesh_files)})")
            mesh_files = mesh_files[start_index:]
            print(f"[LoadMeshBatch] Skipping first {start_index} files")

        if max_meshes > 0:
            mesh_files = mesh_files[:max_meshes]
            print(f"[LoadMeshBatch] Loading up to {max_meshes} meshes")

        # Load all meshes
        loaded_meshes = []
        for i, filename in enumerate(mesh_files):
            file_path = os.path.join(full_folder_path, filename)
            try:
                loaded_mesh, error = mesh_ops.load_mesh_file(file_path)
                if loaded_mesh is None:
                    print(f"[LoadMeshBatch] Warning: Failed to load {filename}: {error}")
                    continue

                loaded_meshes.append(loaded_mesh)
                print(f"[LoadMeshBatch] [{i+1}/{len(mesh_files)}] Loaded {filename}: "
                      f"{len(loaded_mesh.vertices)} vertices, {len(loaded_mesh.faces)} faces")
            except Exception as e:
                print(f"[LoadMeshBatch] Warning: Error loading {filename}: {e}")
                continue

        if len(loaded_meshes) == 0:
            raise ValueError(f"Failed to load any meshes from folder: {full_folder_path}")

        print(f"[LoadMeshBatch] Successfully loaded {len(loaded_meshes)} meshes")

        return (loaded_meshes,)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "GeomPackLoadMeshBatch": LoadMeshBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeomPackLoadMeshBatch": "Load Mesh Batch",
}
