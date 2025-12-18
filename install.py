#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack Installer

Orchestrates modular install scripts for ComfyUI-GeometryPack.
Blender is optional - run 'python blender_install.py' separately if needed.
"""

import sys

from install_scripts import (
    install_system_dependencies,
    install_python_dependencies,
    install_cumesh,
)


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Installation")
    print("="*60 + "\n")
    print("This installer will set up:")
    print("  1. System dependencies (OpenGL libraries on Linux)")
    print("  2. Python dependencies (trimesh, pymeshlab, etc.)")
    print("  3. CuMesh (optional, GPU-accelerated mesh processing)")
    print("")
    print("Note: Blender is optional. If you need Blender nodes")
    print("(UV Unwrap, Remesh, etc.), run: python blender_install.py")
    print("")

    results = {
        'system_deps': False,
        'python_deps': False,
        'cumesh': False,
    }

    # Install in order
    results['system_deps'] = install_system_dependencies()
    results['python_deps'] = install_python_dependencies()
    results['cumesh'] = install_cumesh()

    # Print summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    print(f"  System Dependencies: {'+ Success' if results['system_deps'] else 'x Failed'}")
    print(f"  Python Dependencies: {'+ Success' if results['python_deps'] else 'x Failed'}")
    print(f"  CuMesh (GPU):        {'+ Success' if results['cumesh'] else '~ Skipped'}")
    print("="*60 + "\n")

    if results['python_deps']:
        print("Installation completed successfully!")
        print("You can now use ComfyUI-GeometryPack nodes in ComfyUI.")
        print("")
        print("For Blender-based nodes (UV Unwrap, Remesh, etc.), run:")
        print("  python blender_install.py")
        print("")
        sys.exit(0)
    else:
        print("Installation completed with issues.")
        if not results['python_deps']:
            print("\nPython dependencies failed to install. You can:")
            print("  1. Try running: pip install -r requirements.txt")
            print("  2. Check your Python environment and permissions")
        print("")
        sys.exit(1)


if __name__ == "__main__":
    main()
