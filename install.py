#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
GeometryPack Installer

Installs system dependencies and Python packages for ComfyUI-GeometryPack.
Blender is optional - run 'python blender_install.py' separately if needed.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def install_system_dependencies():
    """Install required system dependencies (Linux only)."""
    plat, _ = get_platform_info()

    if plat != "linux":
        return True

    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: System Dependencies")
    print("="*60 + "\n")

    print("[Install] Checking for required OpenGL libraries...")
    print("[Install] These are needed for PyMeshLab remeshing to work properly.")

    try:
        critical_packages = ["libgl1", "libopengl0", "libglu1-mesa", "libglx-mesa0"]
        optional_packages = ["libosmesa6"]

        all_packages = critical_packages + optional_packages
        print(f"[Install] Installing OpenGL libraries: {', '.join(all_packages)}")
        print("[Install] You may be prompted for your sudo password...")

        print("[Install] Updating apt cache...")
        update_result = subprocess.run(
            ['sudo', 'apt-get', 'update'],
            capture_output=True,
            text=True,
            timeout=120
        )

        if update_result.returncode != 0:
            print("[Install] Warning: Failed to update apt cache")
            print(f"[Install] You may need to run manually: sudo apt-get update")

        installed_packages = []
        failed_packages = []
        critical_failed = []

        print("[Install] Installing critical OpenGL libraries...")
        for package in critical_packages:
            result = subprocess.run(
                ['sudo', 'apt-get', 'install', '-y', package],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                installed_packages.append(package)
                print(f"[Install]   + {package}")
            else:
                failed_packages.append(package)
                critical_failed.append(package)
                print(f"[Install]   x {package} (failed)")

        print("[Install] Installing optional OpenGL libraries...")
        for package in optional_packages:
            result = subprocess.run(
                ['sudo', 'apt-get', 'install', '-y', package],
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode == 0:
                installed_packages.append(package)
                print(f"[Install]   + {package}")
            else:
                failed_packages.append(package)
                print(f"[Install]   ~ {package} (optional, skipped)")

        print("[Install] Verifying OpenGL libraries...")
        opengl_works = False
        try:
            import ctypes
            ctypes.CDLL("libOpenGL.so.0")
            opengl_works = True
            print("[Install]   + libOpenGL.so.0 loaded successfully")
        except OSError as e:
            print(f"[Install]   x libOpenGL.so.0 failed to load: {e}")

        if installed_packages:
            print(f"[Install] Installed: {', '.join(installed_packages)}")

        if failed_packages:
            print(f"[Install] Failed to install: {', '.join(failed_packages)}")

        if critical_failed:
            print(f"[Install] ERROR: Critical packages failed to install: {', '.join(critical_failed)}")
            print(f"[Install] PyMeshLab remeshing will NOT work!")
            print(f"[Install] You may need to run manually:")
            print(f"[Install]   sudo apt-get install {' '.join(critical_failed)}")
            return False
        elif not opengl_works:
            print("[Install] ERROR: OpenGL libraries installed but cannot be loaded!")
            print("[Install] PyMeshLab remeshing will NOT work!")
            print("[Install] Try running: sudo ldconfig")
            return False
        else:
            print("[Install] OpenGL libraries installed and verified successfully!")
            return True

    except subprocess.TimeoutExpired:
        print("[Install] Warning: Installation timed out")
        print(f"[Install] You may need to run manually:")
        print(f"[Install]   sudo apt-get install libgl1 libopengl0 libglu1-mesa libglx-mesa0")
        return False
    except FileNotFoundError:
        print("[Install] Warning: apt-get not found (not a Debian/Ubuntu system?)")
        print("[Install] Please install OpenGL libraries manually for your distribution")
        return True
    except KeyboardInterrupt:
        print("\n[Install] Installation cancelled by user")
        print(f"[Install] You can install OpenGL libraries later with:")
        print(f"[Install]   sudo apt-get install libgl1 libopengl0 libglu1-mesa libglx-mesa0")
        return False
    except Exception as e:
        print(f"[Install] Warning: Could not install system dependencies: {e}")
        print(f"[Install] PyMeshLab remeshing may not work without OpenGL libraries.")
        print(f"[Install] To fix, run: sudo apt-get update && sudo apt-get install libgl1 libopengl0 libglu1-mesa libglx-mesa0")
        return False


def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Python Dependencies Installation")
    print("="*60 + "\n")

    script_dir = Path(__file__).parent.absolute()
    requirements_file = script_dir / "requirements.txt"

    if not requirements_file.exists():
        print(f"[Install] Warning: requirements.txt not found at {requirements_file}")
        print("[Install] Skipping Python dependencies installation.")
        return True

    print(f"[Install] Installing core Python dependencies...")
    print(f"[Install] This may take a few minutes...\n")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print("\n[Install] All Python dependencies installed successfully!")
            return True
        else:
            print(f"\n[Install] Warning: Some packages failed to install")
            print("[Install] Attempting to install core dependencies without optional packages...")

            result_without_optional = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "requests>=2.25.0", "tqdm>=4.60.0",
                 "numpy>=1.21.0", "scipy>=1.7.0",
                 "trimesh>=3.15.0", "pymeshlab>=2022.2",
                 "matplotlib>=3.5.0", "Pillow>=9.0.0",
                 "point-cloud-utils>=0.30.0",
                 "fast-simplification>=0.1.5",
                 "xatlas>=0.0.11",
                 "skeletor>=1.2.0"],
                capture_output=True,
                text=True,
                timeout=600
            )

            if result_without_optional.returncode == 0:
                print("\n[Install] Core Python dependencies installed successfully!")
                print("[Install] Note: Some optional packages (like cgal) may not be available")
                print("[Install] You can install them manually later if needed")
                return True
            else:
                print(f"\n[Install] Error installing Python dependencies:")
                print(result_without_optional.stderr)
                print("\n[Install] You can try installing manually with:")
                print(f"[Install]   pip install -r {requirements_file}")
                return False

    except subprocess.TimeoutExpired:
        print("\n[Install] Error: Installation timed out after 10 minutes")
        print("[Install] You can try installing manually with:")
        print(f"[Install]   pip install -r {requirements_file}")
        return False
    except Exception as e:
        print(f"\n[Install] Error installing Python dependencies: {e}")
        print("[Install] You can try installing manually with:")
        print(f"[Install]   pip install -r {requirements_file}")
        return False


def main():
    """Entry point."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Installation")
    print("="*60 + "\n")
    print("This installer will set up:")
    print("  1. System dependencies (OpenGL libraries on Linux)")
    print("  2. Python dependencies (trimesh, pymeshlab, etc.)")
    print("")
    print("Note: Blender is optional. If you need Blender nodes")
    print("(UV Unwrap, Remesh, etc.), run: python blender_install.py")
    print("")

    results = {
        'system_deps': False,
        'python_deps': False,
    }

    results['system_deps'] = install_system_dependencies()
    results['python_deps'] = install_python_dependencies()

    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    print(f"  System Dependencies: {'+ Success' if results['system_deps'] else 'x Failed'}")
    print(f"  Python Dependencies: {'+ Success' if results['python_deps'] else 'x Failed'}")
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
