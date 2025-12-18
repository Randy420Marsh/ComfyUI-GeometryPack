# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
System Dependencies Installation - OpenGL libraries for PyMeshLab.
"""

import platform
import subprocess


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
