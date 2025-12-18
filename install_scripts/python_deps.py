# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Python Dependencies Installation - pip requirements for GeometryPack.
"""

import sys
import subprocess
from pathlib import Path


def install_python_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n" + "="*60)
    print("ComfyUI-GeometryPack: Python Dependencies Installation")
    print("="*60 + "\n")

    # Look for requirements.txt relative to this script's parent directory
    script_dir = Path(__file__).parent.parent.absolute()
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
                 "skeletor>=1.2.0",
                 "libigl>=2.6.1"],
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
