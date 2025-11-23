# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Mesh validation utilities for workflow testing.

Provides validators to check mesh outputs from workflow execution.
"""

import numpy as np
import trimesh


class MeshValidator:
    """Validates mesh geometry and topology."""

    @staticmethod
    def is_valid_mesh(mesh):
        """
        Check if a mesh has valid basic geometry.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            tuple: (is_valid, error_message)
        """
        if not isinstance(mesh, trimesh.Trimesh):
            return False, f"Not a Trimesh object: {type(mesh)}"

        if len(mesh.vertices) == 0:
            return False, "Mesh has no vertices"

        if len(mesh.faces) == 0:
            return False, "Mesh has no faces"

        if np.any(np.isnan(mesh.vertices)):
            return False, "Mesh contains NaN vertices"

        if np.any(np.isinf(mesh.vertices)):
            return False, "Mesh contains infinite vertices"

        return True, None

    @staticmethod
    def check_vertex_count(mesh, min_vertices=3, max_vertices=None):
        """
        Check if mesh has acceptable vertex count.

        Args:
            mesh: trimesh.Trimesh object
            min_vertices: Minimum vertex count
            max_vertices: Maximum vertex count (None = no limit)

        Returns:
            tuple: (is_valid, error_message)
        """
        count = len(mesh.vertices)

        if count < min_vertices:
            return False, f"Too few vertices: {count} < {min_vertices}"

        if max_vertices and count > max_vertices:
            return False, f"Too many vertices: {count} > {max_vertices}"

        return True, None

    @staticmethod
    def check_face_count(mesh, min_faces=1, max_faces=None):
        """
        Check if mesh has acceptable face count.

        Args:
            mesh: trimesh.Trimesh object
            min_faces: Minimum face count
            max_faces: Maximum face count (None = no limit)

        Returns:
            tuple: (is_valid, error_message)
        """
        count = len(mesh.faces)

        if count < min_faces:
            return False, f"Too few faces: {count} < {min_faces}"

        if max_faces and count > max_faces:
            return False, f"Too many faces: {count} > {max_faces}"

        return True, None

    @staticmethod
    def check_manifold(mesh):
        """
        Check if mesh is manifold (watertight, no edges shared by >2 faces).

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            tuple: (is_manifold, error_message)
        """
        try:
            is_manifold = mesh.is_watertight
            if not is_manifold:
                return False, "Mesh is not watertight/manifold"
            return True, None
        except Exception as e:
            return False, f"Error checking manifold: {e}"

    @staticmethod
    def check_bounding_box(mesh, expected_min=None, expected_max=None, tolerance=0.1):
        """
        Check if mesh bounding box is within expected range.

        Args:
            mesh: trimesh.Trimesh object
            expected_min: Expected minimum bounds (x, y, z)
            expected_max: Expected maximum bounds (x, y, z)
            tolerance: Tolerance for comparison

        Returns:
            tuple: (is_valid, error_message)
        """
        bounds = mesh.bounds
        actual_min = bounds[0]
        actual_max = bounds[1]

        if expected_min is not None:
            diff = np.abs(actual_min - np.array(expected_min))
            if np.any(diff > tolerance):
                return False, f"Bounds min mismatch: {actual_min} vs {expected_min}"

        if expected_max is not None:
            diff = np.abs(actual_max - np.array(expected_max))
            if np.any(diff > tolerance):
                return False, f"Bounds max mismatch: {actual_max} vs {expected_max}"

        return True, None

    @staticmethod
    def check_has_uv_coordinates(mesh):
        """
        Check if mesh has UV coordinates.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            tuple: (has_uvs, error_message)
        """
        if not hasattr(mesh.visual, 'uv'):
            return False, "Mesh has no UV coordinates"

        if mesh.visual.uv is None or len(mesh.visual.uv) == 0:
            return False, "Mesh UV coordinates are empty"

        return True, None

    @staticmethod
    def check_vertex_attributes(mesh, expected_attributes):
        """
        Check if mesh has expected vertex attributes.

        Args:
            mesh: trimesh.Trimesh object
            expected_attributes: List of attribute names

        Returns:
            tuple: (has_attributes, error_message)
        """
        if not hasattr(mesh, 'vertex_attributes'):
            return False, "Mesh has no vertex_attributes"

        for attr in expected_attributes:
            if attr not in mesh.vertex_attributes:
                return False, f"Missing vertex attribute: {attr}"

        return True, None

    @classmethod
    def validate_all(cls, mesh, checks=None):
        """
        Run all specified validation checks on a mesh.

        Args:
            mesh: trimesh.Trimesh object
            checks: List of check names to run (None = run basic checks)

        Returns:
            tuple: (all_valid, list of error messages)
        """
        if checks is None:
            checks = ['is_valid_mesh', 'check_vertex_count', 'check_face_count']

        errors = []

        for check_name in checks:
            if not hasattr(cls, check_name):
                errors.append(f"Unknown check: {check_name}")
                continue

            check_method = getattr(cls, check_name)

            try:
                # Call with just mesh for basic checks
                is_valid, error = check_method(mesh)
                if not is_valid:
                    errors.append(f"{check_name}: {error}")
            except TypeError:
                # Check requires additional parameters, skip
                continue
            except Exception as e:
                errors.append(f"{check_name} raised exception: {e}")

        return len(errors) == 0, errors
