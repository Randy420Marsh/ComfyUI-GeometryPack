# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Workflow-based testing infrastructure for ComfyUI-GeometryPack.

This package provides tools for executing ComfyUI workflow JSON files
via the ComfyUI API and validating their outputs.
"""

from .workflow_executor import WorkflowExecutor
from .mesh_validators import MeshValidator

__all__ = ['WorkflowExecutor', 'MeshValidator']
