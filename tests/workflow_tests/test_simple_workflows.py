# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 ComfyUI-GeometryPack Contributors

"""
Tests for simple workflows (2-3 nodes).

These tests execute actual workflow JSON files through the ComfyUI API
and validate their outputs.
"""

import pytest
from pathlib import Path
from .workflow_executor import WorkflowExecutor
from .mesh_validators import MeshValidator

# Path to workflows directory
WORKFLOWS_DIR = Path(__file__).parent.parent.parent / "workflows"

# Simple workflow files (2-3 nodes)
SIMPLE_WORKFLOWS = [
    "preview_mesh.json",
    "side_by_side_viewer.json",
    "transform.json",
]


@pytest.fixture(scope="module")
def workflow_executor():
    """
    Create workflow executor connected to ComfyUI server.

    Note: This requires a running ComfyUI server at 127.0.0.1:8188
    """
    try:
        executor = WorkflowExecutor()
        executor.connect()
        yield executor
        executor.disconnect()
    except Exception as e:
        pytest.skip(f"ComfyUI server not available: {e}")


@pytest.mark.workflow
@pytest.mark.workflow_simple
@pytest.mark.parametrize("workflow_file", SIMPLE_WORKFLOWS)
def test_simple_workflow_execution(workflow_executor, workflow_file):
    """
    Test that simple workflows execute successfully.

    Args:
        workflow_executor: WorkflowExecutor fixture
        workflow_file: Workflow JSON filename
    """
    workflow_path = WORKFLOWS_DIR / workflow_file

    if not workflow_path.exists():
        pytest.skip(f"Workflow file not found: {workflow_path}")

    # Execute workflow
    try:
        outputs = workflow_executor.execute_workflow_file(str(workflow_path), timeout=120)
    except Exception as e:
        pytest.fail(f"Workflow execution failed: {e}")

    # Validate outputs exist
    assert outputs is not None, "Workflow produced no outputs"
    assert len(outputs) > 0, "Workflow output dictionary is empty"


@pytest.mark.workflow
@pytest.mark.workflow_simple
def test_preview_mesh_workflow(workflow_executor):
    """
    Test preview_mesh.json workflow specifically.

    Workflow: LoadMesh → PreviewMesh
    Expected: Mesh loads and preview node executes
    """
    workflow_path = WORKFLOWS_DIR / "preview_mesh.json"

    if not workflow_path.exists():
        pytest.skip(f"Workflow file not found: {workflow_path}")

    outputs = workflow_executor.execute_workflow_file(str(workflow_path), timeout=60)

    # Check outputs exist
    assert outputs is not None
    assert len(outputs) > 0, "No outputs from preview_mesh workflow"

    # Note: Preview nodes typically don't return mesh data in outputs,
    # they just display. Success here means no errors during execution.


@pytest.mark.workflow
@pytest.mark.workflow_simple
def test_transform_workflow(workflow_executor):
    """
    Test transform.json workflow specifically.

    Workflow: CreatePrimitive → TransformMesh → PreviewMeshDual
    Expected: Primitive created, transformed, and previewed
    """
    workflow_path = WORKFLOWS_DIR / "transform.json"

    if not workflow_path.exists():
        pytest.skip(f"Workflow file not found: {workflow_path}")

    outputs = workflow_executor.execute_workflow_file(str(workflow_path), timeout=60)

    # Check outputs exist
    assert outputs is not None
    assert len(outputs) > 0, "No outputs from transform workflow"


@pytest.mark.workflow
@pytest.mark.workflow_simple
def test_side_by_side_viewer_workflow(workflow_executor):
    """
    Test side_by_side_viewer.json workflow specifically.

    Workflow: CreatePrimitive → PreviewMeshDual
    Expected: Primitive created and displayed in dual viewer
    """
    workflow_path = WORKFLOWS_DIR / "side_by_side_viewer.json"

    if not workflow_path.exists():
        pytest.skip(f"Workflow file not found: {workflow_path}")

    outputs = workflow_executor.execute_workflow_file(str(workflow_path), timeout=60)

    # Check outputs exist
    assert outputs is not None
    assert len(outputs) > 0, "No outputs from side_by_side_viewer workflow"


# Validation helper tests (can be used for debugging)
@pytest.mark.unit
def test_mesh_validator_basic():
    """Test MeshValidator basic functionality."""
    import trimesh
    import numpy as np

    # Create a simple valid mesh
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Should pass validation
    is_valid, error = MeshValidator.is_valid_mesh(mesh)
    assert is_valid, f"Valid mesh failed validation: {error}"

    # Check vertex count
    is_valid, error = MeshValidator.check_vertex_count(mesh, min_vertices=3, max_vertices=10)
    assert is_valid, f"Vertex count check failed: {error}"

    # Check face count
    is_valid, error = MeshValidator.check_face_count(mesh, min_faces=1, max_faces=10)
    assert is_valid, f"Face count check failed: {error}"


@pytest.mark.unit
def test_mesh_validator_invalid():
    """Test MeshValidator with invalid meshes."""
    import trimesh
    import numpy as np

    # Create a valid mesh first
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Now manually inject NaN (trimesh may clean it during construction)
    mesh.vertices[1, 0] = np.nan

    is_valid, error = MeshValidator.is_valid_mesh(mesh)
    assert not is_valid, "Mesh with NaN vertices should fail validation"
    assert "NaN" in error


@pytest.mark.unit
def test_workflow_converter_import():
    """Test that WorkflowConverter can be imported."""
    from .workflow_converter import WorkflowConverter

    assert WorkflowConverter is not None
    assert hasattr(WorkflowConverter, 'convert_to_api_format')
    assert hasattr(WorkflowConverter, 'load_workflow')
